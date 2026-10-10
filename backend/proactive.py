"""
Proactive Suggestion Engine — the Mark VII principle.

JARVIS launched the suit before Tony asked. This engine watches quiet signals
(reminder deadlines, time-of-day transitions, session fatigue, finished
dossiers) and lets Ada interject with ONE short spoken line when a signal
crosses a threshold — never nagging, never during an active conversation.

Design rules learned from the films:
  - Anticipate, don't interrupt: she only speaks when Sir is idle.
  - Every signal has its own cooldown so nothing repeats; urgent reminders
    (due within 2 minutes, or just overdue) may bypass the hourly ceiling
    ONCE each — a missed meeting is worse than a fourth line in an hour.
  - Global rate ceiling per hour for everythingROUTINE; quiet hours are
    sacred except for urgent reminders, which are whispered through.
  - Signals are priority-ordered: overdue > due-now > dossier > morning
    briefing > clock transitions > fatigue. First signal that clears its
    gate wins the cycle — at most one interjection per cycle.
  - No CAD signals — that domain is explicitly off-limits.
"""
import time
import asyncio
import os
from datetime import datetime


# A reminder due within this window is URGENT: it bypasses the hourly ceiling
# once (tracked per reminder id) and may speak during quiet hours.
URGENT_DUE_S = 120
# A reminder past due within this window escalates once instead of nagging.
OVERDUE_ESCALATE_S = 1800
# Morning briefing window: first cycle after this hour briefs once per day.
MORNING_BRIEF_HOUR = 7


class ProactiveEngine:
    def __init__(self, kernel, speak_fn, interval_s=180.0,
                 quiet_hours=(23, 8), max_per_hour=3, per_key_cooldown_s=4 * 3600):
        self.kernel = kernel
        self.speak_fn = speak_fn          # async: line -> bool (False = skipped)
        # Env-tunable (ADA_PROACTIVE_INTERVAL_S). The cycle itself is local-only
        # (scheduler/dossier/clock checks, no model calls), so the default stays
        # 180s — speaking is still gated by cooldowns + hourly ceiling.
        self.interval_s = float(os.getenv("ADA_PROACTIVE_INTERVAL_S", interval_s))
        self.quiet_hours = quiet_hours    # (start_hour, end_hour) exclusive-ish
        self.max_per_hour = max_per_hour
        self.per_key_cooldown_s = per_key_cooldown_s
        # Per-signal cooldowns: reminders re-arm faster than lunch nudges.
        self.cooldowns = {
            "reminder": 30 * 60,
            "overdue": 60 * 60,
            "dossier": 6 * 3600,
            "briefing": 20 * 3600,
            "lunch": 20 * 3600,
            "wind_down": 20 * 3600,
            "fatigue": 6 * 3600,
            "break": 3 * 3600,
            # Event-driven signals (demand detection): errors re-arm in 2h so
            # a genuinely stuck loop gets a second offer; files pace per item;
            # pace/due risks speak once per day — prophecy, not nagging.
            "errrepeat": 2 * 3600,
            "filelanded": 6 * 3600,
            "focus": 20 * 3600,
            "pace": 20 * 3600,
            "due": 20 * 3600,
        }
        self._last_fired = {}             # signal_key -> ts
        self._hour_stamps = []            # ts of recent interjections
        self._urgent_spent = set()        # reminder ids that used the urgent bypass
        self._briefed_day = ""            # YYYY-MM-DD of last morning briefing
        self._task = None
        self._stop = asyncio.Event()
        self._boot_ts = time.time()
        self._announced_dossier_ts = None

    # ── signal checks (each returns a spoken line or None) ───────────────────
    def _reminder_warnings(self):
        """Upcoming reminders within 15 minutes -> a heads-up now."""
        try:
            now = time.time()
            for item in self.kernel.scheduler.list():
                nr = float(item.get("next_run") or 0)
                lead = nr - now
                if 0 < lead <= 900:
                    descr = (item.get("descr") or "something")[:60]
                    mins = max(1, round(lead / 60))
                    when = "about a minute" if mins <= 1 else f"about {mins} minutes"
                    return (f"reminder_{item.get('id')}",
                            f"[emotion:thoughtful] Sir, a gentle heads-up: {descr} is due {when} from now.")
        except Exception:
            pass
        return None

    def _reminder_overdue(self):
        """A reminder that slipped past due recently -> escalate ONCE, then stop.

        Nagging every cycle about the same overdue item is how assistants get
        muted. One escalation; after that the scheduler owns it."""
        try:
            now = time.time()
            for item in self.kernel.scheduler.list():
                nr = float(item.get("next_run") or 0)
                late = now - nr
                if 0 < late <= OVERDUE_ESCALATE_S:
                    key = f"overdue_{item.get('id')}"
                    if key in self._last_fired:
                        continue
                    descr = (item.get("descr") or "something")[:60]
                    mins = max(1, round(late / 60))
                    return (key,
                            f"[emotion:careful] Sir — {descr} slipped past due by about "
                            f"{mins} minute{'s' if mins != 1 else ''}. Shall I hold it over or stand it down?")
        except Exception:
            pass
        return None

    def _due_within(self, key: str, window_s: float) -> bool:
        try:
            rid = key.split("_", 1)[1]
            now = time.time()
            for item in self.kernel.scheduler.list():
                if str(item.get("id")) == rid:
                    return 0 < float(item.get("next_run") or 0) - now <= window_s
        except Exception:
            pass
        return False

    def _time_transitions(self):
        """Once-a-day nudges tied to the clock: lunch, and evening wind-down."""
        now = datetime.now()
        key, line = None, None
        if now.hour == 13 and now.minute < self.interval_s // 60 + 1:
            key, line = "lunch", ("[emotion:warm] It's past one, Sir. You haven't stopped since "
                                  "morning — shall I hold the fort while you eat?")
        elif now.hour == 22 and now.minute < self.interval_s // 60 + 1:
            key, line = "wind_down", ("[emotion:tender] It's ten, Sir. Anything still running can "
                                      "wait for morning. I'll keep watch either way.")
        return (key, line) if key else None

    def _session_fatigue(self):
        """Long unbroken session -> one hydration/stretch nudge."""
        uptime_h = (time.time() - self._boot_ts) / 3600
        if uptime_h > 4:
            return ("fatigue", "[emotion:careful] Sir, we've been at this over four hours. "
                               "Even I recharge — a five minute break keeps your mind sharp.")
        return None

    def _dossier_ready(self):
        """Overnight research finished -> announce it once."""
        ready = getattr(self.kernel, "dossier", None)
        if ready is None:
            return None
        topic, ts = ready.ready_unannounced()
        if topic:
            self._announced_dossier_ts = ts
            return ("dossier", f"[emotion:confident] Sir, my overnight research on {topic} is "
                               f"complete — the full brief is in your notebook. Shall I give you the short version?")
        return None

    def _morning_briefing(self):
        """First cycle of the day after MORNING_BRIEF_HOUR: ONE combined line —
        schedule load, overnight dossier, sleep-debt note — instead of three
        separate interjections competing for the hourly budget.

        Yash directive (2026-10-02): NO canned briefs, NO pre-seeded
        questions. Openings come naturally from what's live. This signal
        stays OFF unless ADA_MORNING_BRIEF=1 is set explicitly."""
        if os.getenv("ADA_MORNING_BRIEF", "0") != "1":
            return None
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        if now.hour < MORNING_BRIEF_HOUR or self._briefed_day == today:
            return None
        bits = []
        try:
            upcoming = [i for i in self.kernel.scheduler.list()
                        if float(i.get("next_run") or 0) > time.time()]
            if upcoming:
                n = len(upcoming)
                bits.append(f"{n} thing{'s' if n != 1 else ''} on the schedule")
        except Exception:
            pass
        dossier = getattr(self.kernel, "dossier", None)
        try:
            # Peek at the state WITHOUT consuming the announcement — the
            # dossier signal delivers it next cycle so each gets its own turn.
            state = getattr(dossier, "_state", None) or {}
            if dossier is not None and state.get("status") == "ready" \
                    and not state.get("announced"):
                bits.append("overnight research is ready")
        except Exception:
            pass
        wb = getattr(self.kernel, "wellbeing", None)
        try:
            if wb is not None and getattr(wb, "late_night_active_min", 0) >= 45:
                bits.append("you were up late — I'll keep today light where I can")
        except Exception:
            pass
        self._briefed_day = today
        if not bits:
            return None
        return ("briefing",
                "[emotion:warm] Good morning, Sir. " +
                "; ".join(bits)[:220] + ".")

    # ── event-driven demand signals (PASK-style, proof-gated) ──────────
    def _signal_events(self):
        """One event signal per cycle, highest proof first.

        Sources: error repeats (signal bus), landed CAD files (bus),
        long focus (task board), deadline/pace risk (world model).
        A signal without proof NEVER speaks — it stays in the log.
        """
        cands = []
        try:
            bus = getattr(getattr(self, "kernel", None), "signals", None)
            if bus is not None:
                for fn in (bus.error_repeat, bus.file_landed):
                    try:
                        r = fn()
                    except Exception:
                        r = None
                    if r and len(r) == 3:
                        cands.append(r)
                try:
                    board = getattr(getattr(self, "kernel", None),
                                    "task_board_ref", None) or {}
                    r = bus.long_focus(board)
                    if r and len(r) == 3:
                        cands.append(r)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            world = getattr(getattr(self, "kernel", None), "world", None)
            if world is not None:
                for key, line in world.watch():
                    cands.append((key, line, f"world model: {key}"))
        except Exception:
            pass
        if not cands:
            return None
        key, line, proof = cands[0]
        try:
            print(f"[PROACTIVE] signal proof: {proof}")
        except Exception:
            pass
        return (key, line)

    # ── gating ────────────────────────────────────────────────────────────────
    def _in_quiet_hours(self):
        h = datetime.now().hour
        start, end = self.quiet_hours
        return h >= start or h < end if start > end else start <= h < end

    def _cooldown_for(self, key: str) -> float:
        for prefix, cd in self.cooldowns.items():
            if key == prefix or key.startswith(prefix + "_"):
                return cd
        return self.per_key_cooldown_s

    def _gate(self, key, urgent: bool = False):
        now = time.time()
        if now - self._last_fired.get(key, 0) < self._cooldown_for(key):
            return False
        self._hour_stamps = [t for t in self._hour_stamps if now - t < 3600]
        if len(self._hour_stamps) >= self.max_per_hour:
            # Urgent reminders get ONE bypass each; everything else waits.
            if urgent and key not in self._urgent_spent:
                self._urgent_spent.add(key)
                return True
            return False
        return True

    def _urgent_bypass_available(self, key: str) -> bool:
        """A reminder due within URGENT_DUE_S earns the bypass; overdue
        escalation already consumed its bypass when it was created."""
        if key.startswith("overdue_"):
            return True
        return self._due_within(key, URGENT_DUE_S)

    # ── main loop ─────────────────────────────────────────────────────────────
    def _prioritised_checks(self):
        """Highest-stakes signals first. First gate that clears wins the cycle."""
        checks = (self._reminder_overdue, self._reminder_warnings,
                  self._signal_events,
                  self._dossier_ready, self._morning_briefing,
                  self._time_transitions, self._session_fatigue)
        # Wellbeing signals (Blood Toxicity principle) ride the same loop
        wb = getattr(self.kernel, "wellbeing", None)
        if wb is not None:
            checks = checks + (wb.break_nudge,)
        return checks

    async def _cycle(self):
        for check in self._prioritised_checks():
            result = check()
            if not result:
                continue
            key, line = result
            urgent = (key.startswith("overdue_") or
                      (key.startswith("reminder_") and self._due_within(key, URGENT_DUE_S)))
            if not self._gate(key, urgent=urgent and self._urgent_bypass_available(key)):
                continue
            spoke = await self.speak_fn(line)
            if spoke:
                self._last_fired[key] = time.time()
                self._hour_stamps.append(time.time())
                print(f"[PROACTIVE] {key}: {line[:70]}")
            return  # at most one interjection per cycle

    async def _quiet_urgent_cycle(self):
        """During quiet hours only urgent reminders may whisper through —
        everything else waits for morning. A missed 2am medication reminder
        matters; a lunch nudge at 2am does not."""
        for check in (self._reminder_overdue, self._reminder_warnings):
            result = check()
            if not result:
                continue
            key, line = result
            if not (key.startswith("overdue_") or self._due_within(key, URGENT_DUE_S)):
                continue
            if not self._gate(key, urgent=self._urgent_bypass_available(key)):
                continue
            spoke = await self.speak_fn(line)
            if spoke:
                self._last_fired[key] = time.time()
                self._hour_stamps.append(time.time())
                print(f"[PROACTIVE] {key} (quiet-hours urgent): {line[:70]}")
            return

    async def _loop(self):
        await asyncio.sleep(self.interval_s)   # settle after boot
        while not self._stop.is_set():
            try:
                if not self._in_quiet_hours():
                    await self._cycle()
                else:
                    await self._quiet_urgent_cycle()
            except Exception as e:
                print(f"[PROACTIVE] cycle failed: {e}")
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
