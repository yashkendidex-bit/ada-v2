"""
Proactive Suggestion Engine — the Mark VII principle, policy-driven edition.

What changed (Feature 8): the fixed 180s timer and the fixed per-key
cooldowns / hourly ceiling no longer decide send-vs-suppress. They are
candidate plumbing now. The DECISION lives in `proactivity_policy`:
a Horvitz-style heuristic (v1, active day one) that blends into
Beta-Binomial Thompson sampling per (suggestion_type x context_bucket)
(v2, once a bucket has >= 5 samples). Every decision and every observed
reaction is logged in SQLite, so the policy learns when to stay silent —
the learned policy keeps discovering that "send nothing" is optimal.

The loop is INVERTED: SignalBus detections wake the evaluation
immediately (subscribe, in the F1 runtime style — guarded, works
standalone). A lengthened poll (600s) remains as backstop for
time-based checks only (reminders, clock transitions, dossier, briefing).

Hard rails (never overridden): active voice turn, another human voice
present (speaker_present() hook, Feature 7 fills it), hush cooldown,
quiet hours (fixed window prior + learned per-hour soft-quiet),
max 3 proactive messages/day, per-key cooldowns.

Voice-direct UX only — no popups. Proactive speech goes through
the existing `speak_fn`, which still refuses mid-conversation.
"""
import time
import asyncio
import os
from datetime import datetime

from proactivity_policy import (
    InterruptionPolicy, PolicyStore, RELEVANCE,
    REACTION_ENGAGED, REACTION_DISMISSED, REACTION_IGNORED, REACTION_SNOOZED,
    speaker_present as _speaker_present_hook,
    DEFAULT_THRESHOLD, DEFAULT_MAX_PER_DAY, FEEDBACK_WINDOW_S,
    HUSH_COOLDOWN_S,
)

# A reminder due within this window is URGENT: it bypasses quiet hours, the
# daily cap and cooldowns ONCE each (tracked per reminder id) — a missed
# reminder is worse than a fourth line in a day. NEVER bypasses the
# voice-turn or speaker rails.
URGENT_DUE_S = 120
# A reminder past due within this window escalates once instead of nagging.
OVERDUE_ESCALATE_S = 1800
# Morning briefing window: first cycle after this hour briefs once per day.
MORNING_BRIEF_HOUR = 7
# Backstop poll for time-based checks only (signal events wake us sooner).
BACKSTOP_POLL_S = 600.0


class ProactiveEngine:
    def __init__(self, kernel, speak_fn, interval_s=BACKSTOP_POLL_S,
                 quiet_hours=(23, 8), max_per_day=DEFAULT_MAX_PER_DAY,
                 threshold=DEFAULT_THRESHOLD,
                 per_key_cooldown_s=4 * 3600,
                 voice_turn_active_fn=None, speaker_present_fn=None,
                 db_path=None, policy=None, **_legacy):
        self.kernel = kernel
        self.speak_fn = speak_fn          # async: line -> bool (False = skipped)
        # Env-tunable (ADA_PROACTIVE_INTERVAL_S). This is now only the
        # backstop poll for time-based checks; signal events wake the loop.
        self.interval_s = float(os.getenv("ADA_PROACTIVE_INTERVAL_S", interval_s))
        self.quiet_hours = quiet_hours    # (start_hour, end_hour) — the prior
        self.threshold = threshold
        # Learned policy (v1 heuristic -> v2 Thompson sampling).
        self.policy = policy or InterruptionPolicy(
            store=PolicyStore(db_path=db_path),
            threshold=threshold, max_per_day=max_per_day)
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
        self._last_sent_ts = 0.0          # ts of last actual interjection
        self._urgent_spent = set()        # reminder ids that used the urgent bypass
        self._briefed_day = ""            # YYYY-MM-DD of last morning briefing
        self._task = None
        self._stop = asyncio.Event()
        self._wake = asyncio.Event()      # set by SignalBus detections
        self._boot_ts = time.time()
        self._announced_dossier_ts = None
        self._hush_until = 0.0            # hush cooldown expiry
        self._last_sent_iid = None        # interruption id of the last send
        self._last_sent_at = 0.0
        self._voice_turn_active_fn = voice_turn_active_fn
        self._speaker_present_fn = speaker_present_fn
        # F1-runtime-style subscription to the SignalBus (guarded; standalone OK).
        self._subscribed_bus = None
        try:
            self.attach_signal_bus()
        except Exception:
            pass

    # ── signal checks (each returns a spoken line or None) ───────────────────
    # These are CANDIDATE generators now — the policy decides send/suppress.
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
        long focus (task board), deadline/pace risk (F3 world_risk_signals).
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
                        cands.append((r[0], r[1], r[2], "signal"))
                try:
                    board = getattr(getattr(self, "kernel", None),
                                    "task_board_ref", None) or {}
                    r = bus.long_focus(board)
                    if r and len(r) == 3:
                        cands.append((r[0], r[1], r[2], "signal"))
                except Exception:
                    pass
        except Exception:
            pass
        # F3 world model: structured deadline/deviation signals as candidates.
        try:
            world = getattr(getattr(self, "kernel", None), "world", None)
            if world is not None and hasattr(world, "world_risk_signals"):
                for sig in world.world_risk_signals():
                    if not isinstance(sig, dict):
                        continue
                    sid = str(sig.get("id", "world_risk"))
                    text = str(sig.get("text", ""))[:300]
                    if not text:
                        continue
                    sev = str(sig.get("severity", "warn"))
                    proof = (f"world risk: {sig.get('kind', 'info')}/"
                             f"{sig.get('project', '')} ({sev})")
                    cands.append((sid, f"[emotion:thoughtful] {text}",
                                  proof, "world_risk"))
        except Exception:
            pass
        if not cands:
            return None
        key, line, proof, stype = cands[0]
        try:
            print(f"[PROACTIVE] signal proof: {proof}")
        except Exception:
            pass
        return (key, line, stype)

    # ── signal-bus wake subscription (F1 runtime style, guarded) ─────────────
    def attach_signal_bus(self, bus=None):
        """Subscribe to SignalBus detections so they WAKE the evaluation
        instead of waiting for the poll. Safe to call twice; no-op if the
        bus is missing or has no subscribe hook."""
        if self._subscribed_bus is not None:
            return
        try:
            if bus is None:
                bus = getattr(getattr(self, "kernel", None), "signals", None)
            if bus is None or not hasattr(bus, "subscribe"):
                return
            bus.subscribe(self._on_signal_event)
            self._subscribed_bus = bus
        except Exception:
            pass

    def _on_signal_event(self, detection):
        """A SignalBus detection fired — wake the loop (guarded: the loop
        may not exist yet, or no loop may be running in tests)."""
        try:
            self._wake.set()
        except Exception:
            pass

    def wake(self):
        """External wake (tests, runtime hooks)."""
        self._on_signal_event(None)

    # ── rails ───────────────────────────────────────────────────────────────
    def _in_quiet_hours(self):
        h = datetime.now().hour
        start, end = self.quiet_hours
        return h >= start or h < end if start > end else start <= h < end

    def _cooldown_for(self, key: str) -> float:
        for prefix, cd in self.cooldowns.items():
            if key == prefix or key.startswith(prefix + "_"):
                return cd
        return self.per_key_cooldown_s

    def _voice_turn_active(self) -> bool:
        """NEVER interrupt an active voice turn. Prefer the injected probe;
        fall back to probing the kernel for an audio-loop-style object."""
        try:
            if self._voice_turn_active_fn is not None:
                return bool(self._voice_turn_active_fn())
        except Exception:
            pass
        try:
            for attr in ("audio_loop", "ada", "audio"):
                al = getattr(self.kernel, attr, None)
                if al is None:
                    continue
                turn = getattr(al, "_turn_task", None)
                if turn is not None and not getattr(turn, "done", lambda: True)():
                    return True
                lock = getattr(al, "_turn_lock", None)
                if lock is not None and getattr(lock, "locked", lambda: False)():
                    return True
        except Exception:
            pass
        return False

    def _speaker_present(self) -> bool:
        """NEVER interrupt when another human voice is in the room.
        Feature 7 wires real diarization; until then the hook is False."""
        try:
            if self._speaker_present_fn is not None:
                return bool(self._speaker_present_fn())
        except Exception:
            pass
        try:
            return bool(_speaker_present_hook())
        except Exception:
            return False

    def _urgent_bypass_available(self, key: str) -> bool:
        """A reminder due within URGENT_DUE_S earns the bypass; overdue
        escalation already consumed its bypass when it was created."""
        if key.startswith("overdue_"):
            return True
        return self._due_within(key, URGENT_DUE_S)

    def _context_for(self, stype: str, key: str, urgent: bool = False) -> dict:
        msl = None
        if self._last_sent_ts:
            msl = (time.time() - self._last_sent_ts) / 60.0
        now = time.time()
        cooldown_active = (now - self._last_fired.get(key, 0)
                           < self._cooldown_for(key))
        if urgent and key not in self._urgent_spent:
            cooldown_active = False  # urgent bypass spends ONCE below
        return {
            "suggestion_type": stype,
            "relevance": RELEVANCE.get(stype, 0.5),
            "urgent": urgent,
            "voice_turn_active": self._voice_turn_active(),
            "speaker_present": self._speaker_present(),
            "hush_active": time.time() < self._hush_until,
            "quiet_hour": self._in_quiet_hours(),
            "cooldown_active": cooldown_active,
            "minutes_since_last": msl,
            "focus": False,       # filled by focus probes where available
            "typing": False,      # local activity signals; filled when wired
            "calendar_busy": False,
        }

    # ── the policy decision ─────────────────────────────────────────────────
    def _decide(self, key: str, stype: str, urgent: bool):
        ctx = self._context_for(stype, key, urgent=urgent)
        decision = self.policy.decide(ctx)
        if urgent and decision["action"] == "sent":
            self._urgent_spent.add(key)   # the bypass is spent exactly once
        self._emit_decision_event(key, stype, decision, ctx)
        return decision

    def _emit_decision_event(self, key: str, stype: str, decision: dict,
                             ctx: dict):
        """Every send/suppress decision lands on the F4 perception timeline —
        metadata only (the 'suppressed 4 suggestions' trust feed needs this)."""
        try:
            from perception import emit as _tline_emit
            _tline_emit("agent", "policy.decision",
                        {"key": key, "suggestion_type": stype,
                         "action": decision["action"],
                         "reason": decision["reason"],
                         "score": decision["score"],
                         "policy": decision["policy"],
                         "bucket": decision.get("bucket"),
                         "urgent": bool(ctx.get("urgent")),
                         "hour": datetime.now().hour})
        except Exception:
            pass

    async def _maybe_speak(self, key: str, line: str, stype: str) -> bool:
        urgent = (key.startswith("overdue_") or
                  (key.startswith("reminder_")
                   and self._due_within(key, URGENT_DUE_S)))
        decision = self._decide(key, stype, urgent)
        if decision["action"] != "sent":
            try:
                print(f"[PROACTIVE] suppressed {key}: {decision['reason']} "
                      f"(score={decision['score']}, policy={decision['policy']})")
            except Exception:
                pass
            return False
        spoke = await self.speak_fn(line)
        if spoke:
            now = time.time()
            self._last_fired[key] = now
            self._last_sent_ts = now
            self._last_sent_at = now
            self._last_sent_iid = decision["iid"]
            try:
                print(f"[PROACTIVE] {key}: {line[:70]}")
            except Exception:
                pass
            try:
                from perception import emit as _tline_emit
                _tline_emit("agent", "agent.interjection",
                            {"said": line[:300], "why": key,
                             "policy": decision["policy"],
                             "score": decision["score"],
                             "interruption_id": decision["iid"]})
            except Exception:
                pass
        return bool(spoke)

    # ── main loop (event-woken, poll as backstop) ────────────────────────────
    def _prioritised_checks(self):
        """Highest-stakes signals first. First candidate that the policy
        clears wins the cycle — at most one interjection per cycle."""
        checks = (self._reminder_overdue, self._reminder_warnings,
                  self._signal_events,
                  self._dossier_ready, self._morning_briefing,
                  self._time_transitions, self._session_fatigue)
        # Wellbeing signals (Blood Toxicity principle) ride the same loop
        wb = getattr(self.kernel, "wellbeing", None)
        if wb is not None:
            checks = checks + (wb.break_nudge,)
        return checks

    @staticmethod
    def _stype_for_check(check_fn) -> str:
        name = getattr(check_fn, "__name__", "")
        return {
            "_reminder_overdue": "overdue",
            "_reminder_warnings": "reminder",
            "_signal_events": "signal",
            "_dossier_ready": "dossier",
            "_morning_briefing": "briefing",
            "_time_transitions": "signal",
            "_session_fatigue": "fatigue",
        }.get(name, "signal")

    async def _cycle(self):
        for check in self._prioritised_checks():
            result = check()
            if not result:
                continue
            if len(result) == 3:          # event signals carry their stype
                key, line, stype = result
            else:
                key, line = result
                stype = self._stype_for_check(check)
            # Time-transition keys carry their real type in the key.
            if stype == "signal" and key in ("lunch", "wind_down"):
                stype = key
            await self._maybe_speak(key, line, stype)
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
            stype = self._stype_for_check(check)
            if not (key.startswith("overdue_")
                    or self._due_within(key, URGENT_DUE_S)):
                continue
            await self._maybe_speak(key, line, stype)
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
                # Event-woken: signal detections interrupt the wait.
                # Backstop: the timeout keeps time-based checks alive.
                self._wake.clear()
                stop_task = asyncio.ensure_future(self._stop.wait())
                wake_task = asyncio.ensure_future(self._wake.wait())
                try:
                    await asyncio.wait_for(
                        asyncio.wait((stop_task, wake_task),
                                     return_when=asyncio.FIRST_COMPLETED),
                        timeout=self.interval_s)
                except asyncio.TimeoutError:
                    pass
                finally:
                    for _t in (stop_task, wake_task):
                        if not _t.done():
                            _t.cancel()
            except asyncio.TimeoutError:
                pass

    def start(self):
        if self._task is None or self._task.done():
            self._stop.clear()
            self.attach_signal_bus()
            self._task = asyncio.create_task(self._loop())

    async def stop(self):
        self._stop.set()
        if self._task:
            self._task.cancel()

    # ── reactions: the learning loop ────────────────────────────────────────
    def log_reaction(self, interruption_id: str, reaction: str,
                     latency_s: float | None = None) -> bool:
        """Public API: record engaged|dismissed|ignored|snoozed for a
        logged interruption. Updates the bucket's alpha/beta and the
        learned per-hour dismissal rates."""
        try:
            return bool(self.policy.log_reaction(interruption_id, reaction,
                                                 latency_s))
        except Exception:
            return False

    def record_feedback(self, reaction: str) -> bool:
        """Attribute a reaction to the most recent send (within the feedback
        window) — for observe_vote, hush, and 'not now' voice replies that
        arrive without an interruption id."""
        try:
            if (not self._last_sent_iid
                    or time.time() - self._last_sent_at > FEEDBACK_WINDOW_S):
                return False
            latency = time.time() - self._last_sent_at
            row = self.policy.store._conn.execute(
                "SELECT reaction FROM interruptions WHERE id=?",
                (self._last_sent_iid,)).fetchone()
            if row is None or row[0] != "none":
                return False
            return self.log_reaction(self._last_sent_iid, reaction, latency)
        except Exception:
            return False

    def on_hush(self) -> None:
        """'Hush' (button, 'not now', shush): the last line was a dismissal,
        and no new interjections for HUSH_COOLDOWN_S. This is how she learns
        to stop — the research's core lesson."""
        try:
            self.record_feedback(REACTION_DISMISSED)
        except Exception:
            pass
        self._hush_until = time.time() + HUSH_COOLDOWN_S
        try:
            print("[PROACTIVE] hush — quiet for 30 minutes.")
        except Exception:
            pass

    def last_sent_within(self, window_s: float) -> bool:
        return bool(self._last_sent_at and
                    time.time() - self._last_sent_at <= window_s)
