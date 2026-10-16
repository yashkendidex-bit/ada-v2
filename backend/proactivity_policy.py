"""Learned interruption policy — Horvitz EV, not a timer (Feature 8).

The old design polled on a fixed clock and gate-checked per-key cooldowns.
This module inverts that: candidate generators produce *candidates*; the
POLICY decides send/suppress, and every decision + every observed reaction
is logged so the policy learns.

Policy v1 (heuristic, active day one):
    score = relevance x (1 - interrupt_cost)
    interrupt_cost is built from observable local signals only:
      calendar-busy flag, screen active + typing, minutes since last
      suggestion, late night, and a recent-dismissal streak with escalating
      suppression. Send iff score > threshold (default 0.7, conservative;
      the engine starts effectively default-silent).

Policy v2 (learned, when data exists):
    per (suggestion_type x context_bucket) Beta-Binomial Thompson sampling.
    alpha/beta persist in SQLite; sampling is `random.betavariate` (stdlib).
    v2 decides when the bucket has >= 5 samples, else v1 decides.
    Context buckets are coarse on purpose: time-of-day x weekday/weekend
    x focus/idle. One user, little data — coarse is honest.

Quiet hours: the fixed window is the prior, but per-hour dismissal rates
are learned too — an hour with sustained high dismissal becomes soft-quiet
automatically (railed as its own suppression reason, inspectable).

Hard rails (never overridden by any score): active voice turn, another
human voice present, hush cooldown, hard quiet hours (non-urgent),
learned soft-quiet hours (non-urgent), the daily cap (non-urgent),
per-key cooldowns.

Zero local ML model downloads, ever. stdlib + sqlite3 only.
"""
from __future__ import annotations

import json
import os
import random
import sqlite3
import time
from datetime import datetime

# ── defaults (conservative by design) ───────────────────────────────────────
DEFAULT_THRESHOLD = 0.7        # v1 send iff score > 0.7
DEFAULT_MAX_PER_DAY = 3        # proactive messages/day, configurable
MIN_SAMPLES_V2 = 5             # v2 decides once a bucket has >= 5 samples
HOUR_LEARN_MIN_SENDS = 10      # soft-quiet needs >= 10 sends in the hour
HOUR_LEARN_DISMISS_RATE = 0.6  # ...and a dismissal rate this high
STREAK_WINDOW = 8              # recent-dismissal streak looks at last 8 rows
HUSH_COOLDOWN_S = 30 * 60      # a hush buys 30 minutes of silence
FEEDBACK_WINDOW_S = 300        # reactions attribute to a send within 5 min

# Reaction taxonomy (research: direct UI feedback + indirect signals).
REACTION_ENGAGED = "engaged"
REACTION_DISMISSED = "dismissed"
REACTION_IGNORED = "ignored"
REACTION_SNOOZED = "snoozed"
REACTION_NONE = "none"
NEGATIVE_REACTIONS = {REACTION_DISMISSED, REACTION_IGNORED, REACTION_SNOOZED}

# Relevance priors per suggestion type (v1, before any learning).
RELEVANCE = {
    "reminder_urgent": 1.0,
    "reminder": 0.80,
    "overdue": 0.95,
    "dossier": 0.80,
    "briefing": 0.75,
    "signal": 0.70,          # event-driven demand signals (errrepeat/file/focus)
    "world_risk": 0.75,      # F3 world_risk_signals (pace/due deviation)
    "lunch": 0.50,
    "wind_down": 0.45,
    "fatigue": 0.50,
    "break": 0.50,
}


def speaker_present(window_minutes: float = 5.0) -> bool:
    """Feature 7 (filled): True when a non-owner voice was active in the
    last ``window_minutes`` — feeds the proactivity "don't interrupt
    human–human conversation" rail. Any failure (speaker_id unavailable,
    no activity yet) degrades to False, never to an interrupt.
    """
    try:
        from speaker_id import recent_non_owner_active
        return bool(recent_non_owner_active(window_minutes))
    except Exception:
        return False


def context_bucket(ts: float | None = None, focus: bool = False) -> str:
    """Coarse context bucket: time-of-day x weekday/weekend x focus/idle."""
    dt = datetime.fromtimestamp(ts if ts is not None else time.time())
    h = dt.hour
    if 5 <= h < 12:
        tod = "morning"
    elif 12 <= h < 17:
        tod = "afternoon"
    elif 17 <= h < 23:
        tod = "evening"
    else:
        tod = "night"
    day = "weekend" if dt.weekday() >= 5 else "weekday"
    act = "focus" if focus else "idle"
    return f"{tod}/{day}/{act}"


class PolicyStore:
    """SQLite home for the interruption log + learned beliefs.

    `interruptions`: every send/suppress decision.
    `bucket_stats`: alpha/beta per (suggestion_type x context_bucket).
    `hour_stats`: sends/dismissals per local hour -> learned soft-quiet.
    WAL mode; gitignored (*.db). Tests use tmp dirs.
    """

    def __init__(self, db_path: str | None = None):
        # Default is in-memory: production passes an explicit path
        # (ada.py -> ada_kernel/policy.db); nothing here may ever drop a
        # stray policy.db into the repo or a test working directory.
        db_path = db_path or os.getenv("ADA_POLICY_DB") or ":memory:"
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS interruptions(
                 id TEXT PRIMARY KEY, ts REAL, suggestion_type TEXT,
                 context_json TEXT, action TEXT, reaction TEXT,
                 latency_s REAL, reason TEXT)""")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS bucket_stats(
                 bucket TEXT, suggestion_type TEXT, alpha REAL, beta REAL,
                 PRIMARY KEY (bucket, suggestion_type))""")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS hour_stats(
                 hour INTEGER PRIMARY KEY, sends INTEGER DEFAULT 0,
                 dismissals INTEGER DEFAULT 0)""")
        self._conn.commit()

    # ── interruption log ────────────────────────────────────────────────
    def log_decision(self, iid: str, suggestion_type: str, context: dict,
                     action: str, reason: str) -> None:
        self._conn.execute(
            "INSERT INTO interruptions(id, ts, suggestion_type, context_json,"
            " action, reaction, latency_s, reason)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (iid, time.time(), suggestion_type, json.dumps(context or {}),
             action, REACTION_NONE, None, reason))
        self._conn.commit()

    def log_reaction(self, iid: str, reaction: str, latency_s: float | None) -> bool:
        cur = self._conn.execute(
            "SELECT suggestion_type, context_json, ts, reaction FROM interruptions"
            " WHERE id=?", (iid,))
        row = cur.fetchone()
        if row is None:
            return False
        stype, ctx_json, ts, _old = row
        self._conn.execute(
            "UPDATE interruptions SET reaction=?, latency_s=? WHERE id=?",
            (reaction, latency_s, iid))
        try:
            ctx = json.loads(ctx_json or "{}")
        except Exception:
            ctx = {}
        # Learned beliefs: engaged = success; dismissed/ignored/snoozed = failure.
        if reaction in (REACTION_ENGAGED,) or reaction in NEGATIVE_REACTIONS:
            self._update_bucket(stype, ctx, success=reaction == REACTION_ENGAGED)
        # Learned quiet hours: per-hour dismissal rate.
        if reaction in NEGATIVE_REACTIONS:
            hour = datetime.fromtimestamp(ts).hour
            self._conn.execute(
                "INSERT INTO hour_stats(hour, sends, dismissals) VALUES(?,?,?)"
                " ON CONFLICT(hour) DO UPDATE SET dismissals=dismissals+1",
                (hour, 0, 1))
        self._conn.commit()
        return True

    def _update_bucket(self, stype: str, ctx: dict, success: bool) -> None:
        bucket = ctx.get("bucket") or context_bucket(focus=bool(ctx.get("focus")))
        cur = self._conn.execute(
            "SELECT alpha, beta FROM bucket_stats WHERE bucket=? AND suggestion_type=?",
            (bucket, stype))
        row = cur.fetchone()
        if row is None:
            alpha, beta = 1.0, 1.0
        else:
            alpha, beta = float(row[0]), float(row[1])
        if success:
            alpha += 1.0
        else:
            beta += 1.0
        self._conn.execute(
            "INSERT INTO bucket_stats(bucket, suggestion_type, alpha, beta)"
            " VALUES(?,?,?,?)"
            " ON CONFLICT(bucket, suggestion_type)"
            " DO UPDATE SET alpha=excluded.alpha, beta=excluded.beta",
            (bucket, stype, alpha, beta))
        self._conn.commit()

    def record_send(self, hour: int) -> None:
        self._conn.execute(
            "INSERT INTO hour_stats(hour, sends, dismissals) VALUES(?,?,?)"
            " ON CONFLICT(hour) DO UPDATE SET sends=sends+1",
            (hour, 1, 0))
        self._conn.commit()

    def bucket_samples(self, bucket: str, stype: str) -> tuple[float, float]:
        cur = self._conn.execute(
            "SELECT alpha, beta FROM bucket_stats WHERE bucket=? AND suggestion_type=?",
            (bucket, stype))
        row = cur.fetchone()
        if row is None:
            return 1.0, 1.0
        return float(row[0]), float(row[1])

    def recent_streak(self, window: int = STREAK_WINDOW) -> int:
        """Consecutive recent negative reactions (most recent first)."""
        cur = self._conn.execute(
            "SELECT reaction FROM interruptions WHERE action='sent'"
            " ORDER BY ts DESC LIMIT ?", (window,))
        streak = 0
        for (reaction,) in cur.fetchall():
            if reaction in NEGATIVE_REACTIONS:
                streak += 1
            elif reaction == REACTION_NONE:
                continue  # not answered yet — doesn't break the streak
            else:
                break
        return streak

    def sends_today(self) -> int:
        start = datetime.now().replace(hour=0, minute=0, second=0,
                                       microsecond=0).timestamp()
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM interruptions WHERE action='sent' AND ts>=?",
            (start,))
        return int(cur.fetchone()[0])

    def hour_dismissal_rate(self, hour: int) -> float:
        cur = self._conn.execute(
            "SELECT sends, dismissals FROM hour_stats WHERE hour=?", (hour,))
        row = cur.fetchone()
        if row is None:
            return 0.0
        sends = int(row[0])
        if sends < HOUR_LEARN_MIN_SENDS:
            return 0.0
        return min(1.0, int(row[1]) / sends)

    def is_soft_quiet_hour(self, hour: int | None = None) -> bool:
        h = datetime.now().hour if hour is None else hour
        return self.hour_dismissal_rate(h) >= HOUR_LEARN_DISMISS_RATE

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


class InterruptionPolicy:
    """The send/suppress decision. Horvitz EV in heuristic form, Thompson
    sampling when the bucket has data.

    ctx keys (all optional, all local):
      focus (bool), typing (bool), calendar_busy (bool),
      minutes_since_last (float|None),
      voice_turn_active (bool), speaker_present (bool), hush_active (bool),
      quiet_hour (bool), urgent (bool), suggestion_type (str), relevance (float)
    """

    def __init__(self, store: PolicyStore | None = None,
                 threshold: float = DEFAULT_THRESHOLD,
                 max_per_day: int = DEFAULT_MAX_PER_DAY,
                 min_samples_v2: int = MIN_SAMPLES_V2):
        self.store = store or PolicyStore()
        self.threshold = float(threshold)
        self.max_per_day = int(max_per_day)
        self.min_samples_v2 = int(min_samples_v2)

    # ── v1: heuristic ───────────────────────────────────────────────────
    def interrupt_cost(self, ctx: dict) -> float:
        cost = 0.0
        now_h = datetime.now().hour
        if 22 <= now_h or now_h < 6:          # late night
            cost += 0.25
        if ctx.get("focus"):
            cost += 0.30
        if ctx.get("typing"):
            cost += 0.20
        if ctx.get("calendar_busy"):
            cost += 0.35
        msl = ctx.get("minutes_since_last")
        if msl is not None and msl < 30:      # recent chatter costs more
            cost += 0.25 * (1.0 - msl / 30.0)
        streak = self.store.recent_streak()   # escalating dismissal suppression
        cost += min(0.45, 0.12 * streak)
        if streak >= 4:
            cost += 0.20
        return min(0.95, max(0.0, cost))

    def v1_score(self, relevance: float, ctx: dict) -> float:
        return float(relevance) * (1.0 - self.interrupt_cost(ctx))

    # ── v2: Thompson sampling ───────────────────────────────────────────
    def v2_decides(self, stype: str, bucket: str) -> bool:
        alpha, beta = self.store.bucket_samples(bucket, stype)
        return (alpha + beta - 2.0) >= self.min_samples_v2

    def v2_sample(self, stype: str, bucket: str) -> float:
        alpha, beta = self.store.bucket_samples(bucket, stype)
        return random.betavariate(alpha, beta)

    # ── decision ────────────────────────────────────────────────────────
    def decide(self, ctx: dict, iid: str | None = None) -> dict:
        """Return {action, reason, score, policy, iid}. Logs the decision."""
        stype = str(ctx.get("suggestion_type", "signal"))
        relevance = float(ctx.get("relevance",
                                 RELEVANCE.get(stype, 0.5)))
        urgent = bool(ctx.get("urgent"))
        iid = iid or f"int_{int(time.time())}_{abs(hash((stype, time.time()))) % 99999}"
        bucket = ctx.get("bucket") or context_bucket(focus=bool(ctx.get("focus")))
        ctx = dict(ctx, bucket=bucket)
        hour = datetime.now().hour

        reason = None
        policy_used = "v1"
        score = self.v1_score(relevance, ctx)

        # ── hard rails (never overridden by any score) ──
        if ctx.get("voice_turn_active"):
            reason = "voice-turn-active"
        elif ctx.get("speaker_present"):
            reason = "speaker-present"
        elif ctx.get("hush_active"):
            reason = "hush-cooldown"
        # ── quiet hours: fixed window prior; learned soft-quiet joins it ──
        elif not urgent and (ctx.get("quiet_hour") or
                             self.store.is_soft_quiet_hour(hour)):
            reason = ("quiet-hours" if ctx.get("quiet_hour")
                      else "soft-quiet-hour")
        # ── daily cap ──
        elif not urgent and self.store.sends_today() >= self.max_per_day:
            reason = "daily-cap"
        # ── per-key cooldown ──
        elif ctx.get("cooldown_active"):
            reason = "cooldown"
        # ── urgent bypass: a due-now reminder is time-critical value, not
        # a learnable preference — it skips the score gate (but never the
        # hard rails above). Spent once per key by the caller. ──
        elif urgent:
            policy_used = "urgent-bypass"
            score = relevance
        # ── the policy itself ──
        elif self.v2_decides(stype, bucket):
            policy_used = "v2"
            sampled = self.v2_sample(stype, bucket)
            score = sampled
            reason = None if sampled > self.threshold else "v2-below-threshold"
        elif score <= self.threshold:
            reason = "v1-below-threshold"

        action = "suppressed" if reason else "sent"
        if action == "sent":
            self.store.record_send(hour)
        self.store.log_decision(iid, stype, ctx, action, reason or "score-ok")
        return {"action": action, "reason": reason, "score": round(score, 4),
                "policy": policy_used, "iid": iid,
                "suggestion_type": stype, "bucket": bucket}

    def log_reaction(self, iid: str, reaction: str,
                     latency_s: float | None = None) -> bool:
        if reaction not in (REACTION_ENGAGED, REACTION_DISMISSED,
                            REACTION_IGNORED, REACTION_SNOOZED, REACTION_NONE):
            return False
        return self.store.log_reaction(iid, reaction, latency_s)
