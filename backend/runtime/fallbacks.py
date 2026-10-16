"""Graceful degradation — the FALLBACKS table.

Maps a service name to the behavior the brain should adopt while that service
is degraded. The Supervisor keeps the live status table current on every
health flip; :class:`FallbackService` translates ``service.degraded`` /
``service.failed`` bus events into ``service.fallback`` events so the brain
can adjust; :func:`health_snapshot` renders the "system health" block the
prompt builder reads.
"""
from __future__ import annotations

import asyncio
import time

FALLBACKS = {
    # Voice output path down -> stop trying to synthesize, deliver text.
    "tts": "Text-only mode: do not attempt TTS; deliver every reply as text.",
    # Web search down -> answer from memory, say so.
    "web_search": "Answer from memory only; tell Sir the answer may be stale.",
    # Proactive engine down -> silence, never a partial nudge.
    "proactive": "Stay silent: no proactive speech until the service recovers.",
    # Scheduler down -> reminders are best-effort only.
    "scheduler": "Reminders are best-effort in-memory only; warn before promising timed delivery.",
    "watch_folder": "Folder watching paused; index files manually on request.",
    "dossier": "Overnight research paused; announce when it recovers.",
    "skill_loop": "Skill practice paused; already-learned skills still work.",
    "dream": "Dream consolidation paused; resumes on recovery.",
    "wellbeing": "Wellbeing nudges paused.",
    "vision": "Ambient vision paused; no screen awareness until recovery.",
    # The autonomy gate is the one service that must fail CLOSED.
    "policy": "CRITICAL: the autonomy gate is unreachable — fail closed and ask Sir out loud for every side-effect.",
}

# Live table, written by the Supervisor on health flips, read by the brain.
# {service: {"status": ..., "error": ..., "ts": ...}}
_latest: dict[str, dict] = {}


def note_status(service: str, status: str, error: str = "") -> None:
    _latest[service] = {"status": status, "error": str(error or ""),
                        "ts": time.time()}


def clear_status(service: str) -> None:
    _latest.pop(service, None)


def active_fallbacks() -> dict[str, str]:
    """Services currently degraded -> their fallback behavior text."""
    out = {}
    for svc, rec in _latest.items():
        if rec["status"] != "healthy" and svc in FALLBACKS:
            out[svc] = FALLBACKS[svc]
    return out


def health_snapshot() -> str:
    """'System health' block source for the prompt builder.

    Returns "" when everything is healthy, so the prompt is untouched in the
    common case. When services are degraded the brain reads this and adjusts.
    """
    bad = {s: r for s, r in _latest.items() if r["status"] != "healthy"}
    if not bad:
        return ""
    lines = ["SYSTEM HEALTH — degraded services, adjust behavior:"]
    for svc, rec in sorted(bad.items()):
        fb = FALLBACKS.get(svc, "no fallback defined — be conservative")
        err = (rec.get("error") or "")[:120]
        lines.append(f"- {svc}: {rec['status']}{f' ({err})' if err else ''}. Fallback: {fb}")
    return "\n".join(lines)


class FallbackService:
    """Control-plane service: turns degraded/failed events into fallbacks.

    Subscribes to the bus; on ``service.degraded`` / ``service.failed`` it
    publishes ``service.fallback`` carrying the fallback behavior text, and on
    ``service.recovered`` it publishes ``service.fallback_cleared``. On
    (re)start it seeds from the supervisor's snapshot so a restart never loses
    the currently-degraded set.
    """

    name = "fallback"
    dependencies = ()
    critical = False

    def __init__(self, bus, snapshot_fn=None):
        self._bus = bus
        self._snapshot_fn = snapshot_fn

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def run(self) -> None:
        # Seed: anything already degraded gets its fallback announced.
        try:
            snap = (self._snapshot_fn or (lambda: {}))()
            for svc, rec in snap.items():
                if rec.get("status") in ("degraded", "down") or rec.get("failed"):
                    fb = FALLBACKS.get(svc)
                    if fb:
                        await self._bus.publish(
                            "service.fallback", source="fallbacks",
                            payload={"service": svc, "fallback": fb,
                                     "trigger": "seed"})
        except Exception:
            pass
        q = self._bus.subscribe()
        try:
            while True:
                ev = await q.get()
                if ev.type in ("service.degraded", "service.failed"):
                    svc = (ev.payload or {}).get("service", "")
                    fb = FALLBACKS.get(svc)
                    if fb:
                        await self._bus.publish(
                            "service.fallback", source="fallbacks",
                            payload={"service": svc, "fallback": fb,
                                     "trigger": ev.type})
                elif ev.type == "service.recovered":
                    svc = (ev.payload or {}).get("service", "")
                    if svc:
                        await self._bus.publish(
                            "service.fallback_cleared", source="fallbacks",
                            payload={"service": svc})
        finally:
            self._bus.unsubscribe(q)

    async def health(self):
        return ("healthy", 0.0, "")
