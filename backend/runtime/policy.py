"""PolicyService — the autonomy gate as a runtime control-plane service.

Feature 0 built the REAL gate: ``AudioLoop._check_confirmation`` in
``backend/ada.py``, driven by the autonomy ladder in ``backend/autonomy.py``,
is the single live dispatch point for every tool call, and it audit-logs
every allow/deny through the existing ``AuditLog``.

This service does NOT reimplement any of that. It wraps the existing gate so
the runtime can see it: ``check()`` delegates straight through to
``_check_confirmation`` (the audit trail is written by the gate itself — no
second log, no divergence), while ``health()`` exposes the current autonomy
level / arm state and lifetime decision counters for ``/health``.
"""
from __future__ import annotations

import inspect
import time

from .services import STATUS_HEALTHY, STATUS_DEGRADED


class PolicyService:
    """Thin audited choke point around the Feature-0 autonomy gate."""

    name = "policy"
    dependencies = ()
    critical = True  # the gate must never silently go missing

    def __init__(self, ada_loop):
        self._ada = ada_loop
        self._decisions = {"approved": 0, "denied": 0}

    async def check(self, tool: str, args=None) -> bool:
        """One audited choke point: delegates to the real F0 gate."""
        res = self._ada._check_confirmation(tool, args)
        ok = await res if inspect.isawaitable(res) else bool(res)
        self._decisions["approved" if ok else "denied"] += 1
        return ok

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def run(self) -> None:
        return None  # tick-driven only; the gate itself has no loop

    async def health(self):
        t0 = time.perf_counter()
        try:
            aut = getattr(getattr(self._ada, "kernel", None), "autonomy", None)
            level = getattr(aut, "level", "?")
            try:
                armed = bool(aut.armed) if aut is not None else False
            except Exception:
                armed = False
            detail = (f"autonomy L{level}{' ARMED' if armed else ''}; "
                      f"decisions approved={self._decisions['approved']} "
                      f"denied={self._decisions['denied']}")
            return (STATUS_HEALTHY, (time.perf_counter() - t0) * 1000.0, detail)
        except Exception as e:
            return (STATUS_DEGRADED, (time.perf_counter() - t0) * 1000.0,
                    f"{type(e).__name__}: {e}"[:200])

    @property
    def decisions(self) -> dict:
        return dict(self._decisions)
