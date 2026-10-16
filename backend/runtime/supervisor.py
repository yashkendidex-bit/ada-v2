"""Supervisor — owns service tasks, restarts with backoff, ticks health.

- ``start_all()``: starts the registry in dependency order, takes ownership of
  every service's ``run()`` coroutine, and does an initial health tick.
- A ``run()`` that raises is restarted with exponential backoff
  (``backoff_base_s * 2**(n-1)``, capped at ``backoff_max_s``). Past
  ``max_restarts`` failures the service is marked failed and a
  ``service.failed`` event goes on the bus.
- Every ``tick_s`` seconds each service's ``health()`` is probed; a flip
  healthy <-> degraded/down publishes ``service.degraded`` /
  ``service.recovered`` on the bus and updates the fallback state.
"""
from __future__ import annotations

import asyncio
import time

from .services import STATUS_HEALTHY, STATUS_DEGRADED, STATUS_DOWN
from .fallbacks import note_status, clear_status


class Supervisor:
    def __init__(self, registry, bus, tick_s: float = 10.0,
                 max_restarts: int = 5,
                 backoff_base_s: float = 1.0,
                 backoff_max_s: float = 60.0):
        self._registry = registry
        self._bus = bus
        self._tick_s = float(tick_s)
        self._max_restarts = int(max_restarts)
        self._backoff_base_s = float(backoff_base_s)
        self._backoff_max_s = float(backoff_max_s)
        self._tasks: dict[str, asyncio.Task] = {}
        self._tick_task: asyncio.Task | None = None
        self._restarts: dict[str, int] = {}
        self._failed: set[str] = set()
        self._last_health: dict[str, tuple] = {}
        self._running = False

    # ── lifecycle ──────────────────────────────────────────────────────────
    async def start_all(self) -> None:
        await self._registry.start_all()
        self._running = True
        for svc in self._registry.ordered():
            self._tasks[svc.name] = asyncio.create_task(
                self._own(svc), name=f"supervisor:{svc.name}")
        await self.tick_once()  # populate the snapshot immediately
        self._tick_task = asyncio.create_task(
            self._tick_loop(), name="supervisor:tick")

    async def stop_all(self) -> None:
        self._running = False
        if self._tick_task is not None:
            self._tick_task.cancel()
        for task in self._tasks.values():
            task.cancel()
        if self._tasks or self._tick_task:
            await asyncio.gather(*self._tasks.values(),
                                 *[t for t in [self._tick_task] if t],
                                 return_exceptions=True)
        self._tasks.clear()
        self._tick_task = None
        await self._registry.stop_all()

    # ── ownership / restart ────────────────────────────────────────────────
    async def _own(self, service) -> None:
        """Own service.run(); restart with backoff on exception."""
        name = service.name
        failures = 0
        while self._running:
            try:
                await service.run()
                return  # clean exit — the service chose to stop
            except asyncio.CancelledError:
                return
            except Exception as e:
                failures += 1
                self._restarts[name] = failures
                if failures > self._max_restarts:
                    await self._mark_failed(service, e, failures)
                    return
                delay = min(self._backoff_base_s * (2 ** (failures - 1)),
                            self._backoff_max_s)
                print(f"[SUPERVISOR] {name} crashed ({type(e).__name__}: {str(e)[:120]}); "
                      f"restart {failures}/{self._max_restarts} in {delay:.2f}s")
                try:
                    await asyncio.sleep(delay)
                except asyncio.CancelledError:
                    return
                if not self._running:
                    return
                try:
                    await service.start()
                except asyncio.CancelledError:
                    return
                except Exception as e2:
                    # A failed restart counts as another crash round.
                    failures += 1
                    self._restarts[name] = failures
                    print(f"[SUPERVISOR] {name} restart failed "
                          f"({type(e2).__name__}: {str(e2)[:120]})")
                    if failures > self._max_restarts:
                        await self._mark_failed(service, e2, failures)
                        return
                    delay = min(self._backoff_base_s * (2 ** (failures - 1)),
                                self._backoff_max_s)
                    try:
                        await asyncio.sleep(delay)
                    except asyncio.CancelledError:
                        return

    async def _mark_failed(self, service, exc: Exception, failures: int) -> None:
        name = service.name
        self._failed.add(name)
        note_status(name, "down", f"{type(exc).__name__}: {exc}")
        print(f"[SUPERVISOR] {name} FAILED after {failures} restarts — giving up")
        try:
            await service.stop()
        except Exception:
            pass
        try:
            await self._bus.publish(
                "service.failed", source="supervisor",
                payload={"service": name,
                         "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                         "restarts": failures})
        except Exception:
            pass

    # ── health ticks ───────────────────────────────────────────────────────
    async def _tick_loop(self) -> None:
        try:
            while self._running:
                await asyncio.sleep(self._tick_s)
                if not self._running:
                    break
                await self.tick_once()
        except asyncio.CancelledError:
            pass

    async def tick_once(self) -> None:
        for svc in self._registry.ordered():
            t0 = time.perf_counter()
            try:
                status, latency_ms, last_error = await svc.health()
            except Exception as e:
                status = STATUS_DOWN
                latency_ms = (time.perf_counter() - t0) * 1000.0
                last_error = f"health() raised {type(e).__name__}: {e}"
            prev = self._last_health.get(svc.name)
            self._last_health[svc.name] = (status, latency_ms, last_error)
            if prev is None or prev[0] == status:
                continue
            # Health flipped — announce it and update fallback state.
            if status in (STATUS_DEGRADED, STATUS_DOWN):
                note_status(svc.name, status, last_error)
                try:
                    await self._bus.publish(
                        "service.degraded", source="supervisor",
                        payload={"service": svc.name, "from": prev[0],
                                 "to": status, "error": str(last_error)[:300]})
                except Exception:
                    pass
                print(f"[SUPERVISOR] {svc.name}: {prev[0]} -> {status} ({str(last_error)[:120]})")
            elif status == STATUS_HEALTHY:
                clear_status(svc.name)
                try:
                    await self._bus.publish(
                        "service.recovered", source="supervisor",
                        payload={"service": svc.name, "from": prev[0]})
                except Exception:
                    pass
                print(f"[SUPERVISOR] {svc.name}: {prev[0]} -> healthy")

    # ── introspection ──────────────────────────────────────────────────────
    def health_payload(self) -> tuple[int, dict]:
        """(status_code, body) for GET /health — pure function of snapshot().

        Kept here (stdlib-only) so the endpoint shape is testable without a
        FastAPI install; server.py's route is a thin wrapper around this.
        """
        snap = self.snapshot()
        bad = [n for n, s in snap.items()
               if s.get("status") in ("degraded", "down") or s.get("failed")]
        return (200, {"status": "degraded" if bad else "ok",
                      "services": snap})

    def snapshot(self) -> dict:
        """Per-service status for /health and the fallback watcher."""
        out: dict[str, dict] = {}
        for name in self._registry.names():
            rec = self._last_health.get(name)
            if rec is None:
                out[name] = {"status": "unknown", "latency_ms": 0.0,
                             "last_error": "", "restarts": self._restarts.get(name, 0),
                             "failed": name in self._failed}
            else:
                status, latency_ms, last_error = rec
                out[name] = {"status": status,
                             "latency_ms": round(float(latency_ms), 1),
                             "last_error": str(last_error or ""),
                             "restarts": self._restarts.get(name, 0),
                             "failed": name in self._failed}
        return out

    @property
    def failed(self) -> set[str]:
        return set(self._failed)

    @property
    def restarts(self) -> dict[str, int]:
        return dict(self._restarts)
