"""Service base class + registry — the Runtime OS lifecycle model.

A Service is anything with a lifetime: async ``start()`` / ``stop()`` and a
``health()`` probe returning ``(status, latency_ms, last_error)``. Services
that run a continuous body implement ``run()``; the Supervisor owns that
coroutine and restarts it on exception. Services that are tick-driven only
(on-demand jobs like the skill loop) leave ``run()`` unimplemented.

``ServiceRegistry`` starts services in dependency order (topological sort over
``Service.dependencies``) and stops them in reverse.
"""
from __future__ import annotations

import asyncio
import time

STATUS_HEALTHY = "healthy"
STATUS_DEGRADED = "degraded"
STATUS_DOWN = "down"
STATUS_STOPPED = "stopped"


class Service:
    """Base class for every supervised unit of the runtime."""

    name = "service"
    # Names of services that must be started before this one.
    dependencies: tuple = ()
    # Non-critical services may stay down without failing the boot.
    critical: bool = True

    async def start(self) -> None:
        """Bring the service up. Must be idempotent."""

    async def stop(self) -> None:
        """Bring the service down. Must be idempotent."""

    async def run(self) -> None:
        """Long-running body owned by the Supervisor.

        The default returns immediately: the service is tick-driven only and
        the Supervisor just health-checks it. Services with a real loop
        override this; an exception out of here triggers restart-with-backoff.
        """
        return None

    async def health(self):
        """Probe. Returns (status, latency_ms, last_error)."""
        return (STATUS_HEALTHY, 0.0, "")


class ServiceRegistry:
    """Holds services, starts them in dependency order, stops in reverse."""

    def __init__(self):
        self._services: dict[str, Service] = {}

    def register(self, service: Service) -> Service:
        if not getattr(service, "name", None):
            raise ValueError("service must define a name")
        if service.name in self._services:
            raise ValueError(f"service already registered: {service.name}")
        self._services[service.name] = service
        return service

    def get(self, name: str):
        return self._services.get(name)

    def names(self):
        return list(self._services)

    def ordered(self):
        """Services topologically sorted so dependencies start first.

        Stable: ties break in registration order. Raises RuntimeError on an
        unknown dependency or a dependency cycle.
        """
        order: list[str] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(name: str):
            if name in visited:
                return
            if name in visiting:
                raise RuntimeError(f"service dependency cycle involving {name!r}")
            if name not in self._services:
                raise RuntimeError(f"unknown service dependency: {name!r}")
            visiting.add(name)
            for dep in self._services[name].dependencies or ():
                visit(dep)
            visiting.discard(name)
            visited.add(name)
            order.append(name)

        for name in self._services:
            visit(name)
        return [self._services[n] for n in order]

    async def start_all(self) -> None:
        for svc in self.ordered():
            try:
                await svc.start()
                print(f"[RUNTIME] service started: {svc.name}")
            except Exception as e:
                print(f"[RUNTIME] service {svc.name} failed to start: {type(e).__name__}: {e}")
                if getattr(svc, "critical", True):
                    raise

    async def stop_all(self) -> None:
        for svc in reversed(self.ordered()):
            try:
                await svc.stop()
            except Exception as e:
                print(f"[RUNTIME] service {svc.name} stop failed: {type(e).__name__}: {e}")
