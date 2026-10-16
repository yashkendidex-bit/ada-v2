"""Thin Service adapters around the existing subsystems.

Delegation only — no subsystem internals are rewritten here. Each adapter
wraps one already-constructed object (ProactiveEngine, Scheduler, WatchFolder,
DreamRunner, SkillLoop, OvernightDossier, WellbeingMonitor, AmbientVision),
forwards start()/stop(), and watches the subsystem's own background task for
liveness so the Supervisor can restart it if it dies.

Sync-or-async start/stop is tolerated: several subsystems define plain
``def start(self)`` while others are ``async def``.
"""
from __future__ import annotations

import asyncio
import inspect
import time

from .services import Service, STATUS_HEALTHY, STATUS_DEGRADED, STATUS_DOWN


async def _call(fn, *args, **kwargs):
    """Call fn; await it if it returns an awaitable (sync/async tolerant)."""
    res = fn(*args, **kwargs)
    if inspect.isawaitable(res):
        return await res
    return res


class SubsystemAdapter(Service):
    """Generic adapter for subsystems with start()/stop() and a _task loop."""

    def __init__(self, name: str, inner, *, dependencies=(), critical=True):
        self.name = name
        self._inner = inner
        self.dependencies = tuple(dependencies)
        self.critical = critical

    # ── lifecycle: pure delegation ─────────────────────────────────────
    async def start(self) -> None:
        await _call(self._inner.start)

    async def stop(self) -> None:
        await _call(self._inner.stop)

    async def run(self) -> None:
        """Liveness watchdog over the subsystem's own background task.

        Raises if that task dies unexpectedly, which is what lets the
        Supervisor restart the subsystem with backoff.
        """
        while True:
            await asyncio.sleep(5)
            task = getattr(self._inner, "_task", None)
            if task is None:
                continue  # not started yet / on-demand — nothing to watch
            if task.done():
                if task.cancelled():
                    return  # we stopped it — clean exit
                exc = None
                try:
                    exc = task.exception()
                except Exception:
                    pass
                raise RuntimeError(
                    f"{self.name}: background task died ({exc!r})")

    async def health(self):
        t0 = time.perf_counter()
        try:
            task = getattr(self._inner, "_task", None)
            if task is not None and task.done() and not task.cancelled():
                exc = None
                try:
                    exc = task.exception()
                except Exception:
                    pass
                return (STATUS_DOWN, (time.perf_counter() - t0) * 1000.0,
                        f"background task dead: {exc!r}"[:300])
            extra = ""
            status_fn = getattr(self._inner, "status", None)
            if callable(status_fn):
                try:
                    extra = str(status_fn())[:160]
                except Exception:
                    pass
            return (STATUS_HEALTHY, (time.perf_counter() - t0) * 1000.0, extra)
        except Exception as e:
            return (STATUS_DOWN, (time.perf_counter() - t0) * 1000.0,
                    f"health probe failed: {type(e).__name__}: {e}"[:300])


class OnDemandAdapter(Service):
    """Adapter for subsystems with no background loop (SkillLoop, DreamRunner).

    They run on demand (rep_now / dream_now); the adapter reports readiness
    instead of liveness.
    """

    def __init__(self, name: str, inner, *, dependencies=(), critical=False):
        self.name = name
        self._inner = inner
        self.dependencies = tuple(dependencies)
        self.critical = critical

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def health(self):
        t0 = time.perf_counter()
        try:
            due_fn = getattr(self._inner, "due", None)
            detail = ""
            if callable(due_fn):
                try:
                    detail = f"due={bool(due_fn())}"
                except Exception:
                    pass
            return (STATUS_HEALTHY, (time.perf_counter() - t0) * 1000.0, detail)
        except Exception as e:
            return (STATUS_DEGRADED, (time.perf_counter() - t0) * 1000.0,
                    f"{type(e).__name__}: {e}"[:200])


# ── concrete adapters (one line each; the classes above do the work) ──────────

class SchedulerService(SubsystemAdapter):
    def __init__(self, scheduler):
        super().__init__("scheduler", scheduler)


class WatchFolderService(SubsystemAdapter):
    def __init__(self, watch):
        super().__init__("watch_folder", watch, critical=False)


class ProactiveService(SubsystemAdapter):
    def __init__(self, engine):
        super().__init__("proactive", engine, critical=False)


class DossierService(SubsystemAdapter):
    def __init__(self, dossier):
        super().__init__("dossier", dossier, critical=False)


class WellbeingService(SubsystemAdapter):
    def __init__(self, monitor):
        super().__init__("wellbeing", monitor, critical=False)


class VisionService(SubsystemAdapter):
    def __init__(self, vision):
        super().__init__("vision", vision, critical=False)


class DreamService(OnDemandAdapter):
    def __init__(self, dreamer):
        super().__init__("dream", dreamer)


class SkillLoopService(OnDemandAdapter):
    def __init__(self, skill_loop):
        super().__init__("skill_loop", skill_loop)


# ── Feature 2: MCP-native tool forge ──────────────────────────────────────────
# The local MCP tool server as a supervised runtime service: lifecycle
# (start/stop), health probe, and supervision come from the F1 runtime.
# GET /health covers it automatically via the supervisor's health payload.

class MCPServerService(Service):
    """Serves the local MCP tool server over Streamable HTTP (127.0.0.1)
    and runs the hot-loader that adds/removes tools at runtime.

    critical=False: if the HTTP serving fails, in-process dispatch
    (``MCPToolClient.call_local``) keeps working — the brain never depends
    on the socket. The supervisor still restarts a dead serve task.
    """

    name = "mcp_server"
    dependencies = ()
    critical = False

    def __init__(self, kernel, server=None, host="127.0.0.1", port=None):
        self.kernel = kernel
        self.host = host
        self.port = port  # None -> DEFAULT_MCP_PORT at start()
        self._server = server      # mcp.server.AdaMCPServer (shared if given)
        self._client = None        # mcp.client.MCPToolClient
        self._hotloader = None     # mcp.hotload.MCPHotLoader
        self._serve_task = None
        self._last_error = ""

    # -- lifecycle ---------------------------------------------------------
    async def start(self) -> None:
        """Idempotent: build (or reuse) the server, start HTTP + hot-load."""
        from mcp.server import create_local_server, DEFAULT_MCP_PORT
        from mcp.client import MCPToolClient
        from mcp.hotload import MCPHotLoader
        if self._server is None:
            self._server = create_local_server(self.kernel)
        if self.port is None:
            self.port = DEFAULT_MCP_PORT
        self._client = MCPToolClient(self._server, kernel=self.kernel)
        forge = getattr(self.kernel, "forge", None)
        kdir = self._server.kernel_dir
        self._hotloader = MCPHotLoader(
            self._server, forge=forge,
            custom_dir=kdir / "custom_tools", mcp_dir=kdir / "mcp_tools")
        await self._hotloader.start()
        if self._serve_task is None or self._serve_task.done():
            self._serve_task = asyncio.create_task(
                self._serve_forever(), name="mcp-serve")
        print(f"[RUNTIME] mcp_server up on {self.host}:{self.port} "
              f"(Streamable HTTP) + hot-load watching")

    async def _serve_forever(self) -> None:
        while True:
            try:
                await self._server.serve(host=self.host, port=self.port)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # bind failure etc: back off, keep trying
                self._last_error = str(e)[:200]
                print(f"[RUNTIME] mcp serve error: {self._last_error} "
                      f"(retry in 5s)")
                await asyncio.sleep(5)

    async def stop(self) -> None:
        if self._hotloader is not None:
            try:
                await self._hotloader.stop()
            except Exception:
                pass
            self._hotloader = None
        if self._serve_task is not None:
            self._serve_task.cancel()
            try:
                await self._serve_task
            except (asyncio.CancelledError, Exception):
                pass
            self._serve_task = None

    async def run(self) -> None:
        """Liveness watchdog: a dead serve task triggers supervisor restart."""
        while True:
            await asyncio.sleep(5)
            task = self._serve_task
            if task is None:
                continue
            if task.done():
                if task.cancelled():
                    return
                exc = task.exception()
                raise RuntimeError(f"mcp serve task died: {exc}")

    # -- health --------------------------------------------------------------
    async def health(self):
        """Probe: in-process list_tools. Returns (status, latency_ms, error)."""
        t0 = time.time()
        try:
            tools = await self._server.list_tools()
            dt_ms = (time.time() - t0) * 1000.0
            if not tools:
                return (STATUS_DEGRADED, dt_ms, "mcp server lists zero tools")
            return (STATUS_HEALTHY, dt_ms, "")
        except Exception as e:
            return (STATUS_DOWN, 0.0, str(e)[:200])

    # -- accessors -------------------------------------------------------------
    @property
    def mcp_server(self):
        return self._server

    @property
    def mcp_client(self):
        return self._client

    @property
    def last_error(self):
        return self._last_error
