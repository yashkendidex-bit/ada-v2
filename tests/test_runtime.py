"""Runtime OS (Feature 1) tests.

Covers: service lifecycle + dependency ordering, supervisor restart-with-
backoff and fail-after-cap, bus fan-out/order/SQLite durability, /health
endpoint shape, fallback publication on degradation, checkpoint/resume
round-trip, PolicyService delegation to the F0 gate, and the SkillLoop
checkpoint migration.
"""
import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).parent.parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from runtime.services import Service, ServiceRegistry, STATUS_HEALTHY, STATUS_DEGRADED
from runtime.bus import EventBus, Event
from runtime.supervisor import Supervisor
from runtime.checkpoints import checkpoint, resume, clear
from runtime import fallbacks


def run(coro):
    """Run one coroutine to completion on a fresh loop (repo convention)."""
    return asyncio.run(coro)


# ── helpers ────────────────────────────────────────────────────────────────

class RecService(Service):
    """Records start/stop order; health is settable."""

    def __init__(self, name, deps=(), log=None):
        self.name = name
        self.dependencies = tuple(deps)
        self._log = log if log is not None else []
        self._health = (STATUS_HEALTHY, 0.5, "")

    def set_health(self, status, err=""):
        self._health = (status, 0.5, err)

    async def start(self):
        self._log.append(f"start:{self.name}")

    async def stop(self):
        self._log.append(f"stop:{self.name}")

    async def health(self):
        return self._health


# ── 1. service lifecycle + dependency ordering ─────────────────────────────

def test_service_lifecycle_ordering():
    async def main():
        log = []
        reg = ServiceRegistry()
        reg.register(RecService("c", deps=("b",), log=log))
        reg.register(RecService("a", log=log))
        reg.register(RecService("b", deps=("a",), log=log))
        await reg.start_all()
        assert log == ["start:a", "start:b", "start:c"]
        st, lat, err = await reg.get("b").health()
        assert st == STATUS_HEALTHY and err == ""
        assert isinstance(lat, float)
        await reg.stop_all()
        assert log[3:] == ["stop:c", "stop:b", "stop:a"]
    run(main())


def test_registry_rejects_unknown_dep_and_cycle():
    reg = ServiceRegistry()
    reg.register(RecService("x", deps=("nope",)))
    with pytest.raises(RuntimeError):
        reg.ordered()
    reg2 = ServiceRegistry()
    reg2.register(RecService("p", deps=("q",)))
    reg2.register(RecService("q", deps=("p",)))
    with pytest.raises(RuntimeError):
        reg2.ordered()
    with pytest.raises(ValueError):
        reg2.register(RecService("p"))  # duplicate name


# ── 2. supervisor: restart with backoff, then recover ─────────────────────

def test_supervisor_restarts_crashing_service_with_backoff():
    class Flaky(Service):
        name = "flaky"

        def __init__(self):
            self.run_calls = 0
            self.start_calls = 0

        async def start(self):
            self.start_calls += 1

        async def run(self):
            self.run_calls += 1
            if self.run_calls == 1:
                raise RuntimeError("boom")
            await asyncio.sleep(3600)

    async def main():
        bus = EventBus(":memory:")
        try:
            reg = ServiceRegistry()
            flaky = Flaky()
            reg.register(flaky)
            sup = Supervisor(reg, bus, tick_s=3600, max_restarts=3,
                             backoff_base_s=0.01, backoff_max_s=0.02)
            t0 = time.perf_counter()
            await sup.start_all()
            try:
                for _ in range(500):
                    if flaky.start_calls == 2:  # boot + one restart
                        break
                    await asyncio.sleep(0.01)
                assert sup.restarts.get("flaky") == 1
                assert "flaky" not in sup.failed
                assert flaky.start_calls == 2
                assert time.perf_counter() - t0 < 5
            finally:
                await sup.stop_all()
        finally:
            await bus.close()
    run(main())


def test_supervisor_marks_failed_after_cap():
    class Doomed(Service):
        name = "doomed"

        async def run(self):
            raise RuntimeError("always dies")

    async def main():
        bus = EventBus(":memory:")
        try:
            failed_q = bus.subscribe("service.failed")
            reg = ServiceRegistry()
            reg.register(Doomed())
            sup = Supervisor(reg, bus, tick_s=3600, max_restarts=2,
                             backoff_base_s=0.01, backoff_max_s=0.02)
            await sup.start_all()
            try:
                for _ in range(500):
                    if "doomed" in sup.failed:
                        break
                    await asyncio.sleep(0.01)
                assert "doomed" in sup.failed
                assert sup.restarts.get("doomed") == 3  # 2 restarts + final
                ev = await asyncio.wait_for(failed_q.get(), timeout=5)
                assert ev.type == "service.failed"
                assert ev.payload["service"] == "doomed"
                assert ev.payload["restarts"] == 3
            finally:
                await sup.stop_all()
                # _mark_failed writes the module-global fallback table; a dead
                # service legitimately stays listed, but tests must not leak it.
                fallbacks.clear_status("doomed")
        finally:
            await bus.close()
    run(main())


# ── 3. bus: fan-out, order, SQLite durability ──────────────────────────────

def test_bus_fanout_order_and_persistence(tmp_path):
    async def main():
        db = tmp_path / "events.db"
        bus = EventBus(db)
        try:
            q_all = bus.subscribe()
            q_ping = bus.subscribe("ping")
            q_other = bus.subscribe("other")
            for i in range(5):
                await bus.publish("ping", source="test", payload={"i": i})
            got_all = [await asyncio.wait_for(q_all.get(), 1) for _ in range(5)]
            got_ping = [await asyncio.wait_for(q_ping.get(), 1) for _ in range(5)]
            assert [e.payload["i"] for e in got_all] == list(range(5))
            assert [e.payload["i"] for e in got_ping] == list(range(5))
            assert all(e.schema_v == 1 for e in got_all)
            assert q_other.empty()  # filtered subscriber gets nothing
        finally:
            await bus.close()
        # Durability: a fresh bus on the same file replays the tail in order.
        bus2 = EventBus(db)
        try:
            replayed = bus2.recent(limit=10, event_type="ping")
            assert [e.payload["i"] for e in replayed] == list(range(5))
            assert replayed[0].source == "test"
        finally:
            await bus2.close()
    run(main())


# ── 4. /health endpoint shape ──────────────────────────────────────────────

def _import_server():
    """Import backend/server.py in this venv, which lacks socketio + uvicorn.

    (That absence is also why test_server_survives_missing_authenticator fails
    pre-existing.) Only the FastAPI app and the /health route are under test
    here — nothing touches the real socket layer, so minimal stubs suffice.
    """
    import types
    if "socketio" not in sys.modules:
        sock = types.ModuleType("socketio")

        class _AsyncServer:
            def __init__(self, *a, **k):
                pass

            def event(self, fn=None, *a, **k):
                # used only as a bare @sio.event decorator
                return (lambda f: f) if fn is None else fn

            async def emit(self, *a, **k):
                pass

        sock.AsyncServer = _AsyncServer
        sock.ASGIApp = lambda *a, **k: object()
        sys.modules["socketio"] = sock
    if "uvicorn" not in sys.modules:
        uv = types.ModuleType("uvicorn")
        uv.run = lambda *a, **k: None
        sys.modules["uvicorn"] = uv
    import server as srv
    return srv


def test_health_payload_shape():
    """The /health response shape, via the real Supervisor code path."""
    async def main():
        bus = EventBus(":memory:")
        try:
            reg = ServiceRegistry()
            ok = RecService("scheduler")
            bad = RecService("proactive")
            bad.set_health(STATUS_DEGRADED, "loop stale")
            reg.register(ok)
            reg.register(bad)
            sup = Supervisor(reg, bus, tick_s=3600)
            await sup.start_all()
            try:
                await sup.tick_once()
                await sup.tick_once()  # second tick: healthy->degraded flip lands
                code, body = sup.health_payload()
                assert code == 200
                assert body["status"] == "degraded"
                svcs = body["services"]
                assert set(svcs) == {"scheduler", "proactive"}
                for _name, s in svcs.items():
                    assert {"status", "latency_ms", "last_error",
                            "restarts", "failed"} <= set(s)
                assert svcs["proactive"]["status"] == "degraded"
                assert svcs["proactive"]["last_error"] == "loop stale"
                assert svcs["scheduler"]["status"] == "healthy"
                # All healthy -> "ok".
                bad.set_health(STATUS_HEALTHY)
                await sup.tick_once()
                code2, body2 = sup.health_payload()
                assert code2 == 200 and body2["status"] == "ok"
            finally:
                await sup.stop_all()
        finally:
            await bus.close()
    run(main())


def test_health_endpoint_route():
    """The actual FastAPI route delegates to health_payload().

    Skipped where FastAPI isn't installed (this venv); the payload shape
    itself is covered by test_health_payload_shape above.
    """
    pytest.importorskip("fastapi")
    srv = _import_server()

    class FakeSupervisor:
        def health_payload(self):
            return (200, {"status": "ok", "services": {}})

    old = srv.runtime_supervisor
    srv.runtime_supervisor = FakeSupervisor()
    try:
        from fastapi.testclient import TestClient
        r = TestClient(srv.app).get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok", "services": {}}
    finally:
        srv.runtime_supervisor = old


def test_health_endpoint_not_started():
    pytest.importorskip("fastapi")
    srv = _import_server()
    old = srv.runtime_supervisor
    srv.runtime_supervisor = None
    try:
        from fastapi.testclient import TestClient
        r = TestClient(srv.app).get("/health")
        assert r.status_code == 503
        assert r.json()["status"] == "not-started"
    finally:
        srv.runtime_supervisor = old


# ── 5. fallback published on degradation ───────────────────────────────────

def test_fallback_published_on_degradation(tmp_path):
    from runtime.fallbacks import FallbackService, health_snapshot

    async def main():
        bus = EventBus(tmp_path / "events.db")
        try:
            fb_q = bus.subscribe("service.fallback")
            reg = ServiceRegistry()
            svc = RecService("proactive")
            reg.register(svc)
            sup = Supervisor(reg, bus, tick_s=3600, max_restarts=2,
                             backoff_base_s=0.01)
            reg.register(FallbackService(bus, snapshot_fn=sup.snapshot))
            await sup.start_all()
            try:
                # Flip healthy -> degraded: supervisor publishes
                # service.degraded, the fallback service translates it.
                svc.set_health(STATUS_DEGRADED, "loop stale")
                await sup.tick_once()
                ev = await asyncio.wait_for(fb_q.get(), timeout=5)
                assert ev.type == "service.fallback"
                assert ev.payload["service"] == "proactive"
                assert "Stay silent" in ev.payload["fallback"]
                # The prompt-builder block reflects it.
                snap = health_snapshot()
                assert "proactive" in snap and "Stay silent" in snap
                # Recover: fallback clears, block goes empty.
                svc.set_health(STATUS_HEALTHY)
                await sup.tick_once()
                assert health_snapshot() == ""
            finally:
                await sup.stop_all()
        finally:
            await bus.close()
            fallbacks.clear_status("proactive")
    run(main())


# ── 6. checkpoint / resume round-trip ──────────────────────────────────────

def test_checkpoint_resume_roundtrip(tmp_path):
    db = tmp_path / "ckpt.db"
    checkpoint("t1", "a", {"x": 1}, db)
    checkpoint("t1", "b", {"y": [1, 2]}, db)
    checkpoint("t1", "c", {"z": "done"}, db)
    got = resume("t1", db)
    assert list(got) == ["a", "b", "c"]
    assert got["a"] == {"x": 1} and got["c"] == {"z": "done"}
    # Overwrite of an existing step replaces the payload.
    checkpoint("t1", "a", {"x": 9}, db)
    assert resume("t1", db)["a"] == {"x": 9}
    # Isolation between tasks; clear drops everything.
    checkpoint("t2", "a", {"q": 1}, db)
    clear("t1", db)
    assert resume("t1", db) == {}
    assert resume("t2", db) == {"a": {"q": 1}}
    assert resume("missing", db) == {}


# ── 7. PolicyService delegates to the F0 gate ──────────────────────────────

def test_policy_service_delegates_to_f0_gate():
    from runtime.policy import PolicyService

    calls = []

    class FakeAda:
        kernel = None  # no autonomy object -> health must still answer

        async def _check_confirmation(self, name, args=None):
            calls.append((name, args))
            return name != "forget"

    async def main():
        p = PolicyService(FakeAda())
        assert await p.check("read_file", {}) is True
        assert await p.check("forget", {}) is False
        assert p.decisions == {"approved": 1, "denied": 1}
        assert calls == [("read_file", {}), ("forget", {})]
        st, _lat, _err = await p.health()
        assert st == "healthy"
    run(main())


# ── 8. SkillLoop checkpoint migration ──────────────────────────────────────

_PROPOSAL = ("NAME: km_to_miles\nPURPOSE: convert km to miles\n"
             "TEST_CALL: {\"km\": 5}\nCODE:\n```python\n"
             "def km_to_miles(km):\n    return km * 0.621371\n```")


class _FakeForge:
    def __init__(self, tools_dir=None):
        self.registry = {}
        self.synth_calls = []
        # Real forges persist every synthesised module to disk (the eval
        # ladder reads the file); the double honors that contract.
        self.tools_dir = Path(tools_dir) if tools_dir else None
        if self.tools_dir:
            self.tools_dir.mkdir(parents=True, exist_ok=True)

    def list_tools(self):
        return list(self.registry)

    def synthesise(self, name, code):
        self.synth_calls.append(name)
        ns = {}
        exec(code, {"__builtins__": {}}, ns)
        self.registry[name] = ns[name]
        if self.tools_dir:
            (self.tools_dir / f"{name}.py").write_text(code)
        return f"Tool '{name}' created and loaded"

    async def run(self, name, kwargs, timeout=20):
        return self.registry[name](**kwargs)


class _FakeKernel:
    def __init__(self, forge):
        self.forge = forge


def _skill_loop(tmp_path, chat_fn):
    from skill_loop import SkillLoop
    return SkillLoop(_FakeKernel(_FakeForge(tmp_path / "ftools")),
                     chat_fn, str(tmp_path))


def test_skill_loop_rep_checkpoints_and_clears(tmp_path, monkeypatch):
    # These two tests exercise checkpointing, not the eval ladder: the L2
    # subset is stubbed green (its own coverage lives in
    # tests/test_eval_gating.py; a tmp state_dir has no repo root to run it
    # from).
    import skill_eval
    monkeypatch.setattr(skill_eval, "run_eval_subset",
                        lambda repo_root, timeout=180: (True, "stubbed"))

    async def chat_fn(prompt):
        return _PROPOSAL

    async def main():
        loop = _skill_loop(tmp_path, chat_fn)
        out = await loop.rep_now(force=True)
        assert "verified and committed" in out
        assert "km_to_miles" in loop._registry()
        # Checkpoints are cleared once the rep commits.
        task_id = f"skill-rep-{datetime.now().strftime('%Y-%m-%d')}"
        db = str(Path(str(tmp_path)) / "skill_checkpoints.db")
        assert resume(task_id, db) == {}
    run(main())


def test_skill_loop_resumes_from_checkpoint(tmp_path, monkeypatch):
    """A crash after PROPOSE resumes without re-asking the coach."""
    import skill_eval
    monkeypatch.setattr(skill_eval, "run_eval_subset",
                        lambda repo_root, timeout=180: (True, "stubbed"))
    task_id = f"skill-rep-{datetime.now().strftime('%Y-%m-%d')}"
    db = str(Path(str(tmp_path)) / "skill_checkpoints.db")
    checkpoint(task_id, "proposal",
               {"name": "km_to_miles", "purpose": "convert",
                "test_call": "{\"km\": 5}",
                "code": "def km_to_miles(km):\n    return km * 0.621371"}, db)

    chat_calls = []

    async def chat_fn(prompt):
        chat_calls.append(prompt)
        raise AssertionError("coach must not be re-asked after a checkpoint")

    async def main():
        loop = _skill_loop(tmp_path, chat_fn)
        out = await loop.rep_now(force=True)
        assert chat_calls == []
        assert "verified and committed" in out
        assert "km_to_miles" in loop._registry()
        assert resume(task_id, db) == {}
    run(main())
