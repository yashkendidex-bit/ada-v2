"""Tests for Feature 2 — MCP-native tool forge (backend/mcp/).

Covers: server exposes all KERNEL_TOOLS (count matches registry); tool call
round-trip through the client; hot-load picks up a new tool file without a
restart; version annotation present; external-server config rejects
non-localhost URLs by default; dispatch still fires `_check_confirmation`
for a side-effect tool (gate -> MCP execute ordering); legacy in-process
path is marked deprecated.
"""
import asyncio
import inspect
import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

BACKEND_DIR = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def kernel_dir(tmp_path):
    d = tmp_path / "ada_kernel"
    (d / "custom_tools").mkdir(parents=True)
    (d / "mcp_tools").mkdir(parents=True)
    # one legacy (pre-Feature-2) custom tool, as the repo ships them
    (d / "custom_tools" / "triple_it.py").write_text(
        "def triple_it(x: int) -> int:\n"
        "    \"\"\"Triple it (legacy custom tool).\"\"\"\n"
        "    return x * 3\n"
    )
    return d


@pytest.fixture()
def stub_kernel(kernel_dir):
    """Minimal kernel double: handle_tool + forge + audit + kernel_dir."""
    from mcp.forge import MCPForge

    class FakeAudit:
        def __init__(self):
            self.rows = []

        def log(self, *a):
            self.rows.append(a)

    audit = FakeAudit()

    async def handle_tool(name, args, permissions=None, default_perm=True,
                          tool_version=None):
        detail = (f"[tool_version={tool_version}] " if tool_version else "") \
            + json.dumps(args)[:300]
        audit.log("kernel", "tool", name, "allow", detail)
        return {"result": f"handled:{name}"}

    forge = MCPForge(kernel_dir / "custom_tools", kernel_dir / "mcp_tools")
    return SimpleNamespace(kernel_dir=kernel_dir, forge=forge,
                           handle_tool=handle_tool,
                           security=SimpleNamespace(audit=audit),
                           _audit=audit)


@pytest.fixture()
def mcp_server(stub_kernel):
    from mcp.server import create_local_server
    return create_local_server(stub_kernel)


@pytest.fixture()
def mcp_client(mcp_server, stub_kernel):
    from mcp.client import MCPToolClient
    return MCPToolClient(mcp_server, kernel=stub_kernel,
                         config_path=stub_kernel.kernel_dir / "mcp_servers.json")


def _run(coro):
    # asyncio.run (fresh loop each call): immune to whatever loop state
    # earlier test modules leave behind (e.g. test_jarvis_suite closes the
    # policy's loop, after which get_event_loop() raises RuntimeError).
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# server surface
# ---------------------------------------------------------------------------

class TestServerSurface:
    def test_exposes_all_kernel_tools(self, mcp_server):
        from agent_kernel import KERNEL_TOOLS
        tools = _run(mcp_server.list_tools())
        kernel_tools = [t for t in tools
                        if (t.meta or {}).get("source") == "kernel"]
        assert len(kernel_tools) == len(KERNEL_TOOLS)
        listed = {t.name for t in tools}
        for entry in KERNEL_TOOLS:
            assert entry["name"] in listed

    def test_exposes_legacy_custom_tools(self, mcp_server):
        assert mcp_server.has_tool("triple_it")
        assert mcp_server.tool_source("triple_it") == "custom-legacy"

    def test_version_annotation_present(self, mcp_server):
        tools = _run(mcp_server.list_tools())
        assert tools, "server lists zero tools"
        for t in tools:
            version = (t.meta or {}).get("version")
            assert version, f"tool {t.name} carries no version annotation"
        assert mcp_server.tool_version("remember") == "1.0.0"
        assert mcp_server.tool_version("triple_it") == "0.1.0"

    def test_kernel_schema_converted_to_json_schema(self, mcp_server):
        info = mcp_server.describe("remember")
        params = info["parameters"]
        assert params["type"] == "object"
        assert params["properties"]["key"]["type"] == "string"
        assert "key" in params.get("required", [])

    def test_manifest_has_version_and_deprecated(self, stub_kernel,
                                                 mcp_server):
        man_path = stub_kernel.kernel_dir / "mcp_manifest.json"
        assert man_path.exists()
        man = json.loads(man_path.read_text())
        entry = man["tools"]["remember"]
        assert entry["version"] == "1.0.0"
        assert entry["deprecated"] is False
        legacy = man["tools"]["triple_it"]
        assert legacy["deprecated"] is True


# ---------------------------------------------------------------------------
# dispatch round-trip
# ---------------------------------------------------------------------------

class TestDispatch:
    def test_tool_call_roundtrip_through_client(self, mcp_client):
        res = _run(mcp_client.call_local("remember",
                                         {"key": "k", "value": "v"},
                                         permissions={"remember": True}))
        assert res == {"result": "handled:remember"}

    def test_audit_row_carries_tool_version(self, mcp_client, stub_kernel):
        _run(mcp_client.call_local("identity_read", {}))
        rows = stub_kernel._audit.rows
        assert rows and "tool_version=1.0.0" in rows[-1][-1]

    def test_legacy_custom_tool_call_roundtrip(self, mcp_client, capsys):
        res = _run(mcp_client.call_local("triple_it", {"x": 7}))
        assert res == {"result": "21"}
        out = capsys.readouterr().out
        assert "DEPRECATED" in out

    def test_unknown_tool_errors_cleanly(self, mcp_client):
        res = _run(mcp_client.call_local("no_such_tool_xyz", {}))
        assert "failed" in res["result"]


# ---------------------------------------------------------------------------
# forge: new tools become MCP definitions, legacy path deprecated
# ---------------------------------------------------------------------------

class TestForge:
    def test_synthesise_writes_mcp_module_not_custom(self, stub_kernel):
        forge = stub_kernel.forge
        msg = forge.synthesise(
            "double_it",
            "def double_it(x: int) -> int:\n"
            "    \"\"\"Double it.\"\"\"\n"
            "    return x * 2\n")
        assert "created" in msg and "MCP tool definition" in msg
        mcp_dir = stub_kernel.kernel_dir / "mcp_tools"
        assert (mcp_dir / "double_it.py").exists()
        assert not (stub_kernel.kernel_dir / "custom_tools"
                    / "double_it.py").exists()
        src = (mcp_dir / "double_it.py").read_text()
        assert "TOOL_NAME = 'double_it'" in src
        assert "TOOL_VERSION" in src

    def test_synthesised_tool_runs(self, stub_kernel):
        forge = stub_kernel.forge
        forge.synthesise("double_it",
                         "def double_it(x: int) -> int:\n    return x * 2\n")
        assert _run(forge.run("double_it", {"x": 21})) == 42

    def test_synthesise_rejects_blocked(self, stub_kernel):
        forge = stub_kernel.forge
        msg = forge.synthesise("pwn", "import subprocess\ndef pwn():\n    return 1\n")
        assert "rejected" in msg.lower()

    def test_legacy_run_warns_deprecated_but_works(self, stub_kernel, capsys):
        forge = stub_kernel.forge
        assert _run(forge.run("triple_it", {"x": 4})) == 12
        assert "DEPRECATED" in capsys.readouterr().out

    def test_legacy_not_used_for_new_tools(self, stub_kernel):
        forge = stub_kernel.forge
        assert "double_it" not in forge.legacy_registry


# ---------------------------------------------------------------------------
# hot-load
# ---------------------------------------------------------------------------

class TestHotLoad:
    def test_hotload_picks_up_new_tool_without_restart(
            self, mcp_server, mcp_client, stub_kernel):
        from mcp.hotload import MCPHotLoader
        loader = MCPHotLoader(mcp_server, forge=stub_kernel.forge)
        loader._snap = loader.snapshot()  # baseline: no replay
        new_file = stub_kernel.kernel_dir / "mcp_tools" / "shout.py"
        new_file.write_text(
            'TOOL_NAME = "shout"\n'
            'TOOL_DESCRIPTION = "Shout it."\n'
            'TOOL_SCHEMA = {"type": "object", '
            '"properties": {"s": {"type": "string"}}, "required": ["s"]}\n'
            'TOOL_VERSION = "2.1.0"\n'
            'TOOL_DEPRECATED = False\n'
            '\n'
            'def shout(s: str) -> str:\n'
            '    return s.upper() + "!"\n'
        )
        changes = loader.check_once()
        assert any("shout" in c for c in changes), changes
        assert mcp_server.has_tool("shout")
        assert mcp_server.tool_version("shout") == "2.1.0"
        res = _run(mcp_client.call_local("shout", {"s": "hey"}))
        assert res == {"result": "HEY!"}

    def test_hotload_removes_deleted_tool(
            self, mcp_server, stub_kernel):
        from mcp.hotload import MCPHotLoader
        loader = MCPHotLoader(mcp_server, forge=stub_kernel.forge)
        path = stub_kernel.kernel_dir / "mcp_tools" / "temp_tool.py"
        path.write_text(
            'TOOL_NAME = "temp_tool"\nTOOL_DESCRIPTION = "t"\n'
            'TOOL_SCHEMA = {"type": "object"}\nTOOL_VERSION = "1.0.0"\n'
            'TOOL_DEPRECATED = False\n\ndef temp_tool() -> str:\n    return "x"\n'
        )
        loader.check_once()
        assert mcp_server.has_tool("temp_tool")
        path.unlink()
        changes = loader.check_once()
        assert any("temp_tool" in c for c in changes), changes
        assert not mcp_server.has_tool("temp_tool")


# ---------------------------------------------------------------------------
# external servers: localhost-only without opt-in
# ---------------------------------------------------------------------------

class TestExternalServers:
    def _client_with(self, mcp_server, stub_kernel, servers):
        from mcp.client import MCPToolClient
        cfg = stub_kernel.kernel_dir / "mcp_servers.json"
        cfg.write_text(json.dumps({"servers": servers}))
        return MCPToolClient(mcp_server, kernel=stub_kernel, config_path=cfg)

    def test_rejects_non_localhost_by_default(self, mcp_server, stub_kernel):
        from mcp.client import MCPToolClient, MCPSecurityError
        client = self._client_with(mcp_server, stub_kernel, [
            {"name": "evil", "url": "http://example.com:8000/mcp"}])
        with pytest.raises(MCPSecurityError):
            _run(client.call_external("evil", "tool", {}))

    def test_allows_localhost_without_opt_in(self, mcp_server, stub_kernel):
        client = self._client_with(mcp_server, stub_kernel, [
            {"name": "local", "url": "http://127.0.0.1:8001/mcp"}])
        assert client.check_url_allowed("http://127.0.0.1:8001/mcp")
        assert client.check_url_allowed("http://localhost:8001/mcp")
        assert not client.check_url_allowed("http://example.com/mcp")

    def test_allows_opted_in_remote(self, mcp_server, stub_kernel):
        client = self._client_with(mcp_server, stub_kernel, [
            {"name": "remote", "url": "https://remote.example.com/mcp",
             "opt_in": True}])
        # must not raise MCPSecurityError (it will fail to connect, which
        # becomes a {"result": "failed"} envelope, not a security refusal)
        from mcp.client import MCPSecurityError
        try:
            _run(client.call_external("remote", "tool", {}))
        except MCPSecurityError:
            pytest.fail("opted-in remote server must not be security-refused")

    def test_http_roundtrip_to_local_server(self, mcp_server, stub_kernel):
        from mcp.client import MCPToolClient

        def free_port():
            s = socket.socket()
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
            s.close()
            return port

        async def scenario():
            port = free_port()
            task = asyncio.create_task(
                mcp_server.serve(host="127.0.0.1", port=port))
            await asyncio.sleep(2.0)  # uvicorn startup
            try:
                cfg = stub_kernel.kernel_dir / "mcp_servers.json"
                cfg.write_text(json.dumps({"servers": [
                    {"name": "self",
                     "url": f"http://127.0.0.1:{port}/mcp"}]}))
                client = MCPToolClient(mcp_server, kernel=stub_kernel,
                                       config_path=cfg)
                tools = await client.list_external_tools("self")
                names = [t["name"] for t in tools]
                assert "remember" in names
                res = await client.call_external(
                    "self", "remember", {"key": "k", "value": "v"})
                assert res == {"result": "handled:remember"}
            finally:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        _run(scenario())


# ---------------------------------------------------------------------------
# dispatch: F0 gate still fires before MCP execution (gate -> MCP execute)
# ---------------------------------------------------------------------------

class TestGateOrdering:
    def _loop(self, stub_kernel):
        from ada import AudioLoop
        loop = AudioLoop.__new__(AudioLoop)
        loop.kernel = stub_kernel
        loop.permissions = {"remember": True}
        loop._mcp_server = None
        loop._mcp_client = None
        return loop

    def test_execute_tool_routes_through_mcp_client(self, stub_kernel):
        loop = self._loop(stub_kernel)
        fake_client = SimpleNamespace(
            call_local=AsyncMock(return_value={"result": "via-mcp"}))
        loop._mcp_client = fake_client
        res = _run(loop._execute_tool("remember", {"key": "k"}))
        assert res == {"result": "via-mcp"}
        fake_client.call_local.assert_awaited_once_with(
            "remember", {"key": "k"}, permissions={"remember": True})

    def test_gate_decision_precedes_execution(self, stub_kernel):
        """Mirrors run_one's exact contract: gate first, execute only if allowed."""
        loop = self._loop(stub_kernel)
        loop._check_confirmation = AsyncMock()
        loop._execute_tool = AsyncMock(return_value={"result": "ok"})

        async def run_one_like(name, args):
            # the two statements, in run_one's order
            allowed = await loop._check_confirmation(name, args)
            if not allowed:
                return {"denied": True}
            return await loop._execute_tool(name, args)

        loop._check_confirmation.return_value = False
        out = _run(run_one_like("remember", {"key": "k"}))
        assert out == {"denied": True}
        loop._execute_tool.assert_not_awaited()

        loop._check_confirmation.return_value = True
        out = _run(run_one_like("remember", {"key": "k"}))
        assert out == {"result": "ok"}
        loop._execute_tool.assert_awaited_once()

    def test_run_one_calls_gate_before_execute_in_source(self):
        """Pin the real ordering in the turn loop: gate -> _execute_tool."""
        from ada import AudioLoop
        src = inspect.getsource(AudioLoop)
        i = src.find("async def run_one")
        assert i != -1, "run_one closure not found"
        block = src[i:i + 2500]
        gate_at = block.find("_check_confirmation")
        exec_at = block.find("_execute_tool")
        assert gate_at != -1 and exec_at != -1
        assert gate_at < exec_at, "gate must precede execution in run_one"

    def test_mcp_call_does_not_gate_itself(self):
        """_mcp_call executes only — gating lives at the call site (no dup)."""
        from ada import AudioLoop
        src = inspect.getsource(AudioLoop._mcp_call)
        assert "_check_confirmation(" not in src
