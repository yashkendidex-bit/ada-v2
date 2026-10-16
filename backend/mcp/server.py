"""Local MCP server — Feature 2 (MCP-native tool forge).

One canonical-``mcp``-SDK server (``mcp.server.mcpserver.MCPServer`` — the
SDK v2 successor of ``mcp.server.fastmcp.FastMCP``) that exposes:

* every ``KERNEL_TOOLS`` entry from :mod:`agent_kernel`, auto-wrapped from
  its existing JSON schema (synthesised ``__signature__`` — the registry
  holds schemas, not functions);
* every legacy ``ada_kernel/custom_tools/*.py`` tool, auto-wrapped from its
  real function signature (served through the *deprecated* in-process
  ``exec()`` path — see :mod:`mcp.forge`);
* every forged ``ada_kernel/mcp_tools/*.py`` module (``TOOL_*`` metadata +
  the validated function).

Versioning is Option A from the research (annotations): SDK v2's
``ToolAnnotations`` has no version slot, so the version rides in the tool's
``meta`` annotation namespace (``meta={"version": ...}``) — same semantics —
and is mirrored into the tool manifest
(``ada_kernel/mcp_manifest.json``). Every executed call is audit-logged
with its tool version.

Transport: Streamable HTTP on 127.0.0.1 (``ADA_MCP_PORT``, default 8001).
In-process dispatch (what the brain uses) calls
:meth:`AdaMCPServer.call` directly — no HTTP hop, no extra approval
surface.
"""

from __future__ import annotations

import contextvars
import importlib
import inspect
import json
import os
import re
import time
import weakref
from pathlib import Path
from typing import Any

from ._sdk import SDK  # noqa: F401  (ensures the canonical SDK is loaded)

_MCPServer = importlib.import_module("_mcp_sdk.server.mcpserver").MCPServer

DEFAULT_MCP_PORT = int(os.environ.get("ADA_MCP_PORT", "8001"))
MANIFEST_NAME = "mcp_manifest.json"
DEFAULT_TOOL_VERSION = "1.0.0"

#: Explicit per-tool versions. Everything not listed rides DEFAULT_TOOL_VERSION.
#: Bump a version here when the tool's contract (name/description/schema)
#: changes — the manifest records what the brain last saw.
TOOL_VERSIONS: dict[str, str] = {}

#: Permission maps flow through the in-process MCP boundary on this var, so
#: the MCP protocol surface stays pure while ``kernel.handle_tool`` keeps
#: its existing permission check. Set by :meth:`mcp.client.MCPToolClient.call_local`.
permissions_ctx: contextvars.ContextVar = contextvars.ContextVar(
    "ada_mcp_permissions", default=None
)

_JSON_TO_PY = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}

# Live servers, so the brain's tool list can pick up hot-loaded tools.
_SERVERS: "weakref.WeakSet[AdaMCPServer]" = weakref.WeakSet()


# ---------------------------------------------------------------------------
# schema helpers
# ---------------------------------------------------------------------------

def _to_json_schema(params: dict | None) -> dict:
    """KERNEL_TOOLS use Gemini-ish uppercase types; MCP wants JSON Schema."""
    def conv(node: dict) -> dict:
        if not isinstance(node, dict):
            return {}
        out: dict[str, Any] = {}
        t = str(node.get("type", "")).lower()
        if t:
            out["type"] = t
        for key in ("description", "enum", "default", "minimum", "maximum"):
            if key in node:
                out[key] = node[key]
        if t == "object" and isinstance(node.get("properties"), dict):
            out["properties"] = {k: conv(v) for k, v in node["properties"].items()}
            if node.get("required"):
                out["required"] = list(node["required"])
        if t == "array" and isinstance(node.get("items"), dict):
            out["items"] = conv(node["items"])
        return out

    schema = conv(params or {})
    schema.setdefault("type", "object")
    return schema


def _build_wrapper(name: str, json_schema: dict | None, description: str,
                   executor) -> Any:
    """Build an async tool function whose ``__signature__`` carries the schema.

    The MCP SDK derives the tool's input schema from the function signature,
    but our registry holds JSON schemas, not functions — so we synthesise a
    signature from the schema (keyword-only params, defaults for optionals).
    """
    props = (json_schema or {}).get("properties", {}) or {}
    required = set((json_schema or {}).get("required", []) or [])
    sig_params = []
    for pname, pspec in props.items():
        ann = _JSON_TO_PY.get(str(pspec.get("type", "")).lower(), Any)
        default = inspect.Parameter.empty if pname in required else None
        sig_params.append(inspect.Parameter(
            pname, inspect.Parameter.KEYWORD_ONLY,
            default=default, annotation=ann))
    # Late-binding guard: executor is passed positionally, name bound here.
    tool_name = name

    async def _tool_impl(**kwargs):
        return await executor(tool_name, dict(kwargs))

    _tool_impl.__name__ = re.sub(r"\W+", "_", name) or "tool"
    _tool_impl.__doc__ = description or ""
    _tool_impl.__signature__ = inspect.Signature(sig_params)
    return _tool_impl


def _audit(kernel, name: str, version: str, source: str, args: dict) -> None:
    """One audit row per MCP-executed tool, carrying the tool version."""
    try:
        audit = kernel.security.audit
    except Exception:
        return
    try:
        audit.log("mcp", "tool", name, "allow",
                  f"[tool_version={version}] [source={source}] "
                  f"{json.dumps(args)[:250]}")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# executors (MCP tool body -> ADA internals)
# ---------------------------------------------------------------------------

async def _exec_kernel(kernel, name: str, args: dict, version: str):
    """Kernel tool: keep the existing permission check + audit, add version."""
    perms = permissions_ctx.get()
    res = await kernel.handle_tool(name, args, permissions=perms,
                                   tool_version=version)
    if isinstance(res, dict) and set(res.keys()) == {"result"}:
        return res["result"]
    return res


async def _exec_custom_legacy(forge, kernel, name: str, args: dict, version: str):
    """Legacy custom tool: the deprecated in-process exec() path.

    Kept only for tools forged before Feature 2. ``MCPForge.run`` prints the
    deprecation notice; new tools never take this road.
    """
    res = await forge.run(name, args)
    _audit(kernel, name, version, "custom-legacy", args)
    return res


async def _exec_forged(entry: dict, kernel, name: str, args: dict, version: str):
    fn = entry["fn"]
    if inspect.iscoroutinefunction(fn):
        res = await fn(**args)
    else:
        res = fn(**args)
    _audit(kernel, name, version, "forged", args)
    return res


# ---------------------------------------------------------------------------
# the server
# ---------------------------------------------------------------------------

class AdaMCPServer:
    """ADA's local MCP tool server.

    ``kernel`` is the :class:`agent_kernel.AgentKernel` (or a test double
    exposing ``kernel_dir``, ``handle_tool``, ``forge`` and
    ``security.audit``).
    """

    def __init__(self, kernel):
        self.kernel = kernel
        self.server = _MCPServer("ada-tools")
        self.kernel_dir = Path(getattr(kernel, "kernel_dir",
                                       Path.cwd() / "ada_kernel"))
        self._versions = dict(TOOL_VERSIONS)
        self._sources: dict[str, str] = {}   # tool name -> kernel|custom-legacy|forged
        self._register_kernel_tools()
        self._register_custom_tools()
        self._register_forged_tools()
        self.save_manifest()
        _SERVERS.add(self)
        print(f"[MCP] local server built: {len(self._sources)} tools "
              f"({sum(1 for s in self._sources.values() if s == 'kernel')} kernel, "
              f"{sum(1 for s in self._sources.values() if s == 'custom-legacy')} legacy, "
              f"{sum(1 for s in self._sources.values() if s == 'forged')} forged)")

    # -- registration ------------------------------------------------------
    def _register_kernel_tools(self) -> None:
        from agent_kernel import KERNEL_TOOLS
        for entry in KERNEL_TOOLS:
            name = entry["name"]
            version = self._versions.get(name, DEFAULT_TOOL_VERSION)
            schema = _to_json_schema(entry.get("parameters"))
            desc = entry.get("description", "")

            async def _exec(n, a, _name=name, _v=version):
                return await _exec_kernel(self.kernel, _name, a, _v)

            fn = _build_wrapper(name, schema, desc, _exec)
            self._unregister_quiet(name)
            self.server.add_tool(fn, name=name, description=desc,
                                 meta={"version": version, "source": "kernel"})
            self._sources[name] = "kernel"

    def _register_custom_tools(self) -> None:
        forge = getattr(self.kernel, "forge", None)
        registry = getattr(forge, "legacy_registry", None) or {}
        for name, fn in registry.items():
            self.register_custom_tool(name, fn)

    def _register_forged_tools(self) -> None:
        forge = getattr(self.kernel, "forge", None)
        registry = getattr(forge, "mcp_registry", None) or {}
        for name, entry in registry.items():
            self.register_forged_tool(entry)

    def register_custom_tool(self, name: str, fn, version: str = "0.1.0") -> None:
        """Expose one legacy custom_tools/*.py callable as an MCP tool.

        The schema comes from the REAL function signature (auto-wrapped),
        but execution routes through the deprecated forge path so the
        deprecation notice + versioned audit stay in one place.
        """
        self._unregister_quiet(name)
        forge = getattr(self.kernel, "forge", None)
        kernel = self.kernel

        async def _impl(**kwargs):
            return await _exec_custom_legacy(forge, kernel, name,
                                             dict(kwargs), version)

        _impl.__name__ = re.sub(r"\W+", "_", name) or "tool"
        _impl.__doc__ = (fn.__doc__ or "").strip()
        try:
            _impl.__signature__ = inspect.signature(fn)
        except Exception:
            pass
        self.server.add_tool(_impl, name=name,
                             description=(fn.__doc__ or "").strip(),
                             meta={"version": version, "source": "custom-legacy",
                                   "deprecated": True})
        self._sources[name] = "custom-legacy"
        self.save_manifest()

    def register_forged_tool(self, entry: dict) -> None:
        """Expose one forged mcp_tools/*.py module as an MCP tool."""
        name = entry.get("name") or entry.get("tool_name") or "tool"
        version = entry.get("version", DEFAULT_TOOL_VERSION)
        schema = entry.get("schema") or {"type": "object"}
        desc = entry.get("description", "")

        async def _exec(n, a, _entry=entry, _name=name, _v=version):
            return await _exec_forged(_entry, self.kernel, _name, a, _v)

        fn = _build_wrapper(name, schema, desc, _exec)
        self._unregister_quiet(name)
        self.server.add_tool(fn, name=name, description=desc,
                             meta={"version": version, "source": "forged"})
        self._sources[name] = "forged"
        self.save_manifest()

    def unregister_tool(self, name: str) -> bool:
        """Remove a tool at runtime (hot-load). Returns True if it existed."""
        try:
            self.server.remove_tool(name)
        except Exception:
            return False
        self._sources.pop(name, None)
        self.save_manifest()
        return True

    def _unregister_quiet(self, name: str) -> None:
        try:
            self.server.remove_tool(name)
        except Exception:
            pass
        self._sources.pop(name, None)

    # -- introspection -----------------------------------------------------
    def has_tool(self, name: str) -> bool:
        return self.server._tool_manager.get_tool(name) is not None

    def tool_version(self, name: str) -> str | None:
        tool = self.server._tool_manager.get_tool(name)
        if tool is None:
            return None
        return (tool.meta or {}).get("version")

    def tool_source(self, name: str) -> str | None:
        return self._sources.get(name)

    async def list_tools(self):
        """MCP tool listing (carries meta.version per tool)."""
        return await self.server.list_tools()

    def describe(self, name: str) -> dict | None:
        """name/description/parameters for brain tool-list merging."""
        tool = self.server._tool_manager.get_tool(name)
        if tool is None:
            return None
        return {"name": tool.name, "description": tool.description,
                "parameters": tool.parameters,
                "version": (tool.meta or {}).get("version"),
                "source": self._sources.get(name)}

    # -- manifest ----------------------------------------------------------
    def manifest(self) -> dict:
        tools = {}
        for name, source in sorted(self._sources.items()):
            info = self.describe(name) or {}
            tools[name] = {
                "version": info.get("version") or DEFAULT_TOOL_VERSION,
                "description": (info.get("description") or "")[:300],
                "source": source,
                "deprecated": bool(source == "custom-legacy"),
            }
        return {"generated_ts": time.time(), "tools": tools}

    def save_manifest(self) -> Path:
        path = self.kernel_dir / MANIFEST_NAME
        try:
            path.write_text(json.dumps(self.manifest(), indent=1))
        except Exception as e:
            print(f"[MCP] manifest write failed: {e}")
        return path

    # -- execution / transport ----------------------------------------------
    async def call(self, name: str, args: dict | None):
        """In-process tool call (what the brain's dispatch uses)."""
        return await self.server.call_tool(name, dict(args or {}))

    async def serve(self, host: str = "127.0.0.1",
                    port: int = DEFAULT_MCP_PORT) -> None:
        """Serve Streamable HTTP. Runs until cancelled."""
        print(f"[MCP] serving Streamable HTTP on {host}:{port}")
        await self.server.run_streamable_http_async(host=host, port=port)

    def app(self):
        """Starlette app for mounting into the existing FastAPI server."""
        return self.server.streamable_http_app()


def create_local_server(kernel) -> AdaMCPServer:
    """Build the local MCP server around a kernel (or test double)."""
    return AdaMCPServer(kernel)


def tool_version(name: str) -> str | None:
    """Newest known version of a tool across live servers (None if unknown)."""
    for srv in list(_SERVERS):
        v = srv.tool_version(name)
        if v:
            return v
    return None


def mcp_tool_declarations() -> list[dict]:
    """Hot-loaded (non-kernel) tools in KERNEL_TOOLS dict shape.

    Used to merge custom/forged tools into the brain's tool list without a
    restart. Kernel tools are skipped — they're already in the static list.
    """
    decls: list[dict] = []
    seen: set[str] = set()
    for srv in list(_SERVERS):
        for name, source in srv._sources.items():
            if source == "kernel" or name in seen:
                continue
            info = srv.describe(name)
            if not info:
                continue
            seen.add(name)
            decls.append({"name": name,
                          "description": info.get("description", ""),
                          "parameters": info.get("parameters") or {"type": "object"}})
    return decls
