"""MCP client — Feature 2 (MCP-native tool forge).

Two call paths:

* **Local** (:meth:`MCPToolClient.call_local`): in-process call into the
  local :class:`mcp.server.AdaMCPServer` — this is what the brain's
  ``_execute_tool`` uses. No HTTP hop. The F0 ``_check_confirmation`` gate
  runs at the dispatch call site *before* this; this client never gates,
  it only executes. Permission maps ride the ``permissions_ctx`` var so
  ``kernel.handle_tool`` keeps its existing permission check.
* **External** (:meth:`MCPToolClient.call_external`): Streamable HTTP to a
  third-party MCP server listed in ``ada_kernel/mcp_servers.json``.
  Localhost URLs are allowed freely; anything else needs an explicit
  ``"opt_in": true`` recorded in the config — otherwise
  :class:`MCPSecurityError`.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from urllib.parse import urlparse

import httpx2

from .server import permissions_ctx

_streamable_http_client = importlib.import_module(
    "_mcp_sdk.client.streamable_http").streamable_http_client
_ClientSession = importlib.import_module("_mcp_sdk.client.session").ClientSession

SERVERS_CONFIG_NAME = "mcp_servers.json"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


class MCPSecurityError(Exception):
    """Raised when an external MCP server URL is not localhost and has no
    recorded user opt-in."""


def is_localhost_url(url: str) -> bool:
    """True only for loopback destinations (no extra approval needed)."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    return host in LOCAL_HOSTS


def ensure_default_config(path: Path) -> Path:
    """Create the external-servers config (empty list) if it is missing."""
    path = Path(path)
    if not path.exists():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"servers": []}, indent=1))
        except Exception as e:
            print(f"[MCP] could not write default servers config: {e}")
    return path


def _normalize_result(name: str, res) -> dict:
    """MCP CallToolResult -> ADA's ``{"result": ...}`` envelope."""
    texts: list[str] = []
    is_error = bool(getattr(res, "is_error", False))
    for content in getattr(res, "content", []) or []:
        text = getattr(content, "text", None)
        if text is not None:
            texts.append(text)
        else:
            try:
                texts.append(json.dumps(content))
            except Exception:
                texts.append(str(content))
    text = "".join(texts)
    if is_error and not text.startswith("Tool "):
        text = f"Tool '{name}' failed: {text or 'unknown error'}"
    return {"result": text}


class MCPToolClient:
    """Dispatch client used by AgentKernel / the brain."""

    def __init__(self, server, kernel=None, config_path: Path | None = None):
        self.server = server          # mcp.server.AdaMCPServer
        self.kernel = kernel
        cfg = (Path(config_path) if config_path is not None
               else Path(server.kernel_dir) / SERVERS_CONFIG_NAME)
        self.config_path = ensure_default_config(cfg)

    # -- local (in-process) -------------------------------------------------
    async def call_local(self, name: str, args: dict | None = None,
                         permissions: dict | None = None) -> dict:
        """Execute a local tool through the MCP server, in-process.

        Returns ``{"result": ...}`` like the old direct dispatch. Any MCP
        transport error becomes a ``{"result": "Tool ... failed: ..."}``
        envelope — the turn loop already handles that shape.
        """
        token = permissions_ctx.set(permissions)
        try:
            try:
                res = await self.server.call(name, args or {})
            except Exception as e:
                return {"result": f"Tool '{name}' failed: {str(e)[:200]}"}
            return _normalize_result(name, res)
        finally:
            permissions_ctx.reset(token)

    # -- external servers ----------------------------------------------------
    def _load_config(self) -> dict:
        try:
            data = json.loads(self.config_path.read_text())
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return {"servers": []}

    def _find_server(self, server_name: str) -> dict:
        for entry in self._load_config().get("servers", []) or []:
            if isinstance(entry, dict) and entry.get("name") == server_name:
                return entry
        raise KeyError(f"unknown MCP server '{server_name}' "
                       f"(see {self.config_path})")

    def _checked_url(self, entry: dict) -> str:
        url = str(entry.get("url", "")).strip()
        if not url:
            raise MCPSecurityError(
                f"MCP server '{entry.get('name')}' has no url configured")
        if is_localhost_url(url):
            return url
        if entry.get("opt_in") is True:
            return url
        raise MCPSecurityError(
            f"refusing non-localhost MCP server '{entry.get('name')}' ({url}): "
            f"localhost-only URLs need no approval, anything else needs an "
            f"explicit user opt-in recorded as \"opt_in\": true in "
            f"{self.config_path}")

    async def _session_call(self, url: str, tool_name: str,
                            args: dict, list_only: bool = False):
        # trust_env=False: MCP server traffic never goes through an egress
        # proxy (localhost must not, and env proxy parsing must not be able
        # to break an explicitly opted-in server either).
        http_client = httpx2.AsyncClient(trust_env=False)
        try:
            async with _streamable_http_client(
                    url, http_client=http_client) as streams:
                read, write = streams[0], streams[1]
                async with _ClientSession(read, write) as session:
                    await session.initialize()
                    if list_only:
                        return await session.list_tools()
                    return await session.call_tool(tool_name, dict(args))
        finally:
            try:
                await http_client.aclose()
            except Exception:
                pass

    async def call_external(self, server_name: str, tool_name: str,
                            args: dict | None = None,
                            timeout: float = 30.0) -> dict:
        """Call a tool on an external MCP server over Streamable HTTP."""
        entry = self._find_server(server_name)
        url = self._checked_url(entry)
        try:
            res = await self._session_call(url, tool_name, args or {})
        except MCPSecurityError:
            raise
        except Exception as e:
            return {"result": f"Tool '{tool_name}' failed: {str(e)[:200]}"}
        return _normalize_result(tool_name, res)

    async def list_external_tools(self, server_name: str) -> list[dict]:
        """List tools exposed by an external MCP server."""
        entry = self._find_server(server_name)
        url = self._checked_url(entry)
        res = await self._session_call(url, "", {}, list_only=True)
        out = []
        for tool in getattr(res, "tools", []) or []:
            out.append({"name": getattr(tool, "name", ""),
                        "description": getattr(tool, "description", ""),
                        "input_schema": getattr(tool, "inputSchema", None)
                        or getattr(tool, "input_schema", None)})
        return out

    def check_url_allowed(self, url: str) -> bool:
        """Public probe for tests/UI: is this URL callable without opt-in?"""
        return is_localhost_url(url)
