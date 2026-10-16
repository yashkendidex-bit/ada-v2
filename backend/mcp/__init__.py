"""ADA's MCP-native tool layer — Feature 2 (MCP-native tool forge).

``backend/mcp/`` shadows the top-level ``mcp`` name (``backend/`` is on
``sys.path``), so the canonical PyPI SDK is reached through
:mod:`mcp._sdk` (see that module for the loader mechanics). Everything
below is ADA's own MCP machinery, built on that SDK:

- :mod:`mcp.server` — the local MCP server (Streamable HTTP on
  127.0.0.1), auto-wrapping every ``KERNEL_TOOLS`` entry, every legacy
  ``ada_kernel/custom_tools/*.py`` tool, and every forged
  ``ada_kernel/mcp_tools/*.py`` module.
- :mod:`mcp.client` — the client the brain dispatches through: in-process
  calls to the local server, plus opt-in external MCP servers.
- :mod:`mcp.forge` — ``MCPForge``: newly forged tools are emitted as MCP
  tool-definition modules under ``ada_kernel/mcp_tools/``. The old
  in-process ``exec()`` path survives only as a *deprecated* fallback for
  already-registered legacy tools.
- :mod:`mcp.hotload` — directory watcher: diffs the tool manifest and
  adds/removes tools at runtime, no restart.
"""

from ._sdk import SDK  # noqa: F401  (loads + re-keys the canonical SDK)

from .server import (  # noqa: F401
    AdaMCPServer,
    DEFAULT_MCP_PORT,
    TOOL_VERSIONS,
    create_local_server,
    mcp_tool_declarations,
    tool_version,
)
from .client import (  # noqa: F401
    MCPToolClient,
    MCPSecurityError,
    ensure_default_config,
    is_localhost_url,
)
from .forge import MCPForge  # noqa: F401
from .hotload import MCPHotLoader  # noqa: F401

__all__ = [
    "SDK",
    "AdaMCPServer",
    "DEFAULT_MCP_PORT",
    "TOOL_VERSIONS",
    "create_local_server",
    "mcp_tool_declarations",
    "tool_version",
    "MCPToolClient",
    "MCPSecurityError",
    "ensure_default_config",
    "is_localhost_url",
    "MCPForge",
    "MCPHotLoader",
]
