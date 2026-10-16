# `ada_kernel/mcp_tools/` — forged MCP tool definitions (Feature 2)

One Python module per tool, written by `MCPForge.synthesise()` (see
`backend/mcp/forge.py`). Each module carries a `TOOL_*` contract block:

```python
TOOL_NAME = "celsius_to_fahrenheit"
TOOL_DESCRIPTION = "..."
TOOL_SCHEMA = {"type": "object", "properties": {...}, "required": [...]}
TOOL_VERSION = "1.0.0"
TOOL_DEPRECATED = False

def celsius_to_fahrenheit(celsius: float) -> float: ...
```

The local MCP server (`backend/mcp/server.py`) serves these; the hot-loader
(`backend/mcp/hotload.py`) picks up new/changed/removed files without a
restart. Bump `TOOL_VERSION` when the contract changes.

Do NOT hand-edit the `TOOL_*` block — it is the versioned contract the
manifest (`ada_kernel/mcp_manifest.json`) records. The old
`ada_kernel/custom_tools/` directory is the deprecated pre-Feature-2 home;
new tools never go there.
