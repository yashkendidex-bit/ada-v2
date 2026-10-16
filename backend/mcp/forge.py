"""MCPForge — the ToolForge upgrade (Feature 2: MCP-native tool forge).

Migration contract (from the research brief):

* **New tools** are emitted as MCP tool-definition modules — one file per
  tool under ``ada_kernel/mcp_tools/`` — carrying ``TOOL_NAME`` /
  ``TOOL_DESCRIPTION`` / ``TOOL_SCHEMA`` / ``TOOL_VERSION`` metadata plus
  the validated function. The local MCP server (:mod:`mcp.server`) serves
  them; :mod:`mcp.hotload` picks them up without a restart.
* **The old in-process ``exec()`` path is RETIRED for new tools.**
  ``synthesise`` no longer ``exec()``s fresh code into the live
  ``custom_tools`` registry.
* **Legacy fallback (clearly marked deprecated):** tools forged *before*
  Feature 2 already live in ``ada_kernel/custom_tools/*.py``. They keep
  loading through the original ``exec()`` path and keep running through
  :meth:`MCPForge.run` — every such call prints a ``DEPRECATED`` notice.
  Nothing new may enter through that door.

The same AST whitelist/blacklist as the original forge guards every load
(including hot-loaded files, which may have been hand-edited).
"""

from __future__ import annotations

import ast
import re
import time
from pathlib import Path

from agent_kernel import ToolForge

MCP_TOOL_VERSION_DEFAULT = "1.0.0"

_ANNOTATION_TO_JSON = {
    "int": "integer",
    "str": "string",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
    "List": "array",
    "Dict": "object",
    "Sequence": "array",
    "Mapping": "object",
}


class MCPForge(ToolForge):
    """ToolForge that emits MCP tool definitions instead of exec'ing new code."""

    def __init__(self, tools_dir: Path, mcp_dir: Path):
        self.mcp_dir = Path(mcp_dir)
        self.mcp_dir.mkdir(parents=True, exist_ok=True)
        #: name -> {"fn", "schema", "description", "version", "path"}
        self.mcp_registry: dict[str, dict] = {}
        # Parent __init__ loads the LEGACY custom_tools/*.py dir via the
        # deprecated exec() path (kept for already-registered tools only).
        super().__init__(tools_dir)
        self._load_mcp_dir()

    # ------------------------------------------------------------------
    # legacy surface (deprecated)
    # ------------------------------------------------------------------
    @property
    def legacy_registry(self) -> dict:
        """The pre-Feature-2 in-process registry (custom_tools/*.py)."""
        return self.registry

    def reload_legacy(self, path: Path) -> None:
        """Re-exec one legacy custom_tools file (hot-load). Deprecated path."""
        self._compile_and_register(Path(path))

    def drop_legacy(self, name: str) -> None:
        self.registry.pop(name, None)

    async def run(self, name, args=None, timeout=30):
        """Run a tool.

        Forged (MCP) tools take the current path. Legacy custom_tools take
        the DEPRECATED in-process exec() fallback — the notice below is
        deliberate and loud.
        """
        if name in self.mcp_registry:
            return await self._run_mcp_tool(name, args, timeout)
        if name in self.registry:
            print(f"[FORGE] DEPRECATED: tool '{name}' ran through the legacy "
                  f"in-process exec() path (custom_tools/*.py). Re-forge it "
                  f"as an MCP tool definition to clear this warning.")
            return await super().run(name, args, timeout)
        return f"Tool '{name}' not found."

    def list_tools(self):
        return sorted(set(self.registry) | set(self.mcp_registry))

    # ------------------------------------------------------------------
    # the new path: synthesise -> MCP tool-definition module
    # ------------------------------------------------------------------
    def synthesise(self, name, code, description=None):
        """Forge a NEW tool as an MCP tool definition.

        Validates the code with the original AST rules, extracts the tool
        contract (schema from the function signature, description from the
        docstring), and writes ``ada_kernel/mcp_tools/<name>.py``. The code
        is NOT exec()'d into the legacy registry — the MCP server (via
        hot-load) serves it from the module file.
        """
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name or ""):
            return f"Invalid tool name: {name}"
        if any(b in code for b in self.BLOCKED):
            return "Tool rejected: contains blocked construct."
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return f"Tool rejected: syntax error: {e}"
        try:
            self._validate_ast(tree)
        except ValueError as e:
            return f"Tool rejected at validation: {e}"
        try:
            fn_name, schema, doc = self._extract_contract(name, tree)
        except ValueError as e:
            return f"Tool rejected: {e}"
        desc = (description or doc or f"Forged tool '{fn_name}'").strip()
        version = MCP_TOOL_VERSION_DEFAULT
        path = self.mcp_dir / f"{fn_name}.py"
        path.write_text(self._render_module(fn_name, desc, schema, version,
                                            code.rstrip() + "\n"))
        try:
            # Sanctioned load of the module we just wrote (validated twice:
            # here and inside _load_mcp_module) so the tool is callable
            # immediately, before the hot-loader's next scan.
            self._load_mcp_module(path)
        except Exception as e:
            path.unlink(missing_ok=True)
            self.mcp_registry.pop(fn_name, None)
            return f"Tool rejected at load: {e}"
        return (f"Tool '{fn_name}' created as MCP tool definition "
                f"(mcp_tools/{path.name}, v{version}). You can call it now — "
                f"hot-load serves it without a restart.")

    # ------------------------------------------------------------------
    # module loading (shared by __init__, synthesise, hot-load)
    # ------------------------------------------------------------------
    def _load_mcp_dir(self) -> None:
        for path in sorted(self.mcp_dir.glob("*.py")):
            try:
                self._load_mcp_module(path)
            except Exception as e:
                print(f"[FORGE] skip mcp tool {path.name}: {e}")

    def _load_mcp_module(self, path: Path) -> dict:
        """Validate + exec one mcp_tools/*.py module into mcp_registry."""
        path = Path(path)
        src = path.read_text()
        tree = ast.parse(src)
        self._validate_ast(tree)  # hand-edited files get the same gate
        ns: dict = {}
        exec(compile(tree, str(path), "exec"),
             {"__name__": f"mcp_tools.{path.stem}"}, ns)
        tool_name = str(ns.get("TOOL_NAME") or path.stem)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tool_name):
            raise ValueError(f"bad TOOL_NAME: {tool_name!r}")
        fn = ns.get(tool_name)
        if not callable(fn):
            # fallback: first public callable defined in the module
            for key, obj in ns.items():
                if callable(obj) and not key.startswith("_"):
                    fn = obj
                    break
        if not callable(fn):
            raise ValueError("no callable tool function found")
        schema = ns.get("TOOL_SCHEMA") or {"type": "object"}
        entry = {
            "name": tool_name,
            "fn": fn,
            "schema": schema,
            "description": str(ns.get("TOOL_DESCRIPTION") or
                               (fn.__doc__ or "")).strip(),
            "version": str(ns.get("TOOL_VERSION") or
                           MCP_TOOL_VERSION_DEFAULT),
            "deprecated": bool(ns.get("TOOL_DEPRECATED", False)),
            "path": path,
        }
        self.mcp_registry[tool_name] = entry
        return entry

    async def _run_mcp_tool(self, name, args=None, timeout=30):
        import asyncio
        entry = self.mcp_registry[name]
        fn = entry["fn"]
        args = dict(args or {})
        loop = asyncio.get_event_loop()

        async def _invoke():
            if asyncio.iscoroutinefunction(fn):
                return await fn(**args)
            return await loop.run_in_executor(None, lambda: fn(**args))

        try:
            return await asyncio.wait_for(_invoke(), timeout)
        except asyncio.TimeoutError:
            return f"Tool '{name}' timed out after {timeout}s."
        except TypeError as e:
            return f"Tool '{name}' bad args: {e}"
        except Exception as e:
            return f"Tool '{name}' error: {e}"

    # ------------------------------------------------------------------
    # contract extraction + module rendering
    # ------------------------------------------------------------------
    def _extract_contract(self, name: str, tree: ast.AST):
        """(fn_name, json_schema, docstring) from the synthesised code."""
        target = None
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name == name:
                target = node
                break
        if target is None:
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                        and not node.name.startswith("_"):
                    target = node
                    break
        if target is None:
            raise ValueError("no public function definition found")
        fn_name = target.name
        doc = (ast.get_docstring(target) or "").strip().split("\n")[0]
        properties: dict = {}
        required: list[str] = []
        pos = target.args
        defaults = [None] * (len(pos.args) - len(pos.defaults)) + list(pos.defaults)
        for arg, default in zip(pos.args, defaults):
            if arg.arg in ("self", "cls"):
                continue
            pspec: dict = {}
            if arg.annotation is not None:
                try:
                    ann = ast.unparse(arg.annotation)
                except Exception:
                    ann = ""
                base = re.split(r"[\[\]]", ann.strip().strip("'\""))[0].split(".")[-1]
                jtype = _ANNOTATION_TO_JSON.get(base)
                if jtype:
                    pspec["type"] = jtype
            properties[arg.arg] = pspec
            if default is None:
                required.append(arg.arg)
        schema: dict = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
        return fn_name, schema, doc

    def _render_module(self, name: str, description: str, schema: dict,
                       version: str, code: str) -> str:
        import json as _json
        header = (
            f'"""MCP tool {name!r} — forged by ADA\'s ToolForge (Feature 2).\n\n'
            "Served by backend/mcp/server.py via hot-load. The TOOL_* block "
            "below is the tool contract; bump TOOL_VERSION when it changes.\n"
            '"""\n'
            f"TOOL_NAME = {name!r}\n"
            f"TOOL_DESCRIPTION = {description!r}\n"
            f"TOOL_SCHEMA = {_json.dumps(schema, indent=2)}\n"
            f"TOOL_VERSION = {version!r}\n"
            "TOOL_DEPRECATED = False\n"
            "\n"
        )
        return header + code
