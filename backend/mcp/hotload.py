"""Hot-load — Feature 2 (MCP-native tool forge).

Watches ``ada_kernel/custom_tools/`` (legacy) and ``ada_kernel/mcp_tools/``
(forged) for added/changed/removed ``*.py`` files. On every scan the
watcher diffs against the previous snapshot AND the live tool manifest,
then adds/removes tools on the running :class:`mcp.server.AdaMCPServer`
without a restart.

Run as part of :class:`runtime.adapters.MCPServerService`; ``check_once()``
is the deterministic entry point tests drive.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path


class MCPHotLoader:
    """Directory watcher -> manifest diff -> live add/remove."""

    def __init__(self, mcp_server, forge=None, custom_dir=None, mcp_dir=None,
                 interval: float = 2.0):
        self.server = mcp_server
        self.forge = forge
        kernel_dir = Path(getattr(mcp_server, "kernel_dir", "."))
        self.custom_dir = Path(custom_dir) if custom_dir else kernel_dir / "custom_tools"
        self.mcp_dir = Path(mcp_dir) if mcp_dir else kernel_dir / "mcp_tools"
        self.interval = interval
        self._snap: dict[str, tuple] = {}
        self._task: asyncio.Task | None = None
        self._running = False

    # -- snapshots ---------------------------------------------------------
    def _iter_files(self):
        for directory in (self.custom_dir, self.mcp_dir):
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.py")):
                yield path

    def snapshot(self) -> dict[str, tuple]:
        snap: dict[str, tuple] = {}
        for path in self._iter_files():
            try:
                st = path.stat()
                snap[str(path)] = (st.st_mtime_ns, st.st_size)
            except OSError:
                continue
        return snap

    # -- diff + apply --------------------------------------------------------
    def check_once(self) -> list[str]:
        """One scan: apply adds/changes/removes. Returns human-readable changes."""
        cur = self.snapshot()
        prev = self._snap
        changes: list[str] = []
        for path, sig in cur.items():
            if path not in prev:
                changes.append(self._apply_added(Path(path)))
            elif prev[path] != sig:
                changes.append(self._apply_changed(Path(path)))
        for path in prev:
            if path not in cur:
                changes.append(self._apply_removed(Path(path)))
        self._snap = cur
        # The manifest is the diffable record of the live tool surface.
        try:
            self.server.save_manifest()
        except Exception:
            pass
        return [c for c in changes if c]

    def _in_mcp_dir(self, path: Path) -> bool:
        try:
            path.relative_to(self.mcp_dir)
            return True
        except ValueError:
            return False

    def _apply_added(self, path: Path) -> str:
        return self._load_file(path, "added")

    def _apply_changed(self, path: Path) -> str:
        return self._load_file(path, "changed")

    def _load_file(self, path: Path, kind: str) -> str:
        forge = self.forge
        if self._in_mcp_dir(path):
            if forge is None:
                return ""
            try:
                entry = forge._load_mcp_module(path)
            except Exception as e:
                return f"{kind} {path.name}: rejected ({e})"
            try:
                self.server.register_forged_tool(entry)
            except Exception as e:
                return f"{kind} {path.name}: register failed ({e})"
            return (f"{kind} forged tool '{entry['name']}' "
                    f"v{entry['version']} (hot-loaded)")
        # legacy custom_tools path — deprecated exec(), kept for old tools
        if forge is None:
            return ""
        try:
            forge.reload_legacy(path)
        except Exception as e:
            return f"{kind} {path.name}: legacy reload rejected ({e})"
        for name, fn in (forge.legacy_registry or {}).items():
            try:
                self.server.register_custom_tool(name, fn)
            except Exception:
                pass
        return f"{kind} legacy tool file '{path.name}' (deprecated exec path)"

    def _apply_removed(self, path: Path) -> str:
        name = path.stem
        forge = self.forge
        if self._in_mcp_dir(path):
            if forge is not None:
                forge.mcp_registry.pop(name, None)
        else:
            if forge is not None:
                forge.drop_legacy(name)
        removed = self.server.unregister_tool(name)
        # A legacy file may register several names; sweep stragglers whose
        # source file is gone is best-effort — the manifest is the record.
        return f"removed '{name}'" if removed else ""

    # -- background loop -------------------------------------------------------
    async def start(self) -> None:
        """Begin background scanning. Idempotent."""
        if self._running:
            return
        self._running = True
        self._snap = self.snapshot()  # baseline: don't replay existing files
        self._task = asyncio.create_task(self._loop(), name="mcp-hotload")

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    async def _loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self.interval)
                if self._running:
                    self.check_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"[MCP] hot-load scan failed: {e}")
