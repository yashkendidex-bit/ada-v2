import os
import json
import shutil
import time
from pathlib import Path

class ProjectManager:
    """Single living workspace — NO automatic projects.

    Everything ADA models lives in one workspace/ folder:
      workspace/
        chat_history.jsonl     (conversation log)
        cad/                   (current working files)
        history/               (timestamped snapshots of every model)
    Projects are only created when Sir EXPLICITLY asks for one — never from chat.
    """

    def __init__(self, workspace_root: str):
        self.workspace_root = Path(workspace_root)
        self.workspace = self.workspace_root / "workspace"
        self.projects_dir = self.workspace_root / "projects"  # legacy, unused

        for sub in ("", "cad", "history"):
            p = self.workspace / sub if sub else self.workspace
            p.mkdir(parents=True, exist_ok=True)
        print("[Workspace] Single workspace ready (no auto-projects).")

    # ── Explicit-only project tools (never called automatically) ─────────────
    def create_project(self, name: str):
        return False, ("No need for projects anymore, Sir — everything lives in "
                       "one workspace. I can start a fresh history instead if you like.")

    def switch_project(self, name: str):
        return False, "Single workspace mode — there is nothing to switch."

    def list_projects(self):
        return []

    # ── Workspace paths ──────────────────────────────────────────────────────
    def get_current_project_path(self):
        """Compat alias: the single workspace root."""
        return self.workspace

    @property
    def current_project(self):
        return "workspace"

    # ── Logging & artifacts ──────────────────────────────────────────────────
    def log_chat(self, sender: str, text: str):
        try:
            log_file = self.workspace / "chat_history.jsonl"
            entry = {"timestamp": time.time(), "sender": sender, "text": text}
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"[Workspace] chat log failed: {e}")

    def save_cad_artifact(self, source_path: str, prompt: str):
        """Snapshot a generated model into workspace/history/."""
        if not source_path or not os.path.exists(source_path):
            return None
        timestamp = int(time.time())
        safe_prompt = "".join([c for c in (prompt or "model") if c.isalnum() or c in (' ', '-', '_')])[:30].strip().replace(" ", "_") or "model"
        ext = os.path.splitext(source_path)[1] or ".stl"
        dest_path = self.workspace / "history" / f"{timestamp}_{safe_prompt}{ext}"
        try:
            shutil.copy2(source_path, dest_path)
            print(f"[Workspace] Model snapshot: {dest_path.name}")
            return str(dest_path)
        except Exception as e:
            print(f"[Workspace] snapshot failed: {e}")
            return None

    def recent_models(self, limit: int = 10):
        """Latest model snapshots, newest first."""
        h = self.workspace / "history"
        if not h.exists():
            return []
        files = sorted(h.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        return [str(p) for p in files[:limit]]

    # ── Legacy compat (unused in single-workspace mode) ──────────────────────
    def get_project_context(self, max_file_size: int = 10000) -> str:
        return "Single workspace mode — context comes from kernel memory."

    def get_recent_chat_history(self, limit: int = 10):
        log_file = self.workspace / "chat_history.jsonl"
        if not log_file.exists():
            return []
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            history = []
            for line in lines[-limit:]:
                try:
                    history.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            return history
        except Exception:
            return []
