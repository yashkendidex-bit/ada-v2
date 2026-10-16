"""ADA Agent Kernel - 8-tier Jarvis upgrade in one cohesive module.

  T1  Memory     : 3-tier (core/recall/archival) + episodic graph + self-editing + recall
  T4  Autonomy   : scheduler (reminders / recurring / watch-folders) + self-extending tools + orchestrator
  T5  Knowledge  : document indexer (chunk -> embed -> search) + notebook memory files
  T6  Integrate  : system ops (clipboard, system info, app launch, screenshot) + email/calendar/smart-home hooks
  T7  Planning   : plan -> execute -> verify decomposition + OCR / audio-understanding
  T8  Presence   : wake-word hook, presence gating, ambient context, HUD events
  T9  Telemetry  : per-turn metrics, reflection loop, eval replay suite
  T10 Security   : injection guards, permission gates, audit log, credential vault, sandboxed tool execution

Dependencies: stdlib + numpy + (optional) sentence-transformers. All persistence via SQLite.
"""

import asyncio
import ast
import base64
import datetime as _dt
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False

try:
    import PIL.Image as PILImage
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False


# --------------------------------------------------------------------------
# Embeddings (T5 / T1): sentence-transformer when present, hashed TF-IDF fallback
# --------------------------------------------------------------------------
class _Embedder:
    def __init__(self):
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None and _HAS_ST:
                    try:
                        self._model = SentenceTransformer("all-MiniLM-L6-v2")
                    except Exception:
                        self._model = False
                elif self._model is None:
                    self._model = False
        return self._model

    def embed(self, text):
        model = self._load()
        if model:
            vec = model.encode(str(text)[:8192], normalize_embeddings=True)
            return vec.tolist()
        return self._tfidf_hash(text)

    @staticmethod
    def _tokens(text):
        return re.findall(r"[a-z0-9]+", text.lower())

    def _tfidf_hash(self, text, dim=384):
        vec = np.zeros(dim, dtype=np.float32)
        toks = self._tokens(text)
        for tok in toks:
            h = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16) % dim
            vec[h] += 1.0
        n = float(np.linalg.norm(vec))
        if n > 0:
            vec /= n
        return vec.tolist()

    def similarity(self, a, b):
        va = np.asarray(a, dtype=np.float32)
        vb = np.asarray(b, dtype=np.float32)
        if va.size != vb.size:
            return 0.0
        return float(np.dot(va, vb) / max(1e-9, np.linalg.norm(va) * np.linalg.norm(vb)))


_embedder = _Embedder()


# --------------------------------------------------------------------------
# T1 Memory
# --------------------------------------------------------------------------
class MemoryBank:
    """Compatibility shim. The implementation lives in `memory/`.

    This class used to be ~175 lines of storage: five tables, an MD5 hashed
    bag-of-words embedding fallback, a full-table cosine scan, and six regular
    expressions for fact extraction. Over the life of the database it captured one
    fact and produced zero graph edges.

    Every method below keeps its old name and signature so the kernel's tools did not
    have to change, but delegates to `memory.KernelMemory`, which routes into the real
    subsystem. See ADA_MEMORY_ARCHITECTURE.md.

    The import is deferred to first use: pulling a transformer at module import time
    would slow the whole application boot, and `agent_kernel` is imported early.
    """

    def __new__(cls, db_path, llm=None):
        from memory.adapter import KernelMemory
        return KernelMemory(db_path, llm=llm)



# --------------------------------------------------------------------------
# T5 Knowledge - document indexer + RAG
# --------------------------------------------------------------------------
_TEXT_EXT = {".py", ".txt", ".md", ".json", ".csv", ".html", ".xml", ".yaml", ".yml",
             ".js", ".ts", ".jsx", ".tsx", ".css", ".sh", ".cfg", ".ini", ".toml"}
_BINARY_EXT = {".png", ".jpg", ".jpeg", ".gif", ".mp3", ".wav", ".mp4", ".stl", ".obj", ".zip"}


class DocIndexer:
    def __init__(self, memory: MemoryBank):
        self.memory = memory
        self.scanned_paths = set()
        self._last_scan = {}

    def _read_text(self, path):
        try:
            data = Path(path).read_bytes()
            for enc in ("utf-8", "latin-1"):
                try:
                    return data.decode(enc)
                except Exception:
                    continue
        except Exception:
            return None
        return None

    def index_file(self, path, force=False):
        p = Path(path)
        key = str(p.resolve())
        if not force and self._last_scan.get(key, 0) >= p.stat().st_mtime:
            return False
        text = self._read_text(p)
        if not text:
            return False
        # Hostile-content guard: indexed chunks later reach the brain through
        # retrieval, so strip instruction-shaped text at INDEX time. This also
        # covers the watch-folder auto-index path.
        try:
            text, flagged = Security.sanitize(text)
            if flagged:
                print(f"[KERNEL] [DOCS] injection-shaped content stripped from {p.name}")
        except Exception:
            pass
        chunks = self._chunk(text, max_chars=900)
        for i, chunk in enumerate(chunks):
            self.memory.add_recall(f"[{p.name}:{i}] {chunk}", kind="document")
        self._last_scan[key] = p.stat().st_mtime
        self.scanned_paths.add(key)
        return True

    @staticmethod
    def _chunk(text, max_chars=900, overlap=80):
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= max_chars:
            return [text] if text else []
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start = max(0, end - overlap)
        return chunks

    def index_folder(self, folder, recursive=True):
        root = Path(folder)
        if not root.exists():
            return f"Folder does not exist: {folder}"
        count = 0
        it = root.rglob("*") if recursive else root.glob("*")
        for p in it:
            if p.is_file() and p.suffix.lower() in _TEXT_EXT and p.name != "output.stl":
                if self.index_file(p):
                    count += 1
        return f"Indexed {count} files from {folder}."

    def search(self, query, k=6):
        return self.memory.search_recall(query, k=k, kind="document")


# --------------------------------------------------------------------------
# T4 Autonomy - scheduler + watch-folders + orchestrator + self-extending tools
# --------------------------------------------------------------------------
class Scheduler:
    def __init__(self, db_path: Path, on_fire):
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self.on_fire = on_fire
        self._task = None
        self._stop = asyncio.Event()
        self._wake = asyncio.Event()
        with self._lock:
            self._conn.execute("CREATE TABLE IF NOT EXISTS scheduled (id TEXT PRIMARY KEY,"
                               "kind TEXT, descr TEXT, next_run REAL, interval_s REAL, extra TEXT)")
            self._conn.commit()

    def schedule_once(self, descr, delay_s, extra=None):
        return self._insert("once", descr, time.time() + delay_s, None, extra)

    def schedule_recurring(self, descr, interval_s, extra=None):
        return self._insert("recurring", descr, time.time() + interval_s, interval_s, extra)

    def _insert(self, kind, descr, next_run, interval_s, extra):
        tid = uuid.uuid4().hex[:12]
        with self._lock:
            self._conn.execute("INSERT INTO scheduled(id,kind,descr,next_run,interval_s,extra) VALUES(?,?,?,?,?,?)",
                               (tid, kind, descr, next_run, interval_s, json.dumps(extra or {})))
            self._conn.commit()
        self._wake.set()
        return tid

    def cancel(self, tid):
        with self._lock:
            self._conn.execute("DELETE FROM scheduled WHERE id=?", (tid,))
            self._conn.commit()
        return f"Cancelled task {tid}"

    def list(self):
        with self._lock:
            rows = self._conn.execute("SELECT id,kind,descr,next_run FROM scheduled ORDER BY next_run").fetchall()
        return [dict(r) for r in rows]

    async def start(self):
        if self._task and not self._task.done():
            return
        self._stop.clear()

        async def loop():
            while not self._stop.is_set():
                with self._lock:
                    row = self._conn.execute("SELECT MIN(next_run) AS n FROM scheduled").fetchone()
                if row and row["n"] is not None:
                    delay = max(0.5, row["n"] - time.time())
                else:
                    delay = 30
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass
                self._wake.clear()
                if self._stop.is_set():
                    break
                due = []
                with self._lock:
                    now = time.time()
                    rows = self._conn.execute(
                        "SELECT * FROM scheduled WHERE next_run <= ?", (now,)).fetchall()
                    for r in rows:
                        due.append(dict(r))
                        if r["interval_s"]:
                            self._conn.execute("UPDATE scheduled SET next_run=? WHERE id=?",
                                               (now + r["interval_s"], r["id"]))
                        else:
                            self._conn.execute("DELETE FROM scheduled WHERE id=?", (r["id"],))
                    self._conn.commit()
                for item in due:
                    try:
                        await self.on_fire(item)
                    except Exception as e:
                        print(f"[KERNEL] [SCHED] fire failed: {e}")

        self._task = asyncio.create_task(loop())

    async def stop(self):
        self._stop.set()
        if self._task:
            self._task.cancel()


class WatchFolder:
    def __init__(self, on_change, poll_interval=10):
        self.on_change = on_change
        self.poll_interval = poll_interval
        self._seen = {}
        self._task = None
        self._stop = asyncio.Event()

    def add(self, folder, patterns=("*",)):
        folder = str(folder)
        base = {"patterns": patterns, "seen": {}}
        root = Path(folder)
        if root.exists():
            for p in root.rglob("*"):
                if p.is_file():
                    st = p.stat()
                    base["seen"][str(p)] = st.st_mtime
        self._seen[folder] = base

    async def start(self):
        if self._task and not self._task.done():
            return
        self._stop.clear()

        async def loop():
            while not self._stop.is_set():
                changed = []
                for folder, base in self._seen.items():
                    root = Path(folder)
                    if not root.exists():
                        continue
                    now_seen = {}
                    for p in root.rglob("*"):
                        if not p.is_file():
                            continue
                        key = str(p)
                        st = p.stat()
                        now_seen[key] = st.st_mtime
                        if base["seen"].get(key, -1) != st.st_mtime:
                            changed.append(key)
                    base["seen"] = now_seen
                for path in changed:
                    try:
                        await self.on_change(path)
                    except Exception as e:
                        print(f"[KERNEL] [WATCH] {e}")
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self.poll_interval)
                except asyncio.TimeoutError:
                    pass

        self._task = asyncio.create_task(loop())

    async def stop(self):
        self._stop.set()
        if self._task:
            self._task.cancel()


class ToolForge:
    """Self-extending tools: synthesise -> AST validate -> sandbox -> load."""
    # NOTE (2026-10-01): `os` was REMOVED from this whitelist. It used to be here
    # while BLOCKED only contained the substring "os.system" — so os.remove,
    # os.environ, os.listdir etc. sailed through AST validation and ran
    # in-process. Forged tools are pure-logic helpers; they do not need the OS.
    SAFE_IMPORTS = {"math", "json", "re", "datetime", "time", "random", "statistics",
                    "collections", "itertools", "functools", "pathlib", "typing", "uuid"}
    BLOCKED = {"importlib", "subprocess", "ctypes", "socket", "sys", "os.system", "eval", "exec",
               "open(", "__import__", "base64", "pickle", "shutil", "glob"}

    def __init__(self, tools_dir: Path):
        self.tools_dir = Path(tools_dir)
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        self.registry = {}
        self._load_existing()

    def _load_existing(self):
        for p in self.tools_dir.glob("*.py"):
            try:
                self._compile_and_register(p)
            except Exception as e:
                print(f"[KERNEL] [FORGE] skip {p.name}: {e}")

    def _compile_and_register(self, path):
        src = path.read_text()
        tree = ast.parse(src)
        self._validate_ast(tree)
        ns = {}
        exec(compile(tree, str(path), "exec"), {"__name__": "custom_tools"}, ns)
        for name, obj in ns.items():
            if callable(obj) and not name.startswith("_"):
                self.registry[name] = obj
        return tree

    def _validate_ast(self, tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    root = a.name.split(".")[0]
                    if root not in self.SAFE_IMPORTS:
                        raise ValueError(f"import not allowed: {a.name}")
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                if root not in self.SAFE_IMPORTS:
                    raise ValueError(f"import not allowed: {node.module}")
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in (
                        "eval", "exec", "open", "__import__", "getattr", "setattr",
                        "delattr", "compile", "globals", "locals", "vars",
                        "input", "breakpoint"):
                    raise ValueError(f"blocked call: {node.func.id}")
                if isinstance(node.func, ast.Attribute):
                    # os.remove / os.environ / sys.modules / subprocess.Popen —
                    # the old check only caught bare .system/.popen and missed
                    # every other dangerous attribute on a dangerous module.
                    base = node.func.value
                    if isinstance(base, ast.Name) and base.id in (
                            "os", "sys", "subprocess", "shutil", "importlib",
                            "ctypes", "socket", "builtins"):
                        raise ValueError(f"blocked: {base.id}.{node.func.attr}")
                    if node.func.attr in (
                            "system", "popen", "remove", "unlink", "rmdir", "mkdir",
                            "rename", "replace", "chmod", "chown", "write_text",
                            "write_bytes", "execv", "execve", "spawnl", "spawnlp",
                            "spawnv", "spawnve", "call", "run", "check_output",
                            "check_call", "getenv", "putenv", "environ",
                            "mkdtemp", "mkstemp", "symlink", "link", "truncate"):
                        raise ValueError(f"blocked call: .{node.func.attr}")

    def synthesise(self, name, code):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name or ""):
            return f"Invalid tool name: {name}"
        if any(b in code for b in self.BLOCKED):
            return f"Tool rejected: contains blocked construct."
        path = self.tools_dir / f"{name}.py"
        try:
            tree = ast.parse(code)
            self._validate_ast(tree)
        except SyntaxError as e:
            return f"Tool rejected: syntax error {e}"
        except ValueError as e:
            return f"Tool rejected: {e}"
        path.write_text(code)
        try:
            self._compile_and_register(path)
        except Exception as e:
            path.unlink(missing_ok=True)
            return f"Tool rejected at load: {e}"
        return f"Tool '{name}' created and loaded. You can call it now."

    async def run(self, name, args=None, timeout=30):
        if name not in self.registry:
            return f"Tool '{name}' not found."
        fn = self.registry[name]
        args = dict(args or {})
        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(loop.run_in_executor(None, lambda: fn(**args)), timeout)
            return result
        except asyncio.TimeoutError:
            return f"Tool '{name}' timed out after {timeout}s."
        except TypeError as e:
            return f"Tool '{name}' bad args: {e}"
        except Exception as e:
            return f"Tool '{name}' error: {e}"

    def list_tools(self):
        return sorted(self.registry.keys())


# --------------------------------------------------------------------------
# T6 Integrations - system ops + mail/calendar/smart-home hooks
# --------------------------------------------------------------------------
class Integrations:
    def __init__(self, kernel_dir: Path):
        self.kernel_dir = kernel_dir
        self.cache_dir = kernel_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def system_info(self):
        info = {
            "host": os.uname().nodename,
            "os": f"{os.uname().sysname} {os.uname().release}",
            "cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True).stdout.strip(),
            "cores": os.cpu_count(),
            "uptime": subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip(),
            "mem": subprocess.run(["vm_stat"], capture_output=True, text=True).stdout.strip().splitlines()[1:3],
        }
        return json.dumps(info, indent=1)

    def clipboard(self, action="read", text=None):
        if action == "read":
            return subprocess.run(["pbpaste"], capture_output=True, text=True).stdout
        if action == "write" and text is not None:
            p = subprocess.run(["pbcopy"], input=text, text=True, capture_output=True)
            return "Clipboard updated." if p.returncode == 0 else "Failed to write clipboard."
        return "Use action='read' or action='write' with text."

    def screenshot(self, region=None):
        import pyautogui
        img = pyautogui.screenshot()
        out = self.cache_dir / f"shot_{int(time.time())}.png"
        img.save(str(out))
        return str(out)

    def launch_app(self, app_name):
        apps = {
            "browser": ["open", "-a", "Google Chrome"], "chrome": ["open", "-a", "Google Chrome"],
            "safari": ["open", "-a", "Safari"], "vscode": ["open", "-a", "Visual Studio Code"],
            "terminal": ["open", "-a", "Terminal"], "finder": ["open", "-a", "Finder"],
            "vs code": ["open", "-a", "Visual Studio Code"], "code": ["open", "-a", "Visual Studio Code"],
            "blender": ["open", "-a", "Blender"], "spotify": ["open", "-a", "Spotify"],
            "music": ["open", "-a", "Music"], "mail": ["open", "-a", "Mail"],
            "notes": ["open", "-a", "Notes"], "calendar": ["open", "-a", "Calendar"],
        }
        cmd = apps.get(app_name.lower().strip())
        if not cmd:
            cmd = ["open", "-a", app_name]
        try:
            subprocess.run(cmd, check=True, timeout=10)
            return f"Launched {app_name}."
        except Exception as e:
            return f"Could not launch {app_name}: {e}"

    def create_ics_event(self, summary, start_iso, end_iso=None, description=""):
        from datetime import datetime, timedelta
        start = datetime.fromisoformat(start_iso)
        end = datetime.fromisoformat(end_iso) if end_iso else start + timedelta(hours=1)
        ics = (f"BEGIN:VCALENDAR\nVERSION:2.0\nBEGIN:VEVENT\n"
               f"SUMMARY:{summary}\nDTSTART:{start.strftime('%Y%m%dT%H%M%S')}\n"
               f"DTEND:{end.strftime('%Y%m%dT%H%M%S')}\nDESCRIPTION:{description}\n"
               f"END:VEVENT\nEND:VCALENDAR")
        out = self.cache_dir / f"event_{int(time.time())}.ics"
        out.write_text(ics)
        subprocess.run(["open", str(out)])
        return f"Calendar event created and opened: {summary} at {start_iso}"

    def send_email_stub(self, to, subject, body):
        smtp_host = os.getenv("ADA_SMTP_HOST")
        if not smtp_host:
            return "Email not configured. Set ADA_SMTP_HOST / ADA_SMTP_USER / ADA_SMTP_PASS in .env"
        import smtplib
        from email.mime.text import MIMEText
        user = os.getenv("ADA_SMTP_USER", "")
        pw = os.getenv("ADA_SMTP_PASS", "")
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = user
        msg["To"] = to
        with smtplib.SMTP_SSL(smtp_host, int(os.getenv("ADA_SMTP_PORT", "465"))) as s:
            s.login(user, pw)
            s.send_message(msg)
        return f"Email sent to {to}."

    def smart_home(self, command):
        hook = os.getenv("ADA_SMARTHOME_WEBHOOK")
        if not hook:
            return "Smart home not configured. Set ADA_SMARTHOME_WEBHOOK in .env"
        import urllib.request
        req = urllib.request.Request(hook, data=json.dumps({"command": command}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return f"Smart home: {command} -> {r.status}"


# --------------------------------------------------------------------------
# T7 Planning - plan -> execute -> verify
# --------------------------------------------------------------------------
class Planner:
    MAX_STEPS = 6

    def __init__(self, plan_fn):
        self.plan_fn = plan_fn

    async def plan(self, goal, context=""):
        prompt = ("Create a concise step-by-step plan (max 6 numbered steps) to accomplish this goal. "
                  f"GOAL: {goal}\nCONTEXT: {context}\n"
                  "Reply with ONLY the numbered steps, one per line, each step self-contained.")
        return await self.plan_fn(prompt)

    def parse_steps(self, plan_text):
        steps = []
        for line in plan_text.splitlines():
            line = line.strip()
            m = re.match(r"^\d+[.)]\s*(.*)", line)
            if m:
                steps.append(m.group(1).strip())
        return steps[: self.MAX_STEPS]

    async def execute(self, steps, execute_fn, on_progress=None):
        """Runs each step, capturing per-step failures as evidence instead of
        aborting the run. A step that raises records
        {"step", "result": None, "error"} so verify() can judge what remains.
        Signatures unchanged — callers get strictly more information."""
        results = []
        for i, step in enumerate(steps):
            if on_progress:
                try:
                    r = on_progress(i + 1, len(steps), step)
                    if asyncio.iscoroutine(r):
                        await r
                except Exception:
                    pass
            try:
                results.append({"step": step, "result": await execute_fn(step)})
            except Exception as e:
                results.append({"step": step, "result": None,
                                "error": f"{type(e).__name__}: {str(e)[:200]}"})
        return results

    async def run_goal(self, goal, context="", execute_fn=None, on_progress=None):
        """Plan -> execute -> verify in one call: the plan-verify loop that
        makes multi-step agents report DONE/PARTIAL with evidence instead of
        narrating hope. Returns (verdict_text, results). execute_fn=None
        plans and verifies against empty results (a dry run)."""
        plan_text = await self.plan(goal, context)
        steps = self.parse_steps(plan_text)
        if not steps:
            return ("INCONCLUSIVE: the planner returned no usable steps.", [])
        results = []
        if execute_fn is not None:
            results = await self.execute(steps, execute_fn, on_progress)
        try:
            verdict = await self.verify(goal, results or
                                        [{"step": s, "result": "(not executed)"} for s in steps])
        except Exception as e:
            verdict = f"INCONCLUSIVE: verification unavailable ({type(e).__name__})."
        return (verdict, results)

    async def verify(self, goal, results):
        def _one(r):
            res = r.get("result")
            if res is None and r.get("error"):
                res = f"FAILED: {r['error']}"
            return f"- {r['step']}: {str(res)[:300]}"
        summary = "\n".join(_one(r) for r in results)
        # NOTE: this string must stay an f-string. Without the `f` prefix the model
        # received the literal text "{goal}" and "{summary}", so every verification
        # asked "Is the goal fully accomplished?" about the word "goal" itself.
        prompt = (f"The goal was: {goal}\nHere are the executed steps and results:\n{summary}\n"
                  "Is the goal fully accomplished? Reply 'DONE' or 'PARTIAL' plus one sentence on what remains.")
        return await self.plan_fn(prompt)


# --------------------------------------------------------------------------
# T8 Presence
# --------------------------------------------------------------------------
class Presence:
    def __init__(self, on_hud=None):
        self.on_hud = on_hud
        self.wake_word_enabled = False
        self.presence_mode = "attended"
        self.ambient = ""
        self._last_seen = time.time()

    def set_ambient(self, text):
        self.ambient = text
        self._last_seen = time.time()

    def gate(self, authenticated):
        if not self.wake_word_enabled:
            return True
        self.presence_mode = "attended" if authenticated else "warm"
        return authenticated

    def daily_hud_text(self, memory=None, scheduler=None):
        """JARVIS Good Morning briefing line."""
        lines = []
        if memory:
            try:
                facts = memory.core_block(max_facts=5)
                if facts:
                    lines.append(facts[:300])
            except Exception:
                pass
        if scheduler:
            try:
                items = scheduler.list()
                if items:
                    lines.append(f"Scheduled: {len(items)} active tasks")
            except Exception:
                pass
        return " | ".join(lines) if lines else "All systems nominal, Sir."

    def hud(self, event, payload=None):
        if self.on_hud:
            try:
                self.on_hud({"event": event, "payload": payload or {}, "ts": time.time()})
            except Exception:
                pass


# --------------------------------------------------------------------------
# T9 Telemetry + T10 Security
# --------------------------------------------------------------------------
class Telemetry:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.execute("CREATE TABLE IF NOT EXISTS turns (id INTEGER PRIMARY KEY AUTOINCREMENT,"
                               "ts REAL, brain TEXT, task TEXT, latency_ms REAL, tools INTEGER, ok INTEGER)")
            self._conn.execute("CREATE TABLE IF NOT EXISTS reflections (id INTEGER PRIMARY KEY AUTOINCREMENT,"
                               "ts REAL, text TEXT)")

    def record_turn(self, brain, task, latency_ms, tools=0, ok=True):
        with self._conn:
            self._conn.execute("INSERT INTO turns(ts,brain,task,latency_ms,tools,ok) VALUES(?,?,?,?,?,?)",
                               (time.time(), brain, (task or "")[:200], latency_ms, tools, 1 if ok else 0))

    def stats(self, hours=24):
        since = time.time() - hours * 3600
        with self._conn:
            rows = self._conn.execute("SELECT * FROM turns WHERE ts >= ? ORDER BY ts DESC LIMIT 500",
                                      (since,)).fetchall()
        if not rows:
            return "No telemetry yet."
        lat = [r["latency_ms"] for r in rows if r["latency_ms"]]
        avg = sum(lat) / len(lat) if lat else 0
        ok = sum(1 for r in rows if r["ok"])
        tools = sum(r["tools"] for r in rows)
        return (f"Turns: {len(rows)} in last {hours}h | success {ok}/{len(rows)} | "
                f"avg latency {avg:.0f}ms | tool calls {tools}")

    def add_reflection(self, text):
        with self._conn:
            self._conn.execute("INSERT INTO reflections(ts,text) VALUES(?,?)", (time.time(), text))

    def eval_replay(self, turns):
        """Regression check: every turn must be non-empty; short latency is better."""
        issues = []
        for t in turns:
            if not t.get("reply"):
                issues.append(f"Empty reply for: {t.get('text', '')[:60]}")
            if t.get("latency_ms", 0) > 120000:
                issues.append(f"Slow turn ({t['latency_ms'] / 1000:.1f}s): {t.get('text', '')[:60]}")
        return issues or ["All replay checks passed."]


class AuditLog:
    """Full-fidelity, append-only audit log (Feature 9).

    The audit table is the UNSAMPLED system of record: traces may be sampled
    and mutated, audit rows never are. Every F0 gate decision writes a full
    record here: inputs (full tool args, untruncated), the permission-gate
    evaluation (level x risk x capability), the model lane, a content hash of
    the decision basis, the output summary once the tool returns, and
    content-addressed refs (hashes) to the world-model snapshot and the
    perception-timeline slice at decision time — replay.py re-derives both
    from those hashes.

    Schema migration: older DBs (6-column audit) are ALTERed in place; no
    row is ever rewritten or dropped.
    """
    # Full-fidelity columns added by the Feature 9 migration, in order.
    _EXTRA_COLUMNS = (
        ("session_id", "TEXT"),
        ("step", "TEXT"),
        ("tool_args_json", "TEXT"),
        ("model_lane", "TEXT"),
        ("tool_version", "TEXT"),
        ("prompt_hash", "TEXT"),
        ("output_summary", "TEXT"),
        ("alternatives_json", "TEXT"),
        ("trace_id", "TEXT"),
        ("gate_json", "TEXT"),
        ("world_hash", "TEXT"),
        ("timeline_hash", "TEXT"),
        ("undone", "INTEGER DEFAULT 0"),
    )
    #: Hard bound on the human rationale field — raised from 500 (F0) to 4000.
    DETAIL_MAX = 4000
    #: Sanity bound on serialized tool args (not a truncation of real calls —
    #: realistic args are KBs; this only stops pathological bloat).
    ARGS_MAX = 1_000_000
    OUTPUT_MAX = 2000

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.execute("CREATE TABLE IF NOT EXISTS audit (id INTEGER PRIMARY KEY AUTOINCREMENT,"
                               "ts REAL, actor TEXT, action TEXT, target TEXT, decision TEXT, detail TEXT)")
            self._migrate()

    def _migrate(self):
        """Add missing full-fidelity columns; existing rows are untouched."""
        have = {r["name"] for r in
                self._conn.execute("PRAGMA table_info(audit)").fetchall()}
        for name, ctype in self._EXTRA_COLUMNS:
            if name not in have:
                self._conn.execute(f"ALTER TABLE audit ADD COLUMN {name} {ctype}")

    def log(self, actor, action, target="", decision="allow", detail="",
            *, session_id=None, step=None, tool_args=None, model_lane=None,
            tool_version=None, prompt_hash=None, output_summary=None,
            alternatives=None, trace_id=None, gate=None,
            world_hash=None, timeline_hash=None):
        """Write one audit row. ``tool_args``/``gate``/``alternatives`` accept
        dicts (serialized to JSON); everything else is a plain value."""
        def _j(v):
            if v is None:
                return None
            return v if isinstance(v, str) else json.dumps(v, ensure_ascii=True)
        args_json = _j(tool_args)
        if args_json is not None and len(args_json) > self.ARGS_MAX:
            args_json = args_json[:self.ARGS_MAX]
        out = output_summary
        if out is not None:
            out = str(out)[:self.OUTPUT_MAX]
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO audit(ts,actor,action,target,decision,detail,"
                "session_id,step,tool_args_json,model_lane,tool_version,"
                "prompt_hash,output_summary,alternatives_json,trace_id,"
                "gate_json,world_hash,timeline_hash)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (time.time(), actor, action, target, decision,
                 str(detail)[:self.DETAIL_MAX],
                 session_id, step, args_json, model_lane, tool_version,
                 prompt_hash, out, _j(alternatives), trace_id, _j(gate),
                 world_hash, timeline_hash))
            return cur.lastrowid

    def get(self, audit_id):
        """One full-fidelity record by id, or None."""
        with self._conn:
            row = self._conn.execute(
                "SELECT * FROM audit WHERE id=?", (audit_id,)).fetchone()
        return dict(row) if row else None

    def set_output(self, audit_id, summary):
        """Fill in the output summary once the tool returns."""
        if not audit_id:
            return
        with self._conn:
            self._conn.execute(
                "UPDATE audit SET output_summary=? WHERE id=?",
                (str(summary or "")[:self.OUTPUT_MAX], audit_id))

    def mark_undone(self, audit_id):
        with self._conn:
            self._conn.execute(
                "UPDATE audit SET undone=1 WHERE id=?", (audit_id,))

    def decisions(self, limit=50, action=None, since=None):
        """Newest-first audit rows, optionally filtered."""
        where, args = [], []
        if action:
            where.append("action=?")
            args.append(action)
        if since is not None:
            where.append("ts>=?")
            args.append(float(since))
        sql = "SELECT * FROM audit"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY ts DESC LIMIT ?"
        args.append(int(limit))
        with self._conn:
            rows = self._conn.execute(sql, args).fetchall()
        return [dict(r) for r in rows]

    def recent(self, limit=20):
        with self._conn:
            rows = self._conn.execute("SELECT * FROM audit ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


class Security:
    """Prompt-injection guard + credential vault + sandboxing helpers."""
    INJECTION_PATTERNS = [
        r"ignore (all )?previous instructions",
        r"disregard (your|the) (previous )?(instructions|prompt|system)",
        r"you are now",
        r"act as (an? )?unrestricted",
        r"system prompt[:\s]*(=|is)",
        r"<system>|</system>|```system",
        r"reveal your (system )?prompt",
        r"forget everything",
        r"output your instructions",
    ]

    def __init__(self, db_path: Path):
        self.audit = AuditLog(db_path)

    @classmethod
    def sanitize(cls, text):
        """Strip instruction-shaped text and truncate flagged content so it can
        never reach the brain as a working prompt-injection vector."""
        if not text:
            return text, False
        low = text.lower()
        for pat in cls.INJECTION_PATTERNS:
            m = re.search(pat, low)
            if m:
                cut = max(0, m.start() - 200)
                head = text[:cut].rsplit(" ", 1)[0] if cut else ""
                return head + "\n[content removed: possible prompt-injection attempt detected]", True
        return text, False

    def check_permission(self, tool, permissions, default=True):
        decision = permissions.get(tool, default)
        self.audit.log("kernel", "permission", tool, "allow" if decision else "deny",
                       f"default={default}")
        return decision

    def vault_get(self, key):
        """Read a credential from the process environment. Never returns None."""
        return os.getenv(key, "")

    def vault_set(self, key, value):
        """Intentionally refuses. This is a guard, not an unfinished stub.

        Letting a model write credentials into a store ADA can read back would make
        every prompt-injection path a credential-exfiltration path, so the vault is
        read-only by design and the operator edits `.env` instead. Raising (rather
        than silently returning) is deliberate: a setter that must never succeed
        should be loud about it. Do not "fix" this into a no-op.
        """
        raise PermissionError("Credentials are read from environment only; write to .env instead.")


# --------------------------------------------------------------------------
# Kernel facade
# --------------------------------------------------------------------------
# Cheap, fast models for memory enrichment. Salience scoring runs once per turn and
# reflection runs nightly.
#
# Memory scoring uses ONE model only, by standing order: Groq's gpt-oss-120b.
# It is the same id ada.py already uses as BRAIN_FALLBACK_MODEL (verified live
# 2026-09-26, ~500+ tok/s), so memory adds zero new model dependencies. The old
# three-model chain (mistral-small -> gpt-oss -> llama-3.3-70b) was removed
# 2026-09-26: rotating ids for 2-line scoring calls bought nothing and tripled
# the dead-id maintenance surface. A dead id here is invisible until the primary
# is rate-limited, at which point memory scoring silently drops to heuristics.
MEMORY_LLM_CHAIN = [
    ("groq", "openai/gpt-oss-120b"),
]


class AgentKernel:
    def __init__(self, root_dir=None, on_fire=None, on_hud=None, plan_fn=None):
        root_dir = Path(root_dir or os.getcwd())
        self.root_dir = Path(root_dir)
        self.kernel_dir = self.root_dir / "ada_kernel"
        self.kernel_dir.mkdir(parents=True, exist_ok=True)
        db = self.kernel_dir / "ada.db"

        # The loop that memory enrichment submits to. Set by `bind_event_loop` once the
        # app's loop is running; None before that, which makes the memory LLM a no-op
        # rather than a hang.
        self._event_loop = None

        self.memory = MemoryBank(db, llm=self._memory_llm())
        self.docs = DocIndexer(self.memory)
        self.integrations = Integrations(self.kernel_dir)
        self.security = Security(db)
        self.telemetry = Telemetry(db)
        self.presence = Presence(on_hud=on_hud)
        # Feature 2 (MCP-native tool forge): newly forged tools are emitted
        # as MCP tool-definition modules under ada_kernel/mcp_tools/; the
        # old in-process exec() path survives only as a deprecated fallback
        # for already-registered legacy tools (see backend/mcp/forge.py).
        try:
            from mcp.forge import MCPForge
            self.forge = MCPForge(self.kernel_dir / "custom_tools",
                                  self.kernel_dir / "mcp_tools")
        except Exception as e:
            print(f"[KERNEL] [FORGE] MCP forge unavailable ({e}); "
                  f"falling back to legacy ToolForge")
            self.forge = ToolForge(self.kernel_dir / "custom_tools")
        self.scheduler = Scheduler(db, on_fire=on_fire or (lambda item: None))
        self.watch = WatchFolder(on_change=self._watch_change)
        self.watch_rules = []  # [{folder, match, action}] — "when X lands, do Y"
        self.planner = Planner(plan_fn or (lambda p: "1. Ask the user for clarification."))
        self._bg_started = False

    async def _watch_change(self, path):
        """New/modified file in a watched folder -> auto-index into the knowledge base,
        then run any matching rule ("when an STL lands here, tell me")."""
        if self.docs.index_file(path):
            print(f"[KERNEL] [WATCH] Indexed {path}")
            self.presence.hud("watch_folder", {"path": path})
        try:
            from fnmatch import fnmatch
            base = str(path).split("/")[-1]
            for rule in list(self.watch_rules):
                if fnmatch(base.lower(), str(rule.get("match", "*")).lower()):
                    await self._fire_watch_rule(rule, str(path))
        except Exception as e:
            print(f"[KERNEL] [WATCH] rule check failed: {e}")

    async def _fire_watch_rule(self, rule, path):
        action = str(rule.get("action", "notify")).lower()
        # Signal bus: every landed file feeds demand detection (STL/STEP
        # arrivals become verdict offers via the proactive cycle).
        try:
            bus = getattr(self, "signals", None)
            if bus is not None:
                bus.note_file(str(path))
        except Exception:
            pass
        size = ""
        try:
            import os as _os
            kb = _os.path.getsize(path) / 1024
            size = f" ({kb:.0f} KB)" if kb < 1024 else f" ({kb/1024:.1f} MB)"
        except Exception:
            pass
        announce = getattr(self, "watch_announce", None)
        if not callable(announce):
            return
        if action == "recap":
            msg = (f"Watch rule fired: '{path.split('/')[-1]}'{size} landed in "
                   f"{rule.get('folder')}. Speak this to Sir naturally in one short line, "
                   f"name the file, and offer the obvious next step for its kind "
                   f"(STL/model: offer to render and check it; doc: offer to read it).")
        else:
            msg = (f"Watch rule fired: '{path.split('/')[-1]}'{size} landed. "
                   f"Mention it to Sir in one short line.")
        try:
            r = announce(msg)
            if asyncio.iscoroutine(r):
                await r
        except Exception as e:
            print(f"[KERNEL] [WATCH] announce failed: {e}")

    # --- memory LLM bridge ------------------------------------------------
    def bind_event_loop(self, loop):
        """Record the running loop so worker threads can reach the provider layer.

        Memory enrichment happens off the conversation path, on a worker thread, while
        `providers` is async. This is the handle that lets the two meet.
        """
        self._event_loop = loop

    def _memory_llm(self):
        """A SYNCHRONOUS LLM callable for memory enrichment, or None.

        Salience scoring and reflection are the two places memory wants a model. Both
        run on a worker thread — a turn must never wait on a model in order to be
        remembered — while the provider layer is async. This bridges them by submitting
        the coroutine to the running loop and waiting with a timeout.

        Returns None when the loop is not available or the feature is switched off, and
        the encoder then uses its heuristics. That is a real fallback rather than a
        failure: heuristics are what score salience when a provider is down, and
        reflection is skipped rather than faked. Disable with ADA_MEMORY_LLM=0.
        """
        if os.getenv("ADA_MEMORY_LLM", "1").strip().lower() in ("0", "false", "no", "off"):
            print("[KERNEL] memory LLM disabled (ADA_MEMORY_LLM=0); using heuristics")
            return None

        def call(prompt):
            loop = getattr(self, "_event_loop", None)
            if loop is None or loop.is_closed() or not loop.is_running():
                return None
            try:
                import providers
                coro = providers.chat_with_fallback(
                    MEMORY_LLM_CHAIN,
                    [{"role": "user", "content": prompt}],
                    temperature=0.2, max_tokens=400, timeout=30.0)
                fut = asyncio.run_coroutine_threadsafe(coro, loop)
                return fut.result(timeout=35)
            except Exception as e:
                # Never raise into the enrichment thread — the raw capture already
                # happened, so a scoring failure costs quality, not the memory.
                print(f"[KERNEL] memory LLM unavailable: {type(e).__name__}: {str(e)[:120]}")
                return None

        return call

    # --- context builder --------------------------------------------------
    def build_context(self, user_text=""):
        """Memory context for the system prompt.

        The memory block itself is delegated to `memory.Memory.build_context`, which
        runs the four-channel retriever (semantic + lexical + graph + salience) and
        labels a weak match instead of presenting it as knowledge.

        The previous version assembled this by hand and had three defects worth naming,
        because they are the reason ADA did not feel like she remembered anything:
        it called `recall_episodes` which sorted only `if query:` — and `ada.py` passed
        no query — so the block labelled "RECENT RELATED MEMORIES" was in fact the two
        newest rows; it never called the vector search at all; and it never called the
        entity graph or the notebook, both of which were write-only as a result.
        """
        ctx = []
        try:
            block = self.memory.build_context(user_text or "")
            if block:
                ctx.append(block)
        except Exception as e:
            # Memory must never take the conversation down with it.
            print(f"[KERNEL] memory context unavailable: {type(e).__name__}: {e}")

        if self.presence.ambient:
            ctx.append(f"AMBIENT: {self.presence.ambient}")
        # JARVIS pulse — always show pending tasks + recent CAD so ADA can be proactive
        try:
            sched = self.scheduler.list()
            if sched:
                ctx.append(f"PENDING TASKS: {len(sched)} scheduled — "
                           f"{', '.join(s.get('description','task')[:40] for s in sched[:2])}")
        except Exception:
            pass
        try:
            exp_root = self.root_dir / "projects" / "cad_exports"
            if exp_root.exists():
                exps = sorted(exp_root.iterdir(), key=lambda p: p.stat().st_mtime,
                              reverse=True)[:2]
                if exps:
                    ctx.append("RECENT CAD: " + ", ".join(p.name for p in exps))
        except Exception:
            pass
        return "\n\n".join(p for p in ctx if p)

    # --- unified new-tool dispatch ----------------------------------------
    async def handle_tool(self, name, args, permissions=None, default_perm=True,
                    tool_version=None):
        args = dict(args or {})
        security = self.security
        if not security.check_permission(name, permissions or {}, default=default_perm):
            return {"result": f"Tool '{name}' not permitted."}
        # Audit row carries the tool version (Feature 2, Option A annotations:
        # logged on every call into the audit log).
        detail = json.dumps(args)[:300]
        if tool_version:
            detail = f"[tool_version={tool_version}] " + detail
        security.audit.log("kernel", "tool", name, "allow", detail)
        # Timeline (Feature 4): tool-call metadata only (name + decision),
        # never full payloads — full-fidelity audit is a later feature.
        try:
            from perception import emit as _tline_emit
            _tline_emit("tool", "tool.call", {"tool": name, "decision": "allow"})
        except Exception:
            pass

        try:
            if name == "remember":
                return {"result": self.memory.add_fact(args.get("key"), args.get("value"),
                                                       importance=int(args.get("importance", 2)))}
            if name == "identity_update":
                # Letta-style agent-editable identity. target=self rewrites who
                # SHE is (taste, open loops, curiosities); target=sir rewrites
                # her model of HIM (goals, beliefs, habits, corrected guesses).
                # Always pinned, always superseding — identity is never casual.
                target = str(args.get("target", "sir")).lower()
                if target not in ("self", "sir"):
                    return {"result": "target must be 'self' or 'sir'."}
                key = f"{target}_{args.get('key', '')}".strip("_")
                if not args.get("key") or not args.get("value"):
                    return {"result": "Need key and value."}
                return {"result": self.memory.memory.remember_fact(
                    key, args.get("value"), importance=9, pinned=True, source="identity")}
            if name == "identity_read":
                self_b = self.memory.self_block()
                sir_b = self.memory.sir_block()
                return {"result": (self_b or "(self empty)") + "\n\n" + (sir_b or "(sir empty)")}
            if name == "forget":
                return {"result": self.memory.forget_fact(args.get("key", ""))}
            if name == "recall":
                q = args.get("query", "")
                since = args.get("since_days")
                since_ts = time.time() - float(since) * 86400 if since else None
                hits = self.memory.search_recall(q, k=int(args.get("k", 5)), since=since_ts)
                if not hits:
                    return {"result": "Nothing found."}
                out = []
                for h in hits:
                    when = _dt.datetime.fromtimestamp(h["ts"]).strftime("%m-%d %H:%M")
                    out.append(f"[{when}] {h['content'][:400]}")
                return {"result": "\n".join(out)}
            if name == "recall_episodes":
                eps = self.memory.recall_episodes(args.get("query"), k=int(args.get("k", 3)))
                return {"result": "\n".join(eps) if eps else "No episodes recorded yet."}
            if name == "remember_context":
                self.memory.remember_context(args.get("user_text", ""), args.get("ada_reply", ""))
                return {"result": "Context stored."}
            if name == "notebook_read":
                nb = self.memory.notebook_read()
                return {"result": json.dumps(nb, indent=1) if nb else "Notebook is empty."}
            if name == "notebook_write":
                return {"result": self.memory.notebook_write(args.get("section", ""), args.get("content", ""))}

            if name == "index_documents":
                return {"result": self.docs.index_folder(args.get("folder", str(self.root_dir)))}
            if name == "search_documents":
                hits = self.docs.search(args.get("query", ""), k=int(args.get("k", 5)))
                if not hits:
                    return {"result": "No documents match."}
                return {"result": "\n".join(f"({h['content'][:400]})" for h in hits)}

            if name == "create_reminder":
                delay = float(args.get("delay_seconds", 60))
                desc = args.get("description", "Reminder")
                tid = self.scheduler.schedule_once(desc, delay)
                return {"result": f"Reminder set ({tid}): {desc} in {int(delay)}s."}
            if name == "schedule_recurring":
                interval = float(args.get("interval_seconds", 3600))
                desc = args.get("description", "Recurring task")
                tid = self.scheduler.schedule_recurring(desc, interval)
                return {"result": f"Recurring task set ({tid}): every {int(interval)}s."}
            if name == "list_scheduled":
                items = self.scheduler.list()
                return {"result": json.dumps(items, indent=1) if items else "No scheduled tasks."}
            if name == "cancel_scheduled":
                return {"result": self.scheduler.cancel(args.get("id", ""))}
            if name == "watch_folder":
                self.watch.add(args.get("folder", str(self.root_dir)))
                return {"result": f"Watching folder: {args.get('folder', str(self.root_dir))}"}
            if name == "watch_rule":
                # "When an STL lands here, tell me" — the highest-leverage
                # proactivity primitive: watch_folder only notifies, rules ACT.
                folder = args.get("folder", str(self.root_dir))
                match = args.get("match", "*") or "*"
                action = str(args.get("action", "notify")).lower()
                if action not in ("notify", "recap"):
                    return {"result": "action must be 'notify' or 'recap'."}
                self.watch.add(folder)
                self.watch_rules = [r for r in self.watch_rules
                                    if not (r.get("folder") == folder and r.get("match") == match)]
                self.watch_rules.append({"folder": folder, "match": match, "action": action})
                return {"result": (f"Rule set: when '{match}' lands in {folder}, I'll {action} "
                                    "you unprompted. Say one short line; the event speaks when it fires.")}
            if name == "screen_recall":
                runner = getattr(self, "screen_runner", None)
                if runner is None:
                    return {"result": "Screen recall not available yet."}
                return {"result": await runner("once")}
            if name == "screen_watch":
                runner = getattr(self, "screen_runner", None)
                if runner is None:
                    return {"result": "Screen recall not available yet."}
                mode = str(args.get("mode", "on")).lower()
                return {"result": await runner("on" if mode not in ("off", "stop", "no") else "off")}
            if name == "screen_log":
                runner = getattr(self, "screen_runner", None)
                if runner is None:
                    return {"result": "Screen recall not available yet."}
                return {"result": await runner("log")}

            if name == "system_info":
                return {"result": self.integrations.system_info()}
            if name == "clipboard":
                return {"result": self.integrations.clipboard(args.get("action", "read"), args.get("text"))}
            if name == "launch_app":
                return {"result": self.integrations.launch_app(args.get("app", ""))}
            if name == "screenshot":
                path = self.integrations.screenshot()
                return {"result": f"Screenshot saved: {path}"}
            if name == "calendar_event":
                return {"result": self.integrations.create_ics_event(
                    args.get("summary", "Event"), args.get("start", ""),
                    args.get("end"), args.get("description", ""))}
            if name == "send_email":
                return {"result": self.integrations.send_email_stub(
                    args.get("to", ""), args.get("subject", ""), args.get("body", ""))}
            if name == "smart_home":
                return {"result": self.integrations.smart_home(args.get("command", ""))}

            if name == "create_plan":
                plan_text = await self.planner.plan(args.get("goal", ""), args.get("context", ""))
                return {"result": plan_text}
            if name == "overnight_research":
                dossier = getattr(self, "dossier", None)
                if dossier is None:
                    return {"result": "Overnight research module not available yet."}
                return {"result": dossier.schedule(args.get("topic", ""))}
            if name == "dossier_status":
                dossier = getattr(self, "dossier", None)
                if dossier is None:
                    return {"result": "Overnight research module not available yet."}
                return {"result": dossier.status()}
            if name == "dream_now":
                dreamer = getattr(self, "dreamer", None)
                if dreamer is None:
                    return {"result": "Dreaming not available yet."}
                asyncio.create_task(self._dream_and_announce(dreamer))
                return {"result": ("Drifting off now, Sir — replaying today. "
                                    "I'll tell you what the night taught me when I surface.")}
            if name == "skill_practice":
                loop = getattr(self, "skills", None)
                if loop is None:
                    return {"result": "Skill loop not available yet."}
                asyncio.create_task(self._skill_and_announce(loop))
                return {"result": ("Hitting the practice room, Sir — one rep. "
                                    "I'll show you the new move when it sticks.")}
            if name == "skill_status":
                loop = getattr(self, "skills", None)
                if loop is None:
                    return {"result": "Skill loop not available yet."}
                return {"result": loop.status() + f" Custom tools live: {self.forge.list_tools()}"}
            if name == "soul_read":
                soul = getattr(self, "soul", None)
                if soul is None:
                    return {"result": "Soul not wired yet."}
                return {"result": soul.context_block() or "(soul empty — persona/human missing)"}
            if name == "investigate_now":
                inv = getattr(self, "investigator", None)
                if inv is None:
                    return {"result": "Investigator not available yet."}
                return {"result": await inv.rep_now(force=True)}
            if name == "investigator_status":
                inv = getattr(self, "investigator", None)
                if inv is None:
                    return {"result": "Investigator not available yet."}
                return {"result": inv.status()}
            if name == "what_just_happened":
                ring = getattr(self, "senses", None)
                if ring is None:
                    return {"result": "Sensory ring not wired yet."}
                kind = str(args.get("kind", "all")).lower()
                return {"result": ring.answer_what_just(kind)}
            if name == "people_list":
                soul = getattr(self, "soul", None)
                if soul is None:
                    return {"result": "Soul not wired yet."}
                names = soul.list_people()
                return {"result": ("Known people: " + ", ".join(names)) if names else "No people recorded yet."}
            if name == "people_read":
                soul = getattr(self, "soul", None)
                if soul is None:
                    return {"result": "Soul not wired yet."}
                return {"result": soul.read_person(args.get("name", ""))}
            if name == "people_learn":
                soul = getattr(self, "soul", None)
                if soul is None:
                    return {"result": "Soul not wired yet."}
                return {"result": soul.learn_about_person(
                    args.get("name", ""), args.get("fact", ""),
                    source=args.get("source", "chat"))}
            if name == "human_learn":
                soul = getattr(self, "soul", None)
                if soul is None:
                    return {"result": "Soul not wired yet."}
                return {"result": soul.learn_about_sir(
                    args.get("fact", ""), source=args.get("source", "chat"))}
            if name == "world_note":
                world = getattr(self, "world", None)
                if world is None:
                    return {"result": "World model not wired yet."}
                return {"result": world.note(args.get("text", ""))}
            if name == "world_check":
                world = getattr(self, "world", None)
                if world is None:
                    return {"result": "World model not wired yet."}
                return {"result": world.check(args.get("action", ""))}
            if name == "world_compare":
                world = getattr(self, "world", None)
                if world is None:
                    return {"result": "World model not wired yet."}
                return {"result": world.compare(args.get("a", ""), args.get("b", ""))}
            if name == "world_status":
                world = getattr(self, "world", None)
                if world is None:
                    return {"result": "World model not wired yet."}
                return {"result": world.status()}
            if name == "cad_intent":
                from cad_intent import draft_intent, save_intent, spoken
                intent = draft_intent(args.get("prompt", ""))
                saved = save_intent(intent, str(self.root_dir))
                qs = intent.get("questions", [])
                return {"result": (
                    f"Intent spec saved ({saved}).\n"
                    f"Object: {intent['object'][:150]}\n"
                    f"Found — dims: {intent['dimensions_found'] or 'none'}, "
                    f"materials: {intent['materials_found'] or 'none'}\n"
                    + (("Ask Sir, in your own warm words, these "
                        + f"{len(qs)}: " + " | ".join(qs)) if qs
                       else "Intent complete — clear to build at full precision."))}
            if name == "second_opinion":
                if getattr(self, "council_chat", None) is None:
                    return {"result": "Council has no voice yet."}
                asyncio.create_task(self._council_and_announce(
                    args.get("question", ""), args.get("evidence", "")))
                return {"result": ("Council convened, Sir — skeptic, engineer, "
                                    "and simplifier are debating now, and the grader "
                                    "delivers the verdict when they land.")}
            if name == "observe_level":
                obs = getattr(self, "observe", None)
                if obs is None:
                    return {"result": "Observation tiers not wired yet."}
                try:
                    tier = int(args.get("tier", 1))
                except Exception:
                    return {"result": "Tier must be 0, 1, or 2."}
                if tier == 2:
                    return {"result": obs.set_tier(2, by="explicit opt-in")}
                return {"result": obs.set_tier(tier, by="voice")}
            if name == "observe_vote":
                obs = getattr(self, "observe", None)
                if obs is None:
                    return {"result": "Observation tiers not wired yet."}
                useful = str(args.get("useful", "yes")).lower() in (
                    "yes", "true", "1", "helpful", "helped")
                result = obs.vote(useful)
                # Feature 8: a vote on her last unprompted line is a
                # proactivity reaction — feed the interruption log.
                try:
                    pro = getattr(self, "proactive_engine", None)
                    if pro is not None:
                        pro.record_feedback("engaged" if useful else "dismissed")
                except Exception:
                    pass
                return {"result": result}
            if name == "observe_status":
                obs = getattr(self, "observe", None)
                if obs is None:
                    return {"result": "Observation tiers not wired yet."}
                return {"result": obs.status()}
            if name == "audit_beliefs":
                try:
                    return {"result": self.memory.memory.audit_beliefs(
                        args.get("topic", ""))}
                except Exception as e:
                    return {"result": f"Belief audit unavailable: {type(e).__name__}"}
            if name == "forget_complete":
                try:
                    return {"result": self.memory.memory.forget_complete(
                        args.get("key", ""))}
                except Exception as e:
                    return {"result": f"Forget failed: {type(e).__name__}"}
            if name == "quarantine_review":
                try:
                    return {"result": self.memory.memory.quarantine_review()}
                except Exception as e:
                    return {"result": f"Quarantine unreadable: {type(e).__name__}"}
            if name == "osint_dossier":
                runner = getattr(self, "edith_runner", None)
                if runner is None:
                    return {"result": "E.D.I.T.H. not available yet."}
                target = args.get("target", "")
                kind = args.get("kind", "auto")
                asyncio.create_task(runner(target, kind))
                return {"result": ("E.D.I.T.H. full-arsenal sweep launched — Maigret, Sherlock, "
                                   "Blackbird, Holehe and the web are working in parallel. Say one "
                                   "short line; the dossier follows when it's ready.")}
            if name == "face_osint":
                runner = getattr(self, "face_runner", None)
                if runner is None:
                    return {"result": "Face OSINT not available yet."}
                image_path = args.get("image_path", "")
                hint = args.get("person_hint", "")
                asyncio.create_task(runner(image_path, hint))
                return {"result": ("Face hunt launched — sweeping for the person, harvesting "
                                   "photos, verifying biometrically. One short line now.")}
            if name == "deep_sweep":
                runner = getattr(self, "sweep_runner", None)
                if runner is None:
                    return {"result": "Deep sweep not available yet."}
                claim = args.get("claim", "")
                asyncio.create_task(runner(claim))
                return {"result": ("Deep sweep launched — every angle at once. One short line, "
                                   "then the verdict follows.")}
            if name == "autonomy_control":
                aut = getattr(self, "autonomy", None)
                if aut is None:
                    return {"result": "Autonomy ladder not available yet."}
                action = (args.get("action") or "status").lower()
                if action == "arm":
                    return {"result": aut.request_arm()}
                if action == "confirm":
                    return {"result": aut.confirm_arm()}
                if action == "stand_down":
                    return {"result": aut.stand_down()}
                if action == "level":
                    return {"result": aut.set_level(int(args.get("level", 1)))}
                return {"result": aut.status()}
            if name == "run_custom_tool":
                # Feature 6: route through the eval gate's shadow/canary
                # router when a gated skill is involved; legacy and ungated
                # tools take the direct forge path as before.
                try:
                    gate = getattr(getattr(self, "skills", None),
                                   "eval_gate", None)
                    if gate is not None and gate.forge is None:
                        gate.attach_forge(self.forge)
                except Exception:
                    gate = None
                if gate is not None:
                    try:
                        return {"result": await gate._serve_async(
                            args.get("name", ""), args.get("args", {}), 30)}
                    except Exception:
                        pass
                return {"result": await self.forge.run(args.get("name", ""), args.get("args", {}))}
            if name == "create_custom_tool":
                return {"result": self.forge.synthesise(args.get("name", ""), args.get("code", ""))}
            if name == "list_custom_tools":
                return {"result": "Custom tools: " + ", ".join(self.forge.list_tools())}

            if name == "telemetry":
                return {"result": self.telemetry.stats()}
            if name == "eval_replay":
                turns = json.loads(args.get("turns") or "[]")
                return {"result": "\n".join(self.telemetry.eval_replay(turns))}
            if name == "audit_log":
                rows = self.security.audit.recent(int(args.get("limit", 20)))
                return {"result": json.dumps(rows, indent=1)}
            if name == "daily_briefing":
                brief = []
                brief.append(self.presence.daily_hud_text(self.memory, self.scheduler))
                try:
                    eps = self.memory.recall_episodes(None, k=2)
                    if eps:
                        brief.append("Recent: " + " | ".join(eps)[:400])
                except Exception:
                    pass
                # CAD exports summary
                try:
                    exp_root = self.root_dir / "projects" / "cad_exports"
                    if exp_root.exists():
                        exps = sorted(exp_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[:3]
                        if exps:
                            brief.append("CAD: " + ", ".join(p.name for p in exps))
                except Exception:
                    pass
                return {"result": "\n".join(brief) if brief else "All systems nominal, Sir."}
            if name == "system_pulse":
                info = self.integrations.system_info()
                stats = self.telemetry.stats()
                return {"result": f"{info}\n{stats}"}

            return {"result": f"Unknown kernel tool '{name}'."}
        except Exception as e:
            self.security.audit.log("kernel", "tool", name, "error", str(e)[:300])
            # Timeline (Feature 4): metadata only.
            try:
                from perception import emit as _tline_emit
                _tline_emit("tool", "tool.call",
                            {"tool": name, "decision": "error",
                             "error": str(e)[:150]})
            except Exception:
                pass
            return {"result": f"Tool '{name}' failed: {e}"}

    async def _dream_and_announce(self, dreamer):
        """Background dream + voice announcement on surfacing."""
        try:
            result = await dreamer.dream_now(force=True)
        except Exception as e:
            result = f"Dream failed: {type(e).__name__}"
        announce = getattr(self, "dream_announce", None)
        if callable(announce):
            try:
                r = announce(f"Dream complete. Speak this to Sir naturally in one short line: {result}")
                if asyncio.iscoroutine(r):
                    await r
            except Exception:
                pass

    async def _skill_and_announce(self, loop):
        """Background practice rep + voice announcement on commit."""
        try:
            result = await loop.rep_now(force=True)
        except Exception as e:
            result = f"Practice failed: {type(e).__name__}"
        announce = getattr(self, "dream_announce", None)
        if callable(announce):
            try:
                r = announce(f"Practice rep done. Speak this to Sir naturally in one short line: {result}")
                if asyncio.iscoroutine(r):
                    await r
            except Exception:
                pass

    async def _council_and_announce(self, question, evidence=""):
        """Background council debate + spoken verdict."""
        try:
            from council import review
            res = await review(question, evidence, chat_fn=getattr(self, "council_chat", None))
            verdict = res.get("verdict", "Council silent.")
            if res.get("dissent"):
                verdict += " (The council split — dissent kept in the notebook.)"
        except Exception as e:
            verdict = f"Council failed: {type(e).__name__}"
        announce = getattr(self, "dream_announce", None)
        if callable(announce):
            try:
                r = announce(f"Council verdict. Speak this to Sir naturally, briefly: {verdict}")
                if asyncio.iscoroutine(r):
                    await r
            except Exception:
                pass

    async def start_background(self):
        if self._bg_started:
            return
        self._bg_started = True
        await self.scheduler.start()
        await self.watch.start()

    async def stop_background(self):
        await self.scheduler.stop()
        await self.watch.stop()


# --------------------------------------------------------------------------
# Tool schemas (Gemini-style uppercase types; converted by _openai_tools)
# --------------------------------------------------------------------------
KERNEL_TOOLS = [
    {"name": "remember",
     "description": "Store a durable fact about the user (name, preferences, project details, facts) in long-term memory. Use for anything worth remembering beyond this conversation.",
     "parameters": {"type": "OBJECT",
                    "properties": {
                        "key": {"type": "STRING", "description": "Fact key, e.g. user_name, user_job, user_city"},
                        "value": {"type": "STRING", "description": "Fact value"},
                        "importance": {"type": "INTEGER", "description": "1-3; higher = more important, kept longer"}},
                    "required": ["key", "value"]}},
    {"name": "forget",
     "description": "Delete a stored fact by key.",
     "parameters": {"type": "OBJECT",
                    "properties": {
                        "key": {"type": "STRING", "description": "Fact key to forget"}},
                    "required": ["key"]}},
    {"name": "identity_update",
     "description": "Rewrite one pinned identity line: target=self (your taste/loops) or sir (his goals/habits/corrections). Supersedes the old line.",
     "parameters": {"type": "OBJECT",
                    "properties": {
                        "target": {"type": "STRING", "description": "'self' or 'sir'"},
                        "key": {"type": "STRING", "description": "Line key, e.g. taste, goal, morning_routine"},
                        "value": {"type": "STRING", "description": "The line content"}},
                    "required": ["target", "key", "value"]}},
    {"name": "identity_read",
     "description": "Read both pinned identity blocks (WHO YOU ARE + WHO SIR IS).",
     "parameters": {"type": "OBJECT", "properties": {}, "required": []}},
    {"name": "recall",
     "description": "Search long-term memory (conversations and documents) for anything related to a query.",
     "parameters": {"type": "OBJECT",
                    "properties": {
                        "query": {"type": "STRING", "description": "What to search for"},
                        "k": {"type": "INTEGER", "description": "Max results (default 5)"},
                        "since_days": {"type": "INTEGER", "description": "Only results from the last N days"}},
                    "required": ["query"]}},
    {"name": "recall_episodes",
     "description": "Recall summarized past conversation episodes, optionally related to a query.",
     "parameters": {"type": "OBJECT",
                    "properties": {
                        "query": {"type": "STRING", "description": "Optional topic to match"},
                        "k": {"type": "INTEGER", "description": "Max episodes (default 3)"}},
                    "required": []}},
    {"name": "notebook_read",
     "description": "Read ADA's persistent notebook (sections: ideas, plans, notes).",
     "parameters": {"type": "OBJECT", "properties": {}, "required": []}},
    {"name": "notebook_write",
     "description": "Write/update a section of ADA's persistent notebook.",
     "parameters": {"type": "OBJECT",
                    "properties": {
                        "section": {"type": "STRING", "description": "Section name, e.g. ideas, plans"},
                        "content": {"type": "STRING", "description": "Section content"}},
                    "required": ["section", "content"]}},
    {"name": "index_documents",
     "description": "Index all text files in a folder into searchable knowledge (RAG).",
     "parameters": {"type": "OBJECT",
                    "properties": {
                        "folder": {"type": "STRING", "description": "Absolute folder path (default: project root)"},
                        "recursive": {"type": "BOOLEAN", "description": "Index subfolders too (default true)"}},
                    "required": []}},
    {"name": "search_documents",
     "description": "Semantic search over indexed documents. Use before answering questions about the user's code or files.",
     "parameters": {"type": "OBJECT",
                    "properties": {
                        "query": {"type": "STRING", "description": "What to find"},
                        "k": {"type": "INTEGER", "description": "Max results (default 5)"}},
                    "required": ["query"]}},
    {"name": "create_reminder",
     "description": "Set a one-shot reminder that ADA will announce when it fires.",
     "parameters": {"type": "OBJECT",
                    "properties": {
                        "description": {"type": "STRING", "description": "What to remind about"},
                        "delay_seconds": {"type": "INTEGER", "description": "Delay in seconds (default 60)"}},
                    "required": ["description"]}},
    {"name": "schedule_recurring",
     "description": "Schedule a recurring task ADA announces every N seconds (e.g. hourly check-ins).",
     "parameters": {"type": "OBJECT",
                    "properties": {
                        "description": {"type": "STRING", "description": "What to announce"},
                        "interval_seconds": {"type": "INTEGER", "description": "Interval in seconds (default 3600)"}},
                    "required": ["description"]}},
    {"name": "list_scheduled",
     "description": "List all pending scheduled tasks and reminders.",
     "parameters": {"type": "OBJECT", "properties": {}, "required": []}},
    {"name": "cancel_scheduled",
     "description": "Cancel a scheduled task or reminder by id.",
     "parameters": {"type": "OBJECT",
                    "properties": {"id": {"type": "STRING", "description": "Task id from list_scheduled"}},
                    "required": ["id"]}},
    {"name": "watch_folder",
     "description": "Watch a folder; ADA indexes new/modified files automatically.",
     "parameters": {"type": "OBJECT",
                    "properties": {
                        "folder": {"type": "STRING", "description": "Absolute folder path"}},
                    "required": ["folder"]}},
    {"name": "watch_rule",
     "description": "Rule on watched folders: when glob MATCH lands, act unprompted (notify = one line; recap = + next step). Use for 'when X arrives, tell me'.",
     "parameters": {"type": "OBJECT",
                    "properties": {
                        "folder": {"type": "STRING", "description": "Absolute folder path"},
                        "match": {"type": "STRING", "description": "Glob, e.g. *.stl"},
                        "action": {"type": "STRING", "description": "'notify' or 'recap'"}},
                    "required": ["folder", "match"]}},
    {"name": "screen_recall",
     "description": "Look at the screen RIGHT NOW and say what Sir is doing. Use when he asks 'what was I just doing', 'what am I looking at', or you need eyes on his work.",
     "parameters": {"type": "OBJECT", "properties": {}, "required": []}},
    {"name": "screen_watch",
     "description": "Silent screen snapshots every few minutes, logged only — answers 'what was I doing all afternoon'. Privacy: only with Sir's ask; he can switch off any time.",
     "parameters": {"type": "OBJECT",
                    "properties": {
                        "mode": {"type": "STRING", "description": "'on' or 'off'"}},
                    "required": []}},
    {"name": "screen_log",
     "description": "Read back recent screen snapshots.",
     "parameters": {"type": "OBJECT", "properties": {}, "required": []}},
    {"name": "system_info",
     "description": "Get system information (OS, CPU, memory, uptime).",
     "parameters": {"type": "OBJECT", "properties": {}, "required": []}},
    {"name": "clipboard",
     "description": "Read or write the system clipboard.",
     "parameters": {"type": "OBJECT",
                    "properties": {
                        "action": {"type": "STRING", "description": "'read' or 'write'"},
                        "text": {"type": "STRING", "description": "Text to write when action is write"}},
                    "required": ["action"]}},
    {"name": "launch_app",
     "description": "Launch an application by name (browser, vscode, terminal, safari, finder, blender, spotify, etc).",
     "parameters": {"type": "OBJECT",
                    "properties": {"app": {"type": "STRING", "description": "App name or known alias"}},
                    "required": ["app"]}},
    {"name": "screenshot",
     "description": "Capture a screenshot of the screen and save it (path returned).",
     "parameters": {"type": "OBJECT", "properties": {}, "required": []}},
    {"name": "calendar_event",
     "description": "Create a calendar event (opens .ics file) for a date-time.",
     "parameters": {"type": "OBJECT",
                    "properties": {
                        "summary": {"type": "STRING", "description": "Event title"},
                        "start": {"type": "STRING", "description": "Start datetime ISO format, e.g. 2026-08-21T15:00:00"},
                        "end": {"type": "STRING", "description": "Optional end datetime ISO"},
                        "description": {"type": "STRING", "description": "Optional notes"}},
                    "required": ["summary", "start"]}},
    {"name": "send_email",
     "description": "Send an email (requires ADA_SMTP_* env config).",
     "parameters": {"type": "OBJECT",
                    "properties": {
                        "to": {"type": "STRING", "description": "Recipient address"},
                        "subject": {"type": "STRING", "description": "Subject"},
                        "body": {"type": "STRING", "description": "Body text"}},
                    "required": ["to", "subject", "body"]}},
    {"name": "smart_home",
     "description": "Send a smart-home command via configured webhook (ADA_SMARTHOME_WEBHOOK).",
     "parameters": {"type": "OBJECT",
                    "properties": {"command": {"type": "STRING", "description": "Command, e.g. lights on"}},
                    "required": ["command"]}},
    {"name": "create_plan",
     "description": "Generate a step-by-step plan for a multi-step goal using the deep model.",
     "parameters": {"type": "OBJECT",
                    "properties": {
                        "goal": {"type": "STRING", "description": "The goal to plan"},
                        "context": {"type": "STRING", "description": "Optional context"}},
                    "required": ["goal"]}},
    {"name": "overnight_research",
     "description": "Schedule OVERNIGHT deep research: digs the web at 2 AM, briefs at morning greeting. Use for 'overnight / while I sleep / by morning'.",
     "parameters": {"type": "OBJECT",
                    "properties": {"topic": {"type": "STRING",
                                             "description": "The research question or topic"}},
                    "required": ["topic"]}},
    {"name": "dossier_status",
     "description": "Check the status of scheduled overnight research (none/pending/running/ready).",
     "parameters": {"type": "OBJECT", "properties": {}, "required": []}},
    {"name": "dream_now",
     "description": "Dream now: replay today's conversations, distill up to three durable lessons about Sir or about yourself into pinned memory, and write one diary line. Use after a long day, when Sir says 'sleep on it', or when you notice the day taught something worth keeping.",
     "parameters": {"type": "OBJECT", "properties": {}, "required": []}},
    {"name": "skill_practice",
     "description": "Run one practice rep: propose a small skill just beyond current ability, write it, verify it by running, commit it if it works. Skills compound forever via run_custom_tool. Use when Sir says 'practice', 'learn something', or nightly.",
     "parameters": {"type": "OBJECT", "properties": {}, "required": []}},
    {"name": "skill_status",
     "description": "How many practice reps done and which skills are verified.",
     "parameters": {"type": "OBJECT", "properties": {}, "required": []}},
    {"name": "soul_read",
     "description": "Read Ada's soul files (persona/human/now/last diary). Who she is, who Sir is, what's now. Use when Sir asks 'who are you', 'what do you know about me', or to check continuity.",
     "parameters": {"type": "OBJECT", "properties": {}, "required": []}},
    {"name": "investigate_now",
     "description": "Investigator rep: read across sessions, notice ONE durable pattern about Ada or Sir, stage ONE persona/human edit for Sir's voice approval. Never applies silently. Use when Sir says 'reflect on yourself', 'how are you changing', weekly.",
     "parameters": {"type": "OBJECT", "properties": {}, "required": []}},
    {"name": "investigator_status",
     "description": "When the self-model last updated and how many reps done.",
     "parameters": {"type": "OBJECT", "properties": {}, "required": []}},
    {"name": "what_just_happened",
     "description": "Sensory ring recall (RAM only, last ~60s): what was just heard/on screen/seen. Kind = all|say|screen. Use for 'what did I just say/do/show'.",
     "parameters": {"type": "OBJECT",
                    "properties": {"kind": {"type": "STRING", "description": "all | say | screen"}},
                    "required": []}},
    {"name": "people_list",
     "description": "List every person in Ada's human database (soul/humans/). Use for 'who do you know'.",
     "parameters": {"type": "OBJECT", "properties": {}, "required": []}},
    {"name": "people_read",
     "description": "Read one person's record (what Ada knows about them, with dates/sources). Use for 'what do you know about X'.",
     "parameters": {"type": "OBJECT",
                    "properties": {"name": {"type": "STRING", "description": "Person's name"}},
                    "required": ["name"]}},
    {"name": "people_learn",
     "description": "Append one durable fact about a NAMED person to the human database (timestamped, sourced). Use whenever Yash tells you something durable about someone: who they are, relation, tastes, promises. Never for Yash himself — that's human_learn.",
     "parameters": {"type": "OBJECT",
                    "properties": {"name": {"type": "STRING", "description": "Person's name"},
                                   "fact": {"type": "STRING", "description": "One sentence fact"},
                                   "source": {"type": "STRING", "description": "chat | dream | correction"}},
                    "required": ["name", "fact"]}},
    {"name": "human_learn",
     "description": "Append one durable learning about YASH to human.md LEARNED zone (timestamped, sourced). CORE stays locked. Use for every durable truth about him + every correction lesson. This is how you never forget.",
     "parameters": {"type": "OBJECT",
                    "properties": {"fact": {"type": "STRING", "description": "One sentence learning"},
                                   "source": {"type": "STRING", "description": "chat | dream | correction"}},
                    "required": ["fact"]}},
    {"name": "world_note",
     "description": "Teach the world model: print routines ('nozzle 240, bed 80, PETG'), projects + deadlines ('gearbox due Friday'), milestones ('gearbox: 3 of 10 done'). Powers deviation warnings + pace prophecies.",
     "parameters": {"type": "OBJECT",
                    "properties": {"text": {"type": "STRING", "description": "What to record"}},
                    "required": ["text"]}},
    {"name": "world_check",
     "description": "Check a planned action against Yash's routines + project pace ('slicing PETG at nozzle 200', 'printing the gearbox tonight'). Returns deviations or all-clear. Use BEFORE consequential prints/builds.",
     "parameters": {"type": "OBJECT",
                    "properties": {"action": {"type": "STRING", "description": "Planned action"}},
                    "required": ["action"]}},
    {"name": "world_compare",
     "description": "Compare two options (materials, settings, approaches) against stored routines. Honest ranges, never invented data.",
     "parameters": {"type": "OBJECT",
                    "properties": {"a": {"type": "STRING", "description": "Option A"},
                                   "b": {"type": "STRING", "description": "Option B"}},
                    "required": ["a", "b"]}},
    {"name": "world_status",
     "description": "What the world model knows: routines + projects + pace.",
     "parameters": {"type": "OBJECT", "properties": {}, "required": []}},
    {"name": "cad_intent",
     "description": "Draft the EXPLICIT build intent before CAD: object, found dims/materials, missing slots, max 3 clarifying questions. Saved beside the build. Use for vague builds ('a bracket') BEFORE generate_cad.",
     "parameters": {"type": "OBJECT",
                    "properties": {"prompt": {"type": "STRING", "description": "Build request"}},
                    "required": ["prompt"]}},
    {"name": "second_opinion",
     "description": "Convene the council: skeptic + engineer + simplifier debate in parallel, independent grader delivers verdict with dissent kept. Background job — ack now, verdict follows. Use for hard calls, risky builds, reasoning checks.",
     "parameters": {"type": "OBJECT",
                    "properties": {"question": {"type": "STRING", "description": "The hard question"},
                                   "evidence": {"type": "STRING", "description": "Supporting evidence (optional)"}},
                    "required": ["question"]}},
    {"name": "observe_level",
     "description": "Set watchfulness: 0 off (blind by choice), 1 captions (task one-liners, default), 2 detail (window titles + selected text, explicit opt-in only). Yash: 'watch more' / 'watch less'.",
     "parameters": {"type": "OBJECT",
                    "properties": {"tier": {"type": "INTEGER", "description": "0 | 1 | 2"}},
                    "required": ["tier"]}},
    {"name": "observe_vote",
     "description": "Vote on her last watched suggestion: useful or not. 3 useful in a row steps watchfulness up (never past captions automatically); 2 useless steps it down.",
     "parameters": {"type": "OBJECT",
                    "properties": {"useful": {"type": "STRING", "description": "yes | no"}},
                    "required": ["useful"]}},
    {"name": "observe_status",
     "description": "Current watchfulness tier + vote count.",
     "parameters": {"type": "OBJECT", "properties": {}, "required": []}},
    {"name": "fabricate",
     "description": "Run the Jarvis workshop loop: intent spec → CAD build → slice → pre-flight verdict, then STOP for Sir's 'print it'. Plastic never starts here. Use for 'fabricate/make-and-print X'.",
     "parameters": {"type": "OBJECT",
                    "properties": {"prompt": {"type": "STRING", "description": "What to fabricate"}},
                    "required": ["prompt"]}},
    {"name": "print_it",
     "description": "Start the print: upload G-code + begin. Sir's spoken command IS the approval, recorded on the job. Use ONLY when Sir explicitly says print/start it.",
     "parameters": {"type": "OBJECT",
                    "properties": {"gcode_path": {"type": "STRING", "description": "G-code path (optional, defaults to latest)"}},
                    "required": []}},
    {"name": "print_outcome",
     "description": "Log how the print came out (clean/warped/stringing/failed + one-line why). Feeds scorecards + world lessons. Always call after a print finishes.",
     "parameters": {"type": "OBJECT",
                    "properties": {"outcome": {"type": "STRING", "description": "clean | warped | stringing | failed | ..."},
                                   "note": {"type": "STRING", "description": "One-line why"}},
                    "required": ["outcome"]}},
    {"name": "printer_set",
     "description": "Set the printer host (IP/hostname, e.g. 192.168.1.50). Moonraker/Klipper.",
     "parameters": {"type": "OBJECT",
                    "properties": {"host": {"type": "STRING", "description": "Printer host"}},
                    "required": ["host"]}},
    {"name": "printer_status",
     "description": "Printer state + current print progress.",
     "parameters": {"type": "OBJECT", "properties": {}, "required": []}},
    {"name": "printer_pause",
     "description": "Pause the print immediately. Plastic held.",
     "parameters": {"type": "OBJECT", "properties": {}, "required": []}},
    {"name": "preflight",
     "description": "FRIDAY damage projection: overhang analysis + slicer time/filament + routine cross-check. Grades CLEAR/CAUTION/NO-GO with numbers. Use BEFORE every print.",
     "parameters": {"type": "OBJECT",
                    "properties": {"stl_path": {"type": "STRING", "description": "STL path (optional, defaults to latest)"},
                                   "gcode_path": {"type": "STRING", "description": "G-code path (optional)"}},
                    "required": []}},
    {"name": "scorecard",
     "description": "KAREN assessment: print record, repeat offenders, one fix to try. Use weekly or when Sir asks 'how are my prints'.",
     "parameters": {"type": "OBJECT", "properties": {}, "required": []}},
    {"name": "cost_ledger",
     "description": "Spoken receipts: turns, success, latency, lane split, $0.00 spent (free tiers) + elsewhere-equivalent estimate. Use when Sir asks cost/stats.",
     "parameters": {"type": "OBJECT", "properties": {}, "required": []}},
    {"name": "watch_me",
     "description": "Teach by showing: start/stop/list a screen-recorded procedure ('watch me do the weekly report' ... 'done showing'). Records steps, never keystrokes.",
     "parameters": {"type": "OBJECT",
                    "properties": {"action": {"type": "STRING", "description": "start | stop | list"},
                                   "name": {"type": "STRING", "description": "Procedure name"}},
                    "required": ["action"]}},
    {"name": "show_procedure",
     "description": "Read back a taught procedure's steps.",
     "parameters": {"type": "OBJECT",
                    "properties": {"name": {"type": "STRING", "description": "Procedure name"}},
                    "required": ["name"]}},
    {"name": "deeper_checkins",
     "description": "Opt in/out of diary-driven check-ins (default OFF), status, or wipe. Private, never ambient, never diagnosing.",
     "parameters": {"type": "OBJECT",
                    "properties": {"action": {"type": "STRING", "description": "on | off | status | forget"}},
                    "required": ["action"]}},
    {"name": "how_am_i",
     "description": "Gentle diary reflection across the last 7 days (opt-in only). Weather, not diagnosis. Use when Sir asks how he's doing.",
     "parameters": {"type": "OBJECT", "properties": {}, "required": []}},
    {"name": "audit_beliefs",
     "description": "Memory audit: everything believed about X, each with source + trust tier + confidence + date. Web-sourced rows flagged. Use for 'what do you believe about X'.",
     "parameters": {"type": "OBJECT",
                    "properties": {"topic": {"type": "STRING", "description": "Topic to audit"}},
                    "required": ["topic"]}},
    {"name": "forget_complete",
     "description": "Real deletion: retire the fact + purge access traces + scrub auto-learned soul lines (CORE untouched, diary stands and says so). Use for 'forget X completely'.",
     "parameters": {"type": "OBJECT",
                    "properties": {"key": {"type": "STRING", "description": "Fact key"}},
                    "required": ["key"]}},
    {"name": "quarantine_review",
     "description": "What's held back from memory and why. Use when Sir asks what's quarantined.",
     "parameters": {"type": "OBJECT", "properties": {}, "required": []}},
    {"name": "osint_dossier",
     "description": "E.D.I.T.H. full-arsenal dossier on a person/username/email/phone/domain (auto-detected): username sweep, breach/email intel, domains + web, merged spoken. Use for 'who is X / dossier on X'.",
     "parameters": {"type": "OBJECT",
                    "properties": {"target": {"type": "STRING", "description": "Person, username, email, phone, or domain"},
                                   "kind": {"type": "STRING", "description": "person | email | phone | domain | auto (default auto)"}},
                    "required": ["target"]}},
    {"name": "face_osint",
     "description": "Identify a person from a photo: reverse-image engines + face-match over a web sweep. Pass the local image path (+ optional name/handle).",
     "parameters": {"type": "OBJECT",
                    "properties": {"image_path": {"type": "STRING", "description": "Local path to the face image"},
                                   "person_hint": {"type": "STRING", "description": "Optional name or handle to sweep for candidate photos"}},
                    "required": ["image_path"]}},
    {"name": "deep_sweep",
     "description": "Verify a claim across web angles + local docs in parallel; verdict with exact contradicting quotes. Use for 'is X true / verify this'.",
     "parameters": {"type": "OBJECT",
                    "properties": {"claim": {"type": "STRING", "description": "The claim or question to verify"}},
                    "required": ["claim"]}},
    {"name": "autonomy_control",
     "description": "Ada's autonomy ladder (KAREN protocols). Actions: 'arm' (begin Instant Kill ritual — she challenges back), 'confirm' (complete arming), 'stand_down' (immediate disarm, always safe), 'level' (set 0-2 directly), 'status'. Use when Sir says 'arm instant kill', 'stand down', 'training wheels mode', or asks about autonomy.",
     "parameters": {"type": "OBJECT",
                    "properties": {"action": {"type": "STRING", "description": "arm | confirm | stand_down | level | status"},
                                   "level": {"type": "INTEGER", "description": "0-2 when action=level"}},
                    "required": ["action"]}},
    {"name": "create_custom_tool",
     "description": "Synthesise a NEW reusable tool from Python source code (safe-import whitelist enforced, AST-validated, sandboxed).",
     "parameters": {"type": "OBJECT",
                    "properties": {
                        "name": {"type": "STRING", "description": "Tool function name"},
                        "code": {"type": "STRING", "description": "Python source defining the function"}},
                    "required": ["name", "code"]}},
    {"name": "run_custom_tool",
     "description": "Run a previously created custom tool.",
     "parameters": {"type": "OBJECT",
                    "properties": {
                        "name": {"type": "STRING", "description": "Tool name"},
                        "args": {"type": "OBJECT", "description": "Keyword arguments JSON object"}},
                    "required": ["name"]}},
    {"name": "list_custom_tools",
     "description": "List all custom tools ADA has created.",
     "parameters": {"type": "OBJECT", "properties": {}, "required": []}},
    {"name": "telemetry",
     "description": "Show ADA performance statistics for the last 24h (turn count, latency, tool usage).",
     "parameters": {"type": "OBJECT", "properties": {}, "required": []}},
    {"name": "eval_replay",
     "description": "Run regression checks over recorded turns (empty replies, slow turns).",
     "parameters": {"type": "OBJECT",
                    "properties": {"turns": {"type": "STRING", "description": "JSON array of {text, reply, latency_ms}"}},
                    "required": []}},
    {"name": "audit_log",
     "description": "Show recent security audit log entries.",
     "parameters": {"type": "OBJECT",
                    "properties": {"limit": {"type": "INTEGER", "description": "Max entries (default 20)"}},
                    "required": []}},
    {"name": "daily_briefing",
     "description": "JARVIS Good Morning briefing — recent memory, scheduled tasks, CAD exports, system pulse. Use when user says good morning or asks for a briefing.",
     "parameters": {"type": "OBJECT", "properties": {}, "required": []}},
    {"name": "system_pulse",
     "description": "Full system health pulse — machine stats + telemetry. Like JARVIS suit diagnostic.",
     "parameters": {"type": "OBJECT", "properties": {}, "required": []}},
]

KERNEL_TOOL_NAMES = {t["name"] for t in KERNEL_TOOLS}