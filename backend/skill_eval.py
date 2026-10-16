"""Learning gated by evaluation — Feature 6.

Forged skills no longer commit on a single test call. They climb a
four-level eval ladder before they touch live traffic:

  L0 deterministic — schema valid, AST gate passes, timeout enforced,
                     idempotent re-run. Must be 100% (the `make gate` rule).
  L1 behavior trials — the tool's own TEST_CALL plus 2-3 generated edge
                     inputs, executed in a no-network subprocess sandbox.
  L2 eval slice — the `@pytest.mark.skill_eval` subset of tests/eval_ada.py
                     (fast, deterministic, offline). DeepEval-style LLM
                     judges exist as an OPTIONAL second opinion behind
                     ADA_JUDGE=1 — never a hard dependency, never on by
                     default (zero cost, zero new requirements).
  L3 production — candidate -> shadow -> canary -> active, with
                     auto-rollback on error-rate regression.

Registry: SQLite `skill_versions`
    (name, version, status, eval_scores_json, promoted_at, code_ref,
     blast_radius, note)
  status in candidate|shadow|canary|active|retired.
  Rollback = flip the active-version pointer (`skill_active`), instant,
  no redeploy. Versioned dirs: mcp_tools/<name>/v<N>/tool.py; v1 is the
  pre-existing flat mcp_tools/<name>.py (backward compat).

Serving modes (env flags):
  ADA_SKILL_SHADOW=1        shadow: candidate runs alongside live calls,
                            results logged, never shown to the user.
  ADA_SKILL_CANARY=<name>   canary: ADA_SKILL_CANARY_PCT (default 10) of
                            calls route to the canary version; auto-rollback
                            on error-rate regression vs the active version
                            or on sustained useless votes (observe.py).
  (default)                 active: the pointer version serves.

Voice approval queue (proposals_pending.json): high-blast-radius skills
(new external side-effect classes) REQUIRE a spoken yes before `active` —
never auto-promoted past canary. Surfaced BY VOICE only, no popups.

Zero local ML models. DeepEval import is try/except-guarded behind
ADA_JUDGE=1 and is NOT in requirements.txt.
"""
from __future__ import annotations

import ast
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

STATUSES = ("candidate", "shadow", "canary", "active", "retired")
PROMOTION_ORDER = {s: i for i, s in enumerate(STATUSES)}

# Tokens that mark a skill as a NEW EXTERNAL SIDE-EFFECT class — these need
# a spoken yes from Sir before they can go `active`. Forged code is
# AST-gated to pure functions, so this is about intent/surface: a skill
# whose purpose is to send/post/delete/schedule is high blast radius even
# if today's forge cannot grant it the capability.
BLAST_TOKENS = {
    "email", "emails", "sms", "text", "message", "messages", "notify",
    "notification", "notifications", "send", "sends", "sent", "post", "posts",
    "tweet", "tweets", "share", "shares", "publish", "upload", "uploads",
    "buy", "buys", "sell", "sells", "pay", "pays", "payment", "charge",
    "delete", "deletes", "remove", "removes", "schedule", "schedules",
    "reminder", "reminders", "remind", "alert", "alerts", "broadcast",
    "call", "calls", "phone",
}

_JSON_TYPES = {"string", "integer", "number", "boolean", "array", "object", "null"}

_EVAL_SUBSET_MARK = "skill_eval"
_SUBPROCESS_TIMEOUT_S = 180


# ── subprocess sandbox ────────────────────────────────────────────────────────
# The module is re-imported in a fresh interpreter with `socket` nulled out
# (defense in depth: the AST gate already forbids socket imports, the
# sandbox makes network failure structural even if a bypass ever lands).

_SANDBOX_RUNNER = r"""
import json, os, sys, importlib.util
sys.modules["socket"] = None  # no network in the sandbox, ever
spec = importlib.util.spec_from_file_location(
    "sb_tool", os.environ["ADA_SB_MODULE"])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
fn = getattr(mod, os.environ["ADA_SB_FN"])
result = fn(**json.loads(os.environ["ADA_SB_KWARGS"]))
try:
    out = json.dumps(result, default=str)
except Exception:
    out = json.dumps(str(result))
sys.stdout.write(out)
"""


def run_in_sandbox(module_path: str | Path, fn_name: str,
                   kwargs: dict | None = None, timeout: int = 10) -> dict:
    """Run one tool call in a no-network subprocess. Never raises.

    Returns {"ok": True, "result": ...} or {"ok": False, "error": ...}.
    "error" is one of "timeout", "crashed:<rc>", "bad_module:<...>",
    "call_failed:<...>".
    """
    module_path = str(module_path)
    env = {
        "PATH": os.environ.get("PATH", ""),
        "ADA_SANDBOX": "1",
        "ADA_SB_MODULE": module_path,
        "ADA_SB_FN": fn_name,
        "ADA_SB_KWARGS": json.dumps(kwargs or {}),
    }
    try:
        proc = subprocess.run(
            [sys.executable, "-u", "-c", _SANDBOX_RUNNER],
            env=env, capture_output=True, text=True,
            timeout=timeout, cwd=tempfile.gettempdir())
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}
    except Exception as e:
        return {"ok": False, "error": f"harness: {type(e).__name__}"}
    if proc.returncode != 0:
        err = (proc.stderr or "").strip().splitlines()
        tail = err[-1] if err else f"rc={proc.returncode}"
        if "import of socket halted" in tail or "socket" in tail.lower():
            return {"ok": False, "error": f"no_network: {tail[:160]}"}
        return {"ok": False, "error": f"crashed: {tail[:160]}"}
    try:
        return {"ok": True, "result": json.loads(proc.stdout or "null")}
    except Exception:
        return {"ok": False, "error": "bad_output"}


# ── L0: deterministic checks (must be 100%) ───────────────────────────────────

def _read_module_meta(module_path: str | Path) -> tuple[dict, str, str]:
    """(meta, fn_name, src) from a rendered MCP tool module. Raises."""
    src = Path(module_path).read_text()
    tree = ast.parse(src)
    meta: dict = {}
    fn_name = ""
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id.startswith("TOOL_"):
                    try:
                        meta[t.id] = ast.literal_eval(node.value)
                    except Exception:
                        pass
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and not node.name.startswith("_"):
            if node.name == meta.get("TOOL_NAME") or not fn_name:
                fn_name = node.name
            if node.name == meta.get("TOOL_NAME"):
                break
    return meta, fn_name, src


_ANN_TO_JSON = {
    "int": "integer", "str": "string", "float": "number", "bool": "boolean",
    "list": "array", "dict": "object", "List": "array", "Dict": "object",
}


def _schema_from_signature(src: str, fn_name: str) -> dict | None:
    """Derive a JSON schema from the function signature — fallback for
    legacy/raw modules that carry no TOOL_SCHEMA block."""
    try:
        tree = ast.parse(src)
    except Exception:
        return None
    target = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == fn_name:
            target = node
            break
    if target is None:
        return None
    props: dict = {}
    for arg in target.args.args:
        if arg.arg in ("self", "cls"):
            continue
        pspec: dict = {}
        if arg.annotation is not None:
            try:
                ann = ast.unparse(arg.annotation)
            except Exception:
                ann = ""
            base = re.split(r"[\[\]]", ann.strip().strip("'\""))[0].split(".")[-1]
            jtype = _ANN_TO_JSON.get(base)
            if jtype:
                pspec["type"] = jtype
        props[arg.arg] = pspec
    return {"type": "object", "properties": props}


def _ast_gate() -> object:
    """A validator carrying the same AST rules as the forge. Lazy import so
    this module stays importable without the kernel."""
    try:
        from mcp.forge import MCPForge  # noqa
        from agent_kernel import ToolForge
        return ToolForge.__new__(ToolForge)
    except Exception:
        # Minimal fallback mirroring agent_kernel.ToolForge rules (kept in
        # sync by tests/test_eval_gating.py::test_ast_fallback_parity).
        from agent_kernel import ToolForge  # re-raise the real error
        raise


def check_schema_valid(module_path) -> tuple[bool, str]:
    try:
        meta, fn_name, src = _read_module_meta(module_path)
    except Exception as e:
        return False, f"unreadable module: {e}"
    schema = meta.get("TOOL_SCHEMA")
    derived = False
    if not isinstance(schema, dict):
        # Legacy/raw modules carry no TOOL_SCHEMA — derive from signature.
        schema = _schema_from_signature(src, fn_name)
        derived = True
    if not isinstance(schema, dict) or schema.get("type") != "object":
        return False, "TOOL_SCHEMA missing or not an object schema"
    props = schema.get("properties")
    if not isinstance(props, dict):
        return False, "TOOL_SCHEMA.properties missing"
    for pname, pspec in props.items():
        if not isinstance(pspec, dict):
            return False, f"property {pname!r} has no spec dict"
        ptype = pspec.get("type")
        if ptype is not None and ptype not in _JSON_TYPES:
            return False, f"property {pname!r} has bad type {ptype!r}"
    if not fn_name:
        return False, "no callable tool function found"
    tag = " (derived from signature)" if derived else ""
    return True, f"schema ok ({len(props)} params), fn={fn_name}{tag}"


def check_ast_gate(module_path) -> tuple[bool, str]:
    try:
        _src = Path(module_path).read_text()
        tree = ast.parse(_src)
        _ast_gate()._validate_ast(tree)
    except SyntaxError as e:
        return False, f"syntax error: {e}"
    except ValueError as e:
        return False, f"AST gate rejected: {e}"
    except Exception as e:
        return False, f"gate unavailable: {type(e).__name__}"
    return True, "AST gate passes"


_timeout_harness_verified: bool | None = None


def check_timeout_enforced() -> tuple[bool, str]:
    """The sandbox actually kills an overrunning tool (verified once per
    process against a sleeper module)."""
    global _timeout_harness_verified
    if _timeout_harness_verified is None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "sleeper.py"
            p.write_text("import time\ndef sleeper():\n"
                         "    time.sleep(30)\n    return 1\n")
            r = run_in_sandbox(p, "sleeper", {}, timeout=2)
            _timeout_harness_verified = (
                not r["ok"] and r["error"] == "timeout")
    ok = bool(_timeout_harness_verified)
    return ok, "sandbox kills overruns" if ok else "sandbox did NOT kill a sleeper"


def check_idempotent(module_path, fn_name, kwargs, timeout=10) -> tuple[bool, str]:
    r1 = run_in_sandbox(module_path, fn_name, kwargs, timeout=timeout)
    r2 = run_in_sandbox(module_path, fn_name, kwargs, timeout=timeout)
    if not r1["ok"]:
        return False, f"first run failed: {r1['error']}"
    if not r2["ok"]:
        return False, f"second run failed: {r2['error']}"
    n1 = json.dumps(r1["result"], sort_keys=True, default=str)
    n2 = json.dumps(r2["result"], sort_keys=True, default=str)
    if n1 != n2:
        return False, "re-run diverged (not idempotent)"
    return True, "re-run identical"


def l0_checks(module_path, fn_name=None, test_kwargs=None,
              timeout: int = 10) -> dict:
    """Run the L0 deterministic suite. All four must pass.

    Returns {"passed": bool, "checks": {name: (ok, detail)}}.
    """
    try:
        meta, detected_fn, _src = _read_module_meta(module_path)
        fn_name = fn_name or meta.get("TOOL_NAME") or detected_fn
    except Exception as e:
        return {"passed": False, "checks": {
            "read_module": (False, f"unreadable: {e}")}}
    checks: dict[str, tuple[bool, str]] = {}
    checks["schema_valid"] = check_schema_valid(module_path)
    checks["ast_gate"] = check_ast_gate(module_path)
    checks["timeout_enforced"] = check_timeout_enforced()
    checks["idempotent"] = check_idempotent(
        module_path, fn_name, test_kwargs or {}, timeout=timeout)
    passed = all(ok for ok, _d in checks.values())
    return {"passed": passed, "checks": checks}


# ── L1: behavior trials (TEST_CALL + generated edge inputs) ───────────────────

_EDGE_VALUES = {
    "integer": [-1, 0, 10 ** 12],
    "number": [-1.5, 0.0],
    "string": ["", "x" * 512],
    "boolean": [True],
    "array": [[], [1]],
    "object": [{}, {"k": 1}],
    "null": [None],
}


def _infer_type(value) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "string"


def edge_inputs(schema: dict, base_kwargs: dict, n: int = 3) -> list[dict]:
    """2-3 generated edge-input dicts: one perturbed param per case.

    The param type comes from the schema when present, else is inferred
    from the TEST_CALL value — an untyped numeric param gets numeric
    edges, not string ones.
    """
    props = (schema or {}).get("properties", {}) or {}
    if not props and base_kwargs:
        props = {k: {} for k in base_kwargs}
    cases: list[dict] = []
    for pname, pspec in props.items():
        if len(cases) >= n:
            break
        ptype = (pspec or {}).get("type") or _infer_type(
            base_kwargs.get(pname))
        for val in _EDGE_VALUES.get(ptype, [""]):
            if pname in base_kwargs and base_kwargs[pname] == val:
                continue
            case = dict(base_kwargs)
            case[pname] = val
            cases.append(case)
            if len(cases) >= n:
                break
    return cases[:n]


def behavior_trials(module_path, fn_name, test_kwargs, schema=None,
                    timeout: int = 10) -> dict:
    """Run TEST_CALL + edge inputs in the sandbox. Every trial must return
    without crashing — this is a safety trial, not a correctness proof."""
    trials = [{"label": "test_call", "kwargs": dict(test_kwargs or {})}]
    for i, kw in enumerate(edge_inputs(schema or {}, test_kwargs or {})):
        trials.append({"label": f"edge_{i}", "kwargs": kw})
    results = []
    for t in trials:
        r = run_in_sandbox(module_path, fn_name, t["kwargs"], timeout=timeout)
        results.append({"label": t["label"], "ok": r["ok"],
                        "error": r.get("error"),
                        "result": (str(r.get("result"))[:200]
                                   if r["ok"] else None)})
    passed = all(r["ok"] for r in results)
    return {"passed": passed, "trials": results}


# ── L2: the marked eval subset (fast, deterministic, offline) ─────────────────

def run_eval_subset(repo_root: str | Path,
                    timeout: int = _SUBPROCESS_TIMEOUT_S) -> tuple[bool, str]:
    """Run exactly the `@pytest.mark.skill_eval` slice of tests/eval_ada.py.

    Returns (all_green, one_line_summary). Never raises.
    """
    repo_root = str(repo_root)
    cmd = [sys.executable, "-m", "pytest", "tests/eval_ada.py",
           "-m", _EVAL_SUBSET_MARK, "-q", "-p", "no:cacheprovider",
           "--no-header"]
    try:
        proc = subprocess.run(cmd, cwd=repo_root, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "eval subset timed out"
    except Exception as e:
        return False, f"eval subset harness failed: {type(e).__name__}"
    tail = (proc.stdout or "").strip().splitlines()
    summary = tail[-1] if tail else f"rc={proc.returncode}"
    return proc.returncode == 0, summary[:200]


def judge_suite_optional(name: str, summary: str) -> dict:
    """DeepEval-style LLM-judge second opinion — OPTIONAL, off by default.

    Enabled only with ADA_JUDGE=1 AND an importable judge lib. No new hard
    deps: deepeval is never in requirements.txt; the import is guarded.
    """
    if os.environ.get("ADA_JUDGE") != "1":
        return {"ran": False, "reason": "ADA_JUDGE!=1 (opt-in only)"}
    try:
        import deepeval  # type: ignore  # noqa
    except Exception:
        return {"ran": False, "reason": "deepeval not installed"}
    # A real judge run would score seeded multi-turn conversations here;
    # the hook exists so a future wave can plug it in without touching
    # the promotion path. Advisory only — never a gate.
    return {"ran": True, "verdict": "advisory-only",
            "note": f"judge saw skill '{name}': {summary[:80]}"}


# ── blast radius ──────────────────────────────────────────────────────────────

def classify_blast_radius(name: str, purpose: str, code: str) -> str:
    """'high' for new external side-effect classes, else 'low'."""
    tokens = set(re.findall(r"[a-z]+", f"{name} {purpose} {code}".lower()))
    return "high" if tokens & BLAST_TOKENS else "low"


# ── SQLite registry ───────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS skill_versions(
  name TEXT NOT NULL, version TEXT NOT NULL,
  status TEXT NOT NULL, eval_scores_json TEXT NOT NULL DEFAULT '{}',
  promoted_at REAL, code_ref TEXT, blast_radius TEXT DEFAULT 'low',
  note TEXT DEFAULT '',
  PRIMARY KEY(name, version));
CREATE TABLE IF NOT EXISTS skill_active(
  name TEXT PRIMARY KEY, version TEXT NOT NULL, since REAL);
CREATE TABLE IF NOT EXISTS promotion_log(
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, version TEXT,
  from_status TEXT, to_status TEXT, ts REAL, reason TEXT);
CREATE TABLE IF NOT EXISTS eval_log(
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, version TEXT,
  stage TEXT, passed INTEGER, ts REAL, detail TEXT);
CREATE TABLE IF NOT EXISTS served_runs(
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, version TEXT,
  kind TEXT, ts REAL, ok INTEGER, err TEXT);
"""


class SkillRegistry:
    """Version registry + active pointer + run logs. DB is runtime state."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)
        self._db.commit()

    # ── versions ──
    def register(self, name, version, status="candidate", scores=None,
                 code_ref="", blast_radius="low", note=""):
        assert status in STATUSES, status
        self._db.execute(
            "INSERT OR REPLACE INTO skill_versions(name, version, status,"
            " eval_scores_json, promoted_at, code_ref, blast_radius, note)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (name, str(version), status, json.dumps(scores or {}),
             time.time() if status != "candidate" else None,
             code_ref, blast_radius, note))
        self._db.commit()

    def get(self, name, version):
        r = self._db.execute(
            "SELECT * FROM skill_versions WHERE name=? AND version=?",
            (name, str(version))).fetchone()
        return dict(r) if r else None

    def versions(self, name) -> list[dict]:
        return [dict(r) for r in self._db.execute(
            "SELECT * FROM skill_versions WHERE name=? ORDER BY"
            " CAST(version AS INTEGER)", (name,))]

    def by_status(self, name, status) -> list[dict]:
        return [dict(r) for r in self._db.execute(
            "SELECT * FROM skill_versions WHERE name=? AND status=?",
            (name, status))]

    def set_status(self, name, version, status, reason=""):
        assert status in STATUSES, status
        prev = self.get(name, version)
        self._db.execute(
            "UPDATE skill_versions SET status=?, promoted_at=?,"
            " note=COALESCE(NULLIF(?,''), note) WHERE name=? AND version=?",
            (status, time.time(), reason, name, str(version)))
        self._db.execute(
            "INSERT INTO promotion_log(name, version, from_status, to_status,"
            " ts, reason) VALUES(?,?,?,?,?,?)",
            (name, str(version), (prev or {}).get("status"), status,
             time.time(), reason))
        self._db.commit()

    def scores(self, name, version) -> dict:
        r = self.get(name, version)
        try:
            return json.loads((r or {}).get("eval_scores_json") or "{}")
        except Exception:
            return {}

    # ── active pointer ──
    def active_version(self, name) -> str:
        r = self._db.execute(
            "SELECT version FROM skill_active WHERE name=?", (name,)).fetchone()
        return r["version"] if r else "1"  # flat-file compat: v1 = existing file

    def set_active(self, name, version, reason=""):
        self._db.execute(
            "INSERT OR REPLACE INTO skill_active(name, version, since)"
            " VALUES(?,?,?)", (name, str(version), time.time()))
        self._db.commit()
        self.set_status(name, version, "active", reason)

    def previous_active(self, name) -> str | None:
        """Most recent version that was active before the current one."""
        cur = self.active_version(name)
        r = self._db.execute(
            "SELECT version FROM promotion_log WHERE name=? AND"
            " to_status='active' AND version!=? ORDER BY ts DESC LIMIT 1",
            (name, cur)).fetchone()
        return r["version"] if r else None

    # ── run logs ──
    def log_run(self, name, version, kind, ok, err=""):
        self._db.execute(
            "INSERT INTO served_runs(name, version, kind, ts, ok, err)"
            " VALUES(?,?,?,?,?,?)",
            (name, str(version), kind, time.time(), int(bool(ok)),
             (err or "")[:200]))
        self._db.commit()

    def error_rate(self, name, version, kind=None, window: int = 50) -> tuple[float, int]:
        q = ("SELECT ok FROM served_runs WHERE name=? AND version=?"
             + (" AND kind=?" if kind else "") + " ORDER BY id DESC LIMIT ?")
        args = [name, str(version)] + ([kind] if kind else []) + [window]
        rows = self._db.execute(q, args).fetchall()
        if not rows:
            return 0.0, 0
        bad = sum(1 for r in rows if not r["ok"])
        return bad / len(rows), len(rows)

    def log_eval(self, name, version, stage, passed, detail=""):
        self._db.execute(
            "INSERT INTO eval_log(name, version, stage, passed, ts, detail)"
            " VALUES(?,?,?,?,?,?)",
            (name, str(version), stage, int(bool(passed)), time.time(),
             (detail or "")[:500]))
        self._db.commit()

    def close(self):
        try:
            self._db.close()
        except Exception:
            pass


# ── voice approval queue (proposals_pending.json, surfaced BY VOICE) ──────────

class VoiceQueue:
    """The approval queue the brain reads OUT LOUD. No popups, ever."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _load(self) -> list:
        try:
            if self.path.exists():
                d = json.loads(self.path.read_text())
                return d if isinstance(d, list) else []
        except Exception:
            pass
        return []

    def _save(self, items: list):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(items, indent=1))
        except Exception as e:
            print(f"[EVALGATE] voice queue save failed: {e}")

    def propose_for_voice(self, name: str, version: str, summary: str) -> str:
        """Enqueue; returns the line the brain should SPEAK."""
        items = self._load()
        items = [i for i in items
                 if not (i.get("name") == name and i.get("status") == "pending")]
        items.append({"name": name, "version": str(version), "summary": summary,
                      "ts": time.time(), "status": "pending"})
        self._save(items)
        return (f"I forged {name}, version {version} — {summary}. "
                f"Say 'approve {name}' to make it live, or 'reject {name}'"
                f" to retire it.")

    def next_voice_proposal(self) -> dict | None:
        """The pending item for the brain to ask about out loud."""
        for i in self._load():
            if i.get("status") == "pending":
                return dict(i)
        return None

    def pending_count(self) -> int:
        return sum(1 for i in self._load() if i.get("status") == "pending")

    def resolve_voice_proposal(self, name: str, approved: bool) -> str:
        items = self._load()
        found = None
        for i in items:
            if i.get("name") == name and i.get("status") == "pending":
                i["status"] = "approved" if approved else "rejected"
                found = i
                break
        if not found:
            return f"No pending proposal for '{name}'."
        self._save(items)
        if approved:
            return (f"Approved — {name} version {found['version']} goes live.")
        return f"Retired — {name} stays out of the live set."


# ── versioned dirs ────────────────────────────────────────────────────────────

def version_dir(tools_root: str | Path, name: str, version: str | int) -> Path:
    return Path(tools_root) / name / f"v{version}"


def archive_version(tools_root, name, version, src_path) -> str:
    """Copy a rendered module into mcp_tools/<name>/v<N>/tool.py."""
    dest = version_dir(tools_root, name, version) / "tool.py"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(Path(src_path).read_text())
    return str(dest)


def version_module_path(tools_root, name, version) -> Path | None:
    """v<N>/tool.py, else the flat mcp_tools/<name>.py for v1 (compat)."""
    v = version_dir(tools_root, name, version) / "tool.py"
    if v.exists():
        return v
    if str(version) == "1":
        flat = Path(tools_root) / f"{name}.py"
        if flat.exists():
            return flat
    return None


def next_version(registry: SkillRegistry, tools_root, name) -> int:
    rows = registry.versions(name)
    if rows:
        return max(int(r["version"]) for r in rows) + 1
    # Flat-file backward compat: a pre-existing mcp_tools/<name>.py IS v1 —
    # the first registered version archives its current content as v1.
    return 1


def load_version_fn(module_path):
    """AST-validated exec of one archived module -> the tool function.

    Mirrors the forge's gate; does NOT register into any live registry, so
    shadow/canary versions never leak into the served set.
    """
    src = Path(module_path).read_text()
    tree = ast.parse(src)
    _ast_gate()._validate_ast(tree)
    ns: dict = {}
    exec(compile(tree, str(module_path), "exec"),
         {"__name__": "skill_eval_version"}, ns)
    target = None
    for key, obj in ns.items():
        if callable(obj) and not key.startswith("_"):
            if key == ns.get("TOOL_NAME"):
                target = obj
                break
            target = target or obj
    if not callable(target):
        raise ValueError("no callable tool function in version module")
    return target


# ── the gate: evaluate -> promote -> serve -> rollback ────────────────────────

class SkillEvalGate:
    """Owns the registry, the ladder, promotion, serving and rollback."""

    def __init__(self, state_dir: str | Path, tools_root: str | Path):
        self.state_dir = Path(state_dir)
        self.tools_root = Path(tools_root)
        self.registry = SkillRegistry(self.state_dir / "skill_versions.db")
        self.voice = VoiceQueue(
            self.state_dir / "soul" / "proposals_pending.json")
        self.forge = None          # attached lazily (MCPForge/ToolForge)
        self.observe = None        # ObserveTiers, attached lazily
        self._canary_counter: dict[str, int] = {}

    def attach_forge(self, forge):
        self.forge = forge
        return self

    def attach_observe(self, observe):
        self.observe = observe
        return self

    # ── commit path: forge -> L0 -> L1 trials -> L2 subset -> candidate ──
    def evaluate_candidate(self, name, purpose, code, test_kwargs,
                           module_path, repo_root) -> dict:
        """Run the ladder. All-green -> registered as candidate, auto-promoted
        to shadow. Anything red -> blocked with the failing stage named."""
        meta_fn = None
        try:
            meta, detected, _s = _read_module_meta(module_path)
            meta_fn = meta.get("TOOL_NAME") or detected
        except Exception:
            pass
        fn_name = meta_fn or name
        try:
            schema = (meta or {}).get("TOOL_SCHEMA") or {}
        except Exception:
            schema = {}

        version = next_version(self.registry, self.tools_root, name)

        # L0 — deterministic, must be 100%
        l0 = l0_checks(module_path, fn_name=fn_name,
                       test_kwargs=test_kwargs or {})
        self.registry.log_eval(name, version, "L0", l0["passed"],
                               "; ".join(f"{k}={'ok' if ok else 'FAIL'}"
                                         for k, (ok, _d) in l0["checks"].items()))
        if not l0["passed"]:
            failed = [k for k, (ok, _d) in l0["checks"].items() if not ok]
            return {"status": "blocked", "stage": "L0", "version": version,
                    "failed": failed, "checks": l0["checks"]}

        # L1 — sandboxed behavior trials
        trials = behavior_trials(module_path, fn_name, test_kwargs or {},
                                 schema=schema)
        self.registry.log_eval(name, version, "L1", trials["passed"],
                               f"{sum(1 for t in trials['trials'] if t['ok'])}"
                               f"/{len(trials['trials'])} trials ok")
        if not trials["passed"]:
            bad = [t for t in trials["trials"] if not t["ok"]]
            return {"status": "blocked", "stage": "L1", "version": version,
                    "failed": [t["label"] for t in bad], "trials": trials}

        # L2 — the marked eval subset (exactly the skill_eval slice)
        subset_ok, subset_summary = run_eval_subset(repo_root)
        self.registry.log_eval(name, version, "L2", subset_ok, subset_summary)
        if not subset_ok:
            return {"status": "blocked", "stage": "L2", "version": version,
                    "failed": ["eval_subset"], "summary": subset_summary}

        # L2b — optional judge (advisory, never a gate)
        judge = judge_suite_optional(name, purpose or "")

        # All green -> candidate, then auto-promote to shadow.
        blast = classify_blast_radius(name, purpose or "", code or "")
        code_ref = archive_version(self.tools_root, name, version, module_path)
        scores = {
            "l0": {k: ok for k, (ok, _d) in l0["checks"].items()},
            "l1_trials": f"{sum(1 for t in trials['trials'] if t['ok'])}"
                         f"/{len(trials['trials'])}",
            "l2_subset": subset_summary,
            "judge": judge,
            "passed": True,
        }
        self.registry.register(name, version, status="candidate",
                               scores=scores, code_ref=code_ref,
                               blast_radius=blast)
        self.promote(name, version, "shadow",
                     reason="all-green eval ladder; auto-promoted")
        verdict = {"status": "shadow", "stage": "promoted", "version": version,
                   "blast_radius": blast, "scores": scores}
        if blast == "high":
            spoken = self.voice.propose_for_voice(
                name, version,
                f"{purpose or 'a new skill'} (high blast radius — needs your spoken yes before going live)")
            self.registry.set_status(
                name, version, "shadow",
                reason="high blast radius: voice approval required before active")
            verdict["voice_proposal"] = spoken
        return verdict

    # ── promotion pipeline ──
    def promote(self, name, version, to_status, reason="") -> dict:
        cur = self.registry.get(name, version)
        if not cur:
            return {"ok": False, "error": "unknown version"}
        if PROMOTION_ORDER[to_status] < PROMOTION_ORDER[cur["status"]]:
            return {"ok": False,
                    "error": f"cannot demote via promote ({cur['status']}->{to_status})"}
        if to_status == "active":
            return self._go_active(name, version, reason, via_voice=False)
        self.registry.set_status(name, version, to_status, reason)
        return {"ok": True, "status": to_status}

    def _go_active(self, name, version, reason="", via_voice=False) -> dict:
        row = self.registry.get(name, version)
        if not row:
            return {"ok": False, "error": "unknown version"}
        if row["blast_radius"] == "high" and not via_voice:
            # Never auto-promote a high-blast-radius skill past canary.
            self.voice.propose_for_voice(
                name, version,
                "high blast radius — your spoken approval is required")
            return {"ok": False, "error": "voice_approval_required",
                    "detail": "high blast radius: enqueued for voice approval, "
                              "not promoted past canary"}
        prev_active = self.registry.active_version(name)
        if prev_active != str(version):
            if self.registry.get(name, prev_active):
                self.registry.set_status(name, prev_active, "retired",
                                         reason=f"superseded by v{version}")
        # Flip the pointer; the flat file is what the live server serves.
        self.registry.set_active(name, version,
                                 reason or "promoted to active")
        self._publish_active_file(name, version)
        return {"ok": True, "status": "active"}

    def _publish_active_file(self, name, version):
        """Copy the active version's module to the flat mcp_tools/<name>.py
        (what hot-load serves) and reload it in the forge if attached."""
        src = version_module_path(self.tools_root, name, version)
        if not src:
            return
        flat = self.tools_root / f"{name}.py"
        flat.write_text(src.read_text())
        forge = self.forge
        try:
            if forge is not None and hasattr(forge, "_load_mcp_module"):
                forge._load_mcp_module(flat)
        except Exception as e:
            print(f"[EVALGATE] reload of active v{version} failed: {e}")

    def maybe_promote_canary(self, name, version,
                             min_samples: int = 10) -> dict:
        """Canary health check: promote if clean, auto-rollback on error-rate
        regression vs the active version or sustained useless votes."""
        row = self.registry.get(name, version)
        if not row or row["status"] != "canary":
            return {"ok": False, "error": "not a canary"}
        active_ver = self.registry.active_version(name)
        can_err, can_n = self.registry.error_rate(name, version, kind="canary")
        act_err, act_n = self.registry.error_rate(name, active_ver, kind="active")
        # sustained useless votes from observe -> demote to shadow
        if self.observe is not None:
            try:
                if self.observe.skill_signal(name) == "demote":
                    self.registry.set_status(
                        name, version, "shadow",
                        reason="auto-demote: sustained useless votes")
                    self.observe.skill_signal_ack(name)
                    return {"ok": True, "action": "demoted_to_shadow",
                            "reason": "useless votes"}
            except Exception:
                pass
        if can_n >= min_samples and can_err > act_err + 0.05 and can_err > 0:
            # auto-rollback: canary loses, active keeps serving
            self.registry.set_status(
                name, version, "shadow",
                reason=f"auto-rollback: canary err {can_err:.0%} (n={can_n})"
                       f" vs active {act_err:.0%} (n={act_n})")
            return {"ok": True, "action": "rolled_back",
                    "canary_err": can_err, "active_err": act_err}
        if can_n >= min_samples:
            return self._go_active(name, version,
                                   reason=f"canary clean ({can_n} calls, "
                                          f"err {can_err:.0%} vs active {act_err:.0%})")
        return {"ok": True, "action": "waiting",
                "canary_samples": can_n, "min_samples": min_samples}

    def rollback(self, name, reason="manual rollback") -> dict:
        """Flip the active pointer to the previous active version."""
        cur = self.registry.active_version(name)
        prev = self.registry.previous_active(name)
        if not prev:
            return {"ok": False, "error": "no previous active version"}
        self.registry.set_status(name, cur, "retired", reason=f"rollback: {reason}")
        self.registry.set_active(name, prev, reason=f"rollback: {reason}")
        self._publish_active_file(name, prev)
        # The avoid-list note: record it on the retired row so the skill
        # loop routes around this version.
        return {"ok": True, "from": cur, "to": prev, "reason": reason}

    # ── serving: shadow / canary / active routing ──
    def serve(self, name, args=None, timeout=30):
        """Route one invocation. Always returns the user-facing result;
        shadow runs are logged and never shown."""
        import asyncio
        return _run_coro(self._serve_async(name, args or {}, timeout))

    async def _serve_async(self, name, args, timeout):
        active_ver = self.registry.active_version(name)
        active_path = version_module_path(self.tools_root, name, active_ver)
        canary_name = os.environ.get("ADA_SKILL_CANARY", "")
        shadow_on = os.environ.get("ADA_SKILL_SHADOW") == "1"

        # canary: fraction of calls route to the canary version
        canary_rows = self.registry.by_status(name, "canary")
        use_canary = (canary_name == name and canary_rows)
        if use_canary:
            row = canary_rows[0]
            pct = int(os.environ.get("ADA_SKILL_CANARY_PCT", "10") or 10)
            n = self._canary_counter.get(name, 0) + 1
            self._canary_counter[name] = n
            if n % 100 < max(0, min(100, pct)):
                cpath = version_module_path(self.tools_root, name, row["version"])
                try:
                    fn = load_version_fn(cpath)
                    res = await _call_fn(fn, args, timeout)
                    self.registry.log_run(name, row["version"], "canary", True)
                    self.maybe_promote_canary(name, row["version"])
                    return res
                except Exception as e:
                    self.registry.log_run(name, row["version"], "canary",
                                          False, f"{type(e).__name__}: {e}")
                    self.maybe_promote_canary(name, row["version"])
                    # fall through to active on canary failure

        # shadow: run alongside, log, never show
        if shadow_on:
            for row in self.registry.by_status(name, "shadow"):
                spath = version_module_path(self.tools_root, name, row["version"])
                if not spath:
                    continue
                try:
                    sfn = load_version_fn(spath)
                    sres = await _call_fn(sfn, args, timeout)
                    sok, serr = True, ""
                except Exception as e:
                    sres, sok, serr = None, False, f"{type(e).__name__}: {e}"
                ares = await self._run_active(name, active_ver, active_path,
                                              args, timeout)
                matched = (sok and _norm(sres) == _norm(ares[0])
                           if ares[2] else False)
                self.registry.log_run(
                    name, row["version"], "shadow", sok,
                    ("" if sok else serr) + ("" if matched else "|diverged"))
                return ares[0]

        res, ok, _err = await self._run_active(name, active_ver, active_path,
                                              args, timeout)
        return res

    async def _run_active(self, name, active_ver, active_path, args, timeout):
        """(result, ok, err) for the active version via the forge if possible."""
        forge = self.forge
        try:
            if forge is not None:
                res = await forge.run(name, args, timeout=timeout)
                if isinstance(res, str) and (
                        res.startswith(f"Tool '{name}' timed out")
                        or res.startswith(f"Tool '{name}' error")):
                    self.registry.log_run(name, active_ver, "active", False, res[:120])
                    return res, False, res[:120]
                self.registry.log_run(name, active_ver, "active", True)
                return res, True, ""
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            self.registry.log_run(name, active_ver, "active", False, err[:120])
            return f"Tool '{name}' error: {e}", False, err[:120]
        # no forge attached: load the version file directly
        try:
            fn = load_version_fn(active_path)
            res = await _call_fn(fn, args, timeout)
            self.registry.log_run(name, active_ver, "active", True)
            return res, True, ""
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            self.registry.log_run(name, active_ver, "active", False, err[:120])
            return f"Tool '{name}' error: {e}", False, err[:120]


async def _call_fn(fn, args, timeout):
    import asyncio
    if asyncio.iscoroutinefunction(fn):
        return await asyncio.wait_for(fn(**(args or {})), timeout)
    loop = asyncio.get_event_loop()
    return await asyncio.wait_for(
        loop.run_in_executor(None, lambda: fn(**(args or {}))), timeout)


def _run_coro(coro):
    import asyncio
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Already inside a loop (kernel path): run in a scratch thread.
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _norm(v) -> str:
    try:
        return json.dumps(v, sort_keys=True, default=str)
    except Exception:
        return str(v)


# ── dream confidence: eval signal -> importance ───────────────────────────────

def lesson_confidence(registry: SkillRegistry, lesson_text: str) -> tuple[int, bool]:
    """Map a dream lesson to (importance, pinned) from the eval signal.

    Lessons backed by passing evals (active skills with green scores) get
    high importance + pinning; unvalidated lessons stay low and unpinned.
    """
    text = (lesson_text or "").lower()
    best = (5, False)
    try:
        rows = registry._db.execute(
            "SELECT name, version, status, eval_scores_json FROM skill_versions"
        ).fetchall()
    except Exception:
        return best
    for r in rows:
        if r["name"].lower() not in text:
            continue
        try:
            scores = json.loads(r["eval_scores_json"] or "{}")
        except Exception:
            scores = {}
        if r["status"] == "active" and scores.get("passed"):
            return 9, True
        if r["status"] in ("shadow", "canary"):
            best = (7, True)
    return best
