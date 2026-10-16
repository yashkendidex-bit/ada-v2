"""Feature 5 — Simulation & Counterfactual Engine (RAP-lite).

Before a HIGH-blast-radius tool runs, we simulate instead of guessing:

  1. snapshot()   — capture the world slice the action touches (world-model
     entity context, routine deviation check, files/calendar in scope),
     content-addressed with sha256 so predictions and outcomes join later.
  2. dry_run()    — per-domain sandboxes: file writes -> copy-on-write temp
     dir + unified diff (real files NEVER touched); purchases -> budget
     check; schedule changes -> conflict check vs calendar + routine model;
     code/CAD -> syntax/lint in sandbox.
  3. rollout()    — for the top 2-3 candidate actions, the provider router
     runs a "world simulator" prompt (direct effects, side effects, failure
     modes + likelihood, likely user reaction); each is scored on
     reversibility x impact x preference-fit; the winner gets one short
     council.py skeptic critique (RAP-lite: MCTS depth 1-2, branch 2-3).
  4. log_counterfactual() — (snapshot_hash, action, predicted, actual) into
     the F4 timeline as simulation.prediction / simulation.outcome events;
     calibration_report() compares recent predictions vs outcomes so the
     simulation loop improves WITHOUT training (env kill-switch ADA_SIMULATE=0).

Hard constraints:
  - Zero local ML model downloads. Simulation LLM calls go through
    backend/providers.py (cheap CAD_CHAIN), never a new client.
  - Only high-blast-radius actions get simulated (F6's classify_blast_radius);
    low-blast-radius tools skip with zero added latency.
  - Prediction logs are runtime state; tests use tmp dirs; no network in tests.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path

# F6 blast-radius classifier: "high" = new external side-effect classes.
from skill_eval import classify_blast_radius

# Council perspectives for the post-ranking skeptic critique.
from council import PERSPECTIVES

_SKEPTIC_SYS = next((s for n, s in PERSPECTIVES if n == "skeptic"),
                    "Attack feasibility: what breaks, which edge cases kill it.")

SIMULATOR_SYS = (
    "You are Ada's world simulator. Given a structured world STATE and a "
    "candidate ACTION, predict what will happen. Be concrete and brief — "
    "this is read before a voice confirmation, not a research paper.\n\n"
    "Reply in EXACTLY this format (each field on its own line):\n"
    "DIRECT EFFECTS: <what the action will do, 1-2 sentences>\n"
    "SIDE EFFECTS: <unintended consequences, 1-2 sentences>\n"
    "FAILURE MODES: <what could go wrong, each with a likelihood %>\n"
    "USER REACTION: <how the user will likely react: pleased/neutral/annoyed>\n"
    "REVERSIBILITY: <1-10, 10 = trivially undoable>\n"
    "IMPACT: <1-10, 10 = severe if it goes wrong>\n"
    "PREFERENCE FIT: <1-10, 10 = exactly what this user would want>\n"
    "PREDICTED OUTCOME: <one of: smooth, minor-issue, major-issue, user-rejects>\n"
    "PREDICTION: <one line: what will actually happen>"
)

# ── cheap chain, read at call time (mirrors providers.cad_chain override) ──

def _sim_chain() -> list:
    try:
        from providers import cad_chain
        return cad_chain()
    except Exception:
        return [("nvidia", "z-ai/glm-5.3-flash")]


async def _default_chat(messages, **kw):
    from providers import chat_with_fallback
    return await chat_with_fallback(
        _sim_chain(), messages,
        temperature=0.3, max_tokens=600, timeout=60.0, **kw)


def sim_enabled() -> bool:
    """Env kill-switch: ADA_SIMULATE=0 disables all simulation, sync early-out."""
    return os.environ.get("ADA_SIMULATE", "1").strip().lower() not in (
        "0", "false", "no", "off")


def is_high_blast_radius(tool_name: str, args: dict | None = None) -> bool:
    """High-blast-radius = F6's classifier says so, OR the autonomy ladder
    already treats the tool as destructive/external (same vocabulary as the
    F0 voice gate — keeps the hook firing on tool names F6's skill-oriented
    token list was never built for, e.g. write_file). Low-blast-radius
    tools skip simulation with zero added latency."""
    args = dict(args) if args else {}
    purpose = " ".join(str(k) for k in args.keys())
    try:
        if classify_blast_radius(tool_name or "", purpose, "") == "high":
            return True
    except Exception:
        pass
    try:
        from autonomy import risk_class, RISK_DESTRUCTIVE, RISK_EXTERNAL
        return risk_class(tool_name) in (RISK_DESTRUCTIVE, RISK_EXTERNAL)
    except Exception:
        return False


# ── 1. world snapshot ────────────────────────────────────────────────────

def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)


def _file_manifest(paths: list[str]) -> list[dict]:
    manifest = []
    for p in paths or []:
        try:
            fp = Path(p)
            if not fp.exists() or fp.is_dir():
                manifest.append({"path": p, "exists": False})
                continue
            st = fp.stat()
            entry = {"path": p, "exists": True, "size": st.st_size,
                     "mtime": round(st.st_mtime, 1)}
            if st.st_size < 1_048_576:
                h = hashlib.sha256()
                with open(fp, "rb") as f:
                    h.update(f.read())
                entry["sha256"] = h.hexdigest()[:16]
            manifest.append(entry)
        except Exception:
            manifest.append({"path": p, "exists": "unknown"})
    return manifest


def snapshot(action_context: dict | None = None) -> dict:
    """Capture the world slice an action touches. Content-addressed.

    action_context keys (all optional): description (str), entities
    (list[str]), tool, args, files (list[str]), calendar (list[dict]),
    world (WorldModel instance or None).
    """
    ctx = dict(action_context or {})
    world = ctx.get("world")
    entities = [str(e) for e in (ctx.get("entities") or []) if str(e).strip()]
    description = str(ctx.get("description") or "")

    entity_context = {}
    for name in entities[:8]:
        try:
            if world is not None and hasattr(world, "get_entity_context"):
                txt = world.get_entity_context(name)
                if txt:
                    entity_context[name] = txt[:2000]
        except Exception:
            pass

    routine_note = ""
    try:
        if world is not None and hasattr(world, "check") and description:
            routine_note = str(world.check(description))[:800]
    except Exception:
        pass

    files = [str(p) for p in (ctx.get("files") or [])]
    # Also harvest obvious file paths from tool args (best effort).
    for key in ("path", "file", "filepath", "filename", "target"):
        v = (ctx.get("args") or {}).get(key)
        if isinstance(v, str) and v not in files:
            files.append(v)

    snap = {
        "ts": round(time.time(), 1),
        "tool": ctx.get("tool") or "",
        "description": description[:500],
        "entities": entity_context,
        "routine_note": routine_note,
        "file_manifest": _file_manifest(files[:12]),
        "calendar_in_scope": (ctx.get("calendar") or [])[:12],
    }
    snap["snapshot_hash"] = hashlib.sha256(
        _canonical(snap).encode("utf-8")).hexdigest()
    return snap


# ── 2. per-domain dry-run sandboxes ──────────────────────────────────────

_FILE_TOOLS = {"write_file", "edit_file", "save_file", "create_file",
               "append_file", "update_file", "overwrite_file"}
_DELETE_TOOLS = {"delete_file", "remove_file", "delete", "remove",
                 "forget_file", "trash_file"}
_MONEY_TOKENS = {"buy", "buys", "purchase", "purchases", "pay", "pays",
                 "payment", "payments", "charge", "charges", "order",
                 "orders", "subscribe", "sell", "sells", "deposit",
                 "withdraw", "transfer", "invoice"}
_MONEY_ARG_KEYS = {"price", "amount", "cost", "total", "charge", "fee",
                   "budget", "value_usd", "value_inr"}
_SCHED_TOKENS = {"schedule", "schedules", "event", "events", "meeting",
                 "meetings", "reminder", "reminders", "book", "booking",
                 "calendar", "reschedule", "postpone"}
_SCHED_ARG_KEYS = {"datetime", "date", "time", "start", "when", "start_time",
                   "scheduled_at", "at"}
_CODE_TOKENS = {"run_code", "execute_code", "eval_code", "code_exec",
                "generate_gcode", "cad_generate", "gcode"}
_CODE_ARG_KEYS = {"code", "python", "script", "gcode"}


def _classify_domain(tool_name: str, args: dict) -> str:
    toks = set(re.findall(r"[a-z]+", (tool_name or "").lower()))
    keys = {str(k).lower() for k in args.keys()}
    if (tool_name or "") in _FILE_TOOLS or (
            keys & {"path", "file"}
            and keys & {"content", "text", "data", "new_content"}):
        return "file"
    if (tool_name or "") in _DELETE_TOOLS or toks & {"delete", "deletes", "remove", "removes"}:
        return "delete"
    if toks & _MONEY_TOKENS or keys & _MONEY_ARG_KEYS:
        return "money"
    if toks & _SCHED_TOKENS or keys & _SCHED_ARG_KEYS:
        return "schedule"
    if toks & _CODE_TOKENS or keys & _CODE_ARG_KEYS:
        return "code"
    return "unknown"


def _dry_run_delete(tool_name: str, args: dict) -> tuple[bool, str]:
    """What would be deleted — reported, never removed."""
    path = args.get("path") or args.get("file") or ""
    fp = Path(path) if path else None
    if fp is not None and fp.exists() and not fp.is_dir():
        try:
            st = fp.stat()
            head = fp.read_text(encoding="utf-8", errors="replace").splitlines()[:3]
            preview = " / ".join(l[:60] for l in head)
            return False, (f"dry-run: would DELETE {path} "
                           f"({st.st_size} bytes, modified "
                           f"{time.strftime('%Y-%m-%d', time.localtime(st.st_mtime))}). "
                           f"First lines: {preview}. Nothing removed — flagged for voice confirm.")
        except Exception:
            return False, (f"dry-run: would DELETE {path}. "
                           "Nothing removed — flagged for voice confirm.")
    return False, (f"dry-run: would DELETE {path or '(no path given)'} "
                   "(not found on disk). Nothing removed — flagged for voice confirm.")


def _dry_run_file(tool_name: str, args: dict) -> tuple[bool, str]:
    """Copy-on-write: render the write in a temp dir, diff, never touch disk."""
    path = args.get("path") or args.get("file") or ""
    content = args.get("content") or args.get("text") or args.get("data") or args.get("new_content") or ""
    old = ""
    if path and Path(path).exists() and not Path(path).is_dir():
        try:
            old = Path(path).read_text(encoding="utf-8", errors="replace")
        except Exception:
            old = ""
    tmpdir = tempfile.mkdtemp(prefix="ada-sim-")
    copy_name = (Path(path).name if path else "new_file.txt") or "new_file.txt"
    tmp_path = Path(tmpdir) / copy_name
    try:
        tmp_path.write_text(old, encoding="utf-8")
        tmp_path.write_text(str(content), encoding="utf-8")
    except Exception:
        tmp_path.write_text("", encoding="utf-8")
    diff = "".join(difflib.unified_diff(
        old.splitlines(keepends=True), str(content).splitlines(keepends=True),
        fromfile=f"a/{path or 'new-file'}", tofile=f"b/{path or 'new-file'}"))
    if not old and path:
        report = (f"dry-run: would CREATE {path} "
                  f"({len(str(content).splitlines())} lines). "
                  f"Real file untouched — copy rendered in {tmpdir}.")
    else:
        adds = sum(1 for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++"))
        dels = sum(1 for l in diff.splitlines() if l.startswith("-") and not l.startswith("---"))
        report = (f"dry-run: would MODIFY {path}: +{adds}/-{dels} lines. "
                  f"Real file untouched — copy rendered in {tmpdir}.")
        if diff.strip():
            report += "\n" + "\n".join(diff.splitlines()[:25])
    return True, report


def _parse_amount(args: dict) -> float | None:
    for k in ("price", "amount", "total", "cost", "charge", "fee",
              "value_usd", "value_inr"):
        v = args.get(k)
        if v is None:
            continue
        try:
            return float(re.sub(r"[^\d.]", "", str(v).split()[0]) or "nan")
        except Exception:
            continue
    return None


def _dry_run_money(tool_name: str, args: dict, ctx: dict) -> tuple[bool, str]:
    """Read-only budget check: dry-run price + budget vs spending history."""
    amount = _parse_amount(args)
    if amount is None or amount != amount:  # NaN guard
        return True, "dry-run: no parseable amount; nothing to budget-check."
    budget = ctx.get("budget")
    if budget is None:
        raw = os.environ.get("ADA_MONTHLY_BUDGET", "").strip()
        try:
            budget = float(raw) if raw else None
        except Exception:
            budget = None
    history = [float(x) for x in (ctx.get("spending_history") or []) if _is_num(x)]
    report = f"dry-run: purchase of {amount:g} — "
    if budget is not None and amount > budget:
        return False, (report +
                       f"OVER BUDGET: exceeds monthly budget of {budget:g}. "
                       "Blocked by simulation; ask before spending this.")
    if history:
        avg = sum(history) / len(history)
        if amount > 3 * max(avg, 1e-9):
            return False, (report +
                           f"unusual: {amount / max(avg, 1e-9):.1f}x your "
                           f"recent average spend of {avg:.1f}. Flag for review.")
        if budget is None and avg and amount > 0.5 * (budget or float("inf")):
            pass
    ok_note = "within budget and spending norms." if history else "no budget set; no norms to compare."
    return True, report + ok_note


def _is_num(x) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def _parse_when(args: dict) -> str:
    for k in ("datetime", "start", "when", "start_time", "scheduled_at", "at"):
        v = args.get(k)
        if v:
            return str(v)
    d = args.get("date") or ""
    t = args.get("time") or ""
    return f"{d} {t}".strip()


def _dry_run_schedule(tool_name: str, args: dict, ctx: dict) -> tuple[bool, str]:
    """Conflict check vs calendar in scope + routine-model focus-block note."""
    when = _parse_when(args)
    events = ctx.get("calendar") or []
    report = f"dry-run: schedule check for '{when or '(no time given)'}' — "
    if when:
        hits = [e for e in events
                if isinstance(e, dict) and when and str(e.get("start", "") or e.get("when", ""))[:16] == when[:16]]
        if hits:
            titles = ", ".join(str(e.get("title", "?")) for e in hits[:3])
            return False, report + f"CONFLICT with: {titles}. Resolve before booking."
    world = ctx.get("world")
    routine_note = ""
    try:
        if world is not None and hasattr(world, "check") and when:
            routine_note = str(world.check(f"schedule a meeting at {when}"))[:300]
    except Exception:
        pass
    if routine_note:
        report += f"no conflicts found. Routine note: {routine_note}"
    else:
        report += "no conflicts found."
    return True, report


def _dry_run_code(tool_name: str, args: dict) -> tuple[bool, str]:
    """Syntax/lint in sandbox: python compile(), gcode sanity, nothing executed."""
    for key in ("code", "python", "script"):
        src = args.get(key)
        if isinstance(src, str) and src.strip():
            try:
                compile(src, "<sim-sandbox>", "exec")
                return True, f"dry-run: code compiles clean (sandbox, nothing executed)."
            except SyntaxError as e:
                return False, (f"dry-run: code has a syntax error — "
                               f"line {e.lineno}: {e.msg}. Not run.")
    gcode = args.get("gcode")
    if isinstance(gcode, str) and gcode.strip():
        bad = [ln for ln in gcode.splitlines()
               if ln.strip() and not re.match(
                   r"^\s*(;.*|[GM]\d+|X|Y|Z|E|F|S|T\d|\(.*\)|N\d)", ln, re.I)]
        if bad:
            return False, (f"dry-run: gcode has {len(bad)} unrecognized lines "
                           f"(e.g. '{bad[0].strip()[:60]}'). Not sent to printer.")
        return True, "dry-run: gcode sanity check passed (sandbox, not sent)."
    return True, "dry-run: no code payload found; nothing to lint."


def dry_run(tool_name: str, args: dict | None = None,
            ctx: dict | None = None) -> tuple[bool, str]:
    """Run the per-domain sandbox. Never touches real files or spends money.

    Returns (safe, diff_report). `ctx` may carry budget, spending_history,
    calendar, world — all read-only.
    """
    args = dict(args) if args else {}
    ctx = dict(ctx) if ctx else {}
    domain = _classify_domain(tool_name or "", args)
    try:
        if domain == "file":
            return _dry_run_file(tool_name, args)
        if domain == "delete":
            return _dry_run_delete(tool_name, args)
        if domain == "money":
            return _dry_run_money(tool_name, args, ctx)
        if domain == "schedule":
            return _dry_run_schedule(tool_name, args, ctx)
        if domain == "code":
            return _dry_run_code(tool_name, args)
    except Exception as e:
        return True, f"dry-run: sandbox error ({type(e).__name__}) — proceeding on judgment."
    return True, "dry-run: no domain sandbox for this tool; no simulated side effects."


# ── 3. rollout: LLM-as-world-model over top 2-3 candidates ──────────────

def _score_prediction(text: str) -> dict:
    """Parse the simulator's structured output; forgiving, never throws."""
    def num(field):
        m = re.search(rf"{field}\s*:\s*(\d+(?:\.\d+)?)", text, re.I)
        try:
            return max(1.0, min(10.0, float(m.group(1)))) if m else 5.0
        except Exception:
            return 5.0

    def field(name):
        m = re.search(rf"{name}\s*:\s*(.+?)(?:\n[A-Z ]+:\s*|\Z)", text,
                      re.I | re.S)
        return (m.group(1).strip()[:400] if m else "")

    rev, imp, fit = num("REVERSIBILITY"), num("IMPACT"), num("PREFERENCE FIT")
    outcome = field("PREDICTED OUTCOME").lower() or "smooth"
    outcome = next((o for o in ("smooth", "minor-issue", "major-issue",
                                "user-rejects") if o in outcome), "smooth")
    # reversibility x (inverse impact) x preference fit, all 0..1
    score = (rev / 10.0) * ((10.0 - imp) / 10.0) * (fit / 10.0)
    return {
        "direct": field("DIRECT EFFECTS"),
        "side_effects": field("SIDE EFFECTS"),
        "failure_modes": field("FAILURE MODES"),
        "user_reaction": field("USER REACTION"),
        "reversibility": rev,
        "impact": imp,
        "preference_fit": fit,
        "score": round(score, 3),
        "predicted_outcome": outcome,
        "prediction": field("PREDICTION") or text.strip()[:300],
        "raw": text.strip()[:1200],
    }


def _sim_prompt(snapshot: dict, candidate: dict) -> list:
    state = _canonical({
        "entities": snapshot.get("entities", {}),
        "routine_note": snapshot.get("routine_note", ""),
        "files": [f.get("path") for f in snapshot.get("file_manifest", [])],
        "calendar": snapshot.get("calendar_in_scope", []),
    })
    user = (f"WORLD STATE (structured, selective):\n{state}\n\n"
            f"CANDIDATE ACTION:\n{json.dumps(candidate, ensure_ascii=False)[:1500]}\n\n"
            "Predict what will happen. Use the exact format specified.")
    return [{"role": "system", "content": SIMULATOR_SYS},
            {"role": "user", "content": user}]


async def rollout(candidates: list[dict], snapshot: dict | None = None,
                  chat_fn=None, max_candidates: int = 3) -> dict:
    """Roll out the top 2-3 candidate actions through the world simulator.

    chat_fn: async (messages) -> str. Defaults to the provider router over
    the cheap chain. Returns {ranked, winner, skeptic_critique, calls}.
    """
    cands = list(candidates or [])[:max(1, min(3, max_candidates))]
    snap = snapshot or {}
    chat = chat_fn or _default_chat
    calls = 0
    results = []
    for cand in cands:
        desc = (cand or {}).get("description") or str(cand)
        try:
            text = await chat(_sim_prompt(snap, cand or {}))
            calls += 1
        except Exception as e:
            text = ("FAILURE MODES: simulation call failed "
                    f"({type(e).__name__}).\nPREDICTED OUTCOME: smooth\n"
                    "PREDICTION: no simulation available.")
        parsed = _score_prediction(text)
        results.append({"candidate": cand, "description": desc,
                        "simulation": parsed})
    results.sort(key=lambda r: r["simulation"]["score"], reverse=True)
    winner = results[0] if results else None

    critique = ""
    if winner is not None:
        try:
            q = (f"Should Ada take this action? {winner['description']}\n"
                 f"EVIDENCE (world-sim prediction): "
                 f"{winner['simulation']['raw'][:1500]}")
            msg = [{"role": "system", "content": _SKEPTIC_SYS},
                   {"role": "user",
                    "content": f"QUESTION:\n{q}\n\nGive your top 2 objections, "
                               "short. Attack feasibility only."}]
            out = await chat(msg)
            calls += 1
            critique = (out or "").strip()[:600]
        except Exception:
            critique = ""
    return {"ranked": results, "winner": winner,
            "skeptic_critique": critique, "calls": calls}


# ── 4. counterfactual logging + calibration ─────────────────────────────

def log_counterfactual(snapshot_hash: str, action: str,
                       predicted: dict | str, actual: dict | str) -> None:
    """Append prediction + outcome to the F4 timeline. Writers never break."""
    pred = dict(predicted) if isinstance(predicted, dict) else {"outcome": str(predicted)}
    act = dict(actual) if isinstance(actual, dict) else {"outcome": str(actual)}
    pred_out = str(pred.get("predicted_outcome") or pred.get("outcome") or "")
    act_out = str(act.get("predicted_outcome") or act.get("outcome") or "")
    try:
        from perception import emit
        emit("simulation", "simulation.prediction",
             {"snapshot_hash": snapshot_hash, "action": str(action)[:300],
              "predicted": pred, "ts": time.time()})
        emit("simulation", "simulation.outcome",
             {"snapshot_hash": snapshot_hash, "action": str(action)[:300],
              "actual": act,
              "match": bool(pred_out and act_out and pred_out == act_out),
              "ts": time.time()})
    except Exception:
        pass


def calibration_report(n: int = 50) -> dict:
    """Accuracy of recent predictions vs outcomes — the no-training loop.

    Joins simulation.outcome events (each carries `match`) over the last
    `n` events. Returns {events, matched, accuracy, last_10_accuracy}.
    """
    try:
        from perception import get_timeline
        tl = get_timeline()
    except Exception:
        tl = None
    if tl is None:
        return {"events": 0, "matched": 0, "accuracy": None,
                "note": "timeline not initialized"}
    try:
        rows = tl.query(ts_from=time.time() - 90 * 86400, source="simulation",
                        type="simulation.outcome", limit=n)
    except Exception:
        rows = []
    matches, total = [], 0
    for r in rows:
        try:
            payload = r.get("payload_json") if isinstance(r, dict) else "{}"
            p = json.loads(payload) if isinstance(payload, str) else dict(payload)
            matches.append(1 if p.get("match") else 0)
            total += 1
        except Exception:
            continue
    # query() returns oldest-first; calibration wants the most recent window.
    matches = list(reversed(matches))[:n]
    total = len(matches)
    if not total:
        return {"events": 0, "matched": 0, "accuracy": None,
                "note": "no outcome events yet"}
    matched = sum(matches)
    last10 = matches[:10]
    return {"events": total, "matched": matched,
            "accuracy": round(matched / total, 3),
            "last_10_accuracy": round(sum(last10) / len(last10), 3),
            "note": "match = predicted outcome category == actual outcome category"}
