"""Decision replay — inspect what she knew, re-decide with today's rules.

Feature 9. CLI-safe: stdlib only, reads SQLite files directly, no server
needed. Two modes:

  inspect(audit_id)  — reconstruct the decision context: the world-model
                       snapshot + perception-timeline slice at decision time,
                       both content-addressed by the hashes stored in the
                       audit record. Returns a human-readable "what she knew
                       and considered" bundle.
  redecide(audit_id) — re-run the planner/gate logic against the saved
                       snapshot with CURRENT code and policies, and report
                       whether the decision would change. Honestly labelled
                       as a re-run, not a true replay: the world has moved on
                       and the model that decided is not re-invoked.

Also here: the trust-console props builder (decision feed, per-capability
toggles, restraint panel) and the reversible-action undo registry.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path

# Timeline slice kept per decision: the 5 minutes before the gate fired.
TIMELINE_WINDOW_S = 300
TIMELINE_SLICE_LIMIT = 200
WORLD_SNAPSHOT_LIMIT = 500


# ── content addressing ────────────────────────────────────────────────────

def content_hash(obj) -> str:
    """sha256 over canonical JSON. The audit record stores these; inspect
    re-derives them to prove the context has not been rewritten."""
    canon = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def timeline_slice(db_path, ts: float, window_s: float = TIMELINE_WINDOW_S,
                   limit: int = TIMELINE_SLICE_LIMIT) -> list[dict]:
    """Events in [ts-window, ts], oldest first. Bounded, deterministic."""
    p = Path(db_path)
    if not p.exists():
        return []
    try:
        conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT seq, ts_utc, source, type, payload_json, session_id "
            "FROM events WHERE ts_utc >= ? AND ts_utc <= ? "
            "ORDER BY ts_utc ASC, seq ASC LIMIT ?",
            (ts - window_s, ts, limit)).fetchall()
        conn.close()
    except Exception:
        return []
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["payload"] = json.loads(d.pop("payload_json") or "{}")
        except Exception:
            d["payload"] = {}
        out.append(d)
    return out


def timeline_slice_hash(events: list[dict]) -> str:
    canon = [(e.get("seq"), e.get("ts_utc"), e.get("source"), e.get("type"),
              json.dumps(e.get("payload", {}), sort_keys=True,
                         separators=(",", ":"), ensure_ascii=True, default=str))
             for e in events]
    return content_hash(canon)


def world_snapshot(db_path, ts: float,
                   limit: int = WORLD_SNAPSHOT_LIMIT) -> dict:
    """World-model facts valid at ts (F3 bi-temporal edges: valid_from <= ts
    and not yet retired). Bounded, deterministic, read-only."""
    p = Path(db_path)
    if not p.exists():
        return {"edges": [], "entities": []}
    try:
        conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        edges = conn.execute(
            "SELECT e.id, a.name AS a_name, e.relation, b.name AS b_name, "
            "e.valid_from, e.confidence "
            "FROM edges e JOIN entities a ON a.id = e.a_id "
            "JOIN entities b ON b.id = e.b_id "
            "WHERE e.valid_from <= ? AND (e.valid_to IS NULL OR e.valid_to > ?) "
            "ORDER BY e.id ASC LIMIT ?",
            (ts, ts, limit)).fetchall()
        ents = conn.execute(
            "SELECT id, kind, name, attrs_json FROM entities "
            "WHERE first_seen <= ? ORDER BY id ASC LIMIT ?",
            (ts, limit)).fetchall()
        conn.close()
    except Exception:
        return {"edges": [], "entities": []}
    return {
        "edges": [{"a": r["a_name"], "relation": r["relation"],
                   "b": r["b_name"], "valid_from": r["valid_from"],
                   "confidence": r["confidence"]} for r in edges],
        "entities": [{"name": r["name"], "kind": r["kind"],
                      "attrs": r["attrs_json"]} for r in ents],
    }


def world_snapshot_hash(snapshot: dict) -> str:
    return content_hash(snapshot)


def capture_context_hashes(timeline_db, world_db, ts: float | None = None
                           ) -> tuple[str | None, str | None]:
    """Called at gate time: hash the timeline slice and world snapshot that
    the decision is being made against. Bounded queries, never raises —
    (None, None) when the stores are unavailable."""
    ts = ts if ts is not None else time.time()
    tl_hash = wo_hash = None
    try:
        if timeline_db:
            evs = timeline_slice(timeline_db, ts)
            tl_hash = timeline_slice_hash(evs)
    except Exception:
        pass
    try:
        if world_db:
            snap = world_snapshot(world_db, ts)
            wo_hash = world_snapshot_hash(snap)
    except Exception:
        pass
    return tl_hash, wo_hash


# ── audit record access (read-only; AuditLog lives in agent_kernel) ────────

def get_record(audit_db, audit_id: int) -> dict | None:
    p = Path(audit_db)
    if not p.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM audit WHERE id=?",
                           (audit_id,)).fetchone()
        conn.close()
    except Exception:
        return None
    return dict(row) if row else None


def _jload(raw, default):
    if raw is None:
        return default
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return default


# ── inspect ───────────────────────────────────────────────────────────────

def _fmt_when(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)


def inspect(audit_id: int, audit_db, *, timeline_db=None,
            world_db=None) -> dict:
    """Reconstruct the decision context for one audit row.

    Returns the "what she knew and considered" bundle. Timeline/world
    slices are re-derived from the live stores and checked against the
    content hashes in the record: "match" (untouched), "drift" (the world
    moved on or TTL compaction trimmed the slice), or "unavailable".
    """
    rec = get_record(audit_db, audit_id)
    if rec is None:
        return {"ok": False, "audit_id": audit_id,
                "error": "no such audit record"}
    ts = rec.get("ts") or 0
    gate = _jload(rec.get("gate_json"), {}) or {}
    tool_args = _jload(rec.get("tool_args_json"), {}) or {}
    alternatives = _jload(rec.get("alternatives_json"), None)

    tl_events, tl_check = None, "unavailable"
    if timeline_db:
        tl_events = timeline_slice(timeline_db, ts)
        stored = rec.get("timeline_hash")
        if stored:
            tl_check = ("match" if timeline_slice_hash(tl_events) == stored
                        else "drift")
        elif tl_events:
            tl_check = "no-hash-stored"
    wo_snap, wo_check = None, "unavailable"
    if world_db:
        wo_snap = world_snapshot(world_db, ts)
        stored = rec.get("world_hash")
        if stored:
            wo_check = ("match" if world_snapshot_hash(wo_snap) == stored
                        else "drift")
        elif wo_snap.get("edges") or wo_snap.get("entities"):
            wo_check = "no-hash-stored"

    bundle = {
        "ok": True,
        "audit_id": audit_id,
        "ts": ts,
        "when": _fmt_when(ts),
        "actor": rec.get("actor"),
        "action": rec.get("action"),
        "tool": rec.get("target"),
        "decision": rec.get("decision"),
        "step": rec.get("step"),
        "trace_id": rec.get("trace_id"),
        "session_id": rec.get("session_id"),
        "model_lane": rec.get("model_lane"),
        "tool_version": rec.get("tool_version"),
        "prompt_hash": rec.get("prompt_hash"),
        "inputs": tool_args,
        "output_summary": rec.get("output_summary"),
        "rationale": rec.get("detail"),
        "gate": gate,
        "alternatives": alternatives,
        "undone": bool(rec.get("undone")),
        "timeline_slice": tl_events,
        "timeline_events": len(tl_events) if tl_events is not None else None,
        "world_snapshot": wo_snap,
        "world_facts": (len(wo_snap["edges"]) if wo_snap else None),
        "hash_check": {"timeline": tl_check, "world": wo_check},
    }
    bundle["narrative"] = _narrative(bundle)
    return bundle


def _narrative(b: dict) -> str:
    """One plain-English paragraph: what she knew, what she considered."""
    tool = b.get("tool") or "an action"
    decision = b.get("decision") or "?"
    gate = b.get("gate") or {}
    risk = gate.get("risk", "?")
    level = gate.get("level", "?")
    cap = gate.get("capability", "other")
    mode = gate.get("capability_mode", "ladder")
    parts = [f"At {b.get('when')}, Ada {decision} '{tool}' "
             f"(risk: {risk}, autonomy level {level}, capability '{cap}' "
             f"on '{mode}')."]
    parts.append(f"Her stated reason: {b.get('rationale') or '—'}.")
    args = b.get("inputs") or {}
    if args:
        shown = ", ".join(f"{k}={str(v)[:60]}" for k, v in list(args.items())[:6])
        parts.append(f"Inputs: {shown}.")
    alt = b.get("alternatives")
    if alt:
        parts.append(f"Simulation weighed {len(alt)} alternative(s) first.")
    else:
        parts.append("No simulation alternatives were recorded for this call.")
    n_ev = b.get("timeline_events")
    if n_ev is not None:
        kinds = {}
        for e in (b.get("timeline_slice") or [])[:50]:
            kinds[e.get("type", "?")] = kinds.get(e.get("type", "?"), 0) + 1
        top = ", ".join(f"{k} x{v}" for k, v in
                        sorted(kinds.items(), key=lambda kv: -kv[1])[:4])
        parts.append(f"In the 5 minutes before, she had seen {n_ev} timeline "
                     f"event(s){(': ' + top) if top else ''} "
                     f"[timeline hash: {b['hash_check']['timeline']}].")
    n_f = b.get("world_facts")
    if n_f is not None:
        parts.append(f"Her world model held {n_f} live fact(s) at that moment "
                     f"[world hash: {b['hash_check']['world']}].")
    if b.get("output_summary"):
        parts.append(f"Outcome: {str(b['output_summary'])[:200]}")
    if b.get("undone"):
        parts.append("This action was later undone.")
    return " ".join(parts)


# ── re-decide ─────────────────────────────────────────────────────────────

def redecide(audit_id: int, audit_db, *, autonomy=None) -> dict:
    """Re-run the gate logic against the saved inputs with CURRENT code and
    policies. Returns whether the decision would change — labelled honestly
    as a re-run, not a true replay (the world moved on; the deciding model
    is not re-invoked)."""
    from autonomy import evaluate_gate, capability_of  # stdlib-only module
    rec = get_record(audit_db, audit_id)
    if rec is None:
        return {"ok": False, "audit_id": audit_id,
                "error": "no such audit record"}
    tool = rec.get("target") or ""
    gate = _jload(rec.get("gate_json"), {}) or {}
    if not gate:
        return {"ok": False, "audit_id": audit_id,
                "error": "record predates full-fidelity gate context; "
                         "cannot re-decide"}
    orig_verdict = gate.get("verdict") or (
        "voice-ask" if gate.get("needs_voice") else "auto-approved")
    orig_risk = gate.get("risk")

    if autonomy is None:
        from autonomy import AutonomyLevels
        autonomy = AutonomyLevels("/tmp/ada-replay-autonomy")
    level = autonomy.effective_level(tool)
    armed = bool(autonomy.armed) if hasattr(autonomy, "armed") else False
    now = evaluate_gate(tool, level=level, armed=armed,
                        exempt=gate.get("exempt", False))
    # The capability toggle is part of "current policy" too.
    now["capability"] = capability_of(tool)
    now["capability_mode"] = (autonomy.capability_mode(now["capability"])
                              if hasattr(autonomy, "capability_mode") else "ladder")
    now["level"] = level
    new_verdict = now["verdict"]
    changed = (new_verdict != orig_verdict) or (now["risk"] != orig_risk)
    if changed:
        if new_verdict != orig_verdict:
            why = (f"friction changed: was '{orig_verdict}', "
                   f"now '{new_verdict}' (level {gate.get('level')} -> {level}, "
                   f"capability '{now['capability']}' on "
                   f"'{now['capability_mode']}')")
        else:
            why = (f"risk class changed: was '{orig_risk}', "
                   f"now '{now['risk']}' (tool-risk mapping updated)")
    else:
        why = (f"same call today: '{new_verdict}' at effective level {level} "
               f"(risk {now['risk']})")
    return {
        "ok": True,
        "audit_id": audit_id,
        "tool": tool,
        "mode": "re-run — current code/policies against saved inputs, "
                "not a true replay",
        "original": {"verdict": orig_verdict, "risk": orig_risk,
                     "decision": rec.get("decision"),
                     "level": gate.get("level")},
        "now": now,
        "changed": changed,
        "why": why,
    }


# ── restraint (Feature 8 policy.decision events) ───────────────────────────

def restraint_summary(timeline_db, hours: float = 24) -> dict:
    """How much she held back: policy.decision send/suppress counts from the
    perception timeline. This is the trust console's RESTRAINT panel."""
    p = Path(timeline_db) if timeline_db else None
    if not p or not p.exists():
        return {"sent": 0, "suppressed": 0, "hours": hours,
                "unavailable": True}
    cutoff = time.time() - hours * 3600
    try:
        conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT payload_json FROM events WHERE type='policy.decision' "
            "AND ts_utc >= ?", (cutoff,)).fetchall()
        conn.close()
    except Exception:
        return {"sent": 0, "suppressed": 0, "hours": hours,
                "unavailable": True}
    sent = suppressed = 0
    reasons: dict[str, int] = {}
    for r in rows:
        try:
            pl = json.loads(r["payload_json"] or "{}")
        except Exception:
            continue
        if pl.get("action") == "suppressed":
            suppressed += 1
            rsn = str(pl.get("reason") or "unspecified")[:80]
            reasons[rsn] = reasons.get(rsn, 0) + 1
        else:
            sent += 1
    top_reasons = sorted(reasons.items(), key=lambda kv: -kv[1])[:5]
    return {"sent": sent, "suppressed": suppressed, "hours": hours,
            "unavailable": False,
            "top_suppress_reasons": [{"reason": k, "count": v}
                                     for k, v in top_reasons]}


# ── trust-console props ───────────────────────────────────────────────────

def console_snapshot(audit_db, *, timeline_db=None, autonomy=None,
                     voiceprint_db=None,
                     limit: int = 30, hours: float = 24) -> dict:
    """Everything the trust_console widget renders: decision feed (newest
    first), per-capability toggles, restraint panel, voiceprint retention,
    and a day summary."""
    from autonomy import capability_of
    p = Path(audit_db)
    decisions: list[dict] = []
    summary = {"decisions": 0, "confirmations_asked": 0, "undone": 0}
    if p.exists():
        try:
            conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            cutoff = time.time() - hours * 3600
            rows = conn.execute(
                "SELECT id, ts, actor, action, target, decision, detail, "
                "gate_json, undone FROM audit WHERE ts >= ? "
                "ORDER BY ts DESC LIMIT ?", (cutoff, limit)).fetchall()
            conn.close()
        except Exception:
            rows = []
        for r in rows:
            d = dict(r)
            gate = _jload(d.get("gate_json"), {}) or {}
            tool = d.get("target") or ""
            risk = gate.get("risk", "")
            reversible = risk in ("read-only", "state-changing-reversible")
            decisions.append({
                "id": d["id"], "ts": d["ts"], "when": _fmt_when(d["ts"]),
                "tool": tool, "capability": capability_of(tool),
                "decision": d.get("decision"),
                "verdict": gate.get("verdict", ""),
                "risk": risk, "reason": (d.get("detail") or "")[:160],
                "reversible": reversible, "undone": bool(d.get("undone")),
            })
        summary["decisions"] = len(decisions)
        summary["confirmations_asked"] = sum(
            1 for x in decisions if x["verdict"] == "voice-ask")
        summary["undone"] = sum(1 for x in decisions if x["undone"])
    toggles = []
    if autonomy is not None and hasattr(autonomy, "capability_modes"):
        try:
            modes = autonomy.capability_modes()
            toggles = [{"capability": cap, "mode": modes.get(cap, "ladder")}
                       for cap in sorted(modes)]
        except Exception:
            pass
    restraint = restraint_summary(timeline_db, hours=hours)
    summary["suppressed"] = restraint.get("suppressed", 0)
    # Feature 7: voiceprint retention — centroids only, local SQLite,
    # per-person delete by voice. Minimal additive panel.
    voiceprints = {"people": [], "retention": (
        "Voice fingerprints are MFCC centroids only \u2014 never raw audio. "
        "Stored locally in SQLite (gitignored, never committed). "
        "Enrollment only after the spoken ask is answered. "
        "Say \u2018forget my voice\u2019 any time to delete your voiceprint; "
        "voice ID is never used for authentication.")}
    if voiceprint_db is not None:
        try:
            p = Path(voiceprint_db)
            if p.exists():
                from speaker_id import VoiceprintStore
                for v in VoiceprintStore(str(p)).list_known_voices():
                    voiceprints["people"].append({
                        "name": v["name"], "role": v["role"],
                        "enrolled": _fmt_when(v["enrolled_ts"]),
                    })
        except Exception:
            pass
    return {
        "title": "Trust Console",
        "summary": summary,
        "decisions": decisions,
        "toggles": toggles,
        "restraint": restraint,
        "voiceprints": voiceprints,
        "note": ("Decisions are append-only and never sampled. "
                 "Replay one to see exactly what she knew."),
    }


# ── undo ──────────────────────────────────────────────────────────────────

_UNDO_ID_RE = re.compile(r"\(([0-9a-f]{12})\)")


def default_undo_handlers(kernel):
    """Real inverses for the reversible tools the console can undo.

    Only tools with a true inverse are listed: a scheduler entry can be
    cancelled by id, a remembered fact can be forgotten. Everything else
    reports 'not reversible' rather than faking it.
    """
    def _cancel_scheduled(rec):
        out = rec.get("output_summary") or ""
        m = _UNDO_ID_RE.search(out)
        if not m:
            return {"ok": False,
                    "reason": "no scheduler id found in the record's output"}
        tid = m.group(1)
        return {"ok": True,
                "result": kernel.scheduler.cancel(tid)}

    def _forget_fact(rec):
        args = _jload(rec.get("tool_args_json"), {}) or {}
        key = args.get("key")
        if not key:
            return {"ok": False,
                    "reason": "no memory key in the record's inputs"}
        return {"ok": True,
                "result": kernel.memory.forget_fact(str(key)),
                "note": "the fact is removed; a previous value it overwrote "
                        "is not restored"}

    return {
        "create_reminder": _cancel_scheduled,
        "schedule_recurring": _cancel_scheduled,
        "remember": _forget_fact,
    }


def undo_record(audit_id: int, audit_db, handlers: dict) -> dict:
    """One-click undo for a reversible tool record. The undo itself is
    audit-logged (actor 'ada', action 'undo')."""
    rec = get_record(audit_db, audit_id)
    if rec is None:
        return {"ok": False, "audit_id": audit_id,
                "reason": "no such audit record"}
    if rec.get("undone"):
        return {"ok": False, "audit_id": audit_id,
                "reason": "already undone"}
    tool = rec.get("target") or ""
    handler = (handlers or {}).get(tool)
    if handler is None:
        return {"ok": False, "audit_id": audit_id, "tool": tool,
                "reason": f"'{tool}' has no reversible inverse — "
                          "nothing was changed"}
    try:
        res = handler(rec) or {}
    except Exception as e:
        return {"ok": False, "audit_id": audit_id, "tool": tool,
                "reason": f"undo failed: {type(e).__name__}: {e}"}
    if not res.get("ok"):
        return {"ok": False, "audit_id": audit_id, "tool": tool,
                "reason": res.get("reason", "undo refused")}
    # Mark undone + log the undo as its own audit row.
    try:
        conn = sqlite3.connect(str(audit_db))
        with conn:
            conn.execute("UPDATE audit SET undone=1 WHERE id=?", (audit_id,))
            conn.execute(
                "INSERT INTO audit(ts,actor,action,target,decision,detail,"
                "trace_id) VALUES(?,?,?,?,?,?,?)",
                (time.time(), "ada", "undo", tool, "undone",
                 f"undid audit row {audit_id}: {res.get('result', '')}"[:4000],
                 rec.get("trace_id")))
        conn.close()
    except Exception:
        pass
    return {"ok": True, "audit_id": audit_id, "tool": tool,
            "result": res.get("result"), "note": res.get("note", "")}
