"""HUD spec — Ada composes dashboards, never code (declarative generative UI).

She emits JSON: {v, hudId, title, root, nodes:{id: {type, props, children}}}.
The frontend renders from a FIXED registry: unknown types drop, props are
schema-checked, numbers clamped, strings capped. No eval, no innerHTML,
no URLs fetched — an untrusted model cannot XSS the desktop.

Envelope over the existing `hud` socket event:
  hud:start {hudId, title, skeleton}      first pixels instantly
  hud:chunk {hudId, seq, upsert:[...]}   flat node patches, progressive
  hud:data  {hudId, path, value}         live values (/sensors/nozzleTemp)
  hud:final {hudId, seq}                 stream complete
  hud:action (frontend→backend)          button presses, allowlisted

Limits: MAX_NODES=200, MAX_BYTES=64KB per HUD, 50 children max per node.
"""
from __future__ import annotations

import json
import time

V = 1
MAX_NODES = 200
MAX_BYTES = 64 * 1024

# The catalog — the ONLY types she may use. The system prompt carries this
# list; anything else the frontend drops with a logged fallback.
CATALOG = ("HudRoot", "Row", "Column", "Card", "Gauge", "ProgressRing",
           "Sparkline", "Timeline", "SpecCard", "SvgCanvas", "Viewport3D",
           "Button", "LogFeed")

# Actions a HUD button may trigger (frontend→backend). Nothing else runs.
# Feature 9 adds three: replay one audit decision, undo a reversible action,
# and flip a per-capability autonomy toggle. Parameters ride the action
# string as base:param[:param...] (validated backend-side); the frontend
# allow-list mirrors this set.
ACTIONS = ("pause_print", "print_progress", "dismiss_hud", "refresh_data",
           "approve_spec", "run_preflight", "show_scorecard", "show_ledger",
           "replay_decision", "undo_action", "set_autonomy_toggle")

_PROMPT_CATALOG = """HUD CATALOG (the only widgets you may use — pick type+props as JSON, never code):
HudRoot(title) > Row|Column(children ids) > Card(title, children) |
Gauge(label,value,min,max,unit,thresholds[warn,crit]) |
ProgressRing(label,frac 0-1,sub) | Sparkline(label,points[max 100],unit) |
Timeline(steps[{label,state:todo|doing|done|fail}]) |
SpecCard(rows[{k,v}],actions[{label,action}]) |
SvgCanvas(width,height,elements[{shape:circle|rect|line|path|text + numeric attrs}]) |
Viewport3D(objects[{kind:box|cylinder|sphere|line,pos,rot,size,r,color}],explode 0-1,highlight[]) |
Button(label,action,variant) | LogFeed(lines[max 30]).
Rules: flat {id:{type,props,children[]}} + root id. Numbers finite. Text ≤300 chars.
Actions only from [pause_print, print_progress, dismiss_hud, refresh_data, approve_spec, run_preflight, show_scorecard, show_ledger, replay_decision, undo_action, set_autonomy_toggle]."""


def catalog_prompt() -> str:
    return _PROMPT_CATALOG


def _clamp_num(x, lo=-1e6, hi=1e6, default=0):
    try:
        f = float(x)
        if f != f or f in (float("inf"), float("-inf")):
            return default
        return max(lo, min(hi, f))
    except Exception:
        return default


def _clean_str(s, cap=300) -> str:
    return str(s or "")[:cap]


def sanitize_nodes(nodes: dict) -> dict:
    """Validate + clamp a node map. Drops bad nodes, never raises."""
    if not isinstance(nodes, dict):
        return {}
    out = {}
    for nid, node in list(nodes.items())[:MAX_NODES]:
        if not isinstance(nid, str) or not (1 <= len(nid) <= 64):
            continue
        if not isinstance(node, dict):
            continue
        ntype = node.get("type")
        if ntype not in CATALOG:
            continue
        props = node.get("props") if isinstance(node.get("props"), dict) else {}
        clean: dict = {}
        for k, v in list(props.items())[:40]:
            if not isinstance(k, str) or len(k) > 40:
                continue
            if k.startswith("on") or k in ("__proto__", "constructor", "dangerouslySetInnerHTML"):
                continue
            if isinstance(v, bool):
                clean[k] = v
            elif isinstance(v, (int, float)):
                clean[k] = _clamp_num(v)
            elif isinstance(v, str):
                clean[k] = _clean_str(v)
            elif isinstance(v, list):
                clean[k] = _clean_list(v)
            elif isinstance(v, dict):
                clean[k] = {kk[:40]: _clean_str(vv) if isinstance(vv, str)
                            else _clamp_num(vv) if isinstance(vv, (int, float))
                            else (vv if isinstance(vv, bool) else None)
                            for kk, vv in list(v.items())[:20]
                            if isinstance(kk, str)}
            # anything else (nested code-ish) dropped
        children = node.get("children") or []
        if isinstance(children, list):
            children = [c for c in children if isinstance(c, str)][:50]
        else:
            children = []
        out[nid] = {"type": ntype, "props": clean, "children": children}
    return out


def _clean_list(v: list, depth: int = 0) -> list:
    if depth > 2:
        return []
    out = []
    for item in v[:100]:
        if isinstance(item, bool):
            out.append(item)
        elif isinstance(item, (int, float)):
            out.append(_clamp_num(item))
        elif isinstance(item, str):
            out.append(_clean_str(item))
        elif isinstance(item, list):
            out.append(_clean_list(item, depth + 1))
        elif isinstance(item, dict):
            out.append({kk[:40]: (_clean_str(vv) if isinstance(vv, str)
                                  else _clamp_num(vv) if isinstance(vv, (int, float))
                                  else None)
                        for kk, vv in list(item.items())[:20] if isinstance(kk, str)})
    return out


def coerce_spec(raw) -> dict | None:
    """Parse model output into a valid spec (one repair attempt). None = give up."""
    if isinstance(raw, dict):
        spec = raw
    elif isinstance(raw, str):
        txt = raw.strip()
        if txt.startswith("```"):
            txt = txt.split("\n", 1)[1] if "\n" in txt else ""
            if txt.rstrip().endswith("```"):
                txt = txt.rstrip()[:-3]
        try:
            spec = json.loads(txt)
        except Exception:
            # one repair: truncate to last complete brace
            try:
                spec = json.loads(txt[:txt.rfind("}") + 1])
            except Exception:
                return None
    else:
        return None
    if not isinstance(spec, dict):
        return None
    nodes = sanitize_nodes(spec.get("nodes"))
    if not nodes:
        return None
    root = spec.get("root")
    if root not in nodes:
        root = next(iter(nodes))
    return {"v": V, "hudId": _clean_str(spec.get("hudId") or f"hud-{int(time.time())}", 64),
            "title": _clean_str(spec.get("title") or "ADA HUD", 80),
            "root": root, "nodes": nodes}


def budget_ok(spec: dict) -> bool:
    try:
        return len(json.dumps(spec)) <= MAX_BYTES and len(spec.get("nodes", {})) <= MAX_NODES
    except Exception:
        return False


# ── emit helpers (backend → frontend over `hud` event) ─────────────────
async def emit_start(emit_fn, hud_id: str, title: str, skeleton: str = "Card",
                   root: str | None = None):
    await emit_fn("hud", {"t": "hud:start", "v": V, "hudId": hud_id,
                          "title": _clean_str(title, 80),
                          "root": _clean_str(root, 64) if root else None,
                          "skeleton": skeleton if skeleton in CATALOG else "Card"})


async def emit_chunk(emit_fn, hud_id: str, seq: int, upsert: dict, remove: list | None = None):
    clean = sanitize_nodes(upsert)
    if not clean and not remove:
        return
    await emit_fn("hud", {"t": "hud:chunk", "v": V, "hudId": hud_id, "seq": seq,
                          "upsert": clean, "remove": remove or []})


async def emit_data(emit_fn, hud_id: str, path: str, value) -> None:
    if isinstance(value, (int, float)):
        value = _clamp_num(value)
    elif isinstance(value, str):
        value = _clean_str(value)
    await emit_fn("hud", {"t": "hud:data", "v": V, "hudId": hud_id,
                          "path": _clean_str(path, 120), "value": value})


async def emit_final(emit_fn, hud_id: str, seq: int):
    await emit_fn("hud", {"t": "hud:final", "v": V, "hudId": hud_id, "seq": seq})


def demo_spec() -> dict:
    """No-LLM showcase: print-status dashboard proving the whole pipeline."""
    return {
        "v": V, "hudId": f"demo-{int(time.time())}", "title": "Print Status",
        "root": "root",
        "nodes": {
            "root": {"type": "HudRoot", "props": {"title": "Print Status"},
                     "children": ["row1", "timeline1", "log1"]},
            "row1": {"type": "Row", "props": {},
                     "children": ["gauge1", "ring1", "spec1"]},
            "gauge1": {"type": "Gauge",
                       "props": {"label": "Nozzle", "value": 240, "min": 0,
                                 "max": 300, "unit": "°C", "thresholds": [230, 260]},
                       "children": []},
            "ring1": {"type": "ProgressRing",
                      "props": {"label": "Print", "frac": 0.62, "sub": "layer 148/240"},
                      "children": []},
            "spec1": {"type": "SpecCard",
                      "props": {"rows": [
                          {"k": "Material", "v": "PETG"},
                          {"k": "Profile", "v": "0.2mm / 240°C"},
                          {"k": "Filament", "v": "42g"}],
                          "actions": [
                              {"label": "Pause", "action": "pause_print"},
                              {"label": "Progress", "action": "print_progress"}]},
                      "children": []},
            "timeline1": {"type": "Timeline",
                          "props": {"steps": [
                              {"label": "Intent", "state": "done"},
                              {"label": "Build", "state": "done"},
                              {"label": "Slice", "state": "done"},
                              {"label": "Pre-flight", "state": "done"},
                              {"label": "Print", "state": "doing"},
                              {"label": "Verify", "state": "todo"}]},
                          "children": []},
            "log1": {"type": "LogFeed",
                     "props": {"lines": ["Build passed verify", "Sliced in 41s",
                                         "Pre-flight CLEAR — 4% overhang"]},
                     "children": []},
        },
    }
