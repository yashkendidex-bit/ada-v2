"""Widgets — what A.D.A can put on screen, inside and around the orb.

Three tiers, safest first. None of them lets her run code in the app itself.

  1. PRESETS    A fixed set of polished widgets (clock, gauge, chart, list ...).
                She picks a kind and fills in props. Data only.
  2. SCENES     She composes a custom visual from drawing primitives (rings,
                arcs, lines, text, counters, grids, particles, bars) with a few
                animation rules. Still data only; the screen draws it.
  3. CODE       She writes a small HTML/canvas program for something nothing
                above covers. It runs in a sandboxed frame on the screen: no
                network, no access to the app, killed if it stalls. The checks
                here are a first filter; the sandbox is the real wall.

Everything goes to the screen as one `widget` socket event:

  widget:open    {t, id, kind, props, slot, color, title, ttl}
  widget:update  {t, id, props}            shallow merge into the live props
  widget:close   {t, id}                   id "all" clears the screen
  widget:data    {t, id, path, value}      one live value, props path "a.b.0"

The screen validates again; nothing it receives is ever evaluated as code
except tier 3, inside the sandbox.
"""
from __future__ import annotations

import json
import math
import re
import time

V = 1

# ── limits ────────────────────────────────────────────────────────────────
MAX_PROPS_BYTES = 48 * 1024
MAX_CODE_BYTES = 48 * 1024
MAX_STR = 400
MAX_LIST = 120
MAX_DEPTH = 4
MAX_SCENE_ELEMENTS = 160
MAX_TTL = 3600

# ── what she may use ─────────────────────────────────────────────────────
KINDS = (
    "clock", "timer", "gauge", "progress", "power", "tile", "list", "table",
    "chart", "text", "weather", "notification", "status", "steps", "connect",
    "radar", "waveform", "model3d", "image", "hud", "scene", "code",
)
# Kinds that sit inside the orb by default; the others open around it.
INNER_KINDS = ("clock", "timer", "gauge", "progress", "power", "tile",
               "connect", "radar", "waveform", "model3d", "notification", "scene")

SLOTS = ("auto", "inner", "n", "ne", "e", "se", "s", "sw", "w", "nw")

PALETTES = ("teal", "cyan", "blue", "amber", "red", "green", "violet", "white")

_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")

SCENE_SHAPES = ("ring", "arc", "line", "polyline", "rect", "circle", "text",
                "counter", "grid", "particles", "bars", "ticks", "polygon", "pulse")

# Props she must supply, per kind. Missing ones are filled by the screen with a
# harmless default, so this only guards against a call that has nothing at all.
_REQUIRED = {
    "gauge": ("value",), "progress": ("frac",), "power": ("value",),
    "tile": ("value",), "list": ("items",), "table": ("rows",), "chart": ("series",),
    "text": ("body",), "weather": ("temp",), "notification": ("title",),
    "status": ("items",), "steps": ("steps",), "model3d": ("objects",),
    "image": ("src",), "hud": ("nodes",), "scene": ("elements",), "code": ("html",),
}

CATALOG_PROMPT = """WIDGETS — you can put anything on Sir's screen, inside or around the orb.
Use show_widget(kind, props JSON, slot, color). Pick by what the answer IS:
 clock{tz?,hour12?,seconds?}  timer{seconds,label}  gauge{label,value,min,max,unit,thresholds[warn,crit]}
 progress{label,frac 0-1,sub}  power{label,value 0-100,sub}  tile{label,value,unit,sub,trend up|down|flat}
 list{title,items[{text,sub,state todo|doing|done|fail}]}  table{title,columns[],rows[[]]}
 chart{title,type line|bar|area,series[{name,points[]}],labels[],unit}  text{title,body}
 weather{place,temp,unit,cond,hi,lo,details[{k,v}]}  notification{title,body}
 status{title,items[{label,state ok|warn|bad|busy,value}]}  steps{steps[{label,state}]}
 connect{label,sub}  radar{label,blips[{angle 0-360,dist 0-1,label}]}  waveform{label}
 model3d{title,objects[{kind box|cylinder|sphere|line,pos[x,y,z],rot,size,r,color}],spin,explode 0-1}
 image{title,src data:image URL,caption}  hud{title,root,nodes} (the HUD catalog)
Slots: inner (inside the orb: clock, gauge, tile, progress, power, connect, radar, model3d),
 or n ne e se s sw w nw (around it: lists, tables, charts, text, weather). Omit slot and it picks.
Colour: teal (default, the orb's own), cyan, blue, amber, red, green, violet, white, or "#rrggbb".
 Use amber/red for warnings, green for success, a colour that suits the subject otherwise.
Keep it live: update_widget(id, props) while you talk; close_widget(id) or id "all" when done.
Something none of these can show -> build_scene (drawing primitives: ring arc line polyline rect circle
 text counter grid particles bars ticks polygon pulse, with spin/pulse/sweep animation). Only when even that
 cannot do it -> build_widget_code (a small HTML canvas program; sandboxed, no network, no libraries).
Open one to three widgets, not ten. Narrate the headline, never the pixels."""


def catalog_prompt() -> str:
    return CATALOG_PROMPT


# ── cleaning ──────────────────────────────────────────────────────────────
def _num(x, lo=-1e9, hi=1e9, default=0.0):
    try:
        f = float(x)
    except Exception:
        return default
    if not math.isfinite(f):
        return default
    return max(lo, min(hi, f))


def _str(s, cap=MAX_STR) -> str:
    return str(s if s is not None else "")[:cap]


def clean_value(v, depth: int = 0):
    """JSON-ish value -> a safe copy. Numbers clamped, strings capped, depth and
    length limited, keys that look like handlers or prototype tricks dropped."""
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return _num(v)
    if isinstance(v, str):
        return _str(v)
    if depth >= MAX_DEPTH:
        return None
    if isinstance(v, (list, tuple)):
        return [clean_value(x, depth + 1) for x in list(v)[:MAX_LIST]]
    if isinstance(v, dict):
        out = {}
        for k, x in list(v.items())[:60]:
            if not isinstance(k, str) or not (1 <= len(k) <= 40):
                continue
            kl = k.lower()
            if kl.startswith("on") or kl in ("__proto__", "constructor", "prototype",
                                              "innerhtml", "dangerouslysetinnerhtml", "script"):
                continue
            out[k] = clean_value(x, depth + 1)
        return out
    return None


def clean_color(c) -> str:
    """A palette name or #rrggbb. Anything else is the default."""
    if isinstance(c, str):
        c = c.strip()
        if c.lower() in PALETTES:
            return c.lower()
        if _HEX.match(c):
            return c.lower()
    return "teal"


def _parse_json(raw):
    if isinstance(raw, (dict, list)):
        return raw
    if not isinstance(raw, str):
        return None
    txt = raw.strip()
    if txt.startswith("```"):
        txt = txt.split("\n", 1)[1] if "\n" in txt else ""
        txt = txt.rstrip()
        if txt.endswith("```"):
            txt = txt[:-3]
    try:
        return json.loads(txt)
    except Exception:
        pass
    # One repair: take the first complete value and ignore anything after it.
    start = min([i for i in (txt.find("{"), txt.find("[")) if i >= 0], default=-1)
    if start >= 0:
        try:
            return json.JSONDecoder().raw_decode(txt[start:])[0]
        except Exception:
            pass
    cut = max(txt.rfind("}"), txt.rfind("]"))
    if cut > 0:
        try:
            return json.loads(txt[:cut + 1])
        except Exception:
            return None
    return None


def clean_slot(slot, kind: str) -> str:
    slot = (slot or "auto").strip().lower() if isinstance(slot, str) else "auto"
    if slot not in SLOTS:
        slot = "auto"
    return slot


def clean_id(i, kind: str) -> str:
    i = re.sub(r"[^a-zA-Z0-9_\-]", "", str(i or ""))[:48]
    return i or f"{kind}-{int(time.time() * 1000) % 10_000_000}"


def clean_ttl(t) -> int:
    n = int(_num(t, 0, MAX_TTL, 0))
    return n


# ── tier 1: presets ───────────────────────────────────────────────────────
def build_open(kind, props=None, slot=None, color=None, wid=None, title=None, ttl=None):
    """Make a validated widget:open message, or (None, reason)."""
    kind = _str(kind, 20).strip().lower()
    if kind not in KINDS or kind in ("code",):
        return None, f"unknown widget kind '{kind}'. Kinds: {', '.join(k for k in KINDS if k != 'code')}"
    parsed = _parse_json(props) if props not in (None, "") else {}
    if parsed is None:
        return None, "props was not valid JSON"
    if not isinstance(parsed, dict):
        return None, "props must be a JSON object"
    if kind == "scene":
        return None, "use build_scene for scenes"
    clean = clean_value(parsed)
    if kind == "hud":
        from hud_spec import sanitize_nodes
        nodes = sanitize_nodes(clean.get("nodes") if isinstance(clean, dict) else None)
        if not nodes:
            return None, "hud needs a nodes map"
        clean["nodes"] = nodes
        if clean.get("root") not in nodes:
            clean["root"] = next(iter(nodes))
    need = _REQUIRED.get(kind, ())
    if any(clean.get(k) is None for k in need):
        return None, f"{kind} needs: {', '.join(need)}"
    if len(json.dumps(clean)) > MAX_PROPS_BYTES:
        return None, "props too large"
    if title:
        clean["title"] = _str(title, 80)
    kind_slot = clean_slot(slot, kind)
    return {
        "t": "widget:open", "v": V, "id": clean_id(wid, kind), "kind": kind,
        "props": clean, "slot": kind_slot, "color": clean_color(color),
        "ttl": clean_ttl(ttl),
    }, ""


def build_update(wid, props):
    parsed = _parse_json(props)
    if not isinstance(parsed, dict):
        return None, "props must be a JSON object"
    clean = clean_value(parsed)
    if len(json.dumps(clean)) > MAX_PROPS_BYTES:
        return None, "props too large"
    wid = re.sub(r"[^a-zA-Z0-9_\-]", "", str(wid or ""))[:48]
    if not wid:
        return None, "update needs the widget id"
    return {"t": "widget:update", "v": V, "id": wid, "props": clean}, ""


def build_close(wid):
    wid = str(wid or "").strip()
    if wid.lower() in ("all", "*", ""):
        return {"t": "widget:close", "v": V, "id": "all"}
    return {"t": "widget:close", "v": V, "id": re.sub(r"[^a-zA-Z0-9_\-]", "", wid)[:48]}


# ── tier 2: scenes ────────────────────────────────────────────────────────
_SCENE_NUMERIC = (
    "x", "y", "x2", "y2", "r", "r2", "w", "h", "a0", "a1", "n", "size", "width",
    "alpha", "spin", "pulse", "sweep", "value", "speed", "gap", "len", "count",
    "from", "to", "decimals", "dash", "seed", "phase",
)
_SCENE_STRINGS = ("text", "color", "fill", "anchor", "font", "unit", "id", "label", "bind")


def clean_element(el):
    if not isinstance(el, dict):
        return None
    shape = el.get("shape")
    if shape not in SCENE_SHAPES:
        return None
    out = {"shape": shape}
    for k in _SCENE_NUMERIC:
        if k in el:
            out[k] = _num(el[k])
    for k in _SCENE_STRINGS:
        if k not in el:
            continue
        if k in ("color", "fill"):
            c = str(el[k]).strip().lower()
            # "main" is the widget's own colour; dim/bright are its shades.
            out[k] = c if c in PALETTES + ("main", "dim", "bright") or _HEX.match(c) else "main"
        else:
            out[k] = _str(el[k], 80)
    for k in ("points", "values", "labels", "steps"):
        if k in el and isinstance(el[k], list):
            out[k] = [(_num(p) if isinstance(p, (int, float)) else
                       [_num(q) for q in p[:2]] if isinstance(p, list) else _str(p, 40))
                      for p in el[k][:MAX_LIST]]
    return out


def build_scene(title, elements, slot=None, color=None, width=None, height=None,
                wid=None, ttl=None, background=None):
    parsed = _parse_json(elements)
    if isinstance(parsed, dict):
        parsed = parsed.get("elements")
    if not isinstance(parsed, list):
        return None, "elements must be a JSON list of shapes"
    els = [e for e in (clean_element(x) for x in parsed[:MAX_SCENE_ELEMENTS]) if e]
    if not els:
        return None, "no valid shapes (allowed: " + ", ".join(SCENE_SHAPES) + ")"
    props = {
        "title": _str(title or "Scene", 80),
        "width": _num(width or 400, 100, 1000, 400),
        "height": _num(height or 400, 100, 1000, 400),
        "elements": els,
    }
    if background:
        props["background"] = clean_color(background)
    return {
        "t": "widget:open", "v": V, "id": clean_id(wid, "scene"), "kind": "scene",
        "props": props, "slot": clean_slot(slot, "scene"),
        "color": clean_color(color), "ttl": clean_ttl(ttl),
    }, ""


# ── tier 3: sandboxed code ────────────────────────────────────────────────
# A first filter only. The screen runs this in an iframe with no network and no
# access to the app (sandbox + CSP), so these are belt and braces.
_FORBIDDEN = re.compile(
    r"(fetch\s*\(|XMLHttpRequest|WebSocket|EventSource|importScripts|import\s*\(|"
    r"navigator\.sendBeacon|window\.parent|window\.top|\bparent\s*\.|\btop\s*\.|"
    r"document\.cookie|localStorage|sessionStorage|indexedDB|<iframe|<object|<embed|"
    r"<link\b|<meta\b|<base\b|<form\b|window\.open|location\s*=|location\.|"
    r"eval\s*\(|new\s+Function|\bWorker\s*\(|SharedWorker|ServiceWorker|"
    r"src\s*=\s*[\"']?\s*(https?:|//)|href\s*=|@import|url\s*\(\s*[\"']?\s*(https?:|//))",
    re.IGNORECASE,
)


def build_code(title, html, slot=None, wid=None, ttl=None, color=None):
    if not isinstance(html, str) or not html.strip():
        return None, "html is empty"
    if len(html.encode("utf-8", "ignore")) > MAX_CODE_BYTES:
        return None, "program too large (max 48 KB)"
    m = _FORBIDDEN.search(html)
    if m:
        return None, (f"'{m.group(0)[:24]}' is not allowed in a sandboxed widget "
                      "(no network, no navigation, no storage, no frames)")
    return {
        "t": "widget:open", "v": V, "id": clean_id(wid, "code"), "kind": "code",
        "props": {"title": _str(title or "Widget", 80), "html": html},
        "slot": clean_slot(slot, "code"), "color": clean_color(color),
        "ttl": clean_ttl(ttl),
    }, ""
