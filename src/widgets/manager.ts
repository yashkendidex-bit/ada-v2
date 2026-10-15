/**
 * The widget manager: owns everything drawn on top of the orb.
 *
 * Messages come from the backend (`widget:open|update|close|data`, and the
 * older `hud:*` dashboard events) and from the dock. Each widget is made from a
 * fixed set of kinds; nothing she sends is run as code except inside the
 * sandbox of the "code" kind.
 *
 * Placement: dials open inside the orb, panels open around it, on whichever
 * side has room. The orb dims behind an inner widget and churns while a widget
 * reports that a task is running. Each panel is tied to the orb by a thin
 * leader line.
 *
 * Real data only: every widget that shows measured values carries where they
 * came from and how old they are (`src` on the message, set by the backend from
 * the tool that actually ran). The footer under it says LIVE, STALE,
 * UNVERIFIED or SAMPLE; a failed source dims the widget and says NO SIGNAL.
 * Nothing here ever invents or smooths a value.
 */
import type { KindSpec, Props, Slot, WidgetEnv, WidgetInstance, Palette } from "./types";
import { KINDS, INNER_KINDS } from "./kinds";
import { assignPalette, paletteVars, resolvePalette } from "./palette";

export interface OrbLike {
  getLayout(): { cx: number; cy: number; r: number };
  onLayout(fn: () => void): void;
  setFocus(f: { inner?: number; busy?: number }): void;
}

export interface ManagerOptions {
  layer: HTMLElement;
  orb: OrbLike;
  /** An allow-listed action from a widget button. */
  onAction(action: string): void;
}

/** Same list as backend/hud_spec.ACTIONS. Anything else a widget asks for goes nowhere. */
const ACTIONS = new Set([
  "pause_print", "print_progress", "dismiss_hud", "refresh_data", "approve_spec",
  "run_preflight", "show_scorecard", "show_ledger",
]);

const MAX_AROUND = 3;
const MARGIN = 16;
const TOP_RESERVE = 60;
const BOTTOM_RESERVE = 150;
const AUTO_ORDER: Slot[] = ["e", "w", "ne", "nw", "se", "sw", "n", "s"];

/** Kinds that show measured values, so they carry a source line. Clocks, timers, text and scenes do not. */
const DATA_KINDS = new Set(["gauge", "progress", "power", "tile", "list", "table", "chart", "weather", "status", "radar", "hud"]);
/** Seconds before a value counts as old, unless the backend says otherwise. */
const STALE_AFTER: Record<string, number> = { weather: 1800, chart: 1800, table: 600, list: 600 };

type Health = "live" | "stale" | "none" | "unverified" | "sample";
const HEALTH_TAG: Record<Health, string> = { live: "LIVE", stale: "STALE", none: "NO SIGNAL", unverified: "UNVERIFIED", sample: "SAMPLE" };

interface Src { name: string; anchor: number; verified: boolean; staleAfter: number; sample: boolean }

function parseSrc(raw: unknown, kind: string, now: number): Src | null {
  if (!raw || typeof raw !== "object") return null;
  const o = raw as Record<string, unknown>;
  const age = Math.max(0, Number(o.age) || 0);
  const sa = Number(o.stale_after);
  return {
    name: typeof o.name === "string" ? o.name.replace(/[^\w .:/+-]/g, "").slice(0, 32) : "",
    anchor: now - age * 1000,
    verified: o.verified === true,
    staleAfter: Number.isFinite(sa) && sa > 0 ? Math.min(sa, 86400) : (STALE_AFTER[kind] ?? 120),
    sample: o.sample === true,
  };
}

function ageText(sec: number): string {
  if (sec < 90) return `${Math.max(0, Math.floor(sec))}s`;
  if (sec < 5400) return `${Math.round(sec / 60)}m`;
  return `${Math.round(sec / 3600)}h`;
}

const SVG_NS = "http://www.w3.org/2000/svg";

interface Rect { x: number; y: number; w: number; h: number }

interface Entry {
  id: string;
  kind: string;
  spec: KindSpec;
  inst: WidgetInstance;
  props: Props;
  pal: Palette;
  host: HTMLElement;
  fx: HTMLElement;
  inner: boolean;
  slot: Slot;
  rect: Rect;
  manual: { x: number; y: number } | null;
  ttl: number | null;
  closing: boolean;
  ro: ResizeObserver | null;
  src: Src | null;
  health: Health | "";
  prov: { el: HTMLElement; tag: HTMLElement; name: HTMLElement; age: HTMLElement } | null;
  nosig: HTMLElement | null;
  leader: SVGGElement | null;
}

const hit = (a: Rect, b: Rect, pad = 8): boolean =>
  a.x < b.x + b.w + pad && a.x + a.w + pad > b.x && a.y < b.y + b.h + pad && a.y + a.h + pad > b.y;

function setPath(root: Record<string, unknown>, path: string, value: unknown): boolean {
  const segs = path.split(/[./]/).filter(Boolean);
  if (!segs.length || segs.some((s) => s === "__proto__" || s === "constructor" || s === "prototype")) return false;
  let cur: unknown = root;
  for (let i = 0; i < segs.length - 1; i++) {
    if (cur === null || typeof cur !== "object") return false;
    const o = cur as Record<string, unknown>;
    if (o[segs[i]] === undefined || o[segs[i]] === null) o[segs[i]] = /^\d+$/.test(segs[i + 1]) ? [] : {};
    cur = o[segs[i]];
  }
  if (cur === null || typeof cur !== "object") return false;
  (cur as Record<string, unknown>)[segs[segs.length - 1]] = value;
  return true;
}

export class WidgetManager {
  private entries = new Map<string, Entry>();
  private listeners: Array<() => void> = [];
  private t = 0;
  private lastTs = -1;
  private busyAt = 0;
  private raf = 0;
  private destroyed = false;
  private calm = false;
  private hudProps = new Map<string, Props>();
  private leaders: SVGSVGElement;
  private provAt = 0;

  constructor(private opts: ManagerOptions) {
    this.leaders = document.createElementNS(SVG_NS, "svg");
    this.leaders.setAttribute("id", "wd-leaders");
    this.leaders.setAttribute("aria-hidden", "true");
    opts.layer.prepend(this.leaders);
    const mq = window.matchMedia?.("(prefers-reduced-motion: reduce)");
    this.calm = !!mq?.matches;
    document.body.classList.toggle("calm", this.calm);
    opts.orb.onLayout(() => this.relayoutAll());
    window.addEventListener("resize", () => this.relayoutAll());
    this.raf = requestAnimationFrame((ts) => this.frame(ts));
  }

  // ── messages ──────────────────────────────────────────────────────────

  handle(msg: Record<string, unknown>): void {
    const t = String(msg.t ?? "");
    try {
      if (t === "widget:open") this.open(msg);
      else if (t === "widget:update") this.update(String(msg.id ?? ""), (msg.props as Props) ?? {}, msg.src);
      else if (t === "widget:close") this.close(String(msg.id ?? ""));
      else if (t === "widget:data") this.data(String(msg.id ?? ""), String(msg.path ?? ""), msg.value, msg.src);
      else if (t.startsWith("hud:")) this.hud(t, msg);
    } catch (e) {
      console.warn("[widgets] could not handle", t, e);
    }
  }

  /** Open a widget from the screen itself (the dock). */
  openLocal(kind: string, props: Props, slot: Slot = "auto", color = "teal", id?: string, sample = false): string {
    const wid = id ?? `${kind}-local`;
    this.open({
      t: "widget:open", id: wid, kind, props, slot, color, ttl: 0,
      src: sample ? { name: "showcase", sample: true } : undefined,
    });
    return wid;
  }

  isOpen(id: string): boolean {
    const e = this.entries.get(id);
    return !!e && !e.closing;
  }

  count(): number {
    return [...this.entries.values()].filter((e) => !e.closing).length;
  }

  onChange(fn: () => void): void { this.listeners.push(fn); }

  closeAll(): void {
    for (const id of [...this.entries.keys()]) this.close(id);
  }

  /** Close the most recent widget. True if there was one. */
  closeTop(): boolean {
    const live = [...this.entries.values()].filter((e) => !e.closing);
    const e = live[live.length - 1];
    if (!e) return false;
    this.close(e.id);
    return true;
  }

  destroy(): void {
    this.destroyed = true;
    cancelAnimationFrame(this.raf);
    for (const e of this.entries.values()) this.dispose(e);
    this.entries.clear();
  }

  // ── open / update / close ─────────────────────────────────────────────

  private open(msg: Record<string, unknown>): void {
    const kind = String(msg.kind ?? "");
    const spec = KINDS[kind];
    if (!spec) { console.warn("[widgets] unknown kind", kind); return; }
    const id = String(msg.id ?? "").replace(/[^a-zA-Z0-9_\-]/g, "").slice(0, 48) || `${kind}-${Date.now() % 1e7}`;
    const props: Props = msg.props && typeof msg.props === "object" ? { ...(msg.props as Props) } : {};
    const color = msg.color;
    const slotIn = (typeof msg.slot === "string" ? msg.slot : "auto") as Slot;

    const existing = this.entries.get(id);
    if (existing && !existing.closing && existing.kind === kind) {
      existing.props = { ...existing.props, ...props };
      if (props.error === undefined) delete existing.props.error;   // fresh numbers clear an old failure
      this.setSrc(existing, msg.src);
      assignPalette(existing.pal, resolvePalette(color));
      paletteVars(existing.host, existing.pal);
      existing.inst.update(existing.props);
      this.armTtl(existing, Number(msg.ttl) || 0);
      this.layout(existing);
      return;
    }
    if (existing) this.close(id, true);

    const pal = resolvePalette(color);
    const env: WidgetEnv = {
      pal, calm: this.calm,
      action: (n) => { if (ACTIONS.has(n)) this.opts.onAction(n); },
      close: () => this.close(id),
    };
    let inst: WidgetInstance;
    try {
      inst = spec.make(props, env);
    } catch (e) {
      console.warn("[widgets] failed to build", kind, e);
      return;
    }

    const inner = slotIn === "inner" || (slotIn === "auto" && INNER_KINDS.has(kind) && spec.inner !== undefined);
    if (inner) for (const e of this.entries.values()) if (e.inner && !e.closing) this.close(e.id, true);

    const host = document.createElement("div");
    host.className = `wd-host ${inner ? "inner" : "around"}`;
    paletteVars(host, pal);
    const fx = document.createElement("div");
    fx.className = `wd-fx ${spec.panel ? "rect" : "round"}`;
    fx.appendChild(inst.el);
    host.appendChild(fx);
    const x = document.createElement("button");
    x.className = "wd-x";
    x.type = "button";
    x.textContent = "×";
    x.setAttribute("aria-label", "close widget");
    x.addEventListener("click", (ev) => { ev.stopPropagation(); this.close(id); });
    x.addEventListener("pointerdown", (ev) => ev.stopPropagation());
    host.appendChild(x);
    let prov: Entry["prov"] = null;
    let nosig: HTMLElement | null = null;
    if (DATA_KINDS.has(kind)) {
      const pe = document.createElement("div");
      pe.className = "wd-prov";
      const tag = document.createElement("span"); tag.className = "tg";
      const nm = document.createElement("span");
      const ag = document.createElement("span");
      pe.append(tag, nm, ag);
      host.appendChild(pe);
      prov = { el: pe, tag, name: nm, age: ag };
      nosig = document.createElement("div");
      nosig.className = "wd-nosig";
      host.appendChild(nosig);
    }
    host.style.visibility = "hidden";
    this.opts.layer.appendChild(host);

    const entry: Entry = {
      id, kind, spec, inst, props, pal, host, fx, inner,
      slot: inner ? "inner" : slotIn, rect: { x: 0, y: 0, w: 0, h: 0 },
      manual: null, ttl: null, closing: false, ro: null,
      src: null, health: "", prov, nosig, leader: null,
    };
    this.setSrc(entry, msg.src);
    this.entries.set(id, entry);

    if (!inner) this.chooseAround(entry);
    this.layout(entry);
    host.style.visibility = "";
    if (!this.calm) {
      fx.classList.add("in");
      fx.addEventListener("animationend", (ev) => {
        if ((ev as AnimationEvent).animationName === "wd-in") fx.classList.remove("in");
      }, { once: false });
    }
    if (!inner) this.enableDrag(entry);
    if (typeof ResizeObserver !== "undefined") {
      entry.ro = new ResizeObserver(() => { if (!entry.closing) this.layout(entry); });
      entry.ro.observe(inst.el);
    }
    this.armTtl(entry, Number(msg.ttl) || 0);
    this.refreshProv(entry, performance.now(), true);
    this.updateFocus();
    this.emit();
  }

  private setSrc(e: Entry, raw: unknown): void {
    if (!e.prov) return;
    const s = parseSrc(raw, e.kind, performance.now());
    if (s) e.src = s;
  }

  /** LIVE / STALE / UNVERIFIED / SAMPLE / NO SIGNAL, from what the backend said and how long ago. */
  private healthOf(e: Entry, now: number): Health {
    if (typeof e.props.error === "string" && e.props.error) return "none";
    if (!e.src) return "unverified";
    if (e.src.sample) return "sample";
    if (!e.src.verified) return "unverified";
    return (now - e.src.anchor) / 1000 > e.src.staleAfter ? "stale" : "live";
  }

  private refreshProv(e: Entry, now: number, force = false): void {
    if (!e.prov) return;
    const hl = this.healthOf(e, now);
    if (hl !== e.health) {
      e.health = hl;
      for (const k of Object.keys(HEALTH_TAG)) e.host.classList.toggle(`st-${k}`, k === hl);
      e.prov.tag.textContent = HEALTH_TAG[hl];
      if (e.nosig) {
        e.nosig.textContent = "";
        if (hl === "none") {
          const a = document.createElement("div"); a.textContent = "NO SIGNAL";
          const b = document.createElement("div"); b.className = "why";
          b.textContent = String(e.props.error).slice(0, 90);
          e.nosig.append(a, b);
        }
      }
    }
    if (hl === "none" && e.nosig) {
      const why = e.nosig.querySelector(".why");
      if (why && why.textContent !== String(e.props.error).slice(0, 90)) why.textContent = String(e.props.error).slice(0, 90);
    }
    const name = e.src?.name || "none given";
    if (force || e.prov.name.textContent !== `SRC ${name}`) e.prov.name.textContent = `SRC ${name}`;
    const age = e.src && !e.src.sample ? `AGE ${ageText((now - e.src.anchor) / 1000)}` : "";
    if (e.prov.age.textContent !== age) e.prov.age.textContent = age;
  }

  private update(id: string, props: Props, src?: unknown): void {
    const e = this.entries.get(id);
    if (!e || e.closing) return;
    this.setSrc(e, src);
    e.props = { ...e.props, ...props };
    if (props.error === undefined) delete e.props.error;
    if (typeof props.color === "string") { assignPalette(e.pal, resolvePalette(props.color)); paletteVars(e.host, e.pal); }
    e.inst.update(e.props);
  }

  private data(id: string, path: string, value: unknown, src?: unknown): void {
    const e = this.entries.get(id);
    if (!e || e.closing) return;
    this.setSrc(e, src);
    if (setPath(e.props as Record<string, unknown>, path, value)) e.inst.update(e.props);
  }

  close(id: string, fast = false): void {
    if (id === "all" || id === "*") { this.closeAll(); return; }
    const e = this.entries.get(id);
    if (!e || e.closing) return;
    e.closing = true;
    if (e.ttl !== null) { clearTimeout(e.ttl); e.ttl = null; }
    this.updateFocus();
    this.emit();
    const done = () => { this.dispose(e); this.entries.delete(id); this.updateFocus(); this.emit(); };
    if (fast || this.calm) { done(); return; }
    e.fx.classList.remove("in");
    e.fx.classList.add("out");
    e.leader?.classList.add("gone");
    setTimeout(done, 460);
  }

  private dispose(e: Entry): void {
    e.ro?.disconnect();
    if (e.ttl !== null) clearTimeout(e.ttl);
    try { e.inst.destroy(); } catch { /* a widget failing to clean up must not stop the rest */ }
    e.host.remove();
    e.leader?.remove();
  }

  private armTtl(e: Entry, seconds: number): void {
    if (e.ttl !== null) { clearTimeout(e.ttl); e.ttl = null; }
    if (seconds > 0) e.ttl = window.setTimeout(() => this.close(e.id), Math.min(seconds, 3600) * 1000);
  }

  // ── hud dashboards (older event family) ───────────────────────────────

  private hud(t: string, msg: Record<string, unknown>): void {
    const hudId = String(msg.hudId ?? "").replace(/[^a-zA-Z0-9_\-]/g, "").slice(0, 60);
    if (!hudId) return;
    const wid = `hud-${hudId}`;
    if (t === "hud:start") {
      const props: Props = { title: String(msg.title ?? "HUD"), nodes: {}, root: msg.root ?? "", loading: true };
      this.hudProps.set(wid, props);
      this.open({ t: "widget:open", id: wid, kind: "hud", props, slot: "auto", color: "teal", ttl: 0, src: msg.src });
      return;
    }
    const props = this.hudProps.get(wid);
    const e = this.entries.get(wid);
    if (!props || !e) return;
    // A live number pushed into a dashboard from a measured source refreshes its age.
    if (t === "hud:data" || t === "hud:chunk") this.setSrc(e, msg.src ?? (e.src ? { name: e.src.name, verified: e.src.verified, sample: e.src.sample, stale_after: e.src.staleAfter } : undefined));
    const nodes = props.nodes as Record<string, unknown>;
    if (t === "hud:chunk") {
      const up = (msg.upsert ?? {}) as Record<string, unknown>;
      for (const k of Object.keys(up).slice(0, 200)) nodes[k] = up[k];
      for (const k of (Array.isArray(msg.remove) ? msg.remove : []) as unknown[]) delete nodes[String(k)];
    } else if (t === "hud:data") {
      const segs = String(msg.path ?? "").split("/").filter(Boolean);
      const node = segs.length >= 3 ? (nodes[segs[0]] as Record<string, unknown> | undefined) : undefined;
      if (!node) return;
      setPath(node, segs.slice(1).join("/"), msg.value);
    } else if (t === "hud:final") {
      props.loading = false;
    }
    e.props = props;
    e.inst.update(props);
  }

  // ── layout ────────────────────────────────────────────────────────────

  private natural(e: Entry): { w: number; h: number } {
    const el = e.inst.el;
    return { w: el.offsetWidth || e.spec.design[0], h: el.offsetHeight || e.spec.design[1] };
  }

  /** Where a slot sits on this screen, as the region a widget may fill. */
  private region(slot: Slot): Rect {
    const { cx, cy, r } = this.opts.orb.getLayout();
    const W = window.innerWidth, H = window.innerHeight;
    const top = TOP_RESERVE, bottom = H - BOTTOM_RESERVE;
    const gap = r * 1.08;
    const side = cx - gap - MARGIN;
    const wide = side >= 190;
    const left: Rect = { x: MARGIN, y: top, w: Math.max(0, cx - gap - MARGIN), h: bottom - top };
    const right: Rect = { x: cx + gap, y: top, w: Math.max(0, W - MARGIN - (cx + gap)), h: bottom - top };
    const upper = (b: Rect): Rect => ({ ...b, h: Math.max(0, cy - 6 - b.y) });
    const lower = (b: Rect): Rect => ({ ...b, y: cy + 6, h: Math.max(0, b.y + b.h - (cy + 6)) });
    if (wide) {
      switch (slot) {
        case "e": return right;
        case "w": return left;
        case "ne": return upper(right);
        case "se": return lower(right);
        case "nw": return upper(left);
        case "sw": return lower(left);
        case "n": return { x: MARGIN, y: top, w: W - 2 * MARGIN, h: Math.max(0, cy - r * 1.04 - top) };
        default: return { x: MARGIN, y: cy + r * 1.04, w: W - 2 * MARGIN, h: Math.max(0, bottom - (cy + r * 1.04)) };
      }
    }
    // Narrow screen: the sides have no room, so use the strips above and below the orb.
    const topStrip: Rect = { x: MARGIN, y: top, w: W - 2 * MARGIN, h: Math.max(90, cy - r * 1.04 - top) };
    const botStrip: Rect = { x: MARGIN, y: cy + r * 1.04, w: W - 2 * MARGIN, h: Math.max(90, bottom - (cy + r * 1.04)) };
    switch (slot) {
      case "se": case "sw": case "s": return botStrip;
      case "nw": return { ...topStrip, w: topStrip.w / 2 };
      case "ne": return { ...topStrip, x: topStrip.x + topStrip.w / 2, w: topStrip.w / 2 };
      default: return topStrip;
    }
  }

  private fitAround(e: Entry, slot: Slot): Rect {
    const { r } = this.opts.orb.getLayout();
    const R = this.region(slot);
    const { w, h } = this.natural(e);
    const cap = (e.spec.around[0] * r) / w;
    let sc = Math.min(R.w / w, R.h / h, cap, 1.5);
    if (!(sc > 0.4)) sc = 0.4;
    const ww = w * sc, hh = h * sc;
    let x = R.x + (R.w - ww) / 2;
    let y = R.y + (R.h - hh) / 2;
    if (slot === "ne" || slot === "nw") y = R.y + R.h - hh;
    if (slot === "se" || slot === "sw") y = R.y;
    if (slot === "n") y = R.y + R.h - hh;
    if (slot === "s") y = R.y;
    return { x, y, w: ww, h: hh };
  }

  private chooseAround(e: Entry): void {
    const others = () => [...this.entries.values()].filter((o) => o !== e && !o.inner && !o.closing);
    // Never more than three around the orb: the oldest makes way.
    while (others().length >= MAX_AROUND) this.close(others()[0].id, true);
    const order: Slot[] = e.slot !== "auto" && e.slot !== "inner" ? [e.slot] : AUTO_ORDER;
    for (const s of order) {
      const rc = this.fitAround(e, s);
      if (!others().some((o) => hit(rc, o.rect))) { e.slot = s; return; }
    }
    if (order.length === 1) {
      // She chose this slot explicitly: whatever is there gives way.
      const rc = this.fitAround(e, order[0]);
      for (const o of others()) if (hit(rc, o.rect)) this.close(o.id, true);
      e.slot = order[0];
      return;
    }
    const oldest = others()[0];
    if (oldest) this.close(oldest.id, true);
    e.slot = "e";
  }

  private layout(e: Entry): void {
    const { cx, cy, r } = this.opts.orb.getLayout();
    const { w, h } = this.natural(e);
    let x: number, y: number, sc: number;
    if (e.inner) {
      const d = (e.spec.inner ?? 1.5) * r;
      sc = e.spec.panel ? d / Math.hypot(w, h) : d / Math.max(w, h);
      x = cx - (w * sc) / 2; y = cy - (h * sc) / 2;
    } else if (e.manual) {
      const rc = this.fitAround(e, e.slot === "auto" ? "e" : e.slot);
      sc = rc.w / w;
      x = Math.max(4, Math.min(window.innerWidth - rc.w - 4, e.manual.x));
      y = Math.max(4, Math.min(window.innerHeight - rc.h - 4, e.manual.y));
    } else {
      const rc = this.fitAround(e, e.slot === "auto" ? "e" : e.slot);
      sc = rc.w / w; x = rc.x; y = rc.y;
    }
    e.rect = { x, y, w: w * sc, h: h * sc };
    e.host.style.width = `${w}px`;
    e.host.style.height = `${h}px`;
    e.host.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px) scale(${sc.toFixed(4)})`;
    e.host.style.setProperty("--inv", (1 / sc).toFixed(3));
    this.drawLeader(e, sc);
  }

  /**
   * A thin line from the orb's ring to the widget's near edge: out along the ring's
   * direction, one bend, then straight into the title. Only panels around the orb get one.
   */
  private drawLeader(e: Entry, sc: number): void {
    if (e.inner || e.closing) { e.leader?.remove(); e.leader = null; return; }
    const { cx, cy, r } = this.opts.orb.getLayout();
    const rc = e.rect;
    const mx = rc.x + rc.w / 2, my = rc.y + rc.h / 2;
    let px: number, py: number, dir = 0;
    if (mx > cx + r * 0.55) { dir = 1; px = rc.x - 4; py = rc.y + 14 * sc; }
    else if (mx < cx - r * 0.55) { dir = -1; px = rc.x + rc.w + 4; py = rc.y + 14 * sc; }
    else if (my < cy) { px = mx; py = rc.y + rc.h + 4; }
    else { px = mx; py = rc.y - 4; }
    const ang = Math.atan2(py - cy, px - cx);
    const ax = cx + Math.cos(ang) * r * 1.07, ay = cy + Math.sin(ang) * r * 1.07;
    let d: string;
    if (dir !== 0) {
      const run = dir * (px - ax);
      if (run < 10) { e.leader?.remove(); e.leader = null; return; }
      const ex = px - dir * Math.min(30, run * 0.5);
      d = `M ${ax.toFixed(1)} ${ay.toFixed(1)} L ${ex.toFixed(1)} ${py.toFixed(1)} L ${px.toFixed(1)} ${py.toFixed(1)}`;
    } else {
      d = `M ${ax.toFixed(1)} ${ay.toFixed(1)} L ${px.toFixed(1)} ${py.toFixed(1)}`;
    }
    if (!e.leader) {
      const g = document.createElementNS(SVG_NS, "g");
      g.setAttribute("class", "wd-leader");
      const path = document.createElementNS(SVG_NS, "path");
      path.setAttribute("pathLength", "1");
      const dot = document.createElementNS(SVG_NS, "circle");
      dot.setAttribute("r", "3");
      const end = document.createElementNS(SVG_NS, "rect");
      end.setAttribute("width", "4"); end.setAttribute("height", "4");
      g.append(path, dot, end);
      this.leaders.appendChild(g);
      e.leader = g;
      if (!this.calm) {
        g.classList.add("draw");
        path.addEventListener("animationend", () => g.classList.remove("draw"), { once: true });
      }
    }
    e.leader.style.setProperty("--lc", e.pal.c);
    const [path, dot, end] = Array.from(e.leader.children) as [SVGPathElement, SVGCircleElement, SVGRectElement];
    path.setAttribute("d", d);
    dot.setAttribute("cx", ax.toFixed(1)); dot.setAttribute("cy", ay.toFixed(1));
    end.setAttribute("x", (px - 2).toFixed(1)); end.setAttribute("y", (py - 2).toFixed(1));
  }

  private relayoutAll(): void {
    for (const e of this.entries.values()) if (!e.closing) this.layout(e);
  }

  private enableDrag(e: Entry): void {
    let start: { px: number; py: number; ox: number; oy: number } | null = null;
    e.host.addEventListener("pointerdown", (ev) => {
      const tgt = ev.target as HTMLElement;
      if (tgt.closest("button, a, input, iframe, select, textarea")) return;
      start = { px: ev.clientX, py: ev.clientY, ox: e.rect.x, oy: e.rect.y };
      e.host.setPointerCapture(ev.pointerId);
      e.host.classList.add("dragging");
    });
    e.host.addEventListener("pointermove", (ev) => {
      if (!start) return;
      e.manual = { x: start.ox + ev.clientX - start.px, y: start.oy + ev.clientY - start.py };
      this.layout(e);
    });
    const end = (ev: PointerEvent) => {
      if (!start) return;
      start = null;
      e.host.classList.remove("dragging");
      try { e.host.releasePointerCapture(ev.pointerId); } catch { /* already released */ }
    };
    e.host.addEventListener("pointerup", end);
    e.host.addEventListener("pointercancel", end);
  }

  // ── frame loop and orb link ───────────────────────────────────────────

  private updateFocus(): void {
    const innerOpen = [...this.entries.values()].some((e) => e.inner && !e.closing);
    this.opts.orb.setFocus({ inner: innerOpen ? 1 : 0 });
  }

  private frame(ts: number): void {
    if (this.destroyed) return;
    this.raf = requestAnimationFrame((n) => this.frame(n));
    const dt = this.lastTs < 0 ? 0 : Math.min(0.05, (ts - this.lastTs) / 1000);
    this.lastTs = ts;
    this.t += dt;
    let busy = false;
    for (const e of this.entries.values()) {
      if (e.closing) continue;
      try {
        e.inst.tick?.(this.t, dt);
        if (e.inst.busy?.()) busy = true;
      } catch (err) {
        console.warn("[widgets] tick failed, closing", e.kind, err);
        this.close(e.id, true);
      }
    }
    if (ts - this.provAt > 500) {
      this.provAt = ts;
      const now = performance.now();
      for (const e of this.entries.values()) if (!e.closing) this.refreshProv(e, now);
    }
    if (ts - this.busyAt > 120) {
      this.busyAt = ts;
      this.opts.orb.setFocus({ busy: busy ? 1 : 0 });
    }
  }

  private emit(): void {
    for (const fn of this.listeners) fn();
  }
}
