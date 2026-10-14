/** Tiny DOM helpers. Text is only ever set through textContent: nothing she sends is parsed as HTML. */

export function h<K extends keyof HTMLElementTagNameMap>(
  tag: K, cls?: string, text?: string,
): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

const SVG_NS = "http://www.w3.org/2000/svg";

export function s<K extends keyof SVGElementTagNameMap>(
  tag: K, attrs: Record<string, string | number> = {}, parent?: Element,
): SVGElementTagNameMap[K] {
  const e = document.createElementNS(SVG_NS, tag);
  for (const k in attrs) e.setAttribute(k, String(attrs[k]));
  if (parent) parent.appendChild(e);
  return e;
}

export function num(v: unknown, def: number, lo = -1e9, hi = 1e9): number {
  const n = typeof v === "number" ? v : typeof v === "string" ? parseFloat(v) : NaN;
  return Number.isFinite(n) ? Math.max(lo, Math.min(hi, n)) : def;
}

export function str(v: unknown, def = "", cap = 400): string {
  if (typeof v === "string") return v.slice(0, cap);
  if (typeof v === "number" && Number.isFinite(v)) return String(v);
  return def;
}

export function arr(v: unknown, cap = 120): unknown[] {
  return Array.isArray(v) ? v.slice(0, cap) : [];
}

export function obj(v: unknown): Record<string, unknown> {
  return v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : {};
}

/** Close a share of the gap over dt seconds. Same helper as the orb's, kept local so widgets stand alone. */
export function ease(cur: number, target: number, k: number, dt: number): number {
  return cur + (target - cur) * (1 - Math.exp(-k * dt));
}

export function fmt(n: number, decimals = 0): string {
  return n.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
}

/** A canvas sized to its CSS box at the device pixel ratio. */
export function sizedCanvas(w: number, h: number): { cv: HTMLCanvasElement; cx: CanvasRenderingContext2D; dpr: number } {
  const cv = document.createElement("canvas");
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  cv.width = Math.round(w * dpr);
  cv.height = Math.round(h * dpr);
  cv.style.width = `${w}px`;
  cv.style.height = `${h}px`;
  const cx = cv.getContext("2d");
  if (!cx) throw new Error("no 2d context");
  cx.scale(dpr, dpr);
  return { cv, cx, dpr };
}
