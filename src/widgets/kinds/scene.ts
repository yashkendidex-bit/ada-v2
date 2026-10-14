/**
 * Scenes: a custom visual drawn from a list of primitives. She sends data only;
 * this draws it. Coordinates are 0..100 of the scene (x across, y down), radii
 * and widths in the same units, angles in degrees clockwise from the top.
 *
 *   spin   revolutions per second            pulse  breathing size (fraction)
 *   speed  animation rate in Hz              sweep  swing amount
 *
 * Shapes: ring arc line polyline rect circle text counter grid particles bars
 * ticks polygon pulse. An element with an `id` can be changed afterwards with
 * update props { bind: { id: value } }.
 */
import type { KindSpec, Palette, Props, WidgetEnv, WidgetInstance } from "../types";
import { arr, h, num, obj, str } from "../dom";
import { resolvePalette } from "../palette";

type El = Record<string, unknown>;

const MAX_PARTICLES = 700;
const BOX = 400;

function rgba(hex: string, a: number): string {
  const v = parseInt(hex.slice(1, 7), 16);
  return `rgba(${(v >> 16) & 255}, ${(v >> 8) & 255}, ${v & 255}, ${Math.max(0, Math.min(1, a))})`;
}

function colourOf(name: unknown, pal: Palette, alpha: number): string {
  const n = typeof name === "string" ? name.trim().toLowerCase() : "main";
  if (n === "" || n === "main") return rgba(pal.c, alpha);
  if (n === "dim") return rgba(pal.cd, alpha);
  if (n === "bright") return rgba(pal.c2, alpha);
  return rgba(resolvePalette(n).c, alpha);
}

/** Cheap, repeatable pseudo-random in 0..1. */
function rnd(seed: number, i: number): number {
  const x = Math.sin(seed * 127.1 + i * 311.7) * 43758.5453;
  return x - Math.floor(x);
}

function makeScene(props: Props, env: WidgetEnv): WidgetInstance {
  const el = h("div", "wd-scene");
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cv = document.createElement("canvas");
  cv.width = BOX * dpr; cv.height = BOX * dpr;
  cv.style.width = `${BOX}px`; cv.style.height = `${BOX}px`;
  const cx = cv.getContext("2d")!;
  el.appendChild(cv);
  el.style.filter = "drop-shadow(0 0 6px rgba(var(--rgb), 0.4))";

  let els: El[] = [];
  let W = BOX, H = BOX;
  let t0 = -1;
  const bound = new Map<string, unknown>();

  const apply = (p: Props) => {
    const w0 = num(p.width, 400, 100, 1000), h0 = num(p.height, 400, 100, 1000);
    const k = BOX / Math.max(w0, h0);
    W = w0 * k; H = h0 * k;
    if (Array.isArray(p.elements)) els = arr(p.elements, 160).map(obj);
    const b = obj(p.bind);
    for (const id of Object.keys(b).slice(0, 40)) bound.set(id, b[id]);
  };
  apply(props);

  const frame = (t: number) => {
    if (t0 < 0) t0 = t;
    cx.setTransform(dpr, 0, 0, dpr, 0, 0);
    cx.clearRect(0, 0, BOX, BOX);
    cx.translate((BOX - W) / 2, (BOX - H) / 2);
    const ux = W / 100, uy = H / 100, u = Math.min(ux, uy);
    const pal = env.pal;
    let particles = 0;
    const age = t - t0;
    const rad = (a: number) => ((a - 90) * Math.PI) / 180;

    for (const e of els) {
      const shape = str(e.shape, "", 12);
      const alpha = num(e.alpha, 1, 0, 1);
      const x = num(e.x, 50) * ux, y = num(e.y, 50) * uy;
      const speed = num(e.speed, 1, 0, 20);
      const lw = Math.max(0.5, num(e.width, 0.8, 0, 20) * u);
      const stroke = colourOf(e.color, pal, alpha);
      const fillC = e.fill !== undefined ? colourOf(e.fill, pal, alpha * 0.35) : "";
      const id = str(e.id, "", 48);
      const bv = id ? bound.get(id) : undefined;
      const spin = num(e.spin, 0, -20, 20) * age * 360;
      const breathe = 1 + num(e.pulse, 0, 0, 2) * Math.sin(age * Math.PI * 2 * speed + num(e.phase, 0));
      const dash = num(e.dash, 0, 0, 40) * u;

      cx.save();
      cx.lineWidth = lw; cx.strokeStyle = stroke; cx.fillStyle = fillC || stroke;
      cx.lineCap = "round"; cx.lineJoin = "round";
      cx.setLineDash(dash > 0 ? [dash, dash * 1.2] : []);

      switch (shape) {
        case "ring": case "circle": {
          const r = num(e.r, 20, 0, 200) * u * breathe;
          cx.translate(x, y); cx.rotate((spin * Math.PI) / 180);
          cx.beginPath(); cx.arc(0, 0, r, 0, Math.PI * 2);
          if (shape === "circle" && e.fill !== undefined) { cx.fillStyle = colourOf(e.fill, pal, alpha); cx.fill(); }
          if (shape === "ring" || e.width !== undefined || e.fill === undefined) cx.stroke();
          break;
        }
        case "arc": {
          const r = num(e.r, 20, 0, 200) * u * breathe;
          const a0 = num(e.a0, 0), a1 = typeof bv === "number" ? a0 + bv * 360 : num(e.a1, 270);
          const sw = num(e.sweep, 0, 0, 1) * Math.sin(age * Math.PI * 2 * speed) * 0.5;
          cx.translate(x, y); cx.rotate((spin * Math.PI) / 180);
          cx.beginPath(); cx.arc(0, 0, r, rad(a0), rad(a0 + (a1 - a0) * (1 - Math.abs(sw))));
          cx.stroke();
          break;
        }
        case "line": {
          const x2 = num(e.x2, 60) * ux, y2 = num(e.y2, 50) * uy;
          const sw = num(e.sweep, 0, 0, 100) * u * Math.sin(age * Math.PI * 2 * speed);
          const dx = x2 - x, dy = y2 - y, len = Math.hypot(dx, dy) || 1;
          const nx = (-dy / len) * sw, ny = (dx / len) * sw;
          cx.beginPath(); cx.moveTo(x + nx, y + ny); cx.lineTo(x2 + nx, y2 + ny); cx.stroke();
          break;
        }
        case "polyline": case "polygon": {
          const pts = arr(e.points, 160).map((q) => (Array.isArray(q) ? [num(q[0], 0) * ux, num(q[1], 0) * uy] : [0, 0]));
          if (pts.length < 2) break;
          cx.beginPath();
          pts.forEach((q, i) => (i === 0 ? cx.moveTo(q[0], q[1]) : cx.lineTo(q[0], q[1])));
          if (shape === "polygon") { cx.closePath(); if (e.fill !== undefined) { cx.fillStyle = colourOf(e.fill, pal, alpha * 0.3); cx.fill(); } }
          cx.stroke();
          break;
        }
        case "rect": {
          const w = num(e.w, 20) * ux, hh = num(e.h, 20) * uy;
          const sw = num(e.sweep, 0, 0, 100) * u * Math.sin(age * Math.PI * 2 * speed);
          cx.translate(x + sw, y);
          if (e.fill !== undefined) { cx.fillStyle = colourOf(e.fill, pal, alpha * 0.5); cx.fillRect(0, 0, w * breathe, hh * breathe); }
          cx.strokeRect(0, 0, w * breathe, hh * breathe);
          break;
        }
        case "text": case "counter": {
          const size = Math.max(6, num(e.size, 5, 1, 60) * 4 * (u / 4));
          cx.font = `${shape === "counter" ? 200 : 400} ${size}px ${getComputedStyle(el).fontFamily || "sans-serif"}`;
          cx.textBaseline = "middle";
          const anchor = str(e.anchor, "middle", 8);
          cx.textAlign = anchor === "start" ? "left" : anchor === "end" ? "right" : "center";
          cx.fillStyle = colourOf(e.color, pal, alpha);
          cx.shadowColor = colourOf(e.color, pal, 0.8); cx.shadowBlur = 10;
          let text = str(e.text, "", 80);
          if (shape === "counter") {
            const target = typeof bv === "number" ? bv : num(e.value, 0);
            const from = num(e.from, 0);
            const k = Math.min(1, age / 1.6), ease = 1 - Math.pow(1 - k, 3);
            const v = from + (target - from) * ease;
            const dec = Math.round(num(e.decimals, 0, 0, 4));
            text = `${v.toLocaleString(undefined, { minimumFractionDigits: dec, maximumFractionDigits: dec })}${str(e.unit, "", 10)}`;
          } else if (typeof bv === "string") text = bv.slice(0, 80);
          cx.fillText(text, x, y);
          const label = str(e.label, "", 40);
          if (label) {
            cx.font = `400 ${Math.max(7, size * 0.38)}px sans-serif`; cx.shadowBlur = 0;
            cx.fillStyle = colourOf("dim", pal, 1); cx.fillText(label.toUpperCase(), x, y + size * 0.85);
          }
          break;
        }
        case "grid": {
          const w = num(e.w, 40) * ux, hh = num(e.h, 40) * uy, n = Math.round(num(e.n, 6, 1, 30));
          cx.globalAlpha = 0.55; cx.translate(x, y);
          for (let i = 0; i <= n; i++) {
            cx.beginPath(); cx.moveTo((w * i) / n, 0); cx.lineTo((w * i) / n, hh); cx.stroke();
            cx.beginPath(); cx.moveTo(0, (hh * i) / n); cx.lineTo(w, (hh * i) / n); cx.stroke();
          }
          break;
        }
        case "particles": {
          const r = num(e.r, 30, 1, 100) * u;
          const count = Math.min(Math.round(num(e.count, 60, 1, 300)), Math.max(0, MAX_PARTICLES - particles));
          particles += count;
          const seed = num(e.seed, 1), size = Math.max(0.6, num(e.size, 0.7, 0.1, 5) * u);
          for (let i = 0; i < count; i++) {
            const ang = rnd(seed, i) * Math.PI * 2 + age * (0.05 + rnd(seed, i + 900) * 0.25) * speed;
            const rr = Math.sqrt(rnd(seed, i + 400)) * r;
            const tw = 0.4 + 0.6 * Math.abs(Math.sin(age * (0.6 + rnd(seed, i + 77)) * 2 + i));
            cx.fillStyle = colourOf(e.color, pal, alpha * tw);
            cx.beginPath(); cx.arc(x + Math.cos(ang) * rr, y + Math.sin(ang) * rr, size, 0, Math.PI * 2); cx.fill();
          }
          break;
        }
        case "bars": {
          const w = num(e.w, 40) * ux, hh = num(e.h, 20) * uy;
          const raw = (Array.isArray(bv) ? bv : arr(e.values, 60)).map((v) => num(v, 0));
          const n = Math.round(num(e.n, raw.length || 12, 1, 60));
          const mx = Math.max(1e-9, ...raw.map(Math.abs));
          const bw = (w / n) * 0.62;
          cx.translate(x, y);
          for (let i = 0; i < n; i++) {
            const base = raw.length ? Math.abs(raw[i % raw.length]) / mx : 0.3 + 0.7 * Math.abs(Math.sin(age * 2 * speed + i * 0.7));
            const live = num(e.pulse, 0, 0, 1) ? base * (0.75 + 0.25 * Math.sin(age * 3 * speed + i)) : base;
            const bh = Math.max(1, live * hh);
            cx.fillStyle = colourOf(e.color, pal, alpha * 0.9);
            cx.fillRect((w * i) / n, hh - bh, bw, bh);
          }
          break;
        }
        case "ticks": {
          const r0 = num(e.r, 30, 0, 200) * u, r1 = num(e.r2, num(e.r, 30) + 4, 0, 220) * u, n = Math.round(num(e.n, 60, 2, 240));
          cx.translate(x, y); cx.rotate((spin * Math.PI) / 180);
          for (let i = 0; i < n; i++) {
            const a = (i / n) * Math.PI * 2;
            const long = i % 5 === 0;
            cx.beginPath();
            cx.moveTo(Math.cos(a) * r0, Math.sin(a) * r0);
            cx.lineTo(Math.cos(a) * (long ? r1 + (r1 - r0) * 0.5 : r1), Math.sin(a) * (long ? r1 + (r1 - r0) * 0.5 : r1));
            cx.stroke();
          }
          break;
        }
        case "pulse": {
          const r = num(e.r, 30, 1, 200) * u;
          for (let k = 0; k < 3; k++) {
            const f = (age * speed * 0.5 + k / 3) % 1;
            cx.strokeStyle = colourOf(e.color, pal, alpha * (1 - f));
            cx.beginPath(); cx.arc(x, y, r * f, 0, Math.PI * 2); cx.stroke();
          }
          break;
        }
        default: break;
      }
      cx.restore();
    }
  };

  return {
    el,
    update: apply,
    tick(t) { frame(t); },
    busy: () => false,
    destroy() {},
  };
}

export const SCENE: Record<string, KindSpec> = {
  scene: { design: [BOX, BOX], inner: 1.72, around: [1.15, 1.15], panel: false, make: makeScene },
};
