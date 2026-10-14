/**
 * Dials: widgets that sit inside the orb. Each draws its own round shape in a
 * 400 x 400 box (the manager scales it to the orb).
 */
import type { KindSpec, Props, WidgetEnv, WidgetInstance } from "../types";
import { arr, ease, fmt, h, num, obj, s, str } from "../dom";

const WARN = "#ffb43f";
const CRIT = "#ff5a5a";

const polar = (r: number, deg: number): [number, number] => {
  const a = ((deg - 90) * Math.PI) / 180;
  return [200 + r * Math.cos(a), 200 + r * Math.sin(a)];
};

function arcPath(r: number, a0: number, a1: number): string {
  const [x0, y0] = polar(r, a0);
  const [x1, y1] = polar(r, a1);
  const large = Math.abs(a1 - a0) > 180 ? 1 : 0;
  return `M ${x0.toFixed(2)} ${y0.toFixed(2)} A ${r} ${r} 0 ${large} 1 ${x1.toFixed(2)} ${y1.toFixed(2)}`;
}

function shell(): { el: HTMLElement; svg: SVGSVGElement } {
  const el = h("div", "wd");
  const svg = s("svg", { viewBox: "0 0 400 400", width: 400, height: 400, class: "wd-svg" });
  el.appendChild(svg);
  return { el, svg };
}

function ticks(svg: SVGElement, r0: number, r1: number, n: number, cls: string, every = 0, r2 = r1 + 8): SVGGElement {
  const g = s("g", { class: cls }, svg);
  for (let i = 0; i < n; i++) {
    const a = (i / n) * 360;
    const major = every > 0 && i % every === 0;
    const [x0, y0] = polar(r0, a);
    const [x1, y1] = polar(major ? r2 : r1, a);
    s("line", { x1: x0, y1: y0, x2: x1, y2: y1, class: major ? "tk-major" : "tk" }, g);
  }
  return g;
}

/** Text overlay centred in the dial. */
function label(el: HTMLElement, cls: string, top: number, size: number, text = ""): HTMLElement {
  const d = h("div", `wd-t ${cls}`, text);
  d.style.top = `${top}px`;
  d.style.fontSize = `${size}px`;
  el.appendChild(d);
  return d;
}

// ── clock ───────────────────────────────────────────────────────────────

function zoneFormatter(tz: string, hour12: boolean): { parts(d: Date): { h: string; m: string; s: string; ap: string; day: string } } {
  let zone: string | undefined = tz || undefined;
  try {
    new Intl.DateTimeFormat("en-US", { timeZone: zone });
  } catch {
    zone = undefined;
  }
  const f = new Intl.DateTimeFormat("en-US", {
    timeZone: zone, hour: "numeric", minute: "2-digit", second: "2-digit", hour12,
    weekday: "short", month: "short", day: "numeric",
  });
  return {
    parts(d: Date) {
      const o: Record<string, string> = {};
      for (const p of f.formatToParts(d)) o[p.type] = p.value;
      return {
        h: o.hour ?? "", m: o.minute ?? "", s: o.second ?? "",
        ap: (o.dayPeriod ?? "").toUpperCase(),
        day: `${o.weekday ?? ""} ${o.month ?? ""} ${o.day ?? ""}`.trim().toUpperCase(),
      };
    },
  };
}

function makeClock(props: Props, _env: WidgetEnv): WidgetInstance {
  const { el, svg } = shell();
  s("circle", { cx: 200, cy: 200, r: 188, class: "ln-soft" }, svg);
  s("circle", { cx: 200, cy: 200, r: 176, class: "ln" }, svg);
  const spinA = s("g", { class: "spin-a" }, svg);
  s("circle", { cx: 200, cy: 200, r: 166, class: "ln-dash" }, spinA);
  ticks(svg, 150, 156, 60, "ticks", 5, 162);
  const secArc = s("path", { class: "arc-bright" }, svg);
  const scan = s("line", { x1: 40, x2: 360, y1: 200, y2: 200, class: "ln-scan" }, svg);
  const timeEl = label(el, "wd-big", 150, 78);
  const apEl = label(el, "wd-sm", 150, 20);
  apEl.style.transform = "translate(88px, 8px)";
  const secEl = label(el, "wd-sm", 188, 20);
  secEl.style.transform = "translate(88px, 22px)";
  const dayEl = label(el, "wd-cap", 262, 15);
  const nameEl = label(el, "wd-cap dim", 286, 13);
  let fmtr = zoneFormatter(str(props.tz), props.hour12 !== false);
  let showSeconds = props.seconds !== false;
  let lastSec = -1;
  nameEl.textContent = str(props.label) || (str(props.tz).split("/").pop() ?? "").replace(/_/g, " ").toUpperCase();
  return {
    el,
    update(p) {
      fmtr = zoneFormatter(str(p.tz), p.hour12 !== false);
      showSeconds = p.seconds !== false;
      nameEl.textContent = str(p.label) || (str(p.tz).split("/").pop() ?? "").replace(/_/g, " ").toUpperCase();
      lastSec = -1;
    },
    tick(t) {
      const now = new Date();
      const sec = now.getSeconds() + now.getMilliseconds() / 1000;
      secArc.setAttribute("d", arcPath(176, 0, Math.max(0.5, (sec / 60) * 360)));
      scan.setAttribute("y1", String(200 + Math.sin(t * 0.9) * 80));
      scan.setAttribute("y2", String(200 + Math.sin(t * 0.9) * 80));
      spinA.setAttribute("transform", `rotate(${(t * 6) % 360} 200 200)`);
      const whole = Math.floor(sec);
      if (whole === lastSec) return;
      lastSec = whole;
      const p = fmtr.parts(now);
      timeEl.textContent = `${p.h}:${p.m}`;
      apEl.textContent = p.ap;
      secEl.textContent = showSeconds ? p.s : "";
      dayEl.textContent = p.day;
    },
    destroy() {},
  };
}

// ── timer ───────────────────────────────────────────────────────────────

function makeTimer(props: Props, env: WidgetEnv): WidgetInstance {
  const { el, svg } = shell();
  s("circle", { cx: 200, cy: 200, r: 176, class: "ln-soft" }, svg);
  s("path", { d: arcPath(176, 0, 359.9), class: "arc-dim" }, svg);
  const arc = s("path", { class: "arc-bright" }, svg);
  ticks(svg, 150, 156, 60, "ticks", 5, 162);
  const timeEl = label(el, "wd-big", 148, 84);
  const nameEl = label(el, "wd-cap", 262, 16);
  const subEl = label(el, "wd-cap dim", 288, 13);
  let total = num(props.seconds, 60, 1, 86400);
  let up = props.mode === "up";
  let startedAt = performance.now();
  let paused = props.running === false;
  let pausedAt = startedAt;
  let doneFired = false;
  nameEl.textContent = (str(props.label) || (up ? "stopwatch" : "timer")).toUpperCase();
  const elapsed = () => ((paused ? pausedAt : performance.now()) - startedAt) / 1000;
  return {
    el,
    update(p) {
      const sec = num(p.seconds, total, 1, 86400);
      if (sec !== total || (p.mode === "up") !== up) { total = sec; up = p.mode === "up"; startedAt = performance.now(); doneFired = false; }
      const run = p.running !== false;
      if (run && paused) { startedAt += performance.now() - pausedAt; paused = false; }
      if (!run && !paused) { paused = true; pausedAt = performance.now(); }
      nameEl.textContent = (str(p.label) || (up ? "stopwatch" : "timer")).toUpperCase();
    },
    tick() {
      const e = elapsed();
      const left = up ? e : Math.max(0, total - e);
      const m = Math.floor(left / 60), sec = Math.floor(left % 60);
      timeEl.textContent = `${m}:${String(sec).padStart(2, "0")}`;
      const frac = up ? (e % 60) / 60 : left / total;
      arc.setAttribute("d", arcPath(176, 0, Math.max(0.5, frac * 359.9)));
      const done = !up && left <= 0;
      el.classList.toggle("done", done);
      subEl.textContent = done ? "TIME" : paused ? "PAUSED" : "";
      if (done && !doneFired) { doneFired = true; env.action("timer_done"); }
    },
    destroy() {},
  };
}

// ── gauge ───────────────────────────────────────────────────────────────

function makeGauge(props: Props, _env: WidgetEnv): WidgetInstance {
  const { el, svg } = shell();
  const A0 = -135, A1 = 135;
  s("path", { d: arcPath(168, A0, A1), class: "arc-dim thick" }, svg);
  const arc = s("path", { class: "arc-bright thick" }, svg);
  ticks(svg, 140, 146, 55, "ticks", 5, 154);
  const needle = s("line", { x1: 200, y1: 200, x2: 200, y2: 60, class: "needle" }, svg);
  s("circle", { cx: 200, cy: 200, r: 7, class: "dot" }, svg);
  const valEl = label(el, "wd-big", 232, 64);
  const unitEl = label(el, "wd-sm", 300, 20);
  const nameEl = label(el, "wd-cap", 330, 16);
  let min = 0, max = 100, target = 0, shown = 0, unit = "", dec = 0;
  let warn = Infinity, crit = Infinity;
  const apply = (p: Props) => {
    min = num(p.min, 0); max = num(p.max, 100); if (max <= min) max = min + 1;
    target = num(p.value, min); unit = str(p.unit, "", 12);
    const th = arr(p.thresholds).map((x) => num(x, NaN)).filter((x) => !Number.isNaN(x));
    warn = th[0] ?? Infinity; crit = th[1] ?? Infinity;
    dec = Math.abs(target) < 10 && target % 1 !== 0 ? 1 : 0;
    nameEl.textContent = str(p.label).toUpperCase();
    unitEl.textContent = unit;
  };
  apply(props);
  shown = min;
  return {
    el,
    update: apply,
    tick(_t, dt) {
      shown = ease(shown, target, 5, dt);
      const f = Math.max(0, Math.min(1, (shown - min) / (max - min)));
      const a = A0 + f * (A1 - A0);
      arc.setAttribute("d", arcPath(168, A0, Math.max(A0 + 0.5, a)));
      needle.setAttribute("transform", `rotate(${a} 200 200)`);
      valEl.textContent = fmt(shown, dec);
      const col = shown >= crit ? CRIT : shown >= warn ? WARN : "";
      el.style.setProperty("--alert", col);
      el.classList.toggle("alert", col !== "");
    },
    destroy() {},
  };
}

// ── progress ring ───────────────────────────────────────────────────────

function makeProgress(props: Props, _env: WidgetEnv): WidgetInstance {
  const { el, svg } = shell();
  s("circle", { cx: 200, cy: 200, r: 182, class: "ln-soft" }, svg);
  s("path", { d: arcPath(160, 0, 359.9), class: "arc-dim thick" }, svg);
  const arc = s("path", { class: "arc-bright thick" }, svg);
  const head = s("circle", { r: 7, class: "dot" }, svg);
  ticks(svg, 128, 133, 72, "ticks");
  const valEl = label(el, "wd-big", 150, 80);
  const nameEl = label(el, "wd-cap", 252, 17);
  const subEl = label(el, "wd-cap dim", 280, 14);
  let target = 0, shown = 0;
  const apply = (p: Props) => {
    target = Math.max(0, Math.min(1, num(p.frac, p.value !== undefined ? num(p.value, 0) / 100 : 0)));
    nameEl.textContent = str(p.label).toUpperCase();
    subEl.textContent = str(p.sub).toUpperCase();
  };
  apply(props);
  return {
    el,
    update: apply,
    tick(_t, dt) {
      shown = ease(shown, target, 4, dt);
      arc.setAttribute("d", arcPath(160, 0, Math.max(0.5, shown * 359.9)));
      const [x, y] = polar(160, shown * 359.9);
      head.setAttribute("cx", String(x)); head.setAttribute("cy", String(y));
      valEl.textContent = `${Math.round(shown * 100)}%`;
    },
    busy: () => target > 0 && target < 1,
    destroy() {},
  };
}

// ── power: a segmented readout ──────────────────────────────────────────

function makePower(props: Props, _env: WidgetEnv): WidgetInstance {
  const { el, svg } = shell();
  const N = 64;
  s("circle", { cx: 200, cy: 200, r: 186, class: "ln-soft" }, svg);
  const segs: SVGElement[] = [];
  for (let i = 0; i < N; i++) {
    const a = (i / N) * 360, b = a + (360 / N) * 0.62;
    segs.push(s("path", { d: arcPath(162, a, b), class: "seg" }, svg));
  }
  s("circle", { cx: 200, cy: 200, r: 134, class: "ln" }, svg);
  s("circle", { cx: 200, cy: 200, r: 126, class: "ln-dash" }, svg);
  const valEl = label(el, "wd-big", 156, 66);
  const nameEl = label(el, "wd-cap", 240, 17);
  const subEl = label(el, "wd-cap dim", 268, 14);
  let target = 0, shown = 0;
  const apply = (p: Props) => {
    target = Math.max(0, Math.min(100, num(p.value, 0)));
    nameEl.textContent = (str(p.label) || "power level").toUpperCase();
    subEl.textContent = str(p.sub).toUpperCase();
  };
  apply(props);
  return {
    el,
    update: apply,
    tick(t, dt) {
      shown = ease(shown, target, 3, dt);
      const lit = (shown / 100) * N;
      for (let i = 0; i < N; i++) {
        const on = i < Math.floor(lit) ? 1 : i < lit ? lit - i : 0;
        const flick = on === 1 && i > lit - 4 ? 0.75 + 0.25 * Math.sin(t * 9 + i) : 1;
        segs[i].style.opacity = String(0.12 + 0.88 * on * flick);
      }
      valEl.textContent = `${fmt(shown, 1)}%`;
    },
    busy: () => target < 100,
    destroy() {},
  };
}

// ── radar ───────────────────────────────────────────────────────────────

function makeRadar(props: Props, _env: WidgetEnv): WidgetInstance {
  const { el, svg } = shell();
  for (const r of [60, 110, 160, 188]) s("circle", { cx: 200, cy: 200, r, class: r === 188 ? "ln" : "ln-soft" }, svg);
  s("line", { x1: 200, y1: 12, x2: 200, y2: 388, class: "ln-soft" }, svg);
  s("line", { x1: 12, y1: 200, x2: 388, y2: 200, class: "ln-soft" }, svg);
  ticks(svg, 188, 193, 72, "ticks", 6, 198);
  const sweep = s("g", {}, svg);
  for (let i = 0; i < 24; i++) {
    s("path", { d: `${arcPath(188, -i * 1.6, -i * 1.6 + 1.7)} L 200 200 Z`, class: "sweep", style: `opacity:${(0.5 * (1 - i / 24)).toFixed(3)}` }, sweep);
  }
  s("line", { x1: 200, y1: 200, x2: 200, y2: 12, class: "ln-bright" }, sweep);
  const blipLayer = s("g", {}, svg);
  const nameEl = label(el, "wd-cap", 366, 14);
  interface Blip { a: number; d: number; text: string; g: SVGGElement }
  let blips: Blip[] = [];
  const apply = (p: Props) => {
    blipLayer.replaceChildren();
    blips = arr(p.blips, 24).map((b) => {
      const o = obj(b);
      const g = s("g", {}, blipLayer);
      const a = num(o.angle, 0), d = Math.max(0, Math.min(1, num(o.dist, 0.5)));
      const [x, y] = polar(188 * d, a);
      s("circle", { cx: x, cy: y, r: 5, class: "dot" }, g);
      s("circle", { cx: x, cy: y, r: 11, class: "ln-soft" }, g);
      const tx = s("text", { x: x + 14, y: y + 4, class: "svg-t" }, g);
      tx.textContent = str(o.label, "", 24).toUpperCase();
      return { a: ((a % 360) + 360) % 360, d, text: "", g };
    });
    nameEl.textContent = str(p.label).toUpperCase();
  };
  apply(props);
  return {
    el,
    update: apply,
    tick(t) {
      const ang = (t * 70) % 360;
      sweep.setAttribute("transform", `rotate(${ang} 200 200)`);
      for (const b of blips) {
        const since = (ang - b.a + 360) % 360;
        b.g.style.opacity = String(0.25 + 0.75 * Math.max(0, 1 - since / 300));
      }
    },
    destroy() {},
  };
}

// ── connect: two linked rings, as the reference shows while she works ───

function makeConnect(props: Props, _env: WidgetEnv): WidgetInstance {
  const { el, svg } = shell();
  const a = s("g", {}, svg), b = s("g", {}, svg);
  for (const [g, cx] of [[a, 112], [b, 288]] as const) {
    s("circle", { cx, cy: 190, r: 68, class: "ln thick glowy" }, g);
    s("circle", { cx, cy: 190, r: 50, class: "ln-soft" }, g);
    s("circle", { cx, cy: 190, r: 8, class: "dot" }, g);
  }
  s("line", { x1: 180, y1: 190, x2: 220, y2: 190, class: "ln thick glowy" }, svg);
  const pulse = s("circle", { cy: 190, r: 4, class: "dot" }, svg);
  // arcPath draws around (200, 200); each orbit group moves that centre onto its ring.
  const orbitA = s("g", {}, svg);
  const orbitB = s("g", {}, svg);
  s("path", { d: arcPath(78, 0, 90), class: "arc-bright" }, orbitA);
  s("path", { d: arcPath(78, 0, 90), class: "arc-bright" }, orbitB);
  const nameEl = label(el, "wd-cap", 300, 17);
  const subEl = label(el, "wd-cap dim", 328, 14);
  const apply = (p: Props) => {
    nameEl.textContent = (str(p.label) || "connecting").toUpperCase();
    subEl.textContent = str(p.sub).toUpperCase();
  };
  apply(props);
  let done = props.done === true;
  return {
    el,
    update(p) { apply(p); done = p.done === true; },
    tick(t) {
      a.setAttribute("transform", `translate(0 ${Math.sin(t * 1.3) * 3})`);
      b.setAttribute("transform", `translate(0 ${Math.sin(t * 1.3 + 2) * 3})`);
      orbitA.setAttribute("transform", `translate(-88 -10) rotate(${t * 130} 200 200)`);
      orbitB.setAttribute("transform", `translate(88 -10) rotate(${-t * 110} 200 200)`);
      pulse.setAttribute("cx", String(112 + 176 * (0.5 + 0.5 * Math.sin(t * 2.4))));
    },
    busy: () => !done,
    destroy() {},
  };
}

// ── waveform ────────────────────────────────────────────────────────────

function makeWave(props: Props, _env: WidgetEnv): WidgetInstance {
  const { el, svg } = shell();
  const N = 72;
  s("circle", { cx: 200, cy: 200, r: 96, class: "ln" }, svg);
  const bars: SVGElement[] = [];
  for (let i = 0; i < N; i++) bars.push(s("line", { class: "bar" }, svg));
  const nameEl = label(el, "wd-cap", 190, 16, str(props.label).toUpperCase());
  let level = num(props.level, 0.6, 0, 1), lv = level;
  return {
    el,
    update(p) { level = num(p.level, level, 0, 1); nameEl.textContent = str(p.label).toUpperCase(); },
    tick(t, dt) {
      lv = ease(lv, level, 4, dt);
      for (let i = 0; i < N; i++) {
        const a = (i / N) * 360;
        const m = 0.5 + 0.5 * Math.sin(t * 3.1 + i * 0.55) * Math.sin(t * 1.7 + i * 0.21);
        const len = 6 + (0.25 + 0.75 * lv) * 70 * m;
        const [x0, y0] = polar(108, a), [x1, y1] = polar(108 + len, a);
        bars[i].setAttribute("x1", String(x0)); bars[i].setAttribute("y1", String(y0));
        bars[i].setAttribute("x2", String(x1)); bars[i].setAttribute("y2", String(y1));
      }
    },
    busy: () => lv > 0.3,
    destroy() {},
  };
}

const round = (inner: number): Pick<KindSpec, "design" | "inner" | "around" | "panel"> => ({
  design: [400, 400], inner, around: [1.15, 1.15], panel: false,
});

export const DIALS: Record<string, KindSpec> = {
  clock: { ...round(1.72), make: makeClock },
  timer: { ...round(1.6), make: makeTimer },
  gauge: { ...round(1.5), make: makeGauge },
  progress: { ...round(1.55), make: makeProgress },
  power: { ...round(1.72), make: makePower },
  radar: { ...round(1.72), make: makeRadar },
  connect: { design: [400, 400], inner: 1.5, around: [1.1, 1.1], panel: false, make: makeConnect },
  waveform: { ...round(1.5), make: makeWave },
};
