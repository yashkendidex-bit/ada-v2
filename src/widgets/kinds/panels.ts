/**
 * Panels: rectangular widgets that sit around the orb (and two small ones that
 * sit inside it). Text only ever goes in through textContent.
 */
import type { KindSpec, Props, WidgetEnv, WidgetInstance } from "../types";
import { arr, ease, fmt, h, num, obj, s, str } from "../dom";

interface Panel { el: HTMLElement; head: HTMLElement; body: HTMLElement }

function panel(cls: string, title: string): Panel {
  const el = h("div", `wp ${cls}`);
  const head = h("div", "wp-head", title.toUpperCase());
  const body = h("div", "wp-body");
  el.append(head, body);
  return { el, head, body };
}

const STATES = ["todo", "doing", "done", "fail", "ok", "warn", "bad", "busy"];
const state = (v: unknown, def: string): string => {
  const x = str(v, def, 10).toLowerCase();
  return STATES.includes(x) ? x : def;
};

// ── tile: one number, large ─────────────────────────────────────────────

function makeTile(props: Props, _env: WidgetEnv): WidgetInstance {
  const el = h("div", "wp wp-tile");
  const cap = h("div", "wp-cap");
  const val = h("div", "val");
  const num_ = h("span");
  const unit = h("span", "unit");
  const trend = h("span", "trend");
  val.append(num_, unit, trend);
  const sub = h("div", "wp-sub");
  el.append(cap, val, sub);
  let target = 0, shown = 0, numeric = true, text = "", dec = 0;
  const apply = (p: Props) => {
    cap.textContent = str(p.label, "", 40).toUpperCase();
    sub.textContent = str(p.sub, "", 80);
    unit.textContent = str(p.unit, "", 12);
    const raw = p.value;
    numeric = typeof raw === "number" || (typeof raw === "string" && raw.trim() !== "" && Number.isFinite(Number(raw)));
    if (numeric) { target = num(raw, 0); dec = Number.isInteger(target) ? 0 : Math.min(2, String(target).split(".")[1]?.length ?? 1); }
    else text = str(raw, "", 24);
    const tr = str(p.trend, "", 5);
    trend.className = `trend ${tr}`;
    trend.textContent = tr === "up" ? "▲" : tr === "down" ? "▼" : tr === "flat" ? "▬" : "";
  };
  apply(props);
  shown = 0;
  return {
    el,
    update: apply,
    tick(_t, dt) {
      if (!numeric) { num_.textContent = text; return; }
      shown = ease(shown, target, 6, dt);
      if (Math.abs(shown - target) < 0.005 * (Math.abs(target) + 1)) shown = target;
      num_.textContent = fmt(shown, dec);
    },
    destroy() {},
  };
}

// ── notification ────────────────────────────────────────────────────────

function makeNotification(props: Props, _env: WidgetEnv): WidgetInstance {
  const el = h("div", "wp wp-note");
  const ico = h("div", "ico");
  const col = h("div");
  const tt = h("div", "tt");
  const bd = h("div", "bd");
  col.append(tt, bd);
  el.append(ico, col);
  const apply = (p: Props) => { tt.textContent = str(p.title, "", 80); bd.textContent = str(p.body, "", 300); };
  apply(props);
  return { el, update: apply, destroy() {} };
}

// ── list ────────────────────────────────────────────────────────────────

function makeList(props: Props, _env: WidgetEnv): WidgetInstance {
  const p = panel("", str(props.title));
  const rows = h("div", "rows");
  p.body.appendChild(rows);
  let seen = 0;
  const apply = (pr: Props) => {
    p.head.textContent = str(pr.title, "", 60).toUpperCase();
    const items = arr(pr.items, 9);
    rows.replaceChildren();
    items.forEach((it, i) => {
      const o = typeof it === "string" ? { text: it } : obj(it);
      const row = h("div", "row");
      if (i >= seen) { row.classList.add("enter"); row.style.animationDelay = `${(i - seen) * 70}ms`; }
      const st = o.state !== undefined ? state(o.state, "") : "";
      if (st) row.appendChild(h("div", `mk ${st}`));
      else row.appendChild(h("div", "mk"));
      const tx = h("div", "tx");
      tx.appendChild(h("div", st === "todo" ? "todo" : "", str(o.text, "", 120)));
      const sb = str(o.sub, "", 100);
      if (sb) tx.appendChild(h("div", "sb", sb));
      row.appendChild(tx);
      rows.appendChild(row);
    });
    seen = items.length;
  };
  apply(props);
  return { el: p.el, update: apply, destroy() {} };
}

// ── steps ───────────────────────────────────────────────────────────────

function makeSteps(props: Props, _env: WidgetEnv): WidgetInstance {
  const p = panel("", str(props.title, "Progress"));
  const rows = h("div", "steps");
  p.body.appendChild(rows);
  const apply = (pr: Props) => {
    p.head.textContent = str(pr.title, "Progress", 60).toUpperCase();
    rows.replaceChildren();
    arr(pr.steps, 10).forEach((it) => {
      const o = typeof it === "string" ? { label: it } : obj(it);
      const st = state(o.state, "todo");
      const row = h("div", "row");
      row.appendChild(h("div", `mk ${st}`));
      const tx = h("div", `tx ${st === "todo" ? "todo" : ""}`);
      tx.textContent = str(o.label, "", 100);
      row.appendChild(tx);
      rows.appendChild(row);
    });
  };
  apply(props);
  return { el: p.el, update: apply, busy: () => !!rows.querySelector(".mk.doing"), destroy() {} };
}

// ── status ──────────────────────────────────────────────────────────────

function makeStatus(props: Props, _env: WidgetEnv): WidgetInstance {
  const p = panel("", str(props.title, "Status"));
  const rows = h("div", "rows");
  p.body.appendChild(rows);
  const apply = (pr: Props) => {
    p.head.textContent = str(pr.title, "Status", 60).toUpperCase();
    rows.replaceChildren();
    arr(pr.items, 10).forEach((it) => {
      const o = obj(it);
      const st = state(o.state, "ok");
      const row = h("div", "row");
      row.appendChild(h("div", `mk ${st}`));
      row.appendChild(h("div", "tx", str(o.label, "", 80)));
      const v = str(o.value, "", 30);
      if (v) row.appendChild(h("div", "vl", v));
      rows.appendChild(row);
    });
  };
  apply(props);
  return { el: p.el, update: apply, busy: () => !!rows.querySelector(".mk.busy"), destroy() {} };
}

// ── table ───────────────────────────────────────────────────────────────

function makeTable(props: Props, _env: WidgetEnv): WidgetInstance {
  const p = panel("wide", str(props.title));
  const tbl = h("table", "tbl");
  p.body.appendChild(tbl);
  const apply = (pr: Props) => {
    p.head.textContent = str(pr.title, "", 60).toUpperCase();
    tbl.replaceChildren();
    const cols = arr(pr.columns, 8).map((c) => str(c, "", 24));
    const rows = arr(pr.rows, 12).map((r) => arr(r, 8));
    const isNum = (ci: number) => rows.length > 0 && rows.every((r) => r[ci] === undefined || typeof r[ci] === "number" || /^[-+$€£]?[\d.,]+%?$/.test(String(r[ci])));
    if (cols.length) {
      const tr = h("tr");
      cols.forEach((c, ci) => { const th = h("th", ci > 0 && isNum(ci) ? "n" : "", c); tr.appendChild(th); });
      const thead = h("thead"); thead.appendChild(tr); tbl.appendChild(thead);
    }
    const tb = h("tbody");
    rows.forEach((r) => {
      const tr = h("tr");
      const n = Math.max(cols.length, r.length);
      for (let ci = 0; ci < n; ci++) tr.appendChild(h("td", ci > 0 && isNum(ci) ? "n" : "", str(r[ci], "", 60)));
      tb.appendChild(tr);
    });
    tbl.appendChild(tb);
  };
  apply(props);
  return { el: p.el, update: apply, destroy() {} };
}

// ── text, typed out ─────────────────────────────────────────────────────

function makeText(props: Props, env: WidgetEnv): WidgetInstance {
  const p = panel("wp-text", str(props.title));
  const bd = h("div", "bd");
  p.body.appendChild(bd);
  let full = "", shown = 0;
  const apply = (pr: Props) => {
    p.head.textContent = str(pr.title, "", 60).toUpperCase();
    const next = str(pr.body, "", 700);
    if (!next.startsWith(full.slice(0, Math.min(full.length, 12)))) shown = 0;
    full = next;
    bd.classList.toggle("big", full.length < 80);
    if (env.calm) shown = full.length;
  };
  apply(props);
  return {
    el: p.el,
    update: apply,
    tick(_t, dt) {
      if (shown >= full.length) { bd.textContent = full; bd.classList.remove("caret"); return; }
      shown = Math.min(full.length, shown + dt * 110);
      bd.textContent = full.slice(0, Math.floor(shown));
      bd.classList.add("caret");
    },
    destroy() {},
  };
}

// ── weather ─────────────────────────────────────────────────────────────

type Sky = "sun" | "cloud" | "rain" | "snow" | "storm" | "fog" | "night";

function skyOf(cond: string): Sky {
  const c = cond.toLowerCase();
  if (/thunder|storm|lightning/.test(c)) return "storm";
  if (/snow|sleet|flurr|ice/.test(c)) return "snow";
  if (/rain|shower|drizzle/.test(c)) return "rain";
  if (/fog|mist|haze|smoke/.test(c)) return "fog";
  if (/cloud|overcast/.test(c)) return "cloud";
  if (/night|moon|clear night/.test(c)) return "night";
  return "sun";
}

function weatherIcon(kind: Sky): SVGSVGElement {
  const svg = s("svg", { viewBox: "0 0 80 80", class: "ic" });
  const stroke = { fill: "none", stroke: "var(--c2)", "stroke-width": 2.2, "stroke-linecap": "round", "stroke-linejoin": "round" };
  const cloud = () => s("path", { d: "M24 56 a12 12 0 0 1 2-23.8 A16 16 0 0 1 57 36 a10 10 0 0 1-1 20 Z", ...stroke }, svg);
  if (kind === "sun") {
    const g = s("g", { class: "spin" }, svg);
    s("circle", { cx: 40, cy: 40, r: 13, ...stroke }, svg);
    for (let i = 0; i < 8; i++) {
      const a = (i / 8) * Math.PI * 2;
      s("line", { x1: 40 + Math.cos(a) * 20, y1: 40 + Math.sin(a) * 20, x2: 40 + Math.cos(a) * 29, y2: 40 + Math.sin(a) * 29, ...stroke }, g);
    }
    g.setAttribute("style", "transform-origin:40px 40px;animation:wd-spin 24s linear infinite");
  } else if (kind === "night") {
    s("path", { d: "M52 58 A24 24 0 1 1 36 16 A19 19 0 0 0 52 58 Z", ...stroke }, svg);
  } else {
    cloud();
    if (kind === "rain" || kind === "storm") {
      for (let i = 0; i < 3; i++) {
        const l = s("line", { x1: 30 + i * 10, y1: 62, x2: 27 + i * 10, y2: 72, ...stroke }, svg);
        l.setAttribute("style", `animation:wd-drop 1.1s linear ${i * 0.25}s infinite`);
      }
      if (kind === "storm") s("path", { d: "M42 52 l-6 10 h8 l-5 10", ...stroke, stroke: "#ffb43f" }, svg);
    } else if (kind === "snow") {
      for (let i = 0; i < 3; i++) {
        const c = s("circle", { cx: 30 + i * 10, cy: 66, r: 1.8, fill: "var(--c2)" }, svg);
        c.setAttribute("style", `animation:wd-drop 1.8s linear ${i * 0.4}s infinite`);
      }
    } else if (kind === "fog") {
      s("line", { x1: 20, y1: 64, x2: 60, y2: 64, ...stroke }, svg);
      s("line", { x1: 28, y1: 71, x2: 66, y2: 71, ...stroke }, svg);
    }
  }
  return svg;
}

function makeWeather(props: Props, _env: WidgetEnv): WidgetInstance {
  const p = panel("wp-wx", str(props.place, "Weather"));
  const top = h("div", "top");
  const iconBox = h("div");
  const info = h("div");
  const tp = h("div", "tp");
  const cd = h("div", "cd");
  const hl = h("div", "hl");
  info.append(tp, cd, hl);
  top.append(iconBox, info);
  const dt = h("div", "dt");
  p.body.append(top, dt);
  let skyNow: Sky | "" = "";
  let target = 0, shown = 0, unit = "°";
  const apply = (pr: Props) => {
    p.head.textContent = str(pr.place, "Weather", 40).toUpperCase();
    target = num(pr.temp, 0);
    unit = str(pr.unit, "°", 4);
    const cond = str(pr.cond, "", 40);
    cd.textContent = cond;
    const sky = skyOf(cond);
    if (sky !== skyNow) { iconBox.replaceChildren(weatherIcon(sky)); skyNow = sky; }
    const hi = pr.hi !== undefined ? `H ${fmt(num(pr.hi, 0))}${unit}` : "";
    const lo = pr.lo !== undefined ? `L ${fmt(num(pr.lo, 0))}${unit}` : "";
    hl.textContent = [hi, lo].filter(Boolean).join("   ");
    dt.replaceChildren();
    arr(pr.details, 6).forEach((d) => {
      const o = obj(d);
      const row = h("div");
      row.append(h("span", "", str(o.k, "", 20)), h("span", "", str(o.v, "", 20)));
      dt.appendChild(row);
    });
    dt.style.display = dt.childElementCount ? "grid" : "none";
  };
  apply(props);
  shown = target;
  const tNode = h("span");
  const uNode = h("small");
  tp.append(tNode, uNode);
  return {
    el: p.el,
    update: apply,
    tick(_t, dt_) {
      shown = ease(shown, target, 5, dt_);
      tNode.textContent = fmt(Math.abs(shown - target) < 0.05 ? target : shown, Number.isInteger(target) ? 0 : 1);
      uNode.textContent = unit;
    },
    destroy() {},
  };
}

// ── chart (canvas) ──────────────────────────────────────────────────────

const SERIES_COLOURS = ["var(--c)", "var(--c2)", "#ffb43f", "#a98bff", "#4be28c", "#ff5a5a"];

function makeChart(props: Props, env: WidgetEnv): WidgetInstance {
  const p = panel("wide wp-chart", str(props.title));
  const W = 384, H = 190;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cv = document.createElement("canvas");
  cv.width = W * dpr; cv.height = H * dpr; cv.style.width = `${W}px`; cv.style.height = `${H}px`;
  const cx = cv.getContext("2d")!;
  cx.scale(dpr, dpr);
  const legend = h("div", "legend");
  p.body.append(cv, legend);
  interface Series { name: string; pts: number[] }
  let series: Series[] = [];
  let labels: string[] = [];
  let kind = "line";
  let unit = "";
  let prog = env.calm ? 1 : 0;
  let dirty = true;
  const colorOf = (i: number): string => {
    const raw = SERIES_COLOURS[i % SERIES_COLOURS.length];
    if (!raw.startsWith("var(")) return raw;
    return getComputedStyle(p.el).getPropertyValue(raw.slice(4, -1)).trim() || "#3fe0d4";
  };
  const apply = (pr: Props) => {
    p.head.textContent = str(pr.title, "", 60).toUpperCase();
    kind = ["line", "bar", "area"].includes(str(pr.type)) ? str(pr.type) : "line";
    unit = str(pr.unit, "", 8);
    labels = arr(pr.labels, 60).map((l) => str(l, "", 12));
    series = arr(pr.series, 6).map((x, i) => {
      const o = obj(x);
      return { name: str(o.name, `series ${i + 1}`, 24), pts: arr(o.points, 120).map((v) => num(v, 0)) };
    }).filter((x) => x.pts.length > 0);
    legend.replaceChildren();
    if (series.length > 1) series.forEach((x, i) => {
      const it = h("span");
      const sw = h("i"); sw.style.background = colorOf(i);
      it.append(sw, document.createTextNode(x.name));
      legend.appendChild(it);
    });
    legend.style.display = series.length > 1 ? "flex" : "none";
    prog = env.calm ? 1 : 0.35;
    dirty = true;
  };
  apply(props);
  const draw = () => {
    cx.clearRect(0, 0, W, H);
    const L = 40, R = 8, T = 10, B = labels.length ? 22 : 8;
    const pw = W - L - R, ph = H - T - B;
    const all = series.flatMap((x) => x.pts);
    if (!all.length) return;
    let lo = Math.min(...all, kind === "line" ? Infinity : 0), hi = Math.max(...all);
    if (lo === hi) { hi = lo + 1; }
    const pad = (hi - lo) * 0.08; if (kind === "line") lo -= pad; hi += pad;
    const col = colorOf(0);
    cx.font = "10px sans-serif"; cx.textBaseline = "middle";
    for (let g = 0; g <= 4; g++) {
      const y = T + (ph * g) / 4;
      cx.strokeStyle = `rgba(${getComputedStyle(p.el).getPropertyValue("--rgb")}, 0.16)`;
      cx.lineWidth = 1; cx.beginPath(); cx.moveTo(L, y + 0.5); cx.lineTo(W - R, y + 0.5); cx.stroke();
      const v = hi - ((hi - lo) * g) / 4;
      cx.fillStyle = col; cx.globalAlpha = 0.7; cx.textAlign = "right";
      cx.fillText(`${fmt(v, Math.abs(hi - lo) < 10 ? 1 : 0)}${g === 0 ? unit : ""}`, L - 6, y);
      cx.globalAlpha = 1;
    }
    const n = Math.max(...series.map((x) => x.pts.length));
    const xAt = (i: number) => L + (n === 1 ? pw / 2 : (pw * i) / (n - 1));
    const yAt = (v: number) => T + ph - ((v - lo) / (hi - lo)) * ph;
    if (labels.length) {
      cx.fillStyle = col; cx.globalAlpha = 0.7; cx.textAlign = "center"; cx.textBaseline = "alphabetic";
      const step = Math.ceil(labels.length / 8);
      labels.forEach((lb, i) => { if (i % step === 0) cx.fillText(lb, kind === "bar" ? L + (pw * (i + 0.5)) / n : xAt(i), H - 6); });
      cx.globalAlpha = 1;
    }
    series.forEach((sr, si) => {
      const c = colorOf(si);
      cx.save();
      cx.beginPath(); cx.rect(L - 2, 0, (pw + R + 4) * prog, H); cx.clip();
      if (kind === "bar") {
        const bw = (pw / n) * 0.7 / series.length;
        sr.pts.forEach((v, i) => {
          const x = L + (pw * (i + 0.5)) / n - (bw * series.length) / 2 + bw * si;
          const y0 = yAt(Math.max(lo, 0)), y1 = yAt(v);
          cx.fillStyle = c; cx.globalAlpha = 0.85; cx.shadowColor = c; cx.shadowBlur = 8;
          cx.fillRect(x, Math.min(y0, y1), bw - 1, Math.abs(y1 - y0));
        });
      } else {
        cx.beginPath();
        sr.pts.forEach((v, i) => (i === 0 ? cx.moveTo(xAt(i), yAt(v)) : cx.lineTo(xAt(i), yAt(v))));
        if (kind === "area") {
          const grad = cx.createLinearGradient(0, T, 0, T + ph);
          grad.addColorStop(0, c); grad.addColorStop(1, "transparent");
          cx.save(); cx.lineTo(xAt(sr.pts.length - 1), T + ph); cx.lineTo(xAt(0), T + ph); cx.closePath();
          cx.globalAlpha = 0.28; cx.fillStyle = grad; cx.fill(); cx.restore();
          cx.beginPath();
          sr.pts.forEach((v, i) => (i === 0 ? cx.moveTo(xAt(i), yAt(v)) : cx.lineTo(xAt(i), yAt(v))));
        }
        cx.strokeStyle = c; cx.lineWidth = 2; cx.lineJoin = "round"; cx.shadowColor = c; cx.shadowBlur = 8; cx.globalAlpha = 1;
        cx.stroke();
        const lx = xAt(sr.pts.length - 1), ly = yAt(sr.pts[sr.pts.length - 1]);
        cx.fillStyle = c; cx.beginPath(); cx.arc(lx, ly, 3, 0, Math.PI * 2); cx.fill();
      }
      cx.restore();
    });
  };
  return {
    el: p.el,
    update: apply,
    tick(_t, dt) {
      if (prog < 1) { prog = Math.min(1, prog + dt * 1.1); dirty = true; }
      if (dirty) { draw(); dirty = prog < 1; }
    },
    destroy() {},
  };
}

// ── image ───────────────────────────────────────────────────────────────

const DATA_IMAGE = /^data:image\/(png|jpeg|jpg|gif|webp);base64,[A-Za-z0-9+/=]+$/;

function makeImage(props: Props, _env: WidgetEnv): WidgetInstance {
  const p = panel("wp-img", str(props.title));
  const frame = h("div");
  const cap = h("div", "cap");
  p.body.append(frame, cap);
  const apply = (pr: Props) => {
    p.head.textContent = str(pr.title, "", 60).toUpperCase();
    const src = typeof pr.src === "string" ? pr.src : "";
    frame.replaceChildren();
    if (DATA_IMAGE.test(src) && src.length < 6_000_000) {
      const img = new Image();
      img.alt = str(pr.caption, "image", 80);
      img.src = src;
      frame.appendChild(img);
    } else frame.appendChild(h("div", "none", "NO IMAGE"));
    cap.textContent = str(pr.caption, "", 160);
  };
  apply(props);
  return { el: p.el, update: apply, destroy() {} };
}

export const PANELS: Record<string, KindSpec> = {
  tile: { design: [300, 170], inner: 1.15, around: [0.9, 0.55], panel: true, make: makeTile },
  notification: { design: [360, 110], inner: 1.35, around: [1.2, 0.4], panel: true, make: makeNotification },
  list: { design: [340, 280], around: [1.1, 0.95], panel: true, make: makeList },
  steps: { design: [340, 280], around: [1.1, 0.95], panel: true, make: makeSteps },
  status: { design: [340, 260], around: [1.1, 0.9], panel: true, make: makeStatus },
  table: { design: [420, 260], around: [1.35, 0.9], panel: true, make: makeTable },
  text: { design: [340, 220], around: [1.1, 0.8], panel: true, make: makeText },
  weather: { design: [340, 240], around: [1.1, 0.8], panel: true, make: makeWeather },
  chart: { design: [420, 290], around: [1.35, 0.95], panel: true, make: makeChart },
  image: { design: [340, 280], around: [1.1, 0.95], panel: true, make: makeImage },
};
