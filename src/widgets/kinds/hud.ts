/**
 * HUD: a dashboard A.D.A composes from a fixed catalog (see backend/hud_spec.py).
 * She sends a flat map of nodes {id: {type, props, children}}. Unknown types are
 * dropped, props are read defensively, and nothing here parses HTML or runs code.
 * Nodes are kept between updates, so a live gauge animates to its new value
 * instead of starting over.
 */
import type { KindSpec, Props, WidgetEnv, WidgetInstance } from "../types";
import { arr, h, num, obj, s, str } from "../dom";
import { DIALS } from "./dial";
import { MODEL } from "./model3d";

interface Node {
  type: string;
  el: HTMLElement;
  /** Where children go, for containers. */
  host?: HTMLElement;
  update(props: Props, env: WidgetEnv): void;
  tick?(t: number, dt: number): void;
  busy?(): boolean;
  destroy?(): void;
  inner?: WidgetInstance;
}

const KNOWN = new Set([
  "HudRoot", "Row", "Column", "Card", "Gauge", "ProgressRing", "Sparkline", "Timeline",
  "SpecCard", "SvgCanvas", "Viewport3D", "Button", "LogFeed",
]);

const PATH_OK = /^[MmLlHhVvCcSsQqTtAaZz0-9 .,\-eE]{1,600}$/;

function scaled(inst: WidgetInstance, factor: number, box: number): HTMLElement {
  const wrap = h("div", "hud-dial");
  wrap.style.width = `${box}px`; wrap.style.height = `${box}px`;
  const sc = h("div", "scale");
  sc.style.transform = `scale(${factor})`;
  sc.appendChild(inst.el);
  wrap.appendChild(sc);
  return wrap;
}

function make(type: string, props: Props, env: WidgetEnv, act: (a: string) => void): Node {
  switch (type) {
    case "HudRoot": case "Column": {
      const el = h("div", type === "Column" ? "hud-col" : "hud");
      return { type, el, host: el, update() {} };
    }
    case "Row": {
      const el = h("div", "hud-row");
      return { type, el, host: el, update() {} };
    }
    case "Card": {
      const el = h("div", "hud-card");
      const ct = h("div", "ct");
      const host = h("div", "hud-col");
      el.append(ct, host);
      return { type, el, host, update(p) { ct.textContent = str(p.title, "", 60).toUpperCase(); ct.style.display = ct.textContent ? "" : "none"; } };
    }
    case "Gauge": case "ProgressRing": {
      const spec = DIALS[type === "Gauge" ? "gauge" : "progress"];
      const inst = spec.make(props, env);
      return { type, el: scaled(inst, 0.35, 140), inner: inst, update: (p) => inst.update(p), tick: inst.tick?.bind(inst), busy: inst.busy?.bind(inst) };
    }
    case "Viewport3D": {
      const inst = MODEL.model3d.make({ ...props, title: "" }, env);
      return { type, el: scaled(inst, 0.55, 220), inner: inst, update: (p) => inst.update({ ...p, title: "" }), tick: inst.tick?.bind(inst), destroy: () => inst.destroy() };
    }
    case "Sparkline": {
      const el = h("div");
      const svg = s("svg", { viewBox: "0 0 150 46", class: "hud-spark" });
      const line = s("polyline", { fill: "none", stroke: "var(--c)", "stroke-width": 1.8, "stroke-linejoin": "round" }, svg);
      const dotEl = s("circle", { r: 2.6, fill: "var(--c2)" }, svg);
      const cap = h("div", "wp-sub");
      el.append(svg, cap);
      return {
        type, el,
        update(p) {
          const pts = arr(p.points, 100).map((v) => num(v, NaN)).filter((v) => Number.isFinite(v));
          if (pts.length < 2) { line.setAttribute("points", ""); cap.textContent = str(p.label, "", 30); return; }
          const lo = Math.min(...pts), hi = Math.max(...pts), span = hi - lo || 1;
          const xy = pts.map((v, i) => [(i / (pts.length - 1)) * 148 + 1, 42 - ((v - lo) / span) * 36]);
          line.setAttribute("points", xy.map((q) => `${q[0].toFixed(1)},${q[1].toFixed(1)}`).join(" "));
          const last = xy[xy.length - 1];
          dotEl.setAttribute("cx", String(last[0])); dotEl.setAttribute("cy", String(last[1]));
          cap.textContent = `${str(p.label, "", 30)} ${pts[pts.length - 1]}${str(p.unit, "", 8)}`.trim();
        },
      };
    }
    case "Timeline": {
      const el = h("div", "steps");
      return {
        type, el,
        update(p) {
          el.replaceChildren();
          arr(p.steps, 12).forEach((it) => {
            const o = obj(it);
            const st = ["todo", "doing", "done", "fail"].includes(str(o.state)) ? str(o.state) : "todo";
            const row = h("div", "row");
            row.appendChild(h("div", `mk ${st}`));
            row.appendChild(h("div", `tx ${st === "todo" ? "todo" : ""}`, str(o.label, "", 80)));
            el.appendChild(row);
          });
        },
        busy: () => !!el.querySelector(".mk.doing"),
      };
    }
    case "SpecCard": {
      const el = h("div");
      return {
        type, el,
        update(p) {
          el.replaceChildren();
          arr(p.rows, 12).forEach((r) => {
            const o = obj(r);
            const kv = h("div", "hud-kv");
            kv.append(h("span", "", str(o.k, "", 40)), h("span", "", str(o.v, "", 80)));
            el.appendChild(kv);
          });
          const acts = arr(p.actions, 4);
          if (acts.length) {
            const box = h("div");
            box.style.marginTop = "8px";
            acts.forEach((a) => {
              const o = obj(a);
              const b = h("button", "hud-btn", str(o.label, "Do", 24));
              b.type = "button";
              b.addEventListener("click", (e) => { e.stopPropagation(); act(str(o.action, "", 40)); });
              box.appendChild(b);
            });
            el.appendChild(box);
          }
        },
      };
    }
    case "Button": {
      const b = h("button", "hud-btn");
      b.type = "button";
      let action = "";
      b.addEventListener("click", (e) => { e.stopPropagation(); act(action); });
      return {
        type, el: b,
        update(p) { b.textContent = str(p.label, "Do", 24); action = str(p.action, "", 40); b.classList.toggle("danger", str(p.variant) === "danger"); },
      };
    }
    case "LogFeed": {
      const el = h("div", "hud-log");
      return { type, el, update(p) { el.replaceChildren(...arr(p.lines, 30).slice(-6).map((l) => h("div", "", `› ${str(l, "", 120)}`))); } };
    }
    case "SvgCanvas": {
      const svg = s("svg", { class: "hud-svg" });
      svg.style.cssText = "display:block;max-width:100%;height:auto";
      return {
        type, el: svg as unknown as HTMLElement,
        update(p) {
          const w = num(p.width, 200, 20, 600), hh = num(p.height, 120, 20, 400);
          svg.setAttribute("viewBox", `0 0 ${w} ${hh}`); svg.setAttribute("width", String(w)); svg.setAttribute("height", String(hh));
          svg.replaceChildren();
          arr(p.elements, 60).forEach((raw) => {
            const o = obj(raw);
            const common = { stroke: "var(--c)", "stroke-width": num(o.stroke, 1.4, 0, 12), fill: o.fill ? "var(--cd)" : "none" };
            switch (str(o.shape)) {
              case "circle": s("circle", { cx: num(o.cx, 0), cy: num(o.cy, 0), r: num(o.r, 5, 0, 500), ...common }, svg); break;
              case "rect": s("rect", { x: num(o.x, 0), y: num(o.y, 0), width: num(o.width ?? o.w, 10, 0, 600), height: num(o.height ?? o.h, 10, 0, 600), ...common }, svg); break;
              case "line": s("line", { x1: num(o.x1, 0), y1: num(o.y1, 0), x2: num(o.x2, 0), y2: num(o.y2, 0), ...common }, svg); break;
              case "path": { const d = str(o.d, "", 600); if (PATH_OK.test(d)) s("path", { d, ...common }, svg); break; }
              case "text": { const tx = s("text", { x: num(o.x, 0), y: num(o.y, 0), fill: "var(--c2)", "font-size": num(o.size, 11, 6, 60), "font-family": "var(--wd-font)" }, svg); tx.textContent = str(o.text, "", 60); break; }
              default: break;
            }
          });
        },
      };
    }
    default: {
      const el = h("div");
      return { type, el, update() {} };
    }
  }
}

function makeHud(props: Props, env: WidgetEnv): WidgetInstance {
  const el = h("div", "wp xwide");
  const head = h("div", "wp-head");
  const body = h("div", "wp-body");
  const skeleton = h("div", "hud-skel");
  el.append(head, body);
  const live = new Map<string, Node>();
  let rootId = "";
  let loading = true;

  const build = (p: Props) => {
    head.textContent = str(p.title, "", 60).toUpperCase();
    const nodes = obj(p.nodes);
    rootId = str(p.root, "", 64);
    if (!nodes[rootId]) rootId = Object.keys(nodes).find((k) => obj(nodes[k]).type === "HudRoot") ?? Object.keys(nodes)[0] ?? "";
    loading = p.loading === true && !rootId;

    // Drop nodes that went away or changed type; make the new ones.
    for (const [id, n] of live) {
      const spec = obj(nodes[id]);
      if (!spec.type || spec.type !== n.type) { n.destroy?.(); n.el.remove(); live.delete(id); }
    }
    for (const id of Object.keys(nodes).slice(0, 200)) {
      const spec = obj(nodes[id]);
      const type = str(spec.type, "", 20);
      if (!KNOWN.has(type)) continue;
      const pr = obj(spec.props);
      let n = live.get(id);
      if (!n) {
        n = make(type, pr, env, (a) => env.action(a));
        live.set(id, n);
      }
      n.update(pr, env);
    }
    // Wire the tree, children in order. Guard against cycles with a visited set.
    const seen = new Set<string>();
    const attach = (id: string, parent: HTMLElement) => {
      const n = live.get(id);
      if (!n || seen.has(id)) return;
      seen.add(id);
      parent.appendChild(n.el);
      if (n.host) for (const cid of arr(obj(nodes[id]).children, 50)) if (typeof cid === "string") attach(cid, n.host);
    };
    body.replaceChildren();
    if (rootId) attach(rootId, body);
    else if (loading) body.appendChild(skeleton);
  };
  build(props);

  return {
    el,
    update: build,
    tick(t, dt) { for (const n of live.values()) n.tick?.(t, dt); },
    busy: () => [...live.values()].some((n) => n.busy?.()) || loading,
    destroy() { for (const n of live.values()) n.destroy?.(); live.clear(); },
  };
}

export const HUD: Record<string, KindSpec> = {
  hud: { design: [560, 360], around: [1.9, 1.3], panel: true, make: makeHud },
};
