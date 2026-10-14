/**
 * Free build: a small HTML/canvas program she wrote, run in a sandbox.
 *
 * The wall is the browser's, not ours:
 *  - sandbox="allow-scripts" without allow-same-origin: the frame has an opaque
 *    origin, so it cannot touch the app, its storage, cookies or the socket.
 *  - A CSP on the frame (and again inside it) allows inline script and style and
 *    data: images only, so it cannot load or send anything over the network.
 *  - A heartbeat from inside the frame; if it stops (a stalled script), the
 *    frame is removed.
 * The pattern check below is a second filter on top of the backend's.
 */
import type { KindSpec, Props, WidgetEnv, WidgetInstance } from "../types";
import { h, str } from "../dom";

const MAX_BYTES = 48 * 1024;
const CSP = "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; media-src data: blob:; font-src data:";

const FORBIDDEN = new RegExp(
  [
    "fetch\\s*\\(", "XMLHttpRequest", "WebSocket", "EventSource", "importScripts", "import\\s*\\(",
    "navigator\\.sendBeacon", "window\\.parent", "window\\.top", "\\bparent\\s*\\.", "\\btop\\s*\\.",
    "document\\.cookie", "localStorage", "sessionStorage", "indexedDB", "<iframe", "<object", "<embed",
    "<link\\b", "<meta\\b", "<base\\b", "<form\\b", "window\\.open", "location\\s*=", "location\\.",
    "eval\\s*\\(", "new\\s+Function", "\\bWorker\\s*\\(", "SharedWorker", "ServiceWorker",
    "src\\s*=\\s*[\"']?\\s*(https?:|//)", "href\\s*=", "@import", "url\\s*\\(\\s*[\"']?\\s*(https?:|//)",
  ].join("|"),
  "i",
);

let counter = 0;

function wrap(html: string, id: string, env: WidgetEnv): string {
  const p = env.pal;
  return `<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="${CSP}">
<style>:root{--c:${p.c};--c2:${p.c2};--cd:${p.cd};--rgb:${p.rgb}}html,body{margin:0;width:100%;height:100%;overflow:hidden;background:transparent;color:${p.c2};font-family:"Segoe UI","Helvetica Neue",Arial,sans-serif}</style>
<script>(function(){var ID=${JSON.stringify(id)};setInterval(function(){try{parent.postMessage({ada:"hb",id:ID},"*")}catch(e){}},400);
addEventListener("message",function(e){var d=e.data;if(d&&d.ada==="pal"){var r=document.documentElement.style;r.setProperty("--c",d.c);r.setProperty("--c2",d.c2);r.setProperty("--cd",d.cd);r.setProperty("--rgb",d.rgb)}});})();</script>
</head><body>${html}</body></html>`;
}

function makeCode(props: Props, env: WidgetEnv): WidgetInstance {
  const el = h("div", "wd-code");
  const id = `w${++counter}-${Math.random().toString(36).slice(2, 8)}`;
  const html = typeof props.html === "string" ? props.html : "";
  let frame: HTMLIFrameElement | null = null;
  let lastBeat = performance.now();
  let started = performance.now();
  let stopped = false;
  let lastPal = "";

  const refuse = (why: string) => {
    stopped = true;
    el.replaceChildren(h("div", "stopped", why));
  };

  if (!html.trim()) refuse("empty widget");
  else if (new TextEncoder().encode(html).length > MAX_BYTES) refuse("program too large");
  else if (FORBIDDEN.test(html)) refuse("blocked by the sandbox");
  else {
    frame = document.createElement("iframe");
    frame.setAttribute("sandbox", "allow-scripts");
    frame.setAttribute("csp", CSP);
    frame.setAttribute("referrerpolicy", "no-referrer");
    frame.setAttribute("allow", "");
    frame.setAttribute("title", str(props.title, "widget", 60));
    frame.srcdoc = wrap(html, id, env);
    el.appendChild(frame);
  }

  const onMsg = (e: MessageEvent) => {
    if (!frame || e.source !== frame.contentWindow) return;
    const d = e.data as { ada?: string; id?: string } | null;
    if (d && d.ada === "hb" && d.id === id) lastBeat = performance.now();
  };
  window.addEventListener("message", onMsg);

  return {
    el,
    update() { /* a program is replaced, not patched */ },
    tick() {
      if (stopped || !frame) return;
      const now = performance.now();
      if (now - started > 1500 && now - lastBeat > 4000) {
        frame.remove(); frame = null; refuse("stopped: not responding");
        return;
      }
      const key = env.pal.c + env.pal.c2;
      if (key !== lastPal && frame.contentWindow) {
        lastPal = key;
        frame.contentWindow.postMessage({ ada: "pal", c: env.pal.c, c2: env.pal.c2, cd: env.pal.cd, rgb: env.pal.rgb }, "*");
      }
    },
    destroy() {
      window.removeEventListener("message", onMsg);
      if (frame) { frame.src = "about:blank"; frame.remove(); frame = null; }
      started = 0;
    },
  };
}

export const CODE: Record<string, KindSpec> = {
  code: { design: [400, 400], inner: 1.6, around: [1.15, 1.15], panel: false, make: makeCode },
};
