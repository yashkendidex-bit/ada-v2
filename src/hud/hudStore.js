// hudStore — the HUD state outside React (no re-render thrash).
// Backend events: hud:start / hud:chunk / hud:data / hud:final.
// hud:data path uses /a/b segments addressing node props, e.g.
// /gauge1/props/value or /spec1/props/rows/0/v.

const listeners = new Set();
const huds = new Map(); // hudId -> {title, nodes, root, seq, updated, done}
const MAX_HUDS = 3;
// Stable snapshot for useSyncExternalStore: rebuilt ONLY on notify, so the
// reference is referentially stable between events (no render loops).
let cache = [];

function notify() {
  cache = Array.from(huds.entries()).map(([hudId, h]) => ({ hudId, ...h }));
  for (const fn of listeners) {
    try { fn(); } catch (e) {}
  }
}

function setPath(obj, path, value) {
  const segs = String(path || '').split('/').filter(Boolean);
  if (!segs.length) return false;
  let cur = obj;
  for (let i = 0; i < segs.length - 1; i++) {
    const k = segs[i];
    if (cur == null || typeof cur !== 'object') return false;
    if (!(k in cur)) {
      // only extend props/rows arrays and props objects, never invent nodes
      if (k === 'props' || /^\d+$/.test(k)) cur[k] = /^\d+$/.test(segs[i + 1] || '') ? [] : {};
      else return false;
    }
    cur = cur[k];
  }
  const last = segs[segs.length - 1];
  if (cur == null || typeof cur !== 'object') return false;
  cur[last] = value;
  return true;
}

export function applyHudEvent(msg) {
  if (!msg || msg.v !== 1 || !msg.hudId) return;
  const id = String(msg.hudId).slice(0, 64);
  if (msg.t === 'hud:start') {
    huds.set(id, {
      title: String(msg.title || 'ADA HUD').slice(0, 80),
      nodes: {}, root: msg.root || null, seq: 0, updated: Date.now(), done: false,
      skeleton: msg.skeleton || 'Card',
    });
    while (huds.size > MAX_HUDS) {
      const oldest = huds.keys().next().value;
      huds.delete(oldest);
    }
    notify();
    return;
  }
  const h = huds.get(id);
  if (!h) return;
  if (msg.t === 'hud:chunk') {
    const upsert = msg.upsert || {};
    for (const nid of Object.keys(upsert).slice(0, 200)) {
      h.nodes[nid] = upsert[nid];
    }
    for (const nid of (msg.remove || [])) delete h.nodes[nid];
    // Root recovery: first HudRoot-typed node wins, else first node.
    // (hud:start may predate the root id in streaming specs.)
    if (!h.root || !h.nodes[h.root]) {
      const ids = Object.keys(h.nodes);
      h.root = ids.find((k) => h.nodes[k] && h.nodes[k].type === 'HudRoot')
        || ids.find((k) => /root/i.test(k)) || ids[0] || null;
    }
    if (msg.seq) h.seq = msg.seq;
    h.updated = Date.now();
    notify();
  } else if (msg.t === 'hud:data') {
    // path like /gauge1/props/value
    const segs = String(msg.path || '').split('/').filter(Boolean);
    if (segs.length >= 3 && h.nodes[segs[0]]) {
      const node = h.nodes[segs[0]];
      const rest = '/' + segs.slice(1).join('/');
      if (setPath(node, rest, msg.value)) {
        h.updated = Date.now();
        notifyThrottled();
      }
    }
  } else if (msg.t === 'hud:final') {
    h.done = true;
    if (msg.seq) h.seq = msg.seq;
    h.updated = Date.now();
    notify();
  }
}

let throttleTimer = null;
function notifyThrottled() {
  // sensor streams at ~5Hz: batch to ~10fps, no React thrash
  if (throttleTimer) return;
  throttleTimer = setTimeout(() => {
    throttleTimer = null;
    notify();
  }, 100);
}

export function getHuds() {
  return cache;
}

export function getHud(hudId) {
  return huds.get(hudId) || null;
}

export function dismissHud(hudId) {
  huds.delete(hudId);
  notify();
}

export function subscribeHuds(fn) {
  listeners.add(fn);
  return () => { listeners.delete(fn); };
}
