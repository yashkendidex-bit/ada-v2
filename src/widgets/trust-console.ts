/**
 * Trust console: what she did, why, and what she held back.
 *
 * A display widget (never a popup — approvals stay voice-direct). Renders the
 * props built server-side by replay.console_snapshot(): the decision feed
 * (filterable by capability), one-click undo for reversible actions, the
 * per-capability autonomy toggles (ask-first vs act), and the RESTRAINT
 * panel (suggestions she chose not to send). Buttons fire allow-listed
 * actions with parameters: replay_decision:<id>, undo_action:<id>,
 * set_autonomy_toggle:<capability>:<mode>. Text only ever goes in through
 * textContent.
 */
import type { KindSpec, Props, WidgetEnv, WidgetInstance } from "./types";
import { arr, h, num, obj, str } from "./dom";

interface Decision {
  id: number;
  when: string;
  tool: string;
  capability: string;
  decision: string;
  verdict: string;
  risk: string;
  reason: string;
  reversible: boolean;
  undone: boolean;
}

interface Toggle {
  capability: string;
  mode: string;
}

const VERDICT_CLS: Record<string, string> = {
  "auto-approved": "ok",
  "voice-ask": "warn",
  "denied": "bad",
};

const verdictBadge = (d: Decision): string =>
  d.decision === "denied" ? "denied"
    : d.verdict === "voice-ask" ? "voice-ask"
      : "auto-approved";

function makeTrustConsole(props: Props, env: WidgetEnv): WidgetInstance {
  const el = h("div", "wp wtc");
  const head = h("div", "wp-head");
  const body = h("div", "wp-body");
  el.append(head, body);

  const summary = h("div", "wtc-sum");
  const chips = h("div", "wtc-chips");
  const feed = h("div", "wtc-feed");
  const toggleBox = h("div", "wtc-toggles");
  const restraint = h("div", "wtc-restraint");
  body.append(summary, chips, feed, toggleBox, restraint);

  let filter = "all";

  const btn = (label: string, action: string, cls = ""): HTMLButtonElement => {
    const b = h("button", `wtc-btn ${cls}`.trim());
    b.type = "button";
    b.textContent = label;
    b.addEventListener("click", (e) => { e.stopPropagation(); env.action(action); });
    return b;
  };

  const render = (p: Props) => {
    head.textContent = str(p.title, "TRUST CONSOLE", 60).toUpperCase();
    const s = obj(p.summary);
    const bits = [
      `${num(s.decisions, 0)} decisions`,
      `${num(s.confirmations_asked, 0)} asked`,
      `${num(s.suppressed, 0)} held back`,
      `${num(s.undone, 0)} undone`,
    ];
    summary.textContent = bits.join(" · ") + " — today";

    const decisions: Decision[] = arr(p.decisions, 30).map((raw) => {
      const o = obj(raw);
      return {
        id: num(o.id, 0), when: str(o.when, "", 24), tool: str(o.tool, "?", 40),
        capability: str(o.capability, "other", 24), decision: str(o.decision, "", 16),
        verdict: str(o.verdict, "", 16), risk: str(o.risk, "", 40),
        reason: str(o.reason, "", 160), reversible: o.reversible === true,
        undone: o.undone === true,
      };
    });

    // ── capability filter chips ──
    const caps = ["all", ...Array.from(new Set(decisions.map((d) => d.capability)))];
    if (!caps.includes(filter)) filter = "all";
    chips.replaceChildren();
    caps.forEach((c) => {
      const b = btn(c === "all" ? "all" : c, "");
      b.classList.add("chip");
      if (c === filter) b.classList.add("on");
      b.addEventListener("click", () => { filter = c; render(p); });
      chips.appendChild(b);
    });

    // ── decision feed ──
    feed.replaceChildren();
    const shown = decisions.filter((d) => filter === "all" || d.capability === filter);
    if (!shown.length) feed.appendChild(h("div", "wtc-none", "No decisions in range."));
    shown.forEach((d) => {
      const row = h("div", "wtc-row");
      const vb = verdictBadge(d);
      const dot = h("span", `wtc-dot ${VERDICT_CLS[vb] ?? ""}`);
      const main = h("div", "wtc-main");
      const top = h("div", "wtc-top");
      top.append(h("span", "wtc-tool", d.tool), h("span", "wtc-time", d.when));
      const sub = h("div", "wtc-sub", d.undone ? `undone — ${d.reason}` : d.reason);
      main.append(top, sub);
      row.append(dot, main);
      const acts = h("div", "wtc-acts");
      acts.appendChild(btn("replay", `replay_decision:${d.id}`));
      if (d.reversible && !d.undone && d.decision !== "denied") {
        acts.appendChild(btn("undo", `undo_action:${d.id}`, "warn"));
      }
      row.appendChild(acts);
      feed.appendChild(row);
    });

    // ── per-capability toggles ──
    toggleBox.replaceChildren();
    toggleBox.appendChild(h("div", "wtc-sec", "AUTONOMY — ASK FIRST / ACT"));
    arr(p.toggles, 24).forEach((raw) => {
      const t = obj(raw) as unknown as Toggle;
      const cap = str(t.capability, "", 24);
      if (!cap) return;
      const mode = str(t.mode, "ladder", 12);
      const row = h("div", "wtc-trow");
      row.appendChild(h("span", "wtc-tcap", cap));
      const seg = h("div", "wtc-seg");
      const ask = btn("ask first", `set_autonomy_toggle:${cap}:ask-first`);
      const act = btn("act", `set_autonomy_toggle:${cap}:act`);
      if (mode === "ask-first") ask.classList.add("on");
      if (mode === "act") act.classList.add("on");
      if (mode === "ladder") {
        const lad = h("span", "wtc-ladder", "ladder");
        seg.append(lad);
      }
      seg.append(ask, act);
      row.appendChild(seg);
      toggleBox.appendChild(row);
    });

    // ── restraint panel ──
    restraint.replaceChildren();
    const r = obj(p.restraint);
    const n = num(r.suppressed, 0);
    const hours = num(r.hours, 24);
    const line = h("div", "wtc-sec",
      `RESTRAINT — held back ${n} suggestion${n === 1 ? "" : "s"} in ${hours}h`);
    restraint.appendChild(line);
    const why = h("div", "wtc-why");
    if (n > 0) {
      why.textContent = arr(r.top_suppress_reasons, 5)
        .map((x) => { const o = obj(x); return `${str(o.reason, "", 60)} ×${num(o.count, 0)}`; })
        .join(" · ");
    } else {
      why.textContent = r.unavailable ? "timeline unavailable" : "she sent everything she considered.";
    }
    restraint.appendChild(why);
  };

  render(props);
  return { el, update: render, destroy() {} };
}

export const TRUST_CONSOLE: Record<string, KindSpec> = {
  trust_console: { design: [470, 430], around: [1.55, 1.45], panel: true, make: makeTrustConsole },
};
