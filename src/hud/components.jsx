// HUD primitives — hand-rolled SVG, zero chart deps, full streaming control.
// Every component tolerates garbage props (NaN, missing, wrong types) by
// falling back to sane defaults. Nothing here fetches, evals, or injects.

import React from 'react';

const num = (v, dflt = 0) => {
  const f = Number(v);
  return Number.isFinite(f) ? f : dflt;
};
const str = (v, cap = 300) => String(v ?? '').slice(0, cap);
const clamp01 = (v) => Math.max(0, Math.min(1, num(v, 0)));

export function HudRoot({ title, children }) {
  return <div className="flex flex-col gap-2">{children}</div>;
}

export function Row({ children }) {
  return <div className="flex flex-row gap-2 flex-wrap">{children}</div>;
}

export function Column({ children }) {
  return <div className="flex flex-col gap-2">{children}</div>;
}

export function Card({ title, children }) {
  return (
    <div className="flex-1 min-w-[140px] rounded-xl border border-cyan-500/25 bg-black/60 backdrop-blur-md p-3 shadow-[0_0_18px_rgba(6,182,212,0.08)]">
      {title ? <div className="text-[11px] font-bold tracking-widest text-cyan-300/90 mb-2">{str(title, 60).toUpperCase()}</div> : null}
      <div className="flex flex-col gap-2">{children}</div>
    </div>
  );
}

export function Gauge({ label, value, min, max, unit, thresholds }) {
  const lo = num(min, 0), hi = num(max, 100);
  const v = num(value, 0);
  const frac = hi > lo ? Math.max(0, Math.min(1, (v - lo) / (hi - lo))) : 0;
  const th = Array.isArray(thresholds) ? thresholds.map((t) => num(t, NaN)).filter((t) => Number.isFinite(t)) : [];
  // 240° arc gauge
  const R = 44, CX = 55, CY = 55;
  const a0 = Math.PI * 0.75, a1 = Math.PI * 2.25;
  const arc = (f0, f1) => {
    const a = a0 + (a1 - a0) * f0, b = a0 + (a1 - a0) * f1;
    return `M ${CX + R * Math.cos(a)} ${CY + R * Math.sin(a)} A ${R} ${R} 0 ${(b - a) > Math.PI ? 1 : 0} 1 ${CX + R * Math.cos(b)} ${CY + R * Math.sin(b)}`;
  };
  const crit = th.length > 1 && v >= th[1];
  const warn = !crit && th.length > 0 && v >= th[0];
  const color = crit ? '#f87171' : warn ? '#fbbf24' : '#22d3ee';
  return (
    <div className="flex flex-col items-center">
      <svg width="110" height="80" viewBox="0 0 110 80">
        <path d={arc(0, 1)} stroke="#164e63" strokeWidth="9" fill="none" strokeLinecap="round" />
        <path d={arc(0, frac)} stroke={color} strokeWidth="9" fill="none" strokeLinecap="round" />
        {th.map((t, i) => {
          const f = hi > lo ? (t - lo) / (hi - lo) : 0;
          if (f < 0 || f > 1) return null;
          const a = a0 + (a1 - a0) * f;
          return <circle key={i} cx={CX + R * Math.cos(a)} cy={CY + R * Math.sin(a)} r="3" fill={i === 0 ? '#fbbf24' : '#f87171'} />;
        })}
        <text x={CX} y={CY + 2} textAnchor="middle" fill="#e2e8f0" fontSize="15" fontWeight="bold">{str(v)}<tspan fontSize="9" fill="#67e8f9">{str(unit, 12)}</tspan></text>
      </svg>
      {label ? <div className="text-[10px] tracking-widest text-cyan-300/70">{str(label, 40).toUpperCase()}</div> : null}
    </div>
  );
}

export function ProgressRing({ label, frac, sub }) {
  const f = clamp01(frac);
  const R = 30, C = 2 * Math.PI * R;
  return (
    <div className="flex flex-col items-center">
      <svg width="80" height="80" viewBox="0 0 80 80">
        <circle cx="40" cy="40" r={R} stroke="#164e63" strokeWidth="8" fill="none" />
        <circle cx="40" cy="40" r={R} stroke="#22d3ee" strokeWidth="8" fill="none"
          strokeDasharray={`${C * f} ${C}`} strokeLinecap="round" transform="rotate(-90 40 40)" />
        <text x="40" y="45" textAnchor="middle" fill="#e2e8f0" fontSize="14" fontWeight="bold">{Math.round(f * 100)}%</text>
      </svg>
      {label ? <div className="text-[10px] tracking-widest text-cyan-300/70">{str(label, 40).toUpperCase()}</div> : null}
      {sub ? <div className="text-[10px] text-slate-300/80 font-mono">{str(sub, 60)}</div> : null}
    </div>
  );
}

export function Sparkline({ label, points, unit }) {
  const pts = (Array.isArray(points) ? points : []).map((p) => num(p, NaN)).filter((p) => Number.isFinite(p)).slice(-100);
  const W = 150, H = 44;
  if (pts.length < 2) return <div className="text-[10px] text-slate-400">…</div>;
  const lo = Math.min(...pts), hi = Math.max(...pts);
  const span = hi - lo || 1;
  const d = pts.map((p, i) => `${(i / (pts.length - 1)) * W},${H - 4 - ((p - lo) / span) * (H - 10)}`).join(' ');
  return (
    <div className="flex flex-col">
      <svg width={W} height={H}>
        <polyline points={d} fill="none" stroke="#22d3ee" strokeWidth="1.8" />
        <circle cx={W} cy={H - 4 - ((pts[pts.length - 1] - lo) / span) * (H - 10)} r="2.5" fill="#22d3ee" />
      </svg>
      <div className="text-[10px] text-slate-300/80 font-mono">{str(label, 30)} {str(pts[pts.length - 1])}{str(unit, 10)}</div>
    </div>
  );
}

const STEP_COLOR = { done: '#34d399', doing: '#22d3ee', fail: '#f87171', todo: '#475569' };

export function Timeline({ steps }) {
  const list = (Array.isArray(steps) ? steps : []).slice(0, 12);
  return (
    <div className="flex flex-col gap-1.5">
      {list.map((s, i) => {
        const st = STEP_COLOR[s && s.state] ? s.state : 'todo';
        return (
          <div key={i} className="flex items-center gap-2">
            <div className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: STEP_COLOR[st], boxShadow: st === 'doing' ? '0 0 8px #22d3ee' : 'none' }} />
            <div className={`text-[11px] font-mono ${st === 'todo' ? 'text-slate-500' : 'text-slate-200'}`}>{str(s && s.label, 80)}</div>
          </div>
        );
      })}
    </div>
  );
}

export function SpecCard({ rows, actions, onAction }) {
  const list = (Array.isArray(rows) ? rows : []).slice(0, 12);
  const acts = (Array.isArray(actions) ? actions : []).slice(0, 4);
  return (
    <div className="flex flex-col gap-1">
      {list.map((r, i) => (
        <div key={i} className="flex justify-between gap-3 text-[11px] font-mono">
          <span className="text-slate-400">{str(r && r.k, 40)}</span>
          <span className="text-slate-100 text-right">{str(r && r.v, 80)}</span>
        </div>
      ))}
      {acts.length ? (
        <div className="flex gap-2 mt-1 flex-wrap">
          {acts.map((a, i) => (
            <Button key={i} label={(a && a.label) || 'Do'} action={a && a.action} variant={(a && a.variant) || 'ghost'} onAction={onAction} />
          ))}
        </div>
      ) : null}
    </div>
  );
}

export function Button({ label, action, variant, onAction }) {
  const styles = variant === 'danger'
    ? 'border-red-500/50 text-red-300 hover:bg-red-500/15'
    : variant === 'primary'
      ? 'border-cyan-400/70 text-cyan-100 bg-cyan-500/15 hover:bg-cyan-500/25'
      : 'border-cyan-500/30 text-cyan-200 hover:bg-cyan-500/10';
  return (
    <button
      className={`text-[11px] font-mono px-3 py-1 rounded-lg border transition-colors ${styles}`}
      onClick={() => { if (action && onAction) onAction(action); }}
    >{str(label, 40) || 'Do'}</button>
  );
}

export function LogFeed({ lines }) {
  const list = (Array.isArray(lines) ? lines : []).slice(-30);
  return (
    <div className="flex flex-col gap-0.5 max-h-[120px] overflow-hidden">
      {list.map((l, i) => (
        <div key={i} className="text-[10px] font-mono text-slate-300/80 truncate">› {str(l, 160)}</div>
      ))}
    </div>
  );
}

const SVG_SHAPES = new Set(['circle', 'rect', 'line', 'path', 'text']);

export function SvgCanvas({ width, height, elements }) {
  const W = Math.max(50, Math.min(600, num(width, 220)));
  const H = Math.max(50, Math.min(400, num(height, 140)));
  const els = (Array.isArray(elements) ? elements : []).slice(0, 200);
  const numAttr = (o, keys) => {
    const out = {};
    for (const k of keys) {
      if (o && o[k] !== undefined) {
        const f = Number(o[k]);
        if (Number.isFinite(f)) out[k] = Math.max(-2000, Math.min(2000, f));
      }
    }
    return out;
  };
  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="rounded-lg bg-black/40">
      {els.map((el, i) => {
        if (!el || !SVG_SHAPES.has(el.shape)) return null;
        const common = { key: i, stroke: '#22d3ee', fill: 'none', strokeWidth: 1.5 };
        if (el.shape === 'circle') return <circle {...common} {...numAttr(el, ['cx', 'cy', 'r'])} fill={typeof el.fill === 'string' ? el.fill.slice(0, 20) : 'none'} />;
        if (el.shape === 'rect') return <rect {...common} {...numAttr(el, ['x', 'y', 'width', 'height'])} />;
        if (el.shape === 'line') return <line {...common} {...numAttr(el, ['x1', 'y1', 'x2', 'y2'])} />;
        if (el.shape === 'path') return <path {...common} d={typeof el.d === 'string' ? el.d.slice(0, 2000) : ''} />;
        if (el.shape === 'text') return <text {...numAttr(el, ['x', 'y'])} fill="#e2e8f0" fontSize={num(el.size, 11)}>{str(el.text, 80)}</text>;
        return null;
      })}
    </svg>
  );
}
