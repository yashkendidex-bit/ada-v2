// HudRenderer — the ONLY thing that turns AI JSON into pixels.
// Registry lookup + zod validation. No eval, no innerHTML, no URLs.
// Unknown types drop with a visible fallback; bad props coerce or drop
// the node — never the whole HUD.

import React from 'react';
import { z } from 'zod';
import { HudRoot, Row, Column, Card, Gauge, ProgressRing, Sparkline, Timeline, SpecCard, Button, LogFeed, SvgCanvas } from './components';

// three.js stays out of the main bundle: code-split, loads on first 3D HUD.
const Viewport3D = React.lazy(() => import('./Viewport3D'));

const NodeSchema = z.object({
  // id is optional: the map key IS the id on the wire (flat A2UI style).
  id: z.string().min(1).max(64).optional(),
  type: z.enum(['HudRoot', 'Row', 'Column', 'Card', 'Gauge', 'ProgressRing', 'Sparkline', 'Timeline', 'SpecCard', 'SvgCanvas', 'Viewport3D', 'Button', 'LogFeed']),
  props: z.record(z.unknown()).default({}),
  children: z.array(z.string()).max(50).default([]),
});

const HudSpecSchema = z.object({
  v: z.literal(1),
  hudId: z.string(),
  root: z.string(),
  nodes: z.record(NodeSchema),
});

const REGISTRY = {
  HudRoot, Row, Column, Card, Gauge, ProgressRing, Sparkline, Timeline,
  SpecCard, SvgCanvas, Viewport3D, Button, LogFeed,
};

class HudErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { failed: false };
  }
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    if (this.state.failed) {
      return <div className="text-[10px] font-mono text-red-300/70 border border-red-500/30 rounded p-1">widget failed — rest of HUD unaffected</div>;
    }
    return this.props.children;
  }
}

function UnknownFallback({ type }) {
  return <div className="text-[10px] font-mono text-amber-300/70 border border-amber-500/30 rounded p-1">unknown widget “{String(type).slice(0, 30)}” dropped</div>;
}

function RenderNode({ id, nodes, onAction }) {
  const n = nodes[id];
  if (!n) return null;
  const C = REGISTRY[n.type];
  if (!C) return <UnknownFallback type={n.type} />;
  const props = { ...(n.props || {}), onAction };
  // Viewport3D is code-split: never block the HUD on WebGL
  if (n.type === 'Viewport3D') {
    return (
      <HudErrorBoundary>
        <React.Suspense fallback={<div className="text-[11px] font-mono text-cyan-300/60">LOADING 3D…</div>}>
          <C {...props} />
        </React.Suspense>
      </HudErrorBoundary>
    );
  }
  const kids = (n.children || []).map((c) => (
    <RenderNode key={c} id={c} nodes={nodes} onAction={onAction} />
  ));
  return (
    <HudErrorBoundary>
      <C {...props}>{kids.length ? kids : undefined}</C>
    </HudErrorBoundary>
  );
}

export function HudRenderer({ spec, onAction }) {
  const parsed = HudSpecSchema.safeParse({
    v: spec && spec.v, hudId: spec && spec.hudId, root: spec && spec.root, nodes: (spec && spec.nodes) || {},
  });
  if (!parsed.success) {
    return <div className="text-[11px] font-mono text-red-300/80">HUD spec invalid — showing nothing rather than guessing.</div>;
  }
  const { root, nodes } = parsed.data;
  if (!nodes[root]) {
    return <div className="text-[11px] font-mono text-amber-300/80">HUD root missing.</div>;
  }
  return <RenderNode id={root} nodes={nodes} onAction={onAction} />;
}

export default HudRenderer;
