// HudPanel — her stage. Floating right-side panel listing live HUDs.
// Skeleton shimmer while streaming, dismiss per HUD, actions fire back.

import React, { useSyncExternalStore } from 'react';
import { getHuds, subscribeHuds, dismissHud } from './hudStore';
import { HudRenderer } from './HudRenderer';

function useHuds() {
  return useSyncExternalStore(subscribeHuds, getHuds);
}

export function HudPanel({ onAction }) {
  const huds = useHuds();
  if (!huds.length) return null;
  return (
    <div className="fixed right-3 top-[64px] bottom-3 w-[340px] z-40 flex flex-col gap-2 overflow-y-auto pointer-events-auto">
      {huds.map((h) => {
        const hasNodes = h.nodes && Object.keys(h.nodes).length > 0;
        return (
          <div key={h.hudId} className="rounded-2xl border border-cyan-500/25 bg-black/70 backdrop-blur-xl shadow-[0_0_30px_rgba(6,182,212,0.12)] overflow-hidden">
            <div className="flex items-center justify-between px-3 py-1.5 border-b border-cyan-500/15">
              <div className="text-[11px] font-bold tracking-widest text-cyan-200">
                {h.title || 'ADA HUD'}
                {!h.done ? <span className="ml-2 text-cyan-400/60 animate-pulse">● LIVE</span> : null}
              </div>
              <button
                className="text-[11px] font-mono text-slate-400 hover:text-red-300 px-1"
                onClick={() => {
                  dismissHud(h.hudId);
                  if (onAction) onAction('dismiss_hud', { hudId: h.hudId });
                }}
              >✕</button>
            </div>
            <div className="p-2.5">
              {hasNodes ? (
                <HudRenderer spec={{ v: 1, hudId: h.hudId, root: h.root, nodes: h.nodes }} onAction={onAction} />
              ) : (
                <div className="h-[90px] rounded-lg bg-gradient-to-r from-cyan-500/5 via-cyan-500/15 to-cyan-500/5 animate-pulse flex items-center justify-center text-[10px] font-mono text-cyan-300/50">
                  COMPOSING…
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default HudPanel;
