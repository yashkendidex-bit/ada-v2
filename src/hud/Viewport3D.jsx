// Viewport3D — three.js scene from an allowlisted spec (no code, no URLs).
// kinds: box | cylinder | sphere | line. Numbers clamped, colors hex-only.
// Lazy-loaded so the three.js chunk never blocks chat startup.

import React, { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';

const num = (v, dflt = 0) => {
  const f = Number(v);
  return Number.isFinite(f) ? Math.max(-5000, Math.min(5000, f)) : dflt;
};
const vec3 = (v, dflt) => (Array.isArray(v) && v.length >= 3
  ? [num(v[0], dflt[0]), num(v[1], dflt[1]), num(v[2], dflt[2])] : dflt);
const hex = (c, dflt = '#38bdf8') => (/^#[0-9a-fA-F]{6}$/.test(String(c || '')) ? String(c) : dflt);

function Part({ spec, explode }) {
  const kind = spec && spec.kind;
  if (kind !== 'box' && kind !== 'cylinder' && kind !== 'sphere' && kind !== 'line') return null;
  const pos = vec3(spec.pos, [0, 0, 0]);
  const rot = vec3(spec.rot, [0, 0, 0]);
  const ex = Math.max(0, Math.min(1, num(explode, 0)));
  // explode pushes parts radially from origin — Iron Man assembly view
  const ep = [pos[0] * (1 + ex * 1.2), pos[1] * (1 + ex * 1.2), pos[2] * (1 + ex * 1.2)];
  const color = hex(spec.color);
  const hl = Array.isArray(spec.highlightIds) ? false : false;
  void hl;
  const mat = <meshStandardMaterial color={color} roughness={0.35} metalness={0.55} />;
  if (kind === 'box') {
    const s = vec3(spec.size, [10, 10, 10]).map((x) => Math.max(0.5, x));
    return <mesh position={ep} rotation={rot}><boxGeometry args={s} />{mat}</mesh>;
  }
  if (kind === 'cylinder') {
    const r = Math.max(0.5, num(spec.r, 4)), h = Math.max(0.5, num(spec.h, 18));
    return <mesh position={ep} rotation={rot}><cylinderGeometry args={[r, r, h, 24]} />{mat}</mesh>;
  }
  if (kind === 'sphere') {
    return <mesh position={ep}><sphereGeometry args={[Math.max(0.5, num(spec.r, 6)), 20, 20]} />{mat}</mesh>;
  }
  const to = vec3(spec.to, [ep[0] + 10, ep[1], ep[2]]);
  const mid = [(ep[0] + to[0]) / 2, (ep[1] + to[1]) / 2, (ep[2] + to[2]) / 2];
  const len = Math.max(0.5, Math.hypot(to[0] - ep[0], to[1] - ep[1], to[2] - ep[2]));
  return <mesh position={mid}><cylinderGeometry args={[0.4, 0.4, len, 8]} /><meshStandardMaterial color={color} /></mesh>;
}

function Scene({ objects, explode, highlight }) {
  const list = (Array.isArray(objects) ? objects : []).slice(0, 60);
  const hl = new Set(Array.isArray(highlight) ? highlight.map(String) : []);
  return (
    <>
      <ambientLight intensity={0.7} />
      <directionalLight position={[40, 60, 30]} intensity={1.4} />
      <directionalLight position={[-30, -20, -40]} intensity={0.4} color="#67e8f9" />
      {list.map((o, i) => {
        const hot = o && hl.has(String(o.name || i));
        return (
          <group key={i}>
            <Part spec={o} explode={explode} />
            {hot ? <mesh position={vec3(o.pos, [0, 0, 0])}>
              <sphereGeometry args={[3, 12, 12]} />
              <meshBasicMaterial color="#fbbf24" wireframe />
            </mesh> : null}
          </group>
        );
      })}
      <OrbitControls enablePan={false} />
    </>
  );
}

export function Viewport3D({ objects, explode, highlight }) {
  return (
    <div className="w-full h-[220px] rounded-lg bg-black/50 border border-cyan-500/15 overflow-hidden">
      <Suspense fallback={<div className="w-full h-full flex items-center justify-center text-[11px] font-mono text-cyan-300/60">LOADING 3D…</div>}>
        <Canvas camera={{ position: [70, 55, 70], fov: 42 }}>
          <Scene objects={objects} explode={explode} highlight={highlight} />
        </Canvas>
      </Suspense>
    </div>
  );
}

export default Viewport3D;
