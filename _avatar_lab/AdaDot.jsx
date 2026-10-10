import React, { useEffect, useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

// Ada Amber — isolated lab component. NOT imported by App.jsx yet.
// To promote later: copy this file to src/components/AdaDot.jsx, no changes needed.
// Props mirror preview.html controls so what you approve is what ships.

const COLORS = {
  base: '#FFB800',
  deep: '#FF8A00',
  outline: '#7A3E00',
  ink: '#1A1206',
  cream: '#FFF7E0',
  cyan: '#22D3EE',
};

function useBlink(active) {
  const [blinking, setBlinking] = useState(false);
  useEffect(() => {
    if (!active) return;
    let alive = true;
    let t1, t2;
    const loop = () => {
      if (!alive) return;
      const next = 2800 + Math.random() * 2400; // 2.8-5.2s random, not looped
      t1 = setTimeout(() => {
        if (!alive) return;
        setBlinking(true);
        t2 = setTimeout(() => alive && setBlinking(false), 120);
        loop();
      }, next);
    };
    loop();
    return () => { alive = false; clearTimeout(t1); clearTimeout(t2); };
  }, [active]);
  return blinking;
}

function Mouth({ state, audioLevel }) {
  const lvl = Math.max(0, Math.min(1, audioLevel));
  if (state === 'speaking') {
    const h = 4 + lvl * 22;
    return (
      <ellipse cx="100" cy="136" rx={10 + lvl * 6} ry={h / 2}
        fill={COLORS.ink} />
    );
  }
  const d = {
    idle: 'M 82 132 Q 100 144 118 132',
    listening: null, // rendered as ellipse below
    thinking: 'M 86 136 L 114 136',
    working: 'M 84 134 Q 100 128 116 134',
    approval: 'M 82 136 L 90 130 L 98 136 L 106 130 L 114 136',
    success: 'M 78 128 Q 100 160 122 128 Q 100 142 78 128 Z',
    offline: 'M 88 136 L 112 136',
  }[state] || 'M 82 132 Q 100 144 118 132';

  if (state === 'listening') {
    return <ellipse cx="100" cy="136" rx="8" ry="10" fill={COLORS.ink} />;
  }
  if (state === 'success') {
    return <path d={d} fill={COLORS.ink} stroke={COLORS.ink} strokeWidth="4" strokeLinejoin="round" />;
  }
  return (
    <path d={d} fill="none" stroke={COLORS.ink} strokeWidth="5"
      strokeLinecap="round" opacity={state === 'offline' ? 0.4 : 1} />
  );
}

export default function AdaDot({
  state = 'idle',
  audioLevel = 0,
  size = 220,
  accessory = 'visor', // 'visor' | 'headset' | 'none'
  glow = true,
}) {
  const blinking = useBlink(state === 'idle' || state === 'working');
  const lvl = Math.max(0, Math.min(1, audioLevel));

  // Body animation per state (framer-motion 11 syntax)
  const bodyAnimate = (() => {
    switch (state) {
      case 'listening':
        return { scaleY: 1.08, scaleX: 0.98, y: -4 };
      case 'thinking':
        return { scale: [1, 0.93, 1], transition: { duration: 0.6, repeat: Infinity } };
      case 'speaking':
        return { y: -8 * lvl, scale: 1 + 0.12 * lvl };
      case 'working':
        return { scale: 0.88 };
      case 'approval':
        return { x: [0, -6, 6, -4, 4, 0], transition: { duration: 0.4, repeat: Infinity } };
      case 'success':
        return { y: [0, -18, 0], transition: { type: 'spring', stiffness: 400, damping: 12 } };
      case 'offline':
        return { scale: 0.96, y: 0 };
      case 'idle':
      default:
        return { scale: [1, 1.03, 1], y: [0, -6, 0] };
    }
  })();

  const bodyTransition = state === 'idle'
    ? { duration: 2, repeat: Infinity, ease: 'easeInOut' }
    : { type: 'spring', stiffness: 300, damping: 20 };

  const eyeScaleY = blinking ? 0.08 : state === 'listening' ? 1.25 : state === 'working' ? 0.7 : state === 'success' ? 0.6 : 1;
  const eyeScaleX = state === 'approval' ? 1.3 : 1;

  const showVisor = accessory === 'visor';
  const showHeadset = accessory === 'headset' || state === 'listening';

  const glowShadow = !glow ? 'none'
    : state === 'listening' ? '0 0 50px rgba(34,211,238,.6)'
    : state === 'approval' ? '0 0 50px rgba(255,90,0,.7)'
    : state === 'speaking' ? '0 0 60px rgba(255,170,0,.65)'
    : state === 'success' ? '0 0 60px rgba(120,255,150,.6)'
    : state === 'offline' ? 'none'
    : '0 0 40px rgba(255,184,0,.45)';

  return (
    <div style={{ width: size, height: size, position: 'relative' }}>
      {/* thinking orbits */}
      {state === 'thinking' && (
        <motion.div
          style={{ position: 'absolute', inset: -18, pointerEvents: 'none' }}
          animate={{ rotate: 360 }}
          transition={{ duration: 1.2, repeat: Infinity, ease: 'linear' }}
        >
          {[0, 120, 240].map((deg) => (
            <div key={deg} style={{
              position: 'absolute', left: '50%', top: '50%',
              transform: `rotate(${deg}deg) translateX(${size / 2 + 8}px)`,
            }}>
              <div style={{ width: 10, height: 10, borderRadius: '50%', background: COLORS.base }} />
            </div>
          ))}
        </motion.div>
      )}

      {/* body */}
      <motion.div
        animate={bodyAnimate}
        transition={bodyTransition}
        style={{
          width: '100%', height: '100%',
          background: `radial-gradient(circle at 35% 30%, #FFD968 0%, ${COLORS.base} 45%, ${COLORS.deep} 100%)`,
          border: `4px solid ${COLORS.outline}`,
          borderRadius: '46% 54% 52% 48% / 52% 48% 54% 46%',
          boxShadow: glowShadow,
          filter: state === 'offline' ? 'grayscale(1) brightness(0.85)' : 'none',
          position: 'relative', overflow: 'hidden',
        }}
      >
        {/* goo highlight */}
        <div style={{
          position: 'absolute', left: '18%', top: '12%', width: '34%', height: '22%',
          background: 'rgba(255,255,255,.5)', borderRadius: '50%', filter: 'blur(2px)',
        }} />
        {/* working shimmer */}
        {state === 'working' && (
          <motion.div
            style={{ position: 'absolute', top: 0, bottom: 0, width: '40%', background: 'linear-gradient(100deg, transparent, rgba(255,255,255,.5), transparent)' }}
            animate={{ left: ['-40%', '100%'] }}
            transition={{ duration: 1.5, repeat: Infinity }}
          />
        )}

        {/* face */}
        <svg viewBox="0 0 200 200" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}>
          {/* visor */}
          <AnimatePresence>
            {showVisor && (
              <motion.g
                initial={{ y: -50, opacity: 0 }}
                animate={{ y: 0, opacity: 0.96 }}
                exit={{ y: -50, opacity: 0 }}
                transition={{ type: 'spring', stiffness: 260, damping: 22, delay: 0.08 }}
              >
                <rect x="40" y="56" width="120" height="34" rx="17" fill="#0B1B2B" />
                <rect x="50" y="70" width="42" height="5" rx="2.5" fill={COLORS.cyan} />
                <rect x="96" y="70" width="14" height="5" rx="2.5" fill="#0E7490" />
              </motion.g>
            )}
          </AnimatePresence>

          {/* eyes */}
          <motion.g
            animate={state === 'thinking' ? { x: [0, 6, -6, 0] } : { x: 0 }}
            transition={state === 'thinking' ? { duration: 0.5, repeat: Infinity } : {}}
          >
            {[62, 116].map((x) => (
              <motion.g key={x}
                animate={{ scaleY: eyeScaleY, scaleX: eyeScaleX }}
                style={{ originX: `${x + 11}px`, originY: '95px' }}
                transition={{ duration: 0.11 }}
              >
                <rect x={x} y="78" width="22" height="34" rx="11" fill={COLORS.ink}
                  opacity={state === 'offline' ? 0.5 : 1} />
                {!blinking && (
                  <circle cx={x + 7} cy="88" r="4.5" fill={COLORS.cream} />
                )}
              </motion.g>
            ))}
          </motion.g>

          <Mouth state={state} audioLevel={lvl} />

          {/* headset mic */}
          {showHeadset && accessory !== 'none' && (
            <g stroke={COLORS.ink} strokeWidth="5" fill="none" strokeLinecap="round">
              <path d="M 128 96 Q 138 130 112 142" />
              <circle cx="108" cy="143" r="6" fill={COLORS.ink} />
            </g>
          )}
        </svg>
      </motion.div>

      {/* shadow */}
      <div style={{
        position: 'absolute', left: '20%', right: '20%', bottom: -14, height: 14,
        background: 'radial-gradient(ellipse at center, rgba(0,0,0,.35), transparent 70%)',
      }} />
    </div>
  );
}
