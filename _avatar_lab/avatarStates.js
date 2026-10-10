// Ada Amber — avatar state table (lab copy, no Ada imports)
// When promoted, import this from src/components/AdaDot.jsx or a useAvatarState hook.

export const AVATAR_STATES = {
  idle: {
    label: 'Idle / breathing',
    backendTrigger: 'default — no STT, no TTS, no task_board job',
    body: { scale: [1, 1.03, 1], y: [0, -6, 0], duration: 2.0, desc: 'breath + float' },
    radiusMorph: '46% 54% 52% 48% / 52% 48% 54% 46% <-> 54% 46% 48% 52% / 48% 52% 46% 54%, 3s',
    eyes: 'blink every 2.8-5.2s random, 110ms close',
    mouth: 'idle smile M82 132 Q100 144 118 132',
    glow: '0 0 40px rgba(255,184,0,.45)',
  },
  listening: {
    label: 'Listening',
    backendTrigger: 'STT active — Groq Whisper streaming, App.jsx isListening=true',
    body: { scaleY: 1.08, y: -4, duration: 0.25 },
    eyes: 'scale 1.25x, spacing +6px',
    mouth: 'small o ellipse cx100 cy136 rx8 ry10',
    glow: '0 0 50px rgba(34,211,238,.6)',
    accessory: 'headset in',
  },
  thinking: {
    label: 'Thinking',
    backendTrigger: 'brain call — providers.py FAST/DEEP lane, before first TTS chunk',
    body: { scale: 0.93, pulse: '0.6s', desc: 'squash + swirl' },
    eyes: 'dart sine translateX ±6px, 0.5s',
    mouth: 'flat M86 136 L114 136',
    glow: 'amber + 3 orbiting dots, rotate 360 in 1.2s',
  },
  speaking: {
    label: 'Speaking',
    backendTrigger: 'TTS streaming — tts_agent chunk + aiAudioData[64] amplitude 0..1',
    body: { bounce: 'y = -8*level, scale = 1+0.12*level, per-frame from audioLevel prop' },
    eyes: 'normal, highlight scale with level',
    mouth: 'open height = level, path in DESIGN_SPEC.md',
    glow: 'warm bright 0 0 60px rgba(255,170,0,.65)',
  },
  working: {
    label: 'Working in background',
    backendTrigger: 'task_board active, cadData/browserData streaming, overnight_dossier running',
    body: { scale: 0.85, shimmer: 'sweep 1.5s infinite' },
    eyes: 'determined half-lid scaleY 0.7',
    mouth: 'determined M84 134 Q100 128 116 134',
    glow: 'pulsing amber 1s',
  },
  approval: {
    label: 'Needs approval',
    backendTrigger: 'confirmationRequest != null (ConfirmationPopup.jsx visible)',
    body: { shake: 'x [0,-6,6,-4,4,0], 0.4s loop' },
    eyes: 'widen 1.3x + brow up',
    mouth: 'zigzag M82 136 L90 130 L98 136 L106 130 L114 136',
    glow: '0 0 50px rgba(255,90,0,.7)',
  },
  success: {
    label: 'Success',
    backendTrigger: 'CAD success, save_version, print queued, reminder set',
    body: { jump: 'y -18px spring 0.5s, then settle' },
    eyes: 'happy ^^ scaleY 0.6 anchored bottom',
    mouth: 'big open smile, fill dark',
    glow: 'green-gold flash 0 0 60px rgba(120,255,150,.6)',
  },
  offline: {
    label: 'Offline / disconnected',
    backendTrigger: 'socket disconnected or all providers down',
    body: { scale: 0.96, grayscale: true, noFloat: true },
    eyes: 'half closed opacity .5',
    mouth: 'flat dim M88 136 L112 136',
    glow: 'none',
  },
};

// Future hook logic (copy-paste when promoting, do not wire yet):
// function resolveAvatarState({ socketConnected, isListening, isSpeaking, hasJob, needsApproval, lastSuccessAt }) {
//   if (!socketConnected) return 'offline';
//   if (needsApproval) return 'approval';
//   if (isListening) return 'listening';
//   if (isSpeaking) return 'speaking';
//   if (hasJob) return 'working';
//   if (Date.now() - lastSuccessAt < 2500) return 'success';
//   return 'idle';
// }

export const AVATAR_COLORS = {
  base: '#FFB800',
  deep: '#FF8A00',
  outline: '#7A3E00',
  ink: '#1A1206',
  cream: '#FFF7E0',
  cyan: '#22D3EE',
};
