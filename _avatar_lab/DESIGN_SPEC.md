# Ada Amber — Dots-parity Avatar Lab (ISOLATED, not wired to main UI)

This folder is a sandbox. Nothing here touches `src/App.jsx` or `src/components/Visualizer.jsx`.
If you like it, we copy ONE file (`AdaDot.jsx`) into `src/components/` later.

## 1. What we cloned from Alfred (and what we changed)

Alfred (OpenAI Dots, yellow proposal dot):
- warm yellow blob, 1:1, soft gooey border-radius morph
- 2 capsule eyes + 1 bezier mouth, no nose
- 1 accessory, 1 human name, persistent across surfaces
- never fully still: breath + blink + bounce

Ada Amber (this lab):
- color: `Amber #FFB800` base, gradient to `#FF8A00`, deep outline `#7A3E00`
  Why: same warmth/optimism you liked in Alfred, but brass/workshop tone = CAD + engineering, not generic helper. Distinct from Dots rainbow.
- shape: rounded-square blob, not perfect circle. `border-radius: 46% 54% 52% 48% / 52% 48% 54% 46%` morphing. Signals part/blueprint/bolt.
- face: same language as Dots (capsule eyes + bezier mouth) so it reads instantly cute, but eye spacing wider + brow plate for visor mount = own character.
- accessory with job: CAD visor (slides down in CAD mode), headset mic (slides in when listening). Alfred wears style, Ada wears tools.
- name + voice stay: `Ada` + Jane (Voxtral). No renaming.

## 2. Face SVG paths (exact, 200x200 viewBox)

Body is a div (for gooey morph), face is SVG overlay so mouth can stretch with audio.

Eyes (two capsules, not circles — capsules read cuter at small size):
- left eye: `<rect x="62" y="78" width="22" height="34" rx="11" fill="#1A1206"/>`
- right eye: `<rect x="116" y="78" width="22" height="34" rx="11" fill="#1A1206"/>`
- highlight: `<circle cx="69" cy="88" r="4.5" fill="#FFF7E0"/>` mirrored right.
- blink = scaleY 1 -> 0.08 in 110ms, every 2.8–5.2s random (not looped, feels alive).
- listening = scale 1.25x + spacing +6px. thinking = translateX sine dart. approval = widen + shake.

Mouth variants (stroke `#1A1206`, stroke-width 5, round cap, fill none unless noted):
- idle smile: `M 82 132 Q 100 144 118 132`
- speaking open (height = audioLevel): `M 82 130 Q 100 130+18*level 118 130 Q 100 148+18*level 82 130 Z` fill `#1A1206`
- listening small o: ellipse `cx=100 cy=136 rx=8 ry=10`
- thinking flat: `M 86 136 L 114 136` + slight wobble
- working (determined): `M 84 134 Q 100 128 116 134`
- approval (worried zig): `M 82 136 L 90 130 L 98 136 L 106 130 L 114 136`
- success (big open smile): `M 78 128 Q 100 158 122 128 Q 100 140 78 128 Z` fill `#1A1206`
- offline: `M 88 136 L 112 136` opacity 0.4

Visor (CAD mode):
- `<rect x="40" y="58" width="120" height="34" rx="17" fill="#0B1B2B" opacity="0.92"/>` + cyan scan line `<rect x="48" y="72" width="40" height="4" rx="2" fill="#22D3EE"/>`
- slides from y=-50 to y=0 with 80ms lag behind body (secondary motion).

Headset (listening mode):
- arc + mic boom to mouth corner. Only one accessory visible at a time.

## 3. State table (timings match AdaDot.jsx + preview.html)

| state | backend trigger in Ada (future) | body motion | eyes | mouth | glow |
|---|---|---|---|---|---|
| idle | default, no job | breath scale 1±0.03, 2s, float y ±6px, radius morph 3s | blink random | smile | amber soft `0 0 40px rgba(255,184,0,.45)` |
| listening | STT active (Groq Whisper), `isListening` true | perk scaleY 1.08, y -4px | 1.25x wide | small o | cyan rim `0 0 50px rgba(34,211,238,.6)` |
| thinking | brain call (providers.py lane FAST/DEEP) | squash 0.93x, 0.6s pulse | dart sine | flat | amber + 3 orbiting dots |
| speaking | TTS streaming (tts_agent chunk, aiAudioData) | bounce y = -8*level, scale = 1+0.12*level | normal, highlight pop | open height = level | warm bright |
| working | task_board active / cadData / browserData / overnight job | shrink to 0.85x, shimmer sweep 1.5s | determined half-lid | determined | pulsing amber |
| approval | confirmationRequest != null (ConfirmationPopup) | shake x [0,-6,6,-4,4,0] 0.4s loop | widen + brow up | zigzag | amber-red `rgba(255,90,0,.7)` |
| success | CAD success / save_version / print queued | jump y -18px spring, 0.5s | happy ^ ^ (scaleY 0.6 + up) | big smile | green-gold flash |
| offline | providers all down / socket disconnected | grayscale, scale 0.96, no float | half closed opacity .5 | flat dim | none |

Secondary motion rule: accessory + highlights lag body by 80ms. Anticipation rule: squash 120ms before any jump/speak start.

## 4. Files in this lab

- `AdaDot.jsx` — drop-in React component (framer-motion 11 + React 18, same as your package.json). Props: `state, audioLevel (0-1), size, accessory`. No imports from Ada. Copy to `src/components/AdaDot.jsx` when approved.
- `avatarStates.js` — same table as JS + backend event map + framer values.
- `preview.html` — open directly, click states, drag audio slider. Same colors/timings as component so approval = final look.

## 5. How to promote to main UI (DO NOT DO YET)

When you say go:
1. Copy `_avatar_lab/AdaDot.jsx` -> `src/components/AdaDot.jsx`
2. In `App.jsx`, keep `Visualizer` as background glow, render `<AdaDot state={...} audioLevel={...} />` centered above it (10 lines).
3. Add `useAvatarState.js` hook mapping `isListening, aiAudioData, confirmationRequest, cadData, socketConnected` -> state string.
4. Add accessory toggle to `SettingsWindow.jsx`.
No Visualizer deletion, no App rewrite.
