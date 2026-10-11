/**
 * A.D.A orb: the particle sphere that is her face.
 *
 * Derived from ethanplusai/jarvis (frontend/src/orb.ts), Copyright (c) 2026
 * Ethan Rogers, under the JARVIS non-commercial licence
 * (licenses/JARVIS-LICENSE.txt). See NOTICE.md.
 *
 * What changed for responsiveness and smoothness:
 *  - Frame-rate-independent motion. The original moved 2% of the remaining
 *    distance per frame, so its speed depended on the display's refresh
 *    rate. Here every transition is an exponential approach in seconds, and
 *    the particle integrator is scaled by elapsed time.
 *  - An audio envelope with a fast attack and a slow release: the sphere
 *    snaps to a loud word and settles smoothly after it. While she speaks it
 *    is measured from the real TTS output; while you speak, from the mic.
 *  - Connections come from a k-nearest-neighbour graph built once at start.
 *    Each frame is a single linear pass over those edges, instead of the
 *    original O(N^2) pair search. Each edge fades with its current length, so
 *    links fade in and out rather than flickering across a threshold.
 *  - Colours are interpolated in place. Nothing is allocated per frame.
 *  - Adaptive quality: if frames run long, the orb lowers its pixel ratio and
 *    thins the link pass, and recovers when there is headroom again.
 *  - Respects prefers-reduced-motion by scaling the tumble and drift down.
 */

import * as THREE from "three";

export type OrbState = "idle" | "listening" | "thinking" | "speaking" | "offline";

export interface Orb {
  setState(s: OrbState): void;
  setPlaybackAnalyser(a: AnalyserNode | null): void;
  setMicLevel(rms: number): void;
  pulse(): void;
  destroy(): void;
}

// ── Tuning ─────────────────────────────────────────────────────────────────

const N = 2000;                 // particles
const K = 5;                    // neighbours per particle in the link graph
const LINK_DIST = 8;            // a link is at full brightness below this length
const MAX_ELECTRONS = 200;
const MAX_ACTIVE_ELECTRONS = 3;

const TAU_STATE = 0.22;         // seconds: how quickly the orb changes character
const TAU_COLOR = 0.25;
const TAU_ATTACK = 0.05;        // audio envelope rises fast...
const TAU_RELEASE = 0.22;       // ...and settles smoothly
const TAU_TUMBLE = 0.7;         // state-change spin decay
const TAU_PULSE = 0.3;          // user-action pulse decay

const MIC_GAIN = 6;             // mic RMS -> 0..1 envelope
const VOICE_GAIN = 4;           // playback RMS -> 0..1 envelope
const LINK_BRIGHTNESS = 0.35;

const COLOR = {
  base: new THREE.Color(0x4ca8e8),
  think: new THREE.Color(0x6ec4ff),
  speak: new THREE.Color(0x5ab8f0),
  dim: new THREE.Color(0x3a5f8a),
};

interface Target {
  radius: number;
  speed: number;
  bright: number;
  size: number;
  lines: number;
  electronRate: number;
  color: THREE.Color;
}

const TARGETS: Record<OrbState, Target> = {
  idle:      { radius: 28, speed: 0.2,  bright: 0.5,  size: 0.35, lines: 0.15, electronRate: 0,     color: COLOR.base },
  listening: { radius: 22, speed: 0.3,  bright: 0.65, size: 0.4,  lines: 0.4,  electronRate: 0,     color: COLOR.base },
  thinking:  { radius: 16, speed: 0.5,  bright: 0.7,  size: 0.3,  lines: 1.0,  electronRate: 0.015, color: COLOR.think },
  speaking:  { radius: 18, speed: 0.2,  bright: 0.7,  size: 0.4,  lines: 0.8,  electronRate: 0,     color: COLOR.speak },
  // Offline: smaller, slower, dimmer, drawn inward, no links. Reads as "not
  // here", not as "about to answer".
  offline:   { radius: 12, speed: 0.08, bright: 0.28, size: 0.28, lines: 0.0,  electronRate: 0,     color: COLOR.dim },
};

// ── Helpers ────────────────────────────────────────────────────────────────

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);
const approach = (dt: number, tau: number) => 1 - Math.exp(-dt / tau);

/** k-nearest-neighbour edges over the rest positions, as flat index pairs. */
function buildEdges(pos: Float32Array): Uint32Array {
  const seen = new Set<number>();
  const out: number[] = [];
  const bestD = new Float32Array(K);
  const bestJ = new Int32Array(K);
  for (let i = 0; i < N; i++) {
    bestD.fill(Infinity);
    bestJ.fill(-1);
    const xi = pos[i * 3], yi = pos[i * 3 + 1], zi = pos[i * 3 + 2];
    for (let j = 0; j < N; j++) {
      if (j === i) continue;
      const dx = pos[j * 3] - xi, dy = pos[j * 3 + 1] - yi, dz = pos[j * 3 + 2] - zi;
      const d = dx * dx + dy * dy + dz * dz;
      if (d < bestD[K - 1]) {
        let k = K - 1;
        while (k > 0 && bestD[k - 1] > d) {
          bestD[k] = bestD[k - 1];
          bestJ[k] = bestJ[k - 1];
          k--;
        }
        bestD[k] = d;
        bestJ[k] = j;
      }
    }
    for (let k = 0; k < K; k++) {
      const j = bestJ[k];
      if (j < 0) continue;
      const a = Math.min(i, j), b = Math.max(i, j);
      const key = a * N + b;
      if (!seen.has(key)) {
        seen.add(key);
        out.push(a, b);
      }
    }
  }
  return Uint32Array.from(out);
}

function rms(buf: Float32Array): number {
  let s = 0;
  for (let i = 0; i < buf.length; i++) s += buf[i] * buf[i];
  return Math.sqrt(s / buf.length);
}

// ── Orb ────────────────────────────────────────────────────────────────────

export function createOrb(canvas: HTMLCanvasElement): Orb {
  const reduceMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
  const motion = reduceMotion ? 0.35 : 1;

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
  renderer.setClearColor(0x050508, 1);
  const dprCeil = Math.min(window.devicePixelRatio || 1, 2);
  let dprFull = true;
  renderer.setPixelRatio(dprCeil);
  renderer.setSize(window.innerWidth, window.innerHeight, false);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 1, 1000);
  camera.position.z = 92;     // a little back, so the sphere leaves room for the transcript

  const cloud = new THREE.Group();
  scene.add(cloud);

  // ── Particles ──
  const pos = new Float32Array(N * 3);
  const vel = new Float32Array(N * 3);
  const phase = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    const r = Math.pow(Math.random(), 0.5) * 25;
    pos[i * 3] = r * Math.sin(phi) * Math.cos(theta);
    pos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
    pos[i * 3 + 2] = r * Math.cos(phi);
    phase[i] = Math.random() * 1000;
  }
  const pointGeo = new THREE.BufferGeometry();
  pointGeo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  const pointMat = new THREE.PointsMaterial({
    color: COLOR.base, size: 0.4, transparent: true, opacity: 0.6,
    sizeAttenuation: true, blending: THREE.AdditiveBlending, depthWrite: false,
  });
  cloud.add(new THREE.Points(pointGeo, pointMat));

  // ── Links (fixed topology, per-frame brightness) ──
  const edges = buildEdges(pos);
  const E = edges.length / 2;
  const linkPos = new Float32Array(E * 6);
  const linkCol = new Float32Array(E * 6);
  const linkGeo = new THREE.BufferGeometry();
  linkGeo.setAttribute("position", new THREE.BufferAttribute(linkPos, 3));
  linkGeo.setAttribute("color", new THREE.BufferAttribute(linkCol, 3));
  linkGeo.setDrawRange(0, E * 2);
  const linkMat = new THREE.LineBasicMaterial({
    vertexColors: true, transparent: true, opacity: 1,
    blending: THREE.AdditiveBlending, depthWrite: false,
  });
  cloud.add(new THREE.LineSegments(linkGeo, linkMat));
  let linksLit = false;

  // ── Electrons (travel along edges while thinking) ──
  const elPos = new Float32Array(MAX_ELECTRONS * 3);
  const elGeo = new THREE.BufferGeometry();
  elGeo.setAttribute("position", new THREE.BufferAttribute(elPos, 3));
  elGeo.setDrawRange(0, 0);
  const elMat = new THREE.PointsMaterial({
    color: 0xffffff, size: 0.8, transparent: true, opacity: 1,
    sizeAttenuation: true, blending: THREE.AdditiveBlending, depthWrite: false,
  });
  cloud.add(new THREE.Points(elGeo, elMat));
  interface Electron { edge: number; t: number; speed: number; }
  const electrons: Electron[] = [];
  let lastSpawn = -1;

  // ── State ──
  let state: OrbState = "idle";
  let target: Target = TARGETS.idle;
  let radius = TARGETS.idle.radius;
  let speed = TARGETS.idle.speed;
  let bright = TARGETS.idle.bright;
  let size = TARGETS.idle.size;
  let lines = TARGETS.idle.lines;
  let electronRate = 0;
  const colorCur = COLOR.base.clone();
  let tumble = 0;          // decays after a state change: spin and drift
  let pulseEnergy = 0;     // decays after a user action: a brief swell
  let spinX = 0, spinY = 0, spinZ = 0;
  let cloudZ = 0;

  // ── Audio ──
  let playAnalyser: AnalyserNode | null = null;
  let playBuf = new Float32Array(512);
  let micRaw = 0;
  let envelope = 0;        // smoothed 0..1 voice / mic energy

  // ── Quality ──
  let quality = 1;
  let frameEma = 1 / 60;
  let frameIdx = 0;

  let destroyed = false;
  let rafId = 0;
  // Own timer: THREE.Clock is deprecated in recent three.js releases.
  let lastTime = performance.now() / 1000;
  let elapsed = 0;

  function adaptQuality(dt: number): void {
    frameEma += (dt - frameEma) * 0.05;
    if (frameEma > 1 / 40) quality = Math.max(0.3, quality - dt * 0.5);
    else if (frameEma < 1 / 55) quality = Math.min(1, quality + dt * 0.15);
    const full = quality > 0.7;
    if (full !== dprFull) {
      dprFull = full;
      renderer.setPixelRatio(full ? dprCeil : Math.max(1, dprCeil * 0.75));
      renderer.setSize(window.innerWidth, window.innerHeight, false);
    }
  }

  function frame(): void {
    if (destroyed) return;
    rafId = requestAnimationFrame(frame);

    const now = performance.now() / 1000;
    const dt = Math.min(now - lastTime, 0.1);   // clamp: a background tab must not jump
    lastTime = now;
    elapsed += dt;
    const t = elapsed;
    const f = dt * 60;                       // per-frame factor at 60 fps
    frameIdx++;
    adaptQuality(dt);

    // Smooth towards the current state's targets.
    const a = approach(dt, TAU_STATE);
    radius += (target.radius - radius) * a;
    speed += (target.speed - speed) * a;
    bright += (target.bright - bright) * a;
    size += (target.size - size) * a;
    lines += (target.lines - lines) * a;
    electronRate += (target.electronRate - electronRate) * a;
    colorCur.lerp(target.color, approach(dt, TAU_COLOR));
    tumble *= Math.exp(-dt / TAU_TUMBLE);
    pulseEnergy *= Math.exp(-dt / TAU_PULSE);

    // Audio envelope: fast attack, slow release.
    let raw: number;
    if (state === "speaking" && playAnalyser) {
      playAnalyser.getFloatTimeDomainData(playBuf);
      raw = rms(playBuf) * VOICE_GAIN;
    } else {
      raw = micRaw * MIC_GAIN;
    }
    raw = clamp01(raw);
    envelope += (raw - envelope) * approach(dt, raw > envelope ? TAU_ATTACK : TAU_RELEASE);

    // Spin and drift.
    const spinAmt = tumble * motion;
    spinX += spinAmt * 0.012 * Math.sin(t * 1.7) * f;
    spinY += spinAmt * 0.015 * f;
    spinZ += spinAmt * 0.008 * Math.cos(t * 1.3) * f;
    cloud.rotation.set(spinX, spinY, spinZ);

    // Depth breathing.
    let zTarget = Math.sin(t * 0.12) * 8;
    if (state === "thinking") zTarget = Math.sin(t * 0.3) * 15 + Math.sin(t * 0.9) * 6;
    else if (state === "speaking") zTarget = Math.sin(t * 0.15) * 6 - envelope * 10;
    else if (state === "offline") zTarget = -20 + Math.sin(t * 0.08) * 3;
    cloudZ += (zTarget * motion - cloudZ) * approach(dt, 0.6);
    cloud.position.z = cloudZ;

    // Particle physics. Pulled towards the target radius, drifted by noise,
    // pushed outward by the voice envelope.
    const effRadius = radius - pulseEnergy * 5;
    const damp = Math.pow(0.992, f);
    const drift = speed * f * motion;
    const sp = state === "speaking";
    for (let i = 0; i < N; i++) {
      const i3 = i * 3;
      let x = pos[i3], y = pos[i3 + 1], z = pos[i3 + 2];
      const px = phase[i];

      vel[i3] += Math.sin(t * 0.05 + px) * 0.001 * drift;
      vel[i3 + 1] += Math.cos(t * 0.06 + px * 1.3) * 0.001 * drift;
      vel[i3 + 2] += Math.sin(t * 0.055 + px * 0.7) * 0.001 * drift;
      vel[i3] += Math.sin(t * 0.02 + px * 2.1 + y * 0.1) * 0.0008 * drift;
      vel[i3 + 1] += Math.cos(t * 0.025 + px * 1.7 + z * 0.1) * 0.0008 * drift;
      vel[i3 + 2] += Math.sin(t * 0.022 + px * 0.9 + x * 0.1) * 0.0008 * drift;

      const dist = Math.sqrt(x * x + y * y + z * z) || 0.01;
      const pull = Math.max(0, dist - effRadius) * 0.002 + 0.0003;
      vel[i3] -= (x / dist) * pull * f;
      vel[i3 + 1] -= (y / dist) * pull * f;
      vel[i3 + 2] -= (z / dist) * pull * f;

      if (envelope > 0.04) {
        const push = envelope * 0.02 * f;
        vel[i3] += (x / dist) * push;
        vel[i3 + 1] += (y / dist) * push;
        vel[i3 + 2] += (z / dist) * push;
      }
      if (sp && envelope > 0.1) {
        const wave = Math.sin(t * 8 + px) * envelope * 0.012 * f;
        vel[i3] += (x / dist) * wave;
        vel[i3 + 1] += (y / dist) * wave;
      }

      vel[i3] *= damp; vel[i3 + 1] *= damp; vel[i3 + 2] *= damp;
      x += vel[i3] * f; y += vel[i3 + 1] * f; z += vel[i3 + 2] * f;
      pos[i3] = x; pos[i3 + 1] = y; pos[i3 + 2] = z;
    }
    (pointGeo.getAttribute("position") as THREE.BufferAttribute).needsUpdate = true;

    // Links: one linear pass over the fixed edge list.
    const linkAmount = clamp01(lines);
    const updateLinks = quality > 0.6 || (frameIdx & 1) === 0;
    if (linkAmount > 0.01) {
      if (updateLinks) {
        const maxD = LINK_DIST * (1 + envelope * 0.5);
        const inv = 1 / maxD;
        const r = colorCur.r * LINK_BRIGHTNESS, g = colorCur.g * LINK_BRIGHTNESS, b = colorCur.b * LINK_BRIGHTNESS;
        for (let e = 0; e < E; e++) {
          const i3 = edges[e * 2] * 3, j3 = edges[e * 2 + 1] * 3;
          const x1 = pos[i3], y1 = pos[i3 + 1], z1 = pos[i3 + 2];
          const x2 = pos[j3], y2 = pos[j3 + 1], z2 = pos[j3 + 2];
          const o = e * 6;
          linkPos[o] = x1; linkPos[o + 1] = y1; linkPos[o + 2] = z1;
          linkPos[o + 3] = x2; linkPos[o + 4] = y2; linkPos[o + 5] = z2;
          const dx = x2 - x1, dy = y2 - y1, dz = z2 - z1;
          let w = 1 - Math.sqrt(dx * dx + dy * dy + dz * dz) * inv;
          w = w > 0 ? w * w * linkAmount : 0;
          const cr = r * w, cg = g * w, cb = b * w;
          linkCol[o] = cr; linkCol[o + 1] = cg; linkCol[o + 2] = cb;
          linkCol[o + 3] = cr; linkCol[o + 4] = cg; linkCol[o + 5] = cb;
        }
        (linkGeo.getAttribute("position") as THREE.BufferAttribute).needsUpdate = true;
        (linkGeo.getAttribute("color") as THREE.BufferAttribute).needsUpdate = true;
        linksLit = true;
      }
    } else if (linksLit) {
      linkCol.fill(0);
      (linkGeo.getAttribute("color") as THREE.BufferAttribute).needsUpdate = true;
      linksLit = false;
    }

    // Electrons: spawn while thinking, then ride the edges with the particles.
    if (electronRate > 0.005 && E > 0 && electrons.length < MAX_ACTIVE_ELECTRONS && t - lastSpawn > 1.0) {
      electrons.push({ edge: Math.floor(Math.random() * E), t: 0, speed: 0.003 + Math.random() * 0.003 });
      lastSpawn = t;
    }
    let alive = 0;
    for (let k = electrons.length - 1; k >= 0; k--) {
      const el = electrons[k];
      el.t += el.speed * f;
      if (el.t >= 1) { electrons.splice(k, 1); continue; }
      const a3 = edges[el.edge * 2] * 3, b3 = edges[el.edge * 2 + 1] * 3;
      const o = alive * 3;
      elPos[o] = pos[a3] + (pos[b3] - pos[a3]) * el.t;
      elPos[o + 1] = pos[a3 + 1] + (pos[b3 + 1] - pos[a3 + 1]) * el.t;
      elPos[o + 2] = pos[a3 + 2] + (pos[b3 + 2] - pos[a3 + 2]) * el.t;
      alive++;
    }
    elGeo.setDrawRange(0, alive);
    (elGeo.getAttribute("position") as THREE.BufferAttribute).needsUpdate = true;

    // Material.
    pointMat.opacity = clamp01(bright + envelope * 0.15 + pulseEnergy * 0.2);
    pointMat.size = size + envelope * 0.05 + pulseEnergy * 0.08;
    pointMat.color.copy(colorCur);

    // Camera drift.
    camera.position.x = Math.sin(t * 0.02 * motion) * 5;
    camera.position.y = Math.cos(t * 0.03 * motion) * 3;
    camera.lookAt(0, 0, cloudZ * 0.2);

    renderer.render(scene, camera);
  }

  function onResize(): void {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight, false);
  }
  window.addEventListener("resize", onResize);
  rafId = requestAnimationFrame(frame);

  return {
    setState(s: OrbState) {
      if (s === state) return;
      state = s;
      target = TARGETS[s];
      tumble = Math.max(tumble, 1);
    },
    setPlaybackAnalyser(a: AnalyserNode | null) {
      playAnalyser = a;
      playBuf = new Float32Array(a ? a.fftSize : 512);
    },
    setMicLevel(level: number) {
      micRaw = level;
    },
    pulse() {
      pulseEnergy = 1;
    },
    destroy() {
      destroyed = true;
      cancelAnimationFrame(rafId);
      window.removeEventListener("resize", onResize);
      pointGeo.dispose();
      linkGeo.dispose();
      elGeo.dispose();
      renderer.dispose();
    },
  };
}
