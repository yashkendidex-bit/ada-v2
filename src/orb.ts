/**
 * JARVIS — Multi-mode particle visualization.
 *
 * Floating particles with line connections between nearby ones.
 * Lines fade in/out based on state. Transition tumble on state change.
 * Speaking pulls particles closer for denser connections.
 *
 * A.D.A changes (see NOTICE.md): motion is timed in seconds, not per frame, so
 * a high-refresh screen does not run the orb faster; the cloud hears the
 * microphone and swells with her own voice; it breathes and drifts while idle;
 * and a state change sends a visible pulse through the cloud.
 */

import * as THREE from "three";
import { decay, easeTo, follow, unitRange, wander } from "./orb-motion";

// "compacting": the brain is being swapped for a fresh one behind the scenes
// (a context rotation). He answers nothing for a few seconds, and silence with
// no explanation reads as a crash. It used to be followed by a spoken line;
// the user found that annoying, so the orb carries it instead: dim, slow,
// drawn inward, in a cooler colour -- unmistakably not "about to answer".
export type OrbState = "idle" | "listening" | "thinking" | "speaking" | "compacting";

export interface Orb {
  setState(s: OrbState): void;
  setAnalyser(a: AnalyserNode | null): void;
  /** Microphone RMS, called every frame while the mic is live. Silence stops it counting. */
  setMicLevel(rms: number): void;
  destroy(): void;
}

// Colours per state. Kept as objects so the frame loop does not allocate.
const COL_BASE = new THREE.Color(0x4ca8e8);
const COL_THINK = new THREE.Color(0x6ec4ff);
const COL_SPEAK = new THREE.Color(0x5ab8f0);
const COL_HOLD = new THREE.Color(0x3a5f8a);

// Microphone RMS: at or below the floor is room noise; at the ceiling the orb
// is fully responding. Speech from a laptop mic sits between the two.
const MIC_FLOOR = 0.012;
const MIC_CEIL = 0.06;
// The mic loop stops reporting when the mic is off (muted, locked, not wanted).
// After this long without a report, the orb stops hearing it.
const MIC_STALE_MS = 250;
// Her own voice, read from the playback analyser (0..1 across the band).
const VOICE_FLOOR = 0.12;
const VOICE_CEIL = 0.45;

export function createOrb(canvas: HTMLCanvasElement): Orb {
  let destroyed = false;
  const N = 2000;

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setClearColor(0x050508, 1);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 1, 1000);
  camera.position.z = 80;

  // ── Particles ──
  const geo = new THREE.BufferGeometry();
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

  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));

  const mat = new THREE.PointsMaterial({
    color: 0x4ca8e8, size: 0.4, transparent: true, opacity: 0.6,
    sizeAttenuation: true, blending: THREE.AdditiveBlending, depthWrite: false,
  });

  const points = new THREE.Points(geo, mat);
  scene.add(points);

  // ── Connection lines ──
  const MAX_LINES = 8000;
  const linePos = new Float32Array(MAX_LINES * 6);
  const lineGeo = new THREE.BufferGeometry();
  lineGeo.setAttribute("position", new THREE.BufferAttribute(linePos, 3));
  lineGeo.setDrawRange(0, 0);

  const lineMat = new THREE.LineBasicMaterial({
    color: 0x4ca8e8, transparent: true, opacity: 0.0,
    blending: THREE.AdditiveBlending, depthWrite: false,
  });

  const lines = new THREE.LineSegments(lineGeo, lineMat);
  scene.add(lines);

  // ── Electrons — bright dots that travel along connections ──
  const MAX_ELECTRONS = 200;
  const electronGeo = new THREE.BufferGeometry();
  const electronPos = new Float32Array(MAX_ELECTRONS * 3);
  electronGeo.setAttribute("position", new THREE.BufferAttribute(electronPos, 3));
  electronGeo.setDrawRange(0, 0);

  const electronMat = new THREE.PointsMaterial({
    color: 0xffffff, size: 0.8, transparent: true, opacity: 1.0,
    sizeAttenuation: true, blending: THREE.AdditiveBlending, depthWrite: false,
  });

  const electrons = new THREE.Points(electronGeo, electronMat);
  scene.add(electrons);

  // Each electron: start point, end point, progress (0-1), speed
  interface Electron { sx: number; sy: number; sz: number; ex: number; ey: number; ez: number; t: number; speed: number; }
  const activeElectrons: Electron[] = [];
  let electronSpawnRate = 0;
  let targetElectronRate = 0;
  let lastElectronSpawn = 0; // time of the last spawn, in seconds
  // Lines drawn this frame. Electrons pick one straight from linePos, so no
  // per-frame list of connections is built.
  let liveLines = 0;

  // ── State ──
  let state: OrbState = "idle";
  let targetRadius = 25, currentRadius = 25;
  let targetSpeed = 0.3, currentSpeed = 0.3;
  let targetBright = 0.6, currentBright = 0.6;
  let targetSize = 0.4, currentSize = 0.4;
  let targetSpin = 0.03, currentSpin = 0.03;     // slow turn of the whole cloud, rad/s
  let targetBreath = 1.0, currentBreath = 1.0;   // how far the radius breathes
  let lineAmount = 0, targetLineAmount = 0;
  let lineDistance = 8;

  // Transition tumble
  let spinX = 0, spinY = 0, spinZ = 0;
  let transitionEnergy = 0;
  let lastState: OrbState = "idle";

  // Pulses. `kick` swells the cloud outward (she starts or stops talking);
  // `gather` draws it inward (she starts thinking, or the user starts talking).
  // Each one decays on its own, so it reads as an event, not a state.
  let kick = 0, gather = 0;
  let quietFor = 0, onsetLatched = false;

  // Depth Z
  let cloudZ = 0, cloudZVel = 0;

  // ── Ears and voice ──
  let micRms = 0, micAt = 0;
  let hear = 0;   // the user's voice, 0..1, smoothed
  let voice = 0;  // her voice, 0..1, smoothed

  // ── Audio ──
  let analyser: AnalyserNode | null = null;
  let freqData = new Uint8Array(64);
  let bass = 0, mid = 0;

  const clock = new THREE.Clock();
  let t = 0; // seconds of animation time

  function animate() {
    if (destroyed) return;
    requestAnimationFrame(animate);

    // dt is capped so a backgrounded tab does not make the orb jump on return.
    const dt = Math.min(clock.getDelta(), 0.1);
    t += dt;
    // Some constants below were tuned per 60 Hz frame. f converts them.
    const f = dt * 60;

    switch (state) {
      case "idle":
        targetRadius = 28; targetSpeed = 0.2; targetBright = 0.5; targetSize = 0.35;
        targetLineAmount = 0.15; targetElectronRate = 0; targetSpin = 0.03; targetBreath = 1.0; break;
      case "listening":
        targetRadius = 22; targetSpeed = 0.3; targetBright = 0.65; targetSize = 0.4;
        targetLineAmount = 0.4; targetElectronRate = 0; targetSpin = 0.05; targetBreath = 0.6; break;
      case "thinking":
        targetRadius = 16; targetSpeed = 0.5; targetBright = 0.7; targetSize = 0.3;
        targetLineAmount = 1.0; targetElectronRate = 0.015; targetSpin = 0.12; targetBreath = 0.5; break;
      case "speaking":
        targetRadius = 18; targetSpeed = 0.2; targetBright = 0.7; targetSize = 0.4;
        targetLineAmount = 0.8; targetElectronRate = 0; targetSpin = 0.06; targetBreath = 0.8; break;
      case "compacting":
        // Smaller than idle, slower than anything, half the brightness, no
        // connecting lines: a held breath, not a working mind.
        targetRadius = 12; targetSpeed = 0.08; targetBright = 0.28; targetSize = 0.28;
        targetLineAmount = 0.0; targetElectronRate = 0; targetSpin = 0.01; targetBreath = 0.3; break;
    }

    // Settle into the target at a fixed rate per second, whatever the frame rate.
    const K = 1.3;
    currentRadius = easeTo(currentRadius, targetRadius, K, dt);
    currentSpeed = easeTo(currentSpeed, targetSpeed, K, dt);
    currentBright = easeTo(currentBright, targetBright, K, dt);
    currentSize = easeTo(currentSize, targetSize, K, dt);
    currentSpin = easeTo(currentSpin, targetSpin, K, dt);
    currentBreath = easeTo(currentBreath, targetBreath, K, dt);
    lineAmount = easeTo(lineAmount, targetLineAmount, K, dt);
    electronSpawnRate = easeTo(electronSpawnRate, targetElectronRate, K, dt);

    // ── State changes are events ──
    if (state !== lastState) {
      if (state === "speaking") kick = 1;            // her voice leaves the cloud
      else if (state === "thinking") gather = 1;     // she gathers herself
      else if (lastState === "speaking") kick = 0.4; // a small ripple as she stops
      transitionEnergy = 1.0;
      lastState = state;
    }
    transitionEnergy = decay(transitionEnergy, 1.1, dt);
    kick = decay(kick, 0.5, dt);
    gather = decay(gather, 0.7, dt);

    // ── Ears: the microphone, while it is live ──
    // While she talks, the mic mostly hears her own voice coming back, so it counts for much less.
    const micLive = performance.now() - micAt < MIC_STALE_MS;
    const heardNow = micLive ? unitRange(micRms, MIC_FLOOR, MIC_CEIL) : 0;
    hear = follow(hear, state === "speaking" ? heardNow * 0.2 : heardNow, 0.04, 0.3, dt);

    // The user starts talking after a quiet spell: she leans in once.
    if (hear < 0.12) {
      quietFor += dt;
      onsetLatched = false;
    } else {
      if (!onsetLatched && quietFor > 0.5 && state !== "speaking") {
        gather = Math.min(1, gather + 0.5);
        onsetLatched = true;
      }
      quietFor = 0;
    }

    // ── Audio ──
    bass = 0; mid = 0;
    let speechLevel = 0;
    if (analyser) {
      analyser.getByteFrequencyData(freqData);
      let bSum = 0, mSum = 0, aSum = 0;
      for (let i = 0; i < 8; i++) bSum += freqData[i];
      for (let i = 8; i < 24; i++) mSum += freqData[i];
      for (let i = 0; i < 32; i++) aSum += freqData[i];
      bass = bSum / (8 * 255); mid = mSum / (16 * 255);
      speechLevel = aSum / (32 * 255);
    }
    voice = follow(
      voice,
      state === "speaking" ? unitRange(speechLevel, VOICE_FLOOR, VOICE_CEIL) : 0,
      0.03, 0.22, dt
    );

    // Breathing: two unequal slow swells, so it never looks like a metronome.
    const breath = (Math.sin(t * 1.02) * 0.7 + Math.sin(t * 2.05 + 1.1) * 0.3) * currentBreath;

    // Depth Z
    let zTarget = Math.sin(t * 0.12) * 8;
    if (state === "thinking") zTarget = Math.sin(t * 0.3) * 15 + Math.sin(t * 0.9) * 6;
    else if (state === "speaking") zTarget = Math.sin(t * 0.15) * 6 - bass * 10;
    else if (state === "compacting") zTarget = -20 + Math.sin(t * 0.08) * 3;   // withdrawn, barely moving
    if (state !== "compacting") zTarget += hear * 5;                           // leans toward the user
    cloudZVel += (zTarget - cloudZ) * 0.008 * f;
    cloudZVel *= Math.pow(0.94, f);
    cloudZ += cloudZVel * f;

    // Pose: a constant slow turn, the tumble on state change, and a small
    // wandering tilt that brings the cloud a little toward the user.
    spinY += dt * (currentSpin + transitionEnergy * 0.9);
    spinX += dt * transitionEnergy * 0.72 * Math.sin(t * 1.7);
    spinZ += dt * transitionEnergy * 0.48 * Math.cos(t * 1.3);
    const rx = spinX + wander(t, 1.7) * 0.06 + hear * 0.04;
    const ry = spinY;
    const rz = spinZ + wander(t, 5.3) * 0.04;
    points.rotation.set(rx, ry, rz); points.position.z = cloudZ;
    lines.rotation.set(rx, ry, rz); lines.position.z = cloudZ;
    electrons.rotation.set(rx, ry, rz); electrons.position.z = cloudZ;

    // The radius the particles are pulled toward. Everything that moves the
    // cloud outward or inward adds to the state's own radius.
    const R = currentRadius + breath + kick * 2.5 + hear * 1.6 + voice * 1.2 - gather * 3;

    // ── Update particles ──
    const p = geo.getAttribute("position") as THREE.BufferAttribute;
    const a = p.array as Float32Array;
    const damp = Math.pow(0.992, f);
    const s = currentSpeed * f;

    for (let i = 0; i < N; i++) {
      const i3 = i * 3;
      let x = a[i3], y = a[i3 + 1], z = a[i3 + 2];
      const px = phase[i];

      vel[i3] += Math.sin(t * 0.05 + px) * 0.001 * s;
      vel[i3 + 1] += Math.cos(t * 0.06 + px * 1.3) * 0.001 * s;
      vel[i3 + 2] += Math.sin(t * 0.055 + px * 0.7) * 0.001 * s;
      vel[i3] += Math.sin(t * 0.02 + px * 2.1 + y * 0.1) * 0.0008 * s;
      vel[i3 + 1] += Math.cos(t * 0.025 + px * 1.7 + z * 0.1) * 0.0008 * s;
      vel[i3 + 2] += Math.sin(t * 0.022 + px * 0.9 + x * 0.1) * 0.0008 * s;

      const dist = Math.sqrt(x * x + y * y + z * z) || 0.01;
      const pull = (Math.max(0, dist - R) * 0.002 + 0.0003) * f;
      vel[i3] -= (x / dist) * pull;
      vel[i3 + 1] -= (y / dist) * pull;
      vel[i3 + 2] -= (z / dist) * pull;

      if (bass > 0.05) {
        const k = bass * 0.02 * f;
        vel[i3] += (x / dist) * k;
        vel[i3 + 1] += (y / dist) * k;
        vel[i3 + 2] += (z / dist) * k;
      }
      if (state === "speaking" && mid > 0.1) {
        const ripple = Math.sin(t * 8 + px);
        const k = mid * 0.012 * ripple * f;
        vel[i3] += (x / dist) * k;
        vel[i3 + 1] += (y / dist) * k;
      }
      if (hear > 0.05) {
        // Fine shimmer while the user talks: each particle answers a little differently.
        const k = Math.sin(t * 11 + px * 1.7) * hear * 0.006 * f;
        vel[i3] += (x / dist) * k;
        vel[i3 + 1] += (y / dist) * k;
        vel[i3 + 2] += (z / dist) * k;
      }

      vel[i3] *= damp; vel[i3 + 1] *= damp; vel[i3 + 2] *= damp;
      a[i3] += vel[i3] * f; a[i3 + 1] += vel[i3 + 1] * f; a[i3 + 2] += vel[i3 + 2] * f;
    }
    p.needsUpdate = true;

    // ── Update lines ──
    if (lineAmount > 0.01) {
      const lp = lineGeo.getAttribute("position") as THREE.BufferAttribute;
      const la = lp.array as Float32Array;
      let lineCount = 0;
      const maxDist = lineDistance * (1 + bass * 0.5 + hear * 0.25);
      const maxDistSq = maxDist * maxDist;
      const step = Math.max(1, Math.floor(N / 600));

      for (let i = 0; i < N && lineCount < MAX_LINES; i += step) {
        const i3 = i * 3;
        const x1 = a[i3], y1 = a[i3 + 1], z1 = a[i3 + 2];
        for (let j = i + step; j < N && lineCount < MAX_LINES; j += step) {
          const j3 = j * 3;
          const dx = a[j3] - x1, dy = a[j3 + 1] - y1, dz = a[j3 + 2] - z1;
          if (dx * dx + dy * dy + dz * dz < maxDistSq) {
            const idx = lineCount * 6;
            la[idx] = x1; la[idx+1] = y1; la[idx+2] = z1;
            la[idx+3] = a[j3]; la[idx+4] = a[j3+1]; la[idx+5] = a[j3+2];
            lineCount++;
          }
        }
      }
      lineGeo.setDrawRange(0, lineCount * 2);
      lp.needsUpdate = true;
      lineMat.opacity = lineAmount * 0.12;
      liveLines = lineCount;
    } else {
      lineGeo.setDrawRange(0, 0);
      liveLines = 0;
    }

    // ── Update electrons — only during thinking ──
    // One fires off every ~1 second, max 3 alive, takes 2-4s to travel
    if (liveLines > 0 && electronSpawnRate > 0.005) {
      if (activeElectrons.length < 3 && (t - lastElectronSpawn) > 1.0) {
        const ci = Math.floor(Math.random() * Math.min(liveLines, 500)) * 6;
        activeElectrons.push({
          sx: linePos[ci], sy: linePos[ci + 1], sz: linePos[ci + 2],
          ex: linePos[ci + 3], ey: linePos[ci + 4], ez: linePos[ci + 5],
          t: 0,
          speed: 0.003 + Math.random() * 0.003, // progress per 60 Hz frame: about 2-4 s to travel
        });
        lastElectronSpawn = t;
      }
    }

    // Update electron positions
    const ep = electronGeo.getAttribute("position") as THREE.BufferAttribute;
    const ea = ep.array as Float32Array;
    let aliveCount = 0;

    for (let e = activeElectrons.length - 1; e >= 0; e--) {
      const el = activeElectrons[e];
      el.t += el.speed * f;
      if (el.t >= 1) {
        activeElectrons.splice(e, 1);
        continue;
      }
      const ei = aliveCount * 3;
      ea[ei] = el.sx + (el.ex - el.sx) * el.t;
      ea[ei + 1] = el.sy + (el.ey - el.sy) * el.t;
      ea[ei + 2] = el.sz + (el.ez - el.sz) * el.t;
      aliveCount++;
    }

    electronGeo.setDrawRange(0, aliveCount);
    ep.needsUpdate = true;

    // Brightness: the state's level, plus a little on each pulse and a faint wander.
    mat.opacity = Math.min(1, currentBright + bass * 0.08 + kick * 0.12 + gather * 0.04 + wander(t, 0.9) * 0.02);
    mat.size = currentSize + bass * 0.05 + kick * 0.04;

    const colTarget = state === "thinking" ? COL_THINK
      : state === "speaking" ? COL_SPEAK
      : state === "compacting" ? COL_HOLD
      : COL_BASE;
    const cEase = 1 - Math.exp(-(state === "compacting" ? 1.8 : 0.9) * dt);
    mat.color.lerp(colTarget, cEase);
    lineMat.color.lerp(colTarget, cEase);

    camera.position.x = Math.sin(t * 0.02) * 5 + wander(t, 2.2) * 0.8;
    camera.position.y = Math.cos(t * 0.03) * 3 + wander(t, 3.9) * 0.5;
    camera.lookAt(0, 0, cloudZ * 0.2);

    renderer.render(scene, camera);
  }

  function onResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  }

  window.addEventListener("resize", onResize);
  animate();

  return {
    setState(s: OrbState) { state = s; },
    setAnalyser(a: AnalyserNode | null) {
      analyser = a;
      if (a) freqData = new Uint8Array(a.frequencyBinCount);
    },
    setMicLevel(rms: number) {
      micRms = rms;
      micAt = performance.now();
    },
    destroy() {
      destroyed = true;
      window.removeEventListener("resize", onResize);
      renderer.dispose();
    },
  };
}
