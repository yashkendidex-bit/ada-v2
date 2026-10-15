/**
 * The A.D.A orb: a teal particle ring that crumples when she speaks.
 *
 * Drawn to match a reference video of the Iron Man 3 second-screen JARVIS app.
 * Measured from that video, at radius R = 1:
 *   - a dense outer band at 1.00, a second ring at 0.91 and a thin solid ring
 *     at 0.86, all perfect circles when calm;
 *   - a faint ring at 0.64, a scalloped ring near 0.5, and a honeycomb sphere
 *     at 0.30 with a bright rim;
 *   - while a voice plays, the outer rings fold into faceted sheets. The folds
 *     follow the voice's loudness with no visible lag, rise in about three
 *     frames, and snap back between syllables;
 *   - colour: hue about 178 degrees on a near-black ground, with a slanted
 *     teal light column behind the orb.
 *
 * Everything is a GPU point sprite and the folding is computed in the vertex
 * shader, so each frame costs a handful of uniform updates on the CPU. Time
 * comes from the browser's frame timestamp, so the motion is the same at 60 Hz
 * and 144 Hz.
 *
 * No code or imagery from the reference is used. See NOTICE.md.
 */

import * as THREE from "three";
import { VoiceEnvelope, decay, easeTo, follow, rmsOf, unitRange } from "./orb-motion";

// "compacting": the brain is being swapped for a fresh one behind the scenes
// (a context rotation). He answers nothing for a few seconds, and silence with
// no explanation reads as a crash. The orb carries it instead: dim, slow,
// drawn inward -- unmistakably not "about to answer".
export type OrbState = "idle" | "listening" | "thinking" | "speaking" | "compacting";

export interface Orb {
  setState(s: OrbState): void;
  setAnalyser(a: AnalyserNode | null): void;
  /** Microphone RMS, called every frame while the mic is live. Silence stops it counting. */
  setMicLevel(rms: number): void;
  /**
   * What the screen is showing on top of the orb. `inner` (0..1) is how much of
   * the orb's inside is covered by a widget: the core dims behind it. `busy`
   * (0..1) is how hard she is working: the ring churns, as in the reference
   * while a task runs.
   */
  setFocus(focus: { inner?: number; busy?: number }): void;
  /** Where the orb is on screen, in CSS pixels. `r` is the outer ring's radius. */
  getLayout(): { cx: number; cy: number; r: number };
  /** Called whenever the layout changes (window resize). */
  onLayout(fn: () => void): void;
  /** Particle budget, 1 = full. The orb lowers it by itself when frames run long. */
  getQuality(): number;
  destroy(): void;
}

/** Orb centre as a fraction of the window height (the reference sits just above the middle). */
const CENTER_Y = 0.49;
/** Orb diameter: 79% of the width in a tall window (as in the reference), at most 62% of the height. */
const FIT_W = 0.79;
const FIT_H = 0.62;
const FOV = 22;
/** The reference orb is 190 px in radius; point sizes are authored at that scale. */
const REF_RADIUS_PX = 190;

// Microphone RMS: at or below the floor is room noise; at the ceiling the orb
// is fully responding.
const MIC_FLOOR = 0.012;
const MIC_CEIL = 0.06;
const MIC_STALE_MS = 250;
/** How far the ring folds at full drive, in orb radii. Calibrated against the reference (see NOTICE.md). */
const AMP = 0.2;

/**
 * Strengths of the three bright rings, tuned against the reference's measured
 * pixel values at the same scale (see NOTICE.md). Alpha is per sprite; the rings
 * are dense, so many sprites overlap and build up the soft band.
 */
const BAND = {
  a: { radius: 1.0, strands: 24, spacing: 0.0038, jitter: 0.0042, size: 3.4, alpha: 0.019 },
  b: { radius: 0.925, strands: 5, spacing: 0.011, jitter: 0.006, size: 2.6, alpha: 0.055 },
  thin: { radius: 0.862, strands: 1, spacing: 0, jitter: 0.008, size: 3.2, alpha: 0.075 },
};

/** Ring colour. Hue about 178 degrees, as measured. */
const COLOR = new THREE.Color(0.2, 0.88, 0.83);

// ---------------------------------------------------------------------------
// Shaders
// ---------------------------------------------------------------------------

const NOISE = /* glsl */ `
float h2(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123); }
// Noise that is linear around the ring (so its corners read as facets) and
// smooth through time. cells: how many facets around; tt: time in keyframes.
float facet(float th, float cells, float tt, float seed) {
  float x = th / 6.28318530718 * cells;
  float i = floor(x);
  float f = x - i;
  float i0 = mod(i, cells);
  float i1 = mod(i + 1.0, cells);
  float t0 = mod(floor(tt), 1024.0);
  float tf = fract(tt);
  tf = tf * tf * (3.0 - 2.0 * tf);
  float s = seed * 17.0;
  float v0 = mix(h2(vec2(i0 + s, t0)), h2(vec2(i1 + s, t0)), f);
  float v1 = mix(h2(vec2(i0 + s, t0 + 1.0)), h2(vec2(i1 + s, t0 + 1.0)), f);
  return mix(v0, v1, tf) * 2.0 - 1.0;
}
`;

const RING_VERT = /* glsl */ `
precision highp float;
uniform float uTime, uAmp, uWide, uScale, uPx, uGlow, uDust, uQ;
attribute vec4 aP;   // strand, weight, size, alpha
attribute vec2 aR;   // random, kind (0 plain, 1 ticked, 2 dust)
varying float vA;
${NOISE}
void main() {
  float th = position.x, rad = position.y, off = position.z;
  float strand = aP.x, weight = aP.y, size = aP.z, alpha = aP.w;
  float rnd = aR.x, kind = aR.y;

  float tt = uTime * 7.5;
  float low = facet(th, 6.0, tt * 0.4, 1.0);
  float mid = facet(th, 15.0, tt, 2.0);
  float hi  = facet(th, 31.0, tt * 1.3, 3.0);
  float lump = (low + 0.7 * mid + 0.4 * hi) / 2.1;
  // Neighbouring strands run the same motion a few milliseconds apart, so they
  // move together like a curtain; strands far apart move independently.
  float ts = tt + strand * 0.15;
  float own = (facet(th, 11.0, ts * 1.1, 5.0) + 0.5 * facet(th, 23.0, ts * 1.25, 11.0)) / 1.5;
  float d = (lump * 0.4 + own * 0.6) * weight;
  float z = facet(th, 9.0, ts * 0.9, 7.0) * weight;

  // Driven bands splay apart into separate sheets.
  float spread = 1.0 + uWide * 1.4 * abs(own);
  float r = (rad + off * spread) * uScale + d * uAmp;
  if (kind > 1.5) r += uDust * (0.01 + 0.06 * rnd);

  vec4 mv = modelViewMatrix * vec4(cos(th) * r, sin(th) * r, z * uAmp * 0.9, 1.0);
  gl_Position = projectionMatrix * mv;
  gl_PointSize = size * uPx * (0.85 + 0.3 * rnd);

  float tick = (kind > 0.5 && kind < 1.5) ? 0.9 + 0.1 * step(0.5, fract(th * 23.873)) : 1.0;
  float twinkle = 0.85 + 0.15 * sin(uTime * (1.5 + rnd * 3.0) + rnd * 40.0);
  float a = alpha * uGlow * uQ * tick * twinkle;
  vA = kind > 1.5 ? a * uDust * 2.0 : a;
}
`;

const SOFT_FRAG = /* glsl */ `
precision highp float;
uniform vec3 uColor;
varying float vA;
void main() {
  vec2 c = gl_PointCoord - 0.5;
  float d = dot(c, c) * 4.0;
  if (d > 1.0) discard;
  float a = 1.0 - d;
  vec3 col = mix(uColor, vec3(0.8, 1.0, 0.98), pow(a, 3.0) * 0.6);
  gl_FragColor = vec4(col, a * a * vA);
}
`;

const CORE_VERT = /* glsl */ `
precision highp float;
uniform float uTime, uSpin, uTilt, uCore, uPx, uGlow, uSparkle;
attribute vec4 aP;   // size, alpha, kind (0 edge, 1 vertex, 2 sparkle, 3 rim), phase
varying float vA;
varying float vWhite;
void main() {
  vec3 q = position;
  float kind = aP.z;
  float depth = 1.0;
  if (kind < 2.5) {
    float cs = cos(uSpin), sn = sin(uSpin);
    q = vec3(cs * q.x + sn * q.z, q.y, -sn * q.x + cs * q.z);
    float ct = cos(uTilt), st = sin(uTilt);
    q = vec3(q.x, ct * q.y - st * q.z, st * q.y + ct * q.z);
    // Deep shell: the far side drops almost to black, the near side carries the light.
    depth = smoothstep(-0.2, 0.85, q.z);
    depth *= depth;
    // One soft key light from the upper left that slowly drifts, so the sphere
    // has a lit side and a dark side instead of an even glow.
    vec3 L = normalize(vec3(-0.55 + 0.25 * sin(uTime * 0.23), 0.6 + 0.15 * cos(uTime * 0.17), 0.7));
    float lit = 0.18 + 0.95 * pow(max(dot(normalize(q), L), 0.0), 1.6);
    // A faint band of light that travels around the mesh.
    float band = exp(-pow((q.y - 0.6 * sin(uTime * 0.4)) * 3.2, 2.0)) * 0.45;
    depth *= lit + band;
    // Light catching the edge of the shell, front and back, like glass seen at a grazing angle.
    float rim = pow(1.0 - abs(normalize(q).z), 3.0);
    depth += rim * 0.55;
  }
  vec4 mv = modelViewMatrix * vec4(q * uCore, 1.0);
  gl_Position = projectionMatrix * mv;
  bool shell = kind < 2.5;   // the mesh and its sparkles; the rim rings keep their own brightness
  gl_PointSize = aP.x * uPx * (shell ? mix(0.75, 1.5, clamp(depth, 0.0, 1.0)) : 1.0);
  float a = shell ? aP.y * mix(0.04, 1.6, clamp(depth, 0.0, 1.4)) * 2.6 : aP.y;
  vWhite = 0.0;
  if (kind > 1.5 && kind < 2.5) {
    float tw = 0.5 + 0.5 * sin(uTime * (1.2 + aP.w * 2.0) * (1.0 + uSparkle) + aP.w * 50.0);
    a *= pow(tw, 8.0) * 2.4;
    vWhite = 1.0;
  }
  vA = a * uGlow;
}
`;

const CORE_FRAG = /* glsl */ `
precision highp float;
uniform vec3 uColor;
varying float vA;
varying float vWhite;
void main() {
  vec2 c = gl_PointCoord - 0.5;
  float d = dot(c, c) * 4.0;
  if (d > 1.0) discard;
  float a = 1.0 - d;
  gl_FragColor = vec4(mix(uColor, vec3(0.85, 1.0, 0.98), vWhite), a * a * vA);
}
`;

const ARC_VERT = /* glsl */ `
precision highp float;
uniform float uPx, uThink, uHeadA, uHeadB, uScale, uGlow;
attribute vec4 aP;   // which arc (0/1), size, alpha, random
varying float vA;
float tail(float x) {
  x = mod(x, 6.28318530718);
  return exp(-x * 1.7) * (1.0 - smoothstep(5.2, 6.28, x));
}
void main() {
  float th = position.x;
  float r = (position.y + position.z) * uScale;
  vec4 mv = modelViewMatrix * vec4(cos(th) * r, sin(th) * r, 0.0, 1.0);
  gl_Position = projectionMatrix * mv;
  gl_PointSize = aP.y * uPx;
  float t = aP.x < 0.5 ? tail(uHeadA - th) : tail(th - uHeadB);
  vA = aP.z * uThink * t * uGlow;
}
`;

const TEXT_VERT = /* glsl */ `
precision highp float;
uniform float uForm, uTime, uPx, uGlow;
attribute vec4 aP;   // random, random, size, alpha
varying float vA;
void main() {
  float e = uForm * uForm * (3.0 - 2.0 * uForm);
  float ang = aP.x * 6.28318530718 + uTime * 0.5 * (aP.y - 0.5);
  float rr = 0.2 + 0.75 * aP.y;
  vec2 scatter = vec2(cos(ang), sin(ang)) * rr;
  vec2 p = mix(scatter, position.xy, e);
  p.x += sin(uTime * 30.0 + aP.y * 80.0) * 0.004 * e;   // scanline shimmer
  vec4 mv = modelViewMatrix * vec4(p, 0.0, 1.0);
  gl_Position = projectionMatrix * mv;
  gl_PointSize = aP.z * uPx;
  vA = aP.w * e * (0.5 + 0.5 * aP.x) * uGlow;
}
`;

// Background: near black with the slanted teal light column measured from the
// reference. X and Y are in orb radii, from the orb centre, Y downward.
const BG_VERT = /* glsl */ `
void main() { gl_Position = vec4(position.xy, 0.9999, 1.0); }
`;

const BG_FRAG = /* glsl */ `
precision highp float;
uniform vec2 uRes;      // CSS px
uniform vec2 uCenter;   // orb centre, CSS px from the top left
uniform float uRadius;  // orb radius, CSS px
uniform float uPixelRatio;
uniform float uGlow;
float seg(float y, float a, float b, float va, float vb) {
  return mix(va, vb, clamp((y - a) / (b - a), 0.0, 1.0));
}
float hexDist(vec2 p) {
  p = abs(p);
  return max(dot(p, normalize(vec2(1.0, 1.7320508))), p.x);
}
void main() {
  vec2 px = vec2(gl_FragCoord.x, uRes.y * uPixelRatio - gl_FragCoord.y) / uPixelRatio;
  float X = (px.x - uCenter.x) / uRadius;
  float Y = (px.y - uCenter.y) / uRadius;

  // Brightness along the column, in 8-bit green units, from the reference.
  float I = seg(Y, -2.3, -1.83, 5.0, 13.0);
  I = seg(Y, -1.83, -1.51, I, 25.0);
  I = seg(Y, -1.51, -1.19, I, 34.0);
  I = seg(Y, -1.19, 0.0, I, 34.0);
  I = seg(Y, 0.0, 1.23, I, 41.0);
  I = seg(Y, 1.23, 1.6, I, 40.0);
  I = seg(Y, 1.6, 2.4, I, 33.0);

  float xc = -0.096 - 0.35 * Y;                      // the column leans toward lower left
  float dx = X - xc;
  float t = smoothstep(-1.2, 1.2, Y);
  float w = mix(0.8, 1.0, t);
  float beam = exp(-pow(abs(dx / w), 2.2));
  float g = I / 255.0 * beam;

  // A very faint honeycomb at the lower left, as in the reference.
  vec2 hp = vec2(X + 1.2, Y - 1.5) * 5.0;
  vec2 r = vec2(1.0, 1.7320508);
  vec2 a = mod(hp, r) - r * 0.5;
  vec2 b = mod(hp - r * 0.5, r) - r * 0.5;
  vec2 gv = dot(a, a) < dot(b, b) ? a : b;
  float hex = smoothstep(0.05, 0.0, abs(0.5 - hexDist(gv)));
  float hexMask = exp(-pow(length((vec2(X, Y) - vec2(-1.3, 1.5)) / vec2(0.9, 0.7)), 2.0));
  g += hex * hexMask * 0.014;

  // The interior of the ring is lit, not black.
  g += (1.0 - smoothstep(0.7, 1.02, length(vec2(X, Y)))) * 0.03;

  // A soft glow behind the core.
  g += exp(-pow(length(vec2(X, Y)) / 0.32, 2.0)) * 0.05;

  g *= uGlow;
  gl_FragColor = vec4(0.0, g * 0.96, g * 1.06, 1.0);
}
`;

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

/** Grows typed attribute data without knowing the size in advance. */
class Buf {
  pos: number[] = [];
  p: number[] = [];
  r: number[] = [];
  get count(): number { return this.pos.length / 3; }
}

function rnd(): number { return Math.random(); }

/**
 * A ring of points. strands: how many independent sheets the band splits into
 * when it is driven. kind: 0 plain, 1 ticked, 2 dust.
 */
function addRing(
  b: Buf, count: number, radius: number, spacing: number, jitter: number,
  size: number, alpha: number, weight: number, strands: number, kind: number,
): void {
  // Each strand spreads its points evenly around the ring with a small random
  // nudge, instead of dropping them anywhere. Random placement leaves clumps and
  // gaps that read as grain; this reads as a continuous band.
  const per = Math.max(1, Math.round(count / strands));
  for (let strand = 0; strand < strands; strand++) {
    const phase = (strand * 0.6180339887) % 1;
    // Each strand is a thin sheet at its own radial offset. Calm, they stack into
    // one soft band; driven, they part and fold independently.
    const base = (strand - (strands - 1) / 2) * spacing;
    // The outer sheets of the band read brighter than the inner ones.
    const edge = kind === 1 && strands > 1 ? 0.55 + 0.9 * (strand / (strands - 1)) : 1;
    for (let j = 0; j < per; j++) {
      const th = (((j + rnd()) / per + phase) % 1) * Math.PI * 2;
      b.pos.push(th, radius, base + (rnd() * 2 - 1) * jitter);
      b.p.push(strand, weight, size * (0.8 + 0.5 * rnd()), alpha * edge);
      b.r.push(rnd(), kind);
    }
  }
}

function ringGeometry(b: Buf, attrs: { p: string; r?: string }): THREE.BufferGeometry {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(b.pos), 3));
  g.setAttribute(attrs.p, new THREE.BufferAttribute(new Float32Array(b.p), 4));
  if (attrs.r) g.setAttribute(attrs.r, new THREE.BufferAttribute(new Float32Array(b.r), 2));
  return g;
}

/** A geodesic sphere: the triangle mesh of a subdivided icosahedron, as dots along every edge and a brighter dot at every corner. */
function buildCore(): THREE.BufferGeometry {
  const ico = new THREE.IcosahedronGeometry(1, 5); // non-indexed: three vertices per face
  const src = ico.getAttribute("position") as THREE.BufferAttribute;
  const faces = src.count / 3;
  const key = (i: number) =>
    `${Math.round(src.getX(i) * 1e4)},${Math.round(src.getY(i) * 1e4)},${Math.round(src.getZ(i) * 1e4)}`;
  const vid = new Map<string, number>();
  const verts: THREE.Vector3[] = [];
  const id = (i: number) => {
    const k = key(i);
    let v = vid.get(k);
    if (v === undefined) {
      v = verts.length;
      vid.set(k, v);
      verts.push(new THREE.Vector3(src.getX(i), src.getY(i), src.getZ(i)).normalize());
    }
    return v;
  };
  const edges = new Set<string>();
  for (let f = 0; f < faces; f++) {
    const v = [id(f * 3), id(f * 3 + 1), id(f * 3 + 2)];
    for (let k = 0; k < 3; k++) {
      const a = v[k], bb = v[(k + 1) % 3];
      edges.add(a < bb ? `${a}_${bb}` : `${bb}_${a}`);
    }
  }

  const pos: number[] = [];
  const p: number[] = [];
  const dotsPerEdge = 7;
  for (const e of edges) {
    const [ia, ib] = e.split("_").map(Number);
    for (let s = 1; s < dotsPerEdge; s++) {
      const q = verts[ia].clone().lerp(verts[ib], s / dotsPerEdge).normalize();
      pos.push(q.x, q.y, q.z);
      p.push(1.1, 0.2, 0, 0);
    }
  }
  for (const v of verts) {
    pos.push(v.x, v.y, v.z);
    p.push(1.7, 0.3, 1, 0);
  }
  // A few white sparkles that come and go.
  for (let i = 0; i < 14; i++) {
    const th = rnd() * Math.PI * 2, ph = Math.acos(2 * rnd() - 1);
    pos.push(Math.sin(ph) * Math.cos(th), Math.sin(ph) * Math.sin(th), Math.cos(ph));
    p.push(2.6, 0.9, 2, rnd());
  }
  // Bright rim, fixed in front of the spinning sphere, and a faint outer ring.
  for (let i = 0; i < 1500; i++) {
    const th = (i / 1500) * Math.PI * 2;
    pos.push(Math.cos(th) * 1.03, Math.sin(th) * 1.03, 0);
    p.push(1.9, 0.1, 3, 0);
  }
  for (let i = 0; i < 1800; i++) {
    const th = (i / 1800) * Math.PI * 2;
    pos.push(Math.cos(th) * 1.2, Math.sin(th) * 1.2, 0);
    p.push(1.5, 0.04, 3, 0);
  }
  ico.dispose();
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(pos), 3));
  g.setAttribute("aP", new THREE.BufferAttribute(new Float32Array(p), 4));
  return g;
}

/** Points that spell the boot word, in orb radii. */
function buildText(word: string): THREE.BufferGeometry {
  const W = 720, H = 240;
  const cv = document.createElement("canvas");
  cv.width = W; cv.height = H;
  const cx = cv.getContext("2d");
  const pos: number[] = [];
  const p: number[] = [];
  if (cx) {
    cx.fillStyle = "#fff";
    cx.textAlign = "center";
    cx.textBaseline = "middle";
    cx.font = "italic 900 170px 'Arial Black', Impact, 'Helvetica Neue', Arial, sans-serif";
    cx.fillText(word, W / 2, H / 2 + 6);
    const img = cx.getImageData(0, 0, W, H).data;
    const step = 3;
    for (let y = 0; y < H; y += step) {
      for (let x = 0; x < W; x += step) {
        if (img[(y * W + x) * 4 + 3] < 128) continue;
        // 440 canvas px span 1.18 orb radii.
        pos.push((x - W / 2) / 440 * 1.18, -(y - H / 2) / 440 * 1.18, 0);
        p.push(rnd(), rnd(), 1.2 + rnd() * 0.5, 0.5 + 0.5 * rnd());
      }
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(pos), 3));
  g.setAttribute("aP", new THREE.BufferAttribute(new Float32Array(p), 4));
  return g;
}

// ---------------------------------------------------------------------------
// The orb
// ---------------------------------------------------------------------------

interface Look { glow: number; base: number; scale: number; think: number; spin: number; hear: number; }
const LOOK: Record<OrbState, Look> = {
  idle:       { glow: 0.55, base: 0.004, scale: 1.0,  think: 0, spin: 0.12, hear: 0 },
  listening:  { glow: 1.0,  base: 0.006, scale: 1.0,  think: 0, spin: 0.2,  hear: 0.45 },
  thinking:   { glow: 0.95, base: 0.012, scale: 0.985, think: 1, spin: 0.9,  hear: 0.25 },
  speaking:   { glow: 1.0,  base: 0.006, scale: 1.0,  think: 0, spin: 0.35, hear: 0.2 },
  compacting: { glow: 0.32, base: 0.002, scale: 0.8,  think: 0, spin: 0.05, hear: 0 },
};

export function createOrb(canvas: HTMLCanvasElement): Orb {
  let destroyed = false;

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x000000, 1);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(FOV, 1, 0.1, 100);
  const group = new THREE.Group();
  scene.add(group);

  // ── Background ──
  const bgU = {
    uRes: { value: new THREE.Vector2(1, 1) },
    uCenter: { value: new THREE.Vector2(0, 0) },
    uRadius: { value: REF_RADIUS_PX },
    uPixelRatio: { value: 1 },
    uGlow: { value: 1 },
  };
  const bg = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.ShaderMaterial({
      vertexShader: BG_VERT, fragmentShader: BG_FRAG, uniforms: bgU,
      depthTest: false, depthWrite: false,
    }),
  );
  bg.frustumCulled = false;
  bg.renderOrder = -10;
  scene.add(bg);

  // ── Rings ──
  const ringU = {
    uTime: { value: 0 }, uAmp: { value: 0 }, uWide: { value: 0 }, uScale: { value: 1 },
    uPx: { value: 1 }, uGlow: { value: 1 }, uDust: { value: 0 }, uQ: { value: 1 }, uColor: { value: COLOR },
  };
  // The big outer band has its own alpha scale, so its particle count can drop on a slow machine.
  const bandU = { ...ringU, uQ: { value: 1 } };
  const additive = { transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending } as const;
  const ringMat = new THREE.ShaderMaterial({ vertexShader: RING_VERT, fragmentShader: SOFT_FRAG, uniforms: ringU, ...additive });
  const bandMat = new THREE.ShaderMaterial({ vertexShader: RING_VERT, fragmentShader: SOFT_FRAG, uniforms: bandU, ...additive });

  const ba = new Buf();
  const rb = new Buf();
  // Soft, wide sprites at low alpha: the reference is a continuous teal strip, not a dotted line.
  addRing(ba, 70000, BAND.a.radius, BAND.a.spacing, BAND.a.jitter, BAND.a.size, BAND.a.alpha, 1.0, BAND.a.strands, 1); // outer band, twenty-four sheets
  addRing(rb, 12000, BAND.b.radius, BAND.b.spacing, BAND.b.jitter, BAND.b.size, BAND.b.alpha, 0.85, BAND.b.strands, 0); // second ring
  addRing(rb, 5000, BAND.thin.radius, BAND.thin.spacing, BAND.thin.jitter, BAND.thin.size, BAND.thin.alpha, 0.35, 1, 0); // thin solid ring
  addRing(rb, 2500, 0.64, 0, 0.004, 2.2, 0.003, 0.05, 1, 0);                                // faint ring
  addRing(rb, 450, 1.0, 0.0075, 0.003, 1.6, 0.14, 1.0, 12, 2);                                    // dust thrown off the band
  for (let i = 0; i < 150; i++) {                                                       // sparse bright ticks in the band
    rb.pos.push((i / 150) * Math.PI * 2, 0.985, 0);
    rb.p.push(0, 1.0, 2.4, 0.1);
    rb.r.push(rnd(), 0);
  }
  const band = new THREE.Points(ringGeometry(ba, { p: "aP", r: "aR" }), bandMat);
  band.frustumCulled = false;
  group.add(band);
  const bandCount = ba.count;
  const rings = new THREE.Points(ringGeometry(rb, { p: "aP", r: "aR" }), ringMat);
  rings.frustumCulled = false;
  group.add(rings);

  // Scalloped ring around the core: a ring whose radius ripples. Slowly turning, so it is its own object.
  const sb = new Buf();
  for (let i = 0; i < 2600; i++) {
    const th = (i / 2600) * Math.PI * 2 + rnd() * 0.002;
    const r = 0.52 + 0.022 * Math.abs(Math.sin(th * 14));
    sb.pos.push(th, r, (rnd() * 2 - 1) * 0.003);
    sb.p.push(0, 0.02, 1.8 * (0.8 + 0.5 * rnd()), 0.012);
    sb.r.push(rnd(), 0);
  }
  for (let i = 0; i < 1600; i++) {
    const th = (i / 1600) * Math.PI * 2;
    sb.pos.push(th, 0.455, (rnd() * 2 - 1) * 0.003);
    sb.p.push(0, 0.02, 1.6, 0.006);
    sb.r.push(rnd(), 0);
  }
  const scallop = new THREE.Points(ringGeometry(sb, { p: "aP", r: "aR" }), ringMat);
  scallop.frustumCulled = false;
  group.add(scallop);

  // ── Core ──
  const coreU = {
    uTime: { value: 0 }, uSpin: { value: 0 }, uTilt: { value: 0 }, uCore: { value: 0.3 },
    uPx: { value: 1 }, uGlow: { value: 1 }, uSparkle: { value: 0 }, uColor: { value: COLOR },
  };
  const core = new THREE.Points(
    buildCore(),
    new THREE.ShaderMaterial({ vertexShader: CORE_VERT, fragmentShader: CORE_FRAG, uniforms: coreU, ...additive }),
  );
  core.frustumCulled = false;
  group.add(core);

  // ── Thinking arcs: two bright arcs sweeping opposite ways ──
  const ab = new Buf();
  for (let arc = 0; arc < 2; arc++) {
    const radius = arc === 0 ? 0.78 : 0.7;
    for (let i = 0; i < 1500; i++) {
      ab.pos.push(rnd() * Math.PI * 2, radius, (rnd() * 2 - 1) * 0.004);
      ab.p.push(arc, 1.3 * (0.8 + 0.5 * rnd()), 0.9, rnd());
    }
  }
  const arcU = {
    uPx: { value: 1 }, uThink: { value: 0 }, uHeadA: { value: 0 }, uHeadB: { value: 0 },
    uScale: { value: 1 }, uGlow: { value: 1 }, uColor: { value: COLOR },
  };
  const arcs = new THREE.Points(
    ringGeometry(ab, { p: "aP" }),
    new THREE.ShaderMaterial({ vertexShader: ARC_VERT, fragmentShader: SOFT_FRAG, uniforms: arcU, ...additive }),
  );
  arcs.frustumCulled = false;
  group.add(arcs);

  // ── Boot word ──
  const textU = {
    uForm: { value: 0 }, uTime: { value: 0 }, uPx: { value: 1 }, uGlow: { value: 1 }, uColor: { value: COLOR },
  };
  const text = new THREE.Points(
    buildText("A.D.A"),
    new THREE.ShaderMaterial({ vertexShader: TEXT_VERT, fragmentShader: SOFT_FRAG, uniforms: textU, ...additive }),
  );
  text.frustumCulled = false;
  group.add(text);

  // ── Layout ──
  const where = { cx: 0, cy: 0, r: 1 };
  const layoutListeners: Array<() => void> = [];
  function layout(): void {
    const W = window.innerWidth, H = window.innerHeight;
    const pr = renderer.getPixelRatio();
    renderer.setSize(W, H);
    const radius = Math.min(FIT_W * W, FIT_H * H) / 2;
    const hWorld = H / radius;                                    // orb radii visible from top to bottom
    camera.aspect = W / H;
    camera.position.set(0, 0, hWorld / (2 * Math.tan((FOV * Math.PI) / 360)));
    camera.updateProjectionMatrix();
    group.position.y = (0.5 - CENTER_Y) * hWorld;
    const px = pr * (radius / REF_RADIUS_PX);
    ringU.uPx.value = px; coreU.uPx.value = px; arcU.uPx.value = px; textU.uPx.value = px;
    bgU.uRes.value.set(W, H);
    bgU.uCenter.value.set(W / 2, CENTER_Y * H);
    bgU.uRadius.value = radius;
    bgU.uPixelRatio.value = pr;
    where.cx = W / 2; where.cy = CENTER_Y * H; where.r = radius;
    for (const fn of layoutListeners) fn();
  }
  layout();
  window.addEventListener("resize", layout);

  // ── Live state ──
  let state: OrbState = "idle";
  let lastState: OrbState = "idle";
  let glow = LOOK.idle.glow, base = LOOK.idle.base, scale = LOOK.idle.scale;
  let think = 0, spinRate = LOOK.idle.spin;
  let spin = 0;
  let pulse = 0;
  let dust = 0;
  let hear = 0;
  let micRms = 0, micAt = 0;
  let analyser: AnalyserNode | null = null;
  let timeData = new Float32Array(1024);
  const voice = new VoiceEnvelope();
  let headA = 0, headB = 0;
  let focusInner = 0, focusInnerTo = 0, busy = 0, busyTo = 0;
  let quality = 1, slowFor = 0, fastFor = 0, cooldown = 0, ema = 1 / 60;
  let last = -1;
  let t = 0;
  const BOOT_START = 0.4, BOOT_FORM = 1.8, BOOT_HOLD = 2.4, BOOT_FADE = 1.4;

  function setQuality(q: number): void {
    quality = q;
    band.geometry.setDrawRange(0, Math.floor(bandCount * q));
    // Fewer particles, each a little brighter, so the band keeps its weight.
    bandU.uQ.value = 1 / Math.pow(q, 0.8);
    renderer.setPixelRatio(q < 0.5 ? 1 : Math.min(window.devicePixelRatio || 1, 2));
    layout();
  }

  function frame(ts: number): void {
    if (destroyed) return;
    requestAnimationFrame(frame);

    // Time comes from the frame timestamp: steady at any refresh rate.
    if (last < 0) last = ts;
    const dt = Math.min((ts - last) / 1000, 0.05);
    last = ts;
    t += dt;

    // Frame governor: if frames keep running long, draw fewer band particles.
    ema += (dt - ema) * 0.05;
    cooldown = Math.max(0, cooldown - dt);
    if (t > 3) {
      if (ema > 0.024) { slowFor += dt; fastFor = 0; } else if (ema < 0.0185) { fastFor += dt; slowFor = 0; } else { slowFor = 0; fastFor = 0; }
      if (cooldown === 0 && slowFor > 2 && quality > 0.4) { setQuality(quality === 1 ? 0.7 : quality > 0.6 ? 0.5 : 0.4); slowFor = 0; cooldown = 6; }
      else if (cooldown === 0 && fastFor > 10 && quality < 1) { setQuality(quality < 0.6 ? 0.7 : 1); fastFor = 0; cooldown = 12; }
    }
    focusInner = easeTo(focusInner, focusInnerTo, 4, dt);
    busy = easeTo(busy, busyTo, 2.5, dt);

    const look = LOOK[state];
    glow = easeTo(glow, look.glow, 3, dt);
    base = easeTo(base, look.base, 3, dt);
    scale = easeTo(scale, look.scale, 2.2, dt);
    think = easeTo(think, look.think, 3.5, dt);
    spinRate = easeTo(spinRate, look.spin, 2, dt);

    if (state !== lastState) {
      if (state === "speaking") pulse = 1;
      else if (state === "thinking") pulse = 0.5;
      lastState = state;
    }
    pulse = decay(pulse, 0.4, dt);

    // Her voice: loudness of what is playing, straight from the analyser.
    let rms = 0;
    if (analyser) {
      analyser.getFloatTimeDomainData(timeData);
      rms = rmsOf(timeData);
    }
    const voiceEnv = voice.update(rms, dt);

    // The user's voice, while the mic is live.
    const micLive = performance.now() - micAt < MIC_STALE_MS;
    const heardNow = micLive ? unitRange(micRms, MIC_FLOOR, MIC_CEIL) : 0;
    hear = follow(hear, heardNow, 0.03, 0.15, dt);

    // While a task runs the ring churns by itself, like the reference does
    // behind its "connecting" widget: irregular, never still, never at full strength.
    const churn = busy * (0.34 + 0.2 * Math.sin(t * 3.7) * Math.sin(t * 1.3 + 1.0) + 0.1 * Math.sin(t * 9.1));
    const drive = Math.max(state === "compacting" ? 0 : voiceEnv, hear * look.hear, churn);
    dust = follow(dust, Math.max(0, drive - 0.75) * 1.4, 0.03, 0.35, dt);

    // Boot word: forms, holds, dissolves.
    const bt = t - BOOT_START;
    let form = 0;
    if (bt > 0) {
      if (bt < BOOT_FORM) form = bt / BOOT_FORM;
      else if (bt < BOOT_FORM + BOOT_HOLD) form = 1;
      else if (bt < BOOT_FORM + BOOT_HOLD + BOOT_FADE) form = 1 - (bt - BOOT_FORM - BOOT_HOLD) / BOOT_FADE;
    }

    spin += dt * spinRate * (1 + drive * 1.5);
    headA += dt * 2.1;
    headB -= dt * 1.5;

    const g = glow * (1 + 0.45 * drive + 0.35 * pulse);
    ringU.uTime.value = t;
    ringU.uAmp.value = base + AMP * drive;
    ringU.uWide.value = drive;
    ringU.uScale.value = scale;
    ringU.uGlow.value = g;
    ringU.uDust.value = dust;

    coreU.uTime.value = t;
    coreU.uSpin.value = spin;
    coreU.uTilt.value = 0.35 * Math.sin(t * 0.21) + 0.12 * Math.sin(t * 0.53);
    coreU.uCore.value = 0.3 * scale * (1 + 0.035 * drive);
    coreU.uGlow.value = glow * (1 + 0.25 * drive + 0.3 * pulse) * (1 - 0.6 * focusInner);
    coreU.uSparkle.value = drive * 2;

    scallop.visible = focusInner < 0.9;
    scallop.rotation.z = t * 0.03 * (1 + think * 4);

    arcU.uThink.value = think;
    arcU.uHeadA.value = headA;
    arcU.uHeadB.value = headB;
    arcU.uScale.value = scale;
    arcU.uGlow.value = glow * (1 - 0.5 * focusInner);

    textU.uForm.value = form * (1 - focusInner);
    textU.uTime.value = t;
    textU.uGlow.value = glow * 1.4;

    bgU.uGlow.value = 0.55 + 0.45 * glow + 0.1 * drive;

    renderer.render(scene, camera);
  }
  requestAnimationFrame(frame);

  return {
    setState(s: OrbState) { state = s; },
    setAnalyser(a: AnalyserNode | null) {
      analyser = a;
      if (a) {
        // A longer window than the player's default gives a steadier loudness reading.
        a.fftSize = 1024;
        timeData = new Float32Array(a.fftSize);
      }
      voice.reset();
    },
    setMicLevel(rms: number) {
      micRms = rms;
      micAt = performance.now();
    },
    setFocus(f) {
      if (f.inner !== undefined) focusInnerTo = Math.max(0, Math.min(1, f.inner));
      if (f.busy !== undefined) busyTo = Math.max(0, Math.min(1, f.busy));
    },
    getLayout() { return { cx: where.cx, cy: where.cy, r: where.r }; },
    onLayout(fn: () => void) { layoutListeners.push(fn); },
    getQuality() { return quality; },
    destroy() {
      destroyed = true;
      window.removeEventListener("resize", layout);
      for (const o of [band, rings, scallop, core, arcs, text, bg]) {
        o.geometry.dispose();
        (o.material as THREE.Material).dispose();
      }
      renderer.dispose();
    },
  };
}
