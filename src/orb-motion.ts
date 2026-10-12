/**
 * Timing helpers for the orb. Pure functions with no DOM or three.js, so the
 * behaviour can be checked on its own. Everything here works in seconds, so
 * the orb moves the same way at 60 Hz and at 144 Hz.
 */

/**
 * Close a fixed share of the gap to `target` over `dt` seconds. `k` is the rate
 * per second. Splitting dt into smaller steps gives the same result, which is
 * what makes it frame-rate independent.
 */
export function easeTo(current: number, target: number, k: number, dt: number): number {
  return current + (target - current) * (1 - Math.exp(-k * dt));
}

/**
 * Envelope follower: rises with time constant `attack`, falls with `release`
 * (both in seconds). Quick to react, slow to let go, so it reads as breath
 * rather than as flicker.
 */
export function follow(current: number, target: number, attack: number, release: number, dt: number): number {
  const tau = target > current ? attack : release;
  return current + (target - current) * (1 - Math.exp(-dt / tau));
}

/** Map `x` from the range [lo, hi] onto [0, 1], clamped at both ends. */
export function unitRange(x: number, lo: number, hi: number): number {
  const v = (x - lo) / (hi - lo);
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/** Decay a pulse toward zero. `tau` is the time constant in seconds. */
export function decay(v: number, tau: number, dt: number): number {
  return v * Math.exp(-dt / tau);
}

/**
 * Turns the loudness of her voice into how hard the ring is driven.
 *
 * Measured on the reference video: the ring distortion follows the voice's
 * loudness with no visible lag, rises to its peak in about three frames and
 * falls back between syllables in about four. So the follower is fast both
 * ways (a few tens of milliseconds), which keeps it reading as speech rhythm
 * rather than as a slow swell.
 *
 * The output is scaled against recent loudness, so a quiet voice and a loud
 * one drive the ring equally hard. The reference forgets slowly, so a long
 * pause lets it recover sensitivity.
 */
export class VoiceEnvelope {
  private fast = 0;
  private ref = 0.05;

  /** `rms` is the RMS of the playback signal for this frame (0..~0.5). Returns about 0..1.15. */
  update(rms: number, dt: number): number {
    this.fast = follow(this.fast, rms, 0.012, 0.07, dt);
    this.ref = Math.max(this.fast, 0.03, this.ref * Math.exp(-dt / 6));
    const gate = Math.min(0.012, this.ref * 0.15);
    const v = (this.fast - gate) / (this.ref * 0.85 - gate);
    return v < 0 ? 0 : v > 1.15 ? 1.15 : v;
  }

  reset(): void {
    this.fast = 0;
    this.ref = 0.05;
  }
}

/** RMS of a block of samples. */
export function rmsOf(samples: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  return Math.sqrt(sum / samples.length);
}

/**
 * Slow, aperiodic wander in [-1, 1]. Several sines with unrelated frequencies,
 * so it does not repeat on a visible cycle and does not look mechanical.
 */
export function wander(t: number, seed: number): number {
  return (
    Math.sin(t * 0.13 + seed) +
    0.6 * Math.sin(t * 0.047 + seed * 2.3) +
    0.3 * Math.sin(t * 0.31 + seed * 0.7)
  ) / 1.9;
}
