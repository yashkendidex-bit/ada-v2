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
