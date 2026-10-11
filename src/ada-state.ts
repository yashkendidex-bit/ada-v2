/**
 * Shared state between the A.D.A transport (ws.ts) and speech input
 * (ada-voice.ts). Kept apart so neither module has to import the other.
 */

/** A.D.A's backend (backend/server.py): Socket.IO on loopback. */
export const BACKEND_URL = "http://127.0.0.1:8000";

// ── Speech timing ──────────────────────────────────────────────────────────
// The browser does not report when queued audio finishes, so the transport
// estimates it from the duration of each chunk it hands to the player.

let speechEnds = 0; // wall-clock ms when queued speech is expected to end

export function speechEndsAt(): number {
  return speechEnds;
}

export function isAdaSpeaking(): boolean {
  return Date.now() < speechEnds;
}

export function queueSpeech(seconds: number): void {
  speechEnds = Math.max(Date.now(), speechEnds) + seconds * 1000;
}

export function endSpeech(): void {
  speechEnds = 0;
}

// ── Face-auth lock ─────────────────────────────────────────────────────────
// With face auth on, the backend refuses to start audio until the face matches.

let locked = false;
const lockListeners: Array<(locked: boolean) => void> = [];

export function isAdaLocked(): boolean {
  return locked;
}

export function setAdaLocked(value: boolean): void {
  if (value === locked) return;
  locked = value;
  for (const fn of lockListeners) fn(value);
}

export function onAdaLockChange(fn: (locked: boolean) => void): void {
  lockListeners.push(fn);
}

// ── Barge-in ───────────────────────────────────────────────────────────────
// Speech input reports "the user is talking over her"; the transport decides
// what that means for playback.

const bargeInListeners: Array<() => void> = [];

export function onBargeIn(fn: () => void): void {
  bargeInListeners.push(fn);
}

export function requestBargeIn(): void {
  for (const fn of bargeInListeners) fn();
}
