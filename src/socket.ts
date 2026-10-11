/**
 * Socket.IO link to the A.D.A backend (backend/server.py).
 *
 * The event names and payload shapes here are the contract the backend
 * already speaks; they are the same ones the previous React GUI used.
 */

import { io, type Socket } from "socket.io-client";

export const BACKEND_URL = "http://127.0.0.1:8000";

// ── Server -> client payloads ──────────────────────────────────────────────

export interface TtsAudioEvent {
  audio: string;           // base64
  format?: string;         // "pcm" (float32 @24 kHz) | "mp3" | "wav"
  chunkId?: number;
  gen?: number;            // speech generation: bumps once per new turn
  utt?: number;            // utterance id, for playback anchoring
  words?: number;
}

export interface TranscriptionEvent {
  sender: string;          // "ADA" | "You" | other
  text: string;            // streamed fragment; same-sender fragments join
}

export interface AdaQuestionEvent {
  id: string | number;
  question: string;
  choices?: string[];
  default?: string;
}

export interface StatusEvent {
  msg: string;
}

export interface ErrorEvent {
  msg: string;
}

export interface AuthStatusEvent {
  authenticated: boolean;  // face auth: false while the lock screen is up
}

export interface AuthFrameEvent {
  image: string;           // base64 JPEG from the backend's camera
}

// ── Client -> server payloads ──────────────────────────────────────────────

export interface PlayedReport {
  utt: number;
  frac: number;
}

export interface ClientEvents {
  start_audio: { device_index: number | null; device_name: string | null; muted: boolean };
  user_input: { text: string };
  answer_question: { id: string | number; answer: string };
  tts_played: PlayedReport;
}

export function createLink(): Socket {
  return io(BACKEND_URL, {
    reconnection: true,
    reconnectionDelay: 500,
    reconnectionDelayMax: 4000,
  });
}
