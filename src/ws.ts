/**
 * Transport for the JARVIS screen, backed by A.D.A's backend.
 *
 * main.ts was written against JARVIS's websocket protocol. This module presents
 * the same surface (createSocket, send, onMessage) and translates both ways to
 * A.D.A's Socket.IO events in backend/server.py, so the screen code is unchanged.
 */

import { io, type Socket } from "socket.io-client";
import {
  BACKEND_URL,
  endSpeech,
  isAdaLocked,
  isAdaSpeaking,
  onBargeIn,
  queueSpeech,
  setAdaLocked,
  speechEndsAt,
} from "./ada-state";

export type MessageHandler = (msg: Record<string, unknown>) => void;

export interface JarvisSocket {
  send(data: Record<string, unknown>): void;
  onMessage(handler: MessageHandler): void;
  close(): void;
  isConnected(): boolean;
}

// A.D.A payloads (the fields read here).
interface TtsAudioEvent {
  audio: string;
  format?: string;
  chunkId?: number;
  gen?: number;
  utt?: number;
}
interface TranscriptionEvent {
  sender: string;
  text: string;
}
interface AdaQuestionEvent {
  id: string | number;
  question: string;
  choices?: string[];
}
interface AuthStatusEvent {
  authenticated: boolean;
}
interface ErrorEvent {
  msg?: string;
}

/** A.D.A's speech is float32 mono PCM at this rate. */
const SAMPLE_RATE = 24000;
/** After the estimated end of speech, wait this long before going back to listening, so short gaps do not flicker. */
const SPEECH_TAIL_MS = 600;
/** A reply that never starts must not leave the orb thinking forever. */
const THINK_TIMEOUT_MS = 45_000;

/**
 * A.D.A sends raw float32 PCM. The JARVIS player decodes encoded audio, so the
 * chunk is wrapped as 16-bit mono WAV. Also returns its duration in seconds.
 */
function pcmToWav(b64: string): { wav: string; seconds: number } {
  const raw = atob(b64);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  const samples = Math.floor(bytes.length / 4);
  const src = new DataView(bytes.buffer);

  const out = new ArrayBuffer(44 + samples * 2);
  const o = new DataView(out);
  const ascii = (at: number, s: string) => {
    for (let i = 0; i < s.length; i++) o.setUint8(at + i, s.charCodeAt(i));
  };
  ascii(0, "RIFF");
  o.setUint32(4, 36 + samples * 2, true);
  ascii(8, "WAVE");
  ascii(12, "fmt ");
  o.setUint32(16, 16, true);            // fmt chunk size
  o.setUint16(20, 1, true);             // PCM
  o.setUint16(22, 1, true);             // mono
  o.setUint32(24, SAMPLE_RATE, true);
  o.setUint32(28, SAMPLE_RATE * 2, true);
  o.setUint16(32, 2, true);             // block align
  o.setUint16(34, 16, true);            // bits per sample
  ascii(36, "data");
  o.setUint32(40, samples * 2, true);
  for (let i = 0; i < samples; i++) {
    const f = Math.max(-1, Math.min(1, src.getFloat32(i * 4, true)));
    o.setInt16(44 + i * 2, Math.round(f < 0 ? f * 0x8000 : f * 0x7fff), true);
  }

  const wavBytes = new Uint8Array(out);
  let bin = "";
  for (let i = 0; i < wavBytes.length; i += 0x8000) {
    bin += String.fromCharCode(...wavBytes.subarray(i, i + 0x8000));
  }
  return { wav: btoa(bin), seconds: samples / SAMPLE_RATE };
}

export function createSocket(_url: string): JarvisSocket {
  // The URL argument is JARVIS's address. A.D.A's backend address is fixed.
  const handlers: MessageHandler[] = [];
  const toScreen = (msg: Record<string, unknown>) => {
    for (const h of handlers) h(msg);
  };

  const sock: Socket = io(BACKEND_URL, {
    reconnection: true,
    reconnectionDelay: 500,
    reconnectionDelayMax: 4000,
  });

  let connected = false;
  let speechGen = 0;             // bumps once per new turn on the backend
  let lastChunkId = 0;           // drops stale chunks inside a turn
  let suppress = false;          // set by barge-in or Stop; cleared by a newer turn
  let question: string | number | null = null;
  let subtitle = "";             // what she is saying, shown in the status line
  let curUtt = -1;               // utterance currently queued for playback
  let curUttStart = 0;           // estimated wall-clock ms when it starts playing
  let curUttSec = 0;             // seconds of its audio queued so far
  let idleTimer: number | null = null;
  let thinkTimer: number | null = null;
  let errorTimer: number | null = null;
  const errorEl = document.getElementById("error-text");

  function showError(msg: string): void {
    if (!errorEl) return;
    errorEl.textContent = msg;
    errorEl.style.opacity = "1";
    if (errorTimer !== null) clearTimeout(errorTimer);
    errorTimer = window.setTimeout(() => {
      errorEl.style.opacity = "0";
      errorTimer = null;
    }, 5000);
  }

  function scheduleIdle(): void {
    if (idleTimer !== null) clearTimeout(idleTimer);
    const wait = Math.max(0, speechEndsAt() - Date.now()) + SPEECH_TAIL_MS;
    idleTimer = window.setTimeout(() => {
      idleTimer = null;
      if (!isAdaSpeaking()) toScreen({ type: "status", state: "idle" });
    }, wait);
  }

  function startThinking(): void {
    toScreen({ type: "status", state: "thinking" });
    if (thinkTimer !== null) clearTimeout(thinkTimer);
    thinkTimer = window.setTimeout(() => {
      thinkTimer = null;
      if (!isAdaSpeaking()) toScreen({ type: "status", state: "idle" });
    }, THINK_TIMEOUT_MS);
  }

  function stopThinking(): void {
    if (thinkTimer !== null) {
      clearTimeout(thinkTimer);
      thinkTimer = null;
    }
  }

  /** Stop her mid-sentence and tell the backend how much was heard. */
  function interrupt(): void {
    if (curUtt >= 0 && curUttSec > 0) {
      const elapsed = (Date.now() - curUttStart) / 1000;
      const frac = Math.max(0, Math.min(1, elapsed / curUttSec));
      sock.emit("tts_played", { utt: curUtt, frac });
    }
    curUtt = -1;
    suppress = true;
    endSpeech();
    if (idleTimer !== null) {
      clearTimeout(idleTimer);
      idleTimer = null;
    }
  }

  // Speech input reports that the user is talking over her.
  onBargeIn(() => {
    if (!isAdaSpeaking()) return;
    interrupt();
    toScreen({ type: "stop" });
  });

  // ── A.D.A -> screen ──────────────────────────────────────────────────────

  function onTtsAudio(d: TtsAudioEvent): void {
    // A chunk from an older turn is stale. A newer turn lifts the suppression.
    if (d.gen !== undefined) {
      if (d.gen < speechGen) return;
      if (d.gen > speechGen) {
        speechGen = d.gen;
        suppress = false;
      }
    }
    if (suppress) return;
    if (d.chunkId !== undefined) {
      if (d.chunkId <= lastChunkId) return;
      lastChunkId = d.chunkId;
    }
    if (!d.audio) return;

    const utt = Number(d.utt ?? 0);
    if (utt !== curUtt) {
      curUtt = utt;
      curUttStart = Math.max(Date.now(), speechEndsAt());
      curUttSec = 0;
    }
    const isPcm = !d.format || d.format === "pcm";
    const { wav, seconds } = isPcm ? pcmToWav(d.audio) : { wav: d.audio, seconds: 1 };
    curUttSec += seconds;
    queueSpeech(seconds);
    stopThinking();
    toScreen({ type: "audio", data: wav, utt, idx: d.chunkId ?? 0 });
    // The screen's speaking transition clears the status line, so the subtitle
    // is sent again after it. It arrives before the audio, so it would be lost.
    if (subtitle) toScreen({ type: "text", text: subtitle.trim() });
    scheduleIdle();
  }

  function onTranscription(d: TranscriptionEvent): void {
    if (d.sender === "You") return;
    subtitle = `${subtitle}${d.text ?? ""}`.slice(-180);
    toScreen({ type: "text", text: subtitle.trim() });
  }

  function onQuestion(d: AdaQuestionEvent): void {
    question = d.id;
    stopThinking();
    if (!isAdaSpeaking()) toScreen({ type: "status", state: "idle" });
    const hint = d.choices && d.choices.length ? ` (${d.choices.join(" / ")})` : "";
    toScreen({ type: "text", text: `${d.question}${hint}` });
  }

  function onAuthStatus(d: AuthStatusEvent): void {
    if (d.authenticated) {
      if (!isAdaLocked()) return;
      setAdaLocked(false);
      sock.emit("start_audio", { device_index: null, device_name: null, muted: false });
      // Two transitions leave the locked state and re-arm the microphone.
      toScreen({ type: "status", state: "thinking" });
      toScreen({ type: "status", state: "idle" });
      toScreen({ type: "notice", text: "Identity verified" });
      return;
    }
    setAdaLocked(true);
    toScreen({ type: "status", state: "idle" });
    toScreen({ type: "notice", text: "Locked - look at the camera to unlock" });
  }

  sock.on("connect", () => {
    connected = true;
    sock.emit("start_audio", { device_index: null, device_name: null, muted: false });
    if (errorEl) errorEl.style.opacity = "0";
    toScreen({ type: "config", muteMicDuringSpeech: false });
  });

  sock.on("disconnect", () => {
    connected = false;
    stopThinking();
    showError("A.D.A is offline - reconnecting");
    toScreen({ type: "status", state: "idle" });
  });

  sock.on("connect_error", () => {
    showError("A.D.A is offline - reconnecting");
  });

  sock.on("tts_audio", onTtsAudio);
  sock.on("transcription", onTranscription);
  sock.on("ada_question", onQuestion);
  sock.on("auth_status", onAuthStatus);
  // Widgets and HUD dashboards go to the screen as they are; it validates them again.
  sock.on("widget", (d: Record<string, unknown>) => toScreen({ type: "widget", ...d }));
  sock.on("hud", (d: Record<string, unknown>) => toScreen({ type: "hud", ...d }));

  sock.on("error", (d: ErrorEvent) => {
    const msg = d?.msg ?? "";
    // While locked, start_audio is refused on purpose. The lock notice explains it.
    if (isAdaLocked() && /authentication required/i.test(msg)) return;
    stopThinking();
    showError(msg || "Something went wrong");
    if (!isAdaSpeaking()) toScreen({ type: "status", state: "idle" });
  });

  // ── screen -> A.D.A ──────────────────────────────────────────────────────

  function sendUserText(raw: string): void {
    const text = raw.trim();
    if (!text || isAdaLocked()) return;
    if (!connected) {
      showError("A.D.A is offline - reconnecting");
      return;
    }
    if (question !== null) {
      sock.emit("answer_question", { id: question, answer: text });
      question = null;
      startThinking();
      return;
    }
    // Talking over her ends her sentence. Her older chunks stay suppressed until
    // the backend starts a newer turn.
    if (isAdaSpeaking()) {
      interrupt();
      toScreen({ type: "stop" });
    }
    subtitle = "";
    sock.emit("user_input", { text });
    startThinking();
  }

  return {
    send(data) {
      switch (data.type) {
        case "transcript":
          sendUserText(String(data.text ?? ""));
          break;
        case "hush":
          interrupt();
          break;
        case "hud_action":
          if (connected) sock.emit("hud_action", { action: String(data.action ?? "") });
          break;
        default:
          // interim, mic and played carry nothing A.D.A needs.
          break;
      }
    },
    onMessage(handler) {
      handlers.push(handler);
    },
    close() {
      sock.close();
    },
    isConnected() {
      return connected;
    },
  };
}
