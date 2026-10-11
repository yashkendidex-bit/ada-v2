/**
 * A.D.A GUI entry point.
 *
 * The orb is the face. Its state is derived from what A.D.A is actually
 * doing: connection, playback, a turn in flight, mute, or listening. The
 * backend's status messages are free text, so the client keeps its own
 * state rather than parsing them.
 */

import "./style.css";
import { createOrb, type OrbState } from "./orb";
import { VoicePlayer } from "./player";
import { createMic, type MicStatus } from "./mic";
import {
  createLink,
  type AdaQuestionEvent,
  type AuthFrameEvent,
  type AuthStatusEvent,
  type ErrorEvent,
  type StatusEvent,
  type TranscriptionEvent,
  type TtsAudioEvent,
} from "./socket";

// ── DOM ────────────────────────────────────────────────────────────────────

function el<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing element #${id}`);
  return node as T;
}

const canvas = el<HTMLCanvasElement>("orb-canvas");
const connDot = el<HTMLDivElement>("conn");
const btnMute = el<HTMLButtonElement>("btn-mute");
const btnStop = el<HTMLButtonElement>("btn-stop");
const statusEl = el<HTMLDivElement>("status-text");
const clarifyEl = el<HTMLDivElement>("clarify");
const clarifyQ = el<HTMLDivElement>("clarify-q");
const clarifyChoices = el<HTMLDivElement>("clarify-choices");
const transcriptEl = el<HTMLDivElement>("transcript");
const composer = el<HTMLFormElement>("composer");
const composerInput = el<HTMLInputElement>("composer-input");
const toastEl = el<HTMLDivElement>("toast");
const authLockEl = el<HTMLDivElement>("auth-lock");
const authFrameEl = el<HTMLImageElement>("auth-frame");
const authMsgEl = el<HTMLDivElement>("auth-msg");

// ── Components ─────────────────────────────────────────────────────────────

const orb = createOrb(canvas);
const player = new VoicePlayer();
orb.setPlaybackAnalyser(player.analyser);
const link = createLink();

// ── State ──────────────────────────────────────────────────────────────────

let connected = false;
let muted = false;
let thinking = false;
let thinkingTimer: number | null = null;
let micStatus: MicStatus = "listening";
let interim = "";
let speechGen = 0;          // bumps once per new turn on the backend
let lastChunkId = 0;        // drops stale chunks inside a turn
let suppressSpeech = false; // set by barge-in / stop; cleared by a newer turn
let clarifyId: string | number | null = null;
let toastTimer: number | null = null;
let locked = false;         // face auth is on and has not matched yet
let unlockTimer: number | null = null;

// ── Derived state ──────────────────────────────────────────────────────────

function orbState(): OrbState {
  if (!connected) return "offline";
  if (locked) return "idle";
  if (player.playing) return "speaking";
  if (thinking) return "thinking";
  if (muted) return "idle";
  return "listening";
}

function statusText(): string {
  if (!connected) return "reconnecting";
  if (locked) return "locked";
  if (player.playing) return "speaking";
  if (thinking) return "thinking";
  if (muted) return "muted";
  if (micStatus === "hearing") return interim ? `"${interim}"` : "hearing you";
  if (micStatus === "transcribing") return "transcribing";
  return "listening";
}

let lastOrbState: OrbState | null = null;
let lastStatus = "";

/** Re-derive everything visible from state. Cheap; only touches the DOM on change. */
function refresh(): void {
  const s = orbState();
  if (s !== lastOrbState) {
    orb.setState(s);
    lastOrbState = s;
  }
  const stopVisible = player.playing;
  if (btnStop.hidden === stopVisible) btnStop.hidden = !stopVisible;
  btnMute.classList.toggle("muted", muted);
  btnMute.setAttribute("aria-pressed", String(muted));
  connDot.dataset.state = connected ? "on" : "off";
  const text = statusText();
  if (text !== lastStatus) {
    statusEl.textContent = text;
    lastStatus = text;
  }
}

function beginThinking(): void {
  thinking = true;
  if (thinkingTimer !== null) clearTimeout(thinkingTimer);
  // Safety valve: a reply that never arrives must not leave the orb stuck.
  thinkingTimer = window.setTimeout(() => {
    thinking = false;
    thinkingTimer = null;
    refresh();
  }, 45_000);
  refresh();
}

function endThinking(): void {
  if (!thinking) return;
  thinking = false;
  if (thinkingTimer !== null) { clearTimeout(thinkingTimer); thinkingTimer = null; }
  refresh();
}

// ── Transcript ─────────────────────────────────────────────────────────────

let lastSender: string | null = null;
let lastBody: HTMLSpanElement | null = null;

function appendLine(sender: string, text: string, streaming = false): void {
  if (streaming && lastBody && lastSender === sender) {
    lastBody.textContent = (lastBody.textContent ?? "") + text;
  } else {
    const row = document.createElement("div");
    row.className = `line ${sender === "You" ? "you" : sender === "System" ? "system" : "ada"}`;
    const who = document.createElement("span");
    who.className = "who";
    who.textContent = sender;
    const body = document.createElement("span");
    body.className = "body";
    body.textContent = text;
    row.append(who, body);
    transcriptEl.append(row);
    lastBody = body;
    lastSender = sender;
    while (transcriptEl.childElementCount > 80) transcriptEl.firstElementChild?.remove();
  }
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

function showToast(msg: string): void {
  toastEl.textContent = msg;
  toastEl.hidden = false;
  if (toastTimer !== null) clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => { toastEl.hidden = true; }, 4200);
}

// ── Clarifying questions ───────────────────────────────────────────────────

function showClarify(q: AdaQuestionEvent): void {
  clarifyId = q.id;
  clarifyQ.textContent = q.question;
  clarifyChoices.replaceChildren();
  const choices = q.choices ?? [];
  if (choices.length === 0) {
    const hint = document.createElement("span");
    hint.className = "hint";
    hint.textContent = "Answer by voice or in the box below.";
    clarifyChoices.append(hint);
  }
  for (const c of choices) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = c;
    b.addEventListener("click", () => answerClarify(c));
    clarifyChoices.append(b);
  }
  clarifyEl.hidden = false;
  endThinking();
}

function clearClarify(): void {
  clarifyId = null;
  clarifyEl.hidden = true;
}

function answerClarify(answer: string): void {
  if (clarifyId === null || !connected) return;
  link.emit("answer_question", { id: clarifyId, answer });
  appendLine("You", answer);
  clearClarify();
  beginThinking();
  orb.pulse();
}

// ── Speech control ─────────────────────────────────────────────────────────

/** Stop A.D.A mid-sentence and report how much of each utterance was heard. */
function stopSpeaking(): void {
  suppressSpeech = true;
  if (player.busy) {
    const played = player.stopAll(true);
    if (connected) {
      for (const p of played) link.emit("tts_played", p);
    }
  }
  endThinking();
  refresh();
}

function playBackchannel(clip: string): void {
  try {
    const a = new Audio(`${import.meta.env.BASE_URL}audio/backchannel_${clip}.mp3`);
    a.volume = 0.5;
    void a.play().catch(() => {});
  } catch {
    // A missing soft clip is never worth an error.
  }
}

// ── Sending input ──────────────────────────────────────────────────────────

function sendUserText(raw: string): void {
  const text = raw.trim();
  if (!text) return;
  if (!connected) {
    showToast("A.D.A is offline - reconnecting");
    return;
  }
  // The lock screen covers the composer and already says what is needed.
  if (locked) return;
  suppressSpeech = false;
  clearClarify();
  link.emit("user_input", { text });
  appendLine("You", text);
  beginThinking();
  orb.pulse();
}

// ── Microphone ─────────────────────────────────────────────────────────────

const mic = createMic({
  onLevel: (rms) => orb.setMicLevel(rms),
  onInterim: (text) => {
    interim = text;
    refresh();
  },
  onBargeIn: () => stopSpeaking(),
  onUtterance: (text) => sendUserText(text),
  onBackchannel: (clip) => playBackchannel(clip),
  onStatus: (s) => {
    if (s === micStatus) return;
    micStatus = s;
    if (s !== "hearing") interim = "";
    refresh();
  },
  onError: (msg) => showToast(msg),
  isAdaSpeaking: () => player.playing,
});

function toggleMute(): void {
  muted = !muted;
  if (muted) {
    mic.stop();
  } else if (!locked) {
    mic.start();
  }
  refresh();
}

// ── Face-auth lock ─────────────────────────────────────────────────────────
// Only the backend knows whether face auth is on. When it is, the backend says
// "authenticated: false", streams the camera as auth_frame, and refuses
// start_audio until the face matches. When it is off, auth_status is true
// from the start and nothing here changes.

function showLock(): void {
  if (unlockTimer !== null) { clearTimeout(unlockTimer); unlockTimer = null; }
  locked = true;
  authLockEl.classList.remove("verified");
  authMsgEl.textContent = "Look at the camera to unlock";
  authLockEl.hidden = false;
  mic.stop();
  refresh();
}

function onAuthStatus(authenticated: boolean): void {
  if (!authenticated) {
    showLock();
    return;
  }
  if (!locked) return; // face auth off, or already open
  locked = false;
  authLockEl.classList.add("verified");
  authMsgEl.textContent = "Identity verified";
  // The backend does not start the audio loop by itself after a match.
  if (connected) link.emit("start_audio", { device_index: null, device_name: null, muted: false });
  if (!muted) mic.start();
  unlockTimer = window.setTimeout(() => {
    authLockEl.hidden = true;
    unlockTimer = null;
  }, 1500);
  refresh();
}

// ── Backend link ───────────────────────────────────────────────────────────

link.on("connect", () => {
  connected = true;
  // Idempotent on the backend; re-sent on every (re)connect so a lost emit
  // can never leave the brain offline.
  link.emit("start_audio", { device_index: null, device_name: null, muted: false });
  refresh();
});

link.on("disconnect", () => {
  connected = false;
  endThinking();
  refresh();
});

link.on("tts_audio", (d: TtsAudioEvent) => {
  // A chunk from an older turn is stale: drop it for good. The first chunk of
  // a newer turn lifts the barge-in suppression.
  if (d.gen !== undefined) {
    if (d.gen < speechGen) return;
    if (d.gen > speechGen) {
      speechGen = d.gen;
      suppressSpeech = false;
    }
  }
  if (suppressSpeech) return;
  if (d.chunkId !== undefined) {
    if (d.chunkId <= lastChunkId) return;
    lastChunkId = d.chunkId;
  }
  endThinking();
  player.enqueue(d.audio, d.format, { utt: d.utt, words: d.words });
  refresh();
});

link.on("transcription", (d: TranscriptionEvent) => {
  if (d.sender !== "You") endThinking();
  appendLine(d.sender, d.text, true);
});

link.on("ada_question", (d: AdaQuestionEvent) => showClarify(d));

link.on("auth_status", (d: AuthStatusEvent) => onAuthStatus(d.authenticated === true));

link.on("auth_frame", (d: AuthFrameEvent) => {
  if (locked && d.image) authFrameEl.src = `data:image/jpeg;base64,${d.image}`;
});

// Backend status chatter ("A.D.A Started", "Already Running") is logged, not
// shown: the orb and the status line already say what matters.
link.on("status", (d: StatusEvent) => {
  if (d.msg) console.debug("[A.D.A]", d.msg);
});

link.on("error", (d: ErrorEvent) => {
  endThinking();
  // While locked, start_audio is refused on purpose; the lock screen explains it.
  if (locked && /authentication required/i.test(d.msg ?? "")) return;
  showToast(d.msg || "Something went wrong");
});

player.onPlayingChange((playing) => {
  if (playing) endThinking();
  refresh();
});

// ── Controls ───────────────────────────────────────────────────────────────

btnMute.addEventListener("click", toggleMute);
btnStop.addEventListener("click", stopSpeaking);

composer.addEventListener("submit", (e) => {
  e.preventDefault();
  sendUserText(composerInput.value);
  composerInput.value = "";
});

function isTyping(e: KeyboardEvent): boolean {
  const t = e.target as HTMLElement | null;
  return !!t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable);
}

window.addEventListener("keydown", (e: KeyboardEvent) => {
  if (e.key === "Escape") {
    if (player.playing) { e.preventDefault(); stopSpeaking(); }
  } else if ((e.key === "m" || e.key === "M") && !isTyping(e)) {
    toggleMute();
  }
});

// Browsers hold audio contexts suspended until a user gesture.
const resumeAudio = (): void => {
  player.resume();
  mic.resume();
};
document.addEventListener("pointerdown", resumeAudio);
document.addEventListener("keydown", resumeAudio, { once: true });

// ── Start ──────────────────────────────────────────────────────────────────

refresh();
// Give the orb a moment to draw before the microphone comes up.
window.setTimeout(() => {
  if (!muted && !locked) mic.start();
}, 1000);
