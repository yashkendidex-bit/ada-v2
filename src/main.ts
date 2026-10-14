/**
 * JARVIS — Main entry point.
 *
 * Wires together the orb visualization, WebSocket communication,
 * speech recognition, and audio playback into a single experience.
 */

import { createOrb, type OrbState } from "./orb";
import { createAudioPlayer, createMicMonitor } from "./voice";
import { createVoiceInput } from "./ada-voice";
import { createSocket } from "./ws";
import "./style.css";
import "./orb-theme.css";
import "./widgets/widgets.css";
import { WidgetManager } from "./widgets/manager";
import { createDock } from "./widgets/dock";

// ---------------------------------------------------------------------------
// State machine
// ---------------------------------------------------------------------------

type State = "idle" | "listening" | "thinking" | "speaking" | "compacting";
let currentState: State = "idle";
let isMuted = false;

const statusEl = document.getElementById("status-text")!;
const errorEl = document.getElementById("error-text")!;

function showError(msg: string) {
  errorEl.textContent = msg;
  errorEl.style.opacity = "1";
  setTimeout(() => {
    errorEl.style.opacity = "0";
  }, 5000);
}

function updateStatus(state: State) {
  const labels: Record<State, string> = {
    idle: "",
    listening: "listening...",
    thinking: "thinking...",
    speaking: "",
    compacting: "",          // the notice banner carries the words; the orb carries the state
  };
  statusEl.textContent = labels[state];
}

// ---------------------------------------------------------------------------
// Init components
// ---------------------------------------------------------------------------

const canvas = document.getElementById("orb-canvas") as HTMLCanvasElement;
const orb = createOrb(canvas);

// Widgets: what she puts on screen, inside and around the orb.
const widgetLayer = document.createElement("div");
widgetLayer.id = "widget-layer";
document.body.appendChild(widgetLayer);
const widgets = new WidgetManager({
  layer: widgetLayer,
  orb,
  onAction: (action: string) => socket.send({ type: "hud_action", action }),
});
createDock(widgets);

const wsProto = window.location.protocol === "https:" ? "wss:" : "ws:";
const WS_URL = `${wsProto}//${window.location.host}/ws/voice`;
const socket = createSocket(WS_URL);

const audioPlayer = createAudioPlayer();
orb.setAnalyser(audioPlayer.getAnalyser());

let muteMicDuringSpeech = false;

function transition(newState: State) {
  if (newState === currentState) return;
  currentState = newState;
  const btn = document.getElementById("hush");
  if (btn) (btn as HTMLButtonElement).hidden = newState !== "speaking";
  orb.setState(newState as OrbState);
  updateStatus(newState);

  if (isMuted) return;
  if (newState === "speaking" && muteMicDuringSpeech) {
    voiceInput.pause();
  } else {
    voiceInput.resume();
  }
}

// ---------------------------------------------------------------------------
// Voice input
// ---------------------------------------------------------------------------

const voiceInput = createVoiceInput(
  (text: string) => {
    // The server decides whether this is echo, a barge-in, or a new turn.
    micMonitor.sawSpeech();
    socket.send({ type: "transcript", text, isFinal: true });
  },
  (text: string) => {
    micMonitor.sawSpeech();
    socket.send({ type: "interim", text });
  },
  (msg: string) => {
    showError(msg);
  },
  (event: string) => {
    // Mirror the recogniser's lifecycle to the server log. Going deaf is a
    // browser-side failure the server cannot otherwise see at all, and the
    // console it used to be confined to is never open when it happens.
    socket.send({ type: "mic", text: event });
  },
  (rms: number) => {
    // The cloud listens: it swells and leans toward the user while they talk.
    orb.setMicLevel(rms);
  }
);

// A live meter for the microphone itself. If this moves when you speak, the
// microphone is working — whatever else is or is not happening. It answers
// "is it even hearing me?" without a log, a console or anyone to ask.
const micDot = document.createElement("div");
micDot.id = "mic-level";
micDot.title = "microphone input";
document.body.appendChild(micDot);

const micMonitor = createMicMonitor(
  (level: number) => {
    const pct = Math.min(100, Math.round(level * 900));
    micDot.style.setProperty("--level", `${pct}%`);
    micDot.classList.toggle("is-hot", level > 0.02);
  },
  (event: string) => {
    socket.send({ type: "mic", text: event });
    // Proven deaf: sound going in, nothing coming out. Do not wait for the
    // rotation timer to happen along — measured once at 21 seconds, all of
    // it lost. Rebuild the recogniser now.
    if (event.startsWith("DEAF")) voiceInput.restart("deaf: audio in, no results");
  }
);

// ── stopping him ──────────────────────────────────────────────────────────
// Escape, or the button that appears while he is talking. Not a spoken word:
// his voice comes back through the microphone garbled, and a mis-hear that
// looked like "stop" would cut him off at random. A keystroke cannot be
// misheard.
const hushBtn = document.createElement("button");
hushBtn.id = "hush";
hushBtn.type = "button";
hushBtn.textContent = "Stop";
hushBtn.title = "Stop speaking (Esc)";
hushBtn.hidden = true;
document.body.appendChild(hushBtn);

function hush() {
  if (currentState !== "speaking") return;
  // Locally first: the round trip is real and silence should be instant.
  audioPlayer.stop();
  socket.send({ type: "hush" });
  transition(isMuted ? "idle" : "listening");
}

hushBtn.addEventListener("click", hush);
window.addEventListener("keydown", (e: KeyboardEvent) => {
  if (e.key === "Escape") {
    e.preventDefault();
    if (currentState === "speaking") hush();
    else widgets.closeTop();
  }
});

audioPlayer.onPlayed((utt, idx) => {
  socket.send({ type: "played", utt, idx });
});

// End of speech is the server's call (`status: idle` after every chunk is
// acked); a transient empty queue mid-utterance must not flip the UI.
audioPlayer.onFinished(() => {});

audioPlayer.onNeedsGesture(() => {
  showError("Click anywhere to enable audio");
});

// ---------------------------------------------------------------------------
// WebSocket messages
// ---------------------------------------------------------------------------

socket.onMessage((msg) => {
  const type = msg.type as string;

  if (type === "widget" || type === "hud") {
    widgets.handle(msg);
  } else if (type === "config") {
    muteMicDuringSpeech = Boolean(msg.muteMicDuringSpeech);
  } else if (type === "audio") {
    const data = msg.data as string;
    if (data) {
      if (currentState !== "speaking") transition("speaking");
      audioPlayer.enqueue(data, Number(msg.utt), Number(msg.idx));
    }
    if (msg.text) console.log("[JARVIS]", msg.text);
  } else if (type === "stop") {
    audioPlayer.stop();
    transition(isMuted ? "idle" : "listening");
  } else if (type === "drop_queued") {
    audioPlayer.dropQueued();
  } else if (type === "status") {
    const state = msg.state as string;
    if (state === "thinking") transition("thinking");
    else if (state === "speaking") transition("speaking");
    else if (state === "compacting") transition("compacting");
    else if (state === "idle") transition(isMuted ? "idle" : "listening");
  } else if (type === "text") {
    // A chunk TTS could not voice: show it instead of losing it
    console.log("[JARVIS]", msg.text);
    statusEl.textContent = String(msg.text);
  } else if (type === "notice") {
    // Shown, never spoken. The server sends one when it is about to be busy
    // for a few seconds (a context rotation), and an empty string to clear it.
    // Without it the pause looks like a crash.
    const text = String(msg.text ?? "");
    statusEl.textContent = text;
    if (text) console.log("[notice]", text);
  }
});

// ---------------------------------------------------------------------------
// Kick off
// ---------------------------------------------------------------------------

// Start listening after a brief delay for the orb to render
setTimeout(() => {
  voiceInput.start();
  if (currentState !== "speaking") transition("listening");
}, 1000);

// Resume AudioContext on ANY user interaction (browser autoplay policy)
function ensureAudioContext() {
  const ctx = audioPlayer.getAnalyser().context as AudioContext;
  if (ctx.state === "suspended") {
    ctx.resume().then(() => console.log("[audio] context resumed"));
  }
}
document.addEventListener("click", ensureAudioContext);
document.addEventListener("touchstart", ensureAudioContext);
document.addEventListener("keydown", ensureAudioContext, { once: true });

// Try to resume audio context on load
ensureAudioContext();

// ---------------------------------------------------------------------------
// UI Controls
// ---------------------------------------------------------------------------

const btnMute = document.getElementById("btn-mute")!;
const btnMenu = document.getElementById("btn-menu")!;
const menuDropdown = document.getElementById("menu-dropdown")!;
const btnRestart = document.getElementById("btn-restart")!;
const btnFixSelf = document.getElementById("btn-fix-self")!;

btnMute.addEventListener("click", (e) => {
  e.stopPropagation();
  isMuted = !isMuted;
  btnMute.classList.toggle("muted", isMuted);
  if (isMuted) {
    voiceInput.pause();
    transition("idle");
  } else {
    voiceInput.resume();
    transition("listening");
  }
});

btnMenu.addEventListener("click", (e) => {
  e.stopPropagation();
  menuDropdown.style.display = menuDropdown.style.display === "none" ? "block" : "none";
});

document.addEventListener("click", () => {
  menuDropdown.style.display = "none";
});

btnRestart.addEventListener("click", (e) => {
  e.stopPropagation();
  menuDropdown.style.display = "none";
  // A.D.A's backend has no restart route. Restart it from the desktop app.
  statusEl.textContent = "restart is not available in this build";
});

btnFixSelf.addEventListener("click", (e) => {
  e.stopPropagation();
  menuDropdown.style.display = "none";
  // Activate work mode on the WebSocket session (JARVIS becomes Claude Code's voice)
  // Milestone 1 has no tools yet; "Fix yourself" returns as a brain tool later.
  statusEl.textContent = "fix-yourself is not available in this build";
});

// Settings button. A.D.A's settings are not wired into this screen yet.
const btnSettings = document.getElementById("btn-settings")!;
btnSettings.addEventListener("click", (e) => {
  e.stopPropagation();
  menuDropdown.style.display = "none";
  statusEl.textContent = "settings are not available in this build";
});

// ---------------------------------------------------------------------------
// Frame counter. Press F3 (or open with ?fps) to see the real frame rate of the
// orb on this machine.
// ---------------------------------------------------------------------------

const fpsEl = document.createElement("div");
fpsEl.id = "fps-counter";
fpsEl.hidden = !/[?&]fps\b/.test(window.location.search);
document.body.appendChild(fpsEl);
window.addEventListener("keydown", (e: KeyboardEvent) => {
  if (e.key === "F3") { e.preventDefault(); fpsEl.hidden = !fpsEl.hidden; }
});
{
  let frames = 0;
  let since = performance.now();
  const tick = (now: number) => {
    frames++;
    if (now - since >= 500) {
      if (!fpsEl.hidden) fpsEl.textContent = `${Math.round((frames * 1000) / (now - since))} fps`;
      frames = 0;
      since = now;
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}
