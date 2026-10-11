/**
 * Microphone capture for A.D.A: local voice-activity detection, then Groq
 * Whisper transcription through the backend's /stt endpoint.
 *
 * Ported from the previous GUI's startSTT() and kept behaviourally identical
 * (same thresholds, endpointing and barge-in guard), with two additions: a
 * per-frame level callback that drives the orb, and an explicit start/stop
 * that releases the microphone when muted.
 *
 * Why it looks the way it does:
 *   - The noise floor adapts, so a fan does not trigger and a whisper does.
 *   - While A.D.A is speaking her own voice reaches the mic, so barge-in needs
 *     longer sustained voice and a higher bar (700 ms, 1.3x threshold).
 *   - Long utterances endpoint faster, short bursts keep a patient silence
 *     window, and a 30 s cap stops a stuck mic wedging the turn.
 */

import { BACKEND_URL } from "./socket";

export type MicStatus = "listening" | "hearing" | "transcribing";

export interface MicHandlers {
  /** Every frame: RMS of the input signal (0..~1). Drives the orb. */
  onLevel(rms: number): void;
  /** Live caption while the user is mid-utterance. */
  onInterim(text: string): void;
  /** Sustained voice while A.D.A is speaking: she should stop. */
  onBargeIn(): void;
  /** A finished, transcribed utterance. */
  onUtterance(text: string): void;
  /** Soft acknowledgement clip ("mm-hm") during a long user turn. */
  onBackchannel(clip: string): void;
  onStatus(s: MicStatus): void;
  onError(msg: string): void;
  isAdaSpeaking(): boolean;
}

export interface Mic {
  start(): void;
  stop(): void;
  isOn(): boolean;
  /** Browsers suspend audio contexts until a user gesture; call on one. */
  resume(): void;
}

const BACKCHANNEL_CLIPS = ["mmhm", "mmhm", "right", "oh"];

const SPEAK_MS = 400;           // sustained voice before a segment starts (idle)
const BARGE_SPEAK_MS = 700;     // sustained voice to interrupt A.D.A
const SILENCE_MS = 1200;        // patient endpoint for short bursts
const FAST_END_MS = 700;        // endpoint for long utterances
const MIN_SEG_MS = 300;
const MAX_SEG_MS = 30000;
const PARTIAL_EVERY_MS = 4000;
const BACKCHANNEL_DELAY_MS = 1600;
const BACKCHANNEL_GAP_MS = 7000;

interface Pipeline {
  audioCtx: AudioContext;
  analyser: AnalyserNode;
  recorder: MediaRecorder;
  stream: MediaStream;
  chunks: Blob[];
  buf: Uint8Array<ArrayBuffer>;
  raf: number | null;
  noiseFloor: number;
  voiceSince: number;
  speaking: boolean;
  sending: boolean;
  partialBusy: boolean;
  segStart: number;
  lastVoice: number;
  lastPartial: number;
  bcTimer: number | null;
  bcLast: number;
}

export function createMic(h: MicHandlers): Mic {
  let want = false;
  let starting = false;
  let pipe: Pipeline | null = null;

  function mimeOf(p: Pipeline): string {
    return p.recorder.mimeType || "audio/webm";
  }

  function postPartial(p: Pipeline): void {
    if (p.partialBusy || p.chunks.length === 0) return;
    p.partialBusy = true;
    const slice = new Blob(p.chunks, { type: mimeOf(p) });
    const fd = new FormData();
    fd.append("file", slice, "partial.webm");
    fetch(`${BACKEND_URL}/stt?partial=1`, { method: "POST", body: fd })
      .then((r) => r.json())
      .then((res: { text?: string }) => {
        p.partialBusy = false;
        const t = (res.text ?? "").trim();
        if (t && p.speaking) h.onInterim(t.slice(0, 80));
      })
      .catch(() => { p.partialBusy = false; });
  }

  function maybeBackchannel(p: Pipeline): void {
    if (h.isAdaSpeaking()) return;
    const now = performance.now();
    if (now - p.bcLast < BACKCHANNEL_GAP_MS) return;
    p.bcLast = now;
    const pick = BACKCHANNEL_CLIPS[Math.floor(Math.random() * BACKCHANNEL_CLIPS.length)];
    h.onBackchannel(pick);
  }

  function build(stream: MediaStream): Pipeline {
    const AC = window.AudioContext ??
      (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    const audioCtx = new AC();
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 2048;
    audioCtx.createMediaStreamSource(stream).connect(analyser);

    const recorder = new MediaRecorder(stream);
    const p: Pipeline = {
      audioCtx, analyser, recorder, stream,
      chunks: [],
      buf: new Uint8Array(analyser.fftSize),
      raf: null,
      noiseFloor: 0.012,
      voiceSince: 0,
      speaking: false,
      sending: false,
      partialBusy: false,
      segStart: 0,
      lastVoice: 0,
      lastPartial: 0,
      bcTimer: null,
      bcLast: 0,
    };

    recorder.ondataavailable = (e: BlobEvent) => {
      if (e.data.size > 0) p.chunks.push(e.data);
    };

    recorder.onstop = () => {
      if (p.chunks.length === 0) { h.onStatus("listening"); return; }
      const blob = new Blob(p.chunks, { type: mimeOf(p) });
      p.chunks.length = 0;
      p.sending = true;
      const fd = new FormData();
      fd.append("file", blob, "voice.webm");
      fetch(`${BACKEND_URL}/stt`, { method: "POST", body: fd })
        .then((r) => r.json())
        .then((res: { text?: string }) => {
          p.sending = false;
          const text = (res.text ?? "").trim();
          if (text.length > 1) h.onUtterance(text);
          h.onStatus("listening");
        })
        .catch((err: unknown) => {
          p.sending = false;
          console.error("[mic] transcription upload failed:", err);
          h.onError("Voice transcription failed - try again.");
          h.onStatus("listening");
        });
    };

    const loop = () => {
      p.raf = requestAnimationFrame(loop);
      if (!want) return;

      analyser.getByteTimeDomainData(p.buf);
      let sum = 0;
      for (let i = 0; i < p.buf.length; i++) {
        const v = (p.buf[i] - 128) / 128;
        sum += v * v;
      }
      const rms = Math.sqrt(sum / p.buf.length);
      h.onLevel(rms);

      // Adaptive floor: fall quickly toward quiet, climb slowly out of it.
      p.noiseFloor += (rms - p.noiseFloor) * (rms < p.noiseFloor ? 0.3 : 0.002);
      const adaSpeaking = h.isAdaSpeaking();
      const threshold = Math.max(0.025, p.noiseFloor * 2.5) * (adaSpeaking ? 1.3 : 1.0);
      const sustainMs = adaSpeaking ? BARGE_SPEAK_MS : SPEAK_MS;
      const now = performance.now();

      if (rms > threshold) {
        if (!p.speaking && !p.sending) {
          if (!p.voiceSince) p.voiceSince = now;
          // Only sustained voice counts: a door slam or her own loud word
          // through the speakers never cuts her off.
          if (now - p.voiceSince < sustainMs) return;

          h.onBargeIn();
          p.speaking = true;
          p.segStart = now;
          p.lastPartial = now;
          p.lastVoice = now;
          h.onStatus("hearing");
          if (p.recorder.state !== "recording") {
            p.chunks.length = 0;
            try { p.recorder.start(PARTIAL_EVERY_MS); } catch { try { p.recorder.start(); } catch { /* device busy */ } }
          }
          if (p.bcTimer !== null) clearTimeout(p.bcTimer);
          p.bcTimer = window.setTimeout(() => {
            if (p.speaking) maybeBackchannel(p);
          }, BACKCHANNEL_DELAY_MS);
        }
        p.lastVoice = now;
        // Streaming partial: live captions on long utterances.
        if (p.speaking && now - p.lastPartial > PARTIAL_EVERY_MS && now - p.segStart > PARTIAL_EVERY_MS + 500) {
          p.lastPartial = now;
          postPartial(p);
        }
      } else if (p.speaking && (now - p.lastVoice > (now - p.segStart > 2500 ? FAST_END_MS : SILENCE_MS) || now - p.segStart > MAX_SEG_MS)) {
        p.voiceSince = 0;
        if (p.bcTimer !== null) { clearTimeout(p.bcTimer); p.bcTimer = null; }
        if (now - p.segStart > MIN_SEG_MS && !p.sending) {
          p.speaking = false;
          h.onStatus("transcribing");
          if (p.recorder.state === "recording") p.recorder.stop();
        }
      } else if (!p.speaking) {
        p.voiceSince = 0;
        h.onStatus("listening");
      }
    };
    p.raf = requestAnimationFrame(loop);
    return p;
  }

  function release(p: Pipeline): void {
    if (p.raf !== null) cancelAnimationFrame(p.raf);
    if (p.bcTimer !== null) clearTimeout(p.bcTimer);
    // Detach first so tearing down does not upload a half-finished utterance.
    p.recorder.onstop = null;
    if (p.recorder.state !== "inactive") {
      try { p.recorder.stop(); } catch { /* already stopped */ }
    }
    p.stream.getTracks().forEach((t) => t.stop());
    p.audioCtx.close().catch(() => {});
  }

  return {
    start() {
      want = true;
      if (pipe || starting) return;
      starting = true;
      navigator.mediaDevices
        .getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } })
        .then((stream) => {
          starting = false;
          if (!want) { stream.getTracks().forEach((t) => t.stop()); return; }
          pipe = build(stream);
        })
        .catch((err: { name?: string }) => {
          starting = false;
          want = false;
          console.error("[mic] access failed:", err);
          h.onError(`Mic access denied (${err?.name ?? "error"}) - voice input disabled.`);
        });
    },
    stop() {
      want = false;
      if (pipe) { release(pipe); pipe = null; }
    },
    isOn() {
      return want;
    },
    resume() {
      if (pipe && pipe.audioCtx.state === "suspended") {
        pipe.audioCtx.resume().catch(() => {});
      }
    },
  };
}
