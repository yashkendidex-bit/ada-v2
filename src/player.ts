/**
 * Gapless TTS playback for A.D.A through the Web Audio API.
 *
 * Ported from the previous GUI's src/lib/voicePlayer.js and converted to
 * TypeScript. Two additions for the orb:
 *   - the AudioContext is created up front, so `analyser` exists before the
 *     first chunk arrives and the orb can read the real voice level;
 *   - playing-state changes are reported through a single listener.
 *
 * Formats: "pcm" is raw float32 little-endian mono at 24 kHz (streamed
 * Voxtral). Anything else (mp3, wav) goes through decodeAudioData.
 *
 * Barge-in: stopAll(true) halts every scheduled source at once and returns,
 * per utterance, how much of it was actually heard (playback anchoring).
 */

const PCM_SAMPLE_RATE = 24000;

export interface PlayedReport {
  utt: number;
  frac: number;
}

interface Job {
  b64: string;
  fmt: string;
  utt: number | null;
  words: number;
}

interface SourceMeta {
  utt: number | null;
  start: number;
  dur: number;
}

function b64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

export class VoicePlayer {
  readonly ctx: AudioContext;
  readonly analyser: AnalyserNode;
  private readonly master: GainNode;
  private nextTime = 0;
  private readonly sources = new Map<AudioBufferSourceNode, SourceMeta>();
  private queue: Job[] = [];
  private draining = false;
  private pendingPcm: Uint8Array | null = null;
  private _playing = false;
  private listener: ((playing: boolean) => void) | null = null;

  constructor() {
    const AC = window.AudioContext ??
      (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    this.ctx = new AC();
    this.master = this.ctx.createGain();
    this.master.gain.value = 1.0;
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 512;           // short window: the orb should follow the voice closely
    this.analyser.smoothingTimeConstant = 0; // raw energy; the orb applies its own envelope
    this.master.connect(this.analyser);
    this.master.connect(this.ctx.destination);
  }

  get playing(): boolean {
    return this._playing;
  }

  get busy(): boolean {
    return this.sources.size > 0 || this.queue.length > 0;
  }

  onPlayingChange(fn: (playing: boolean) => void): void {
    this.listener = fn;
  }

  /** Autoplay policy: resume on the first user gesture. */
  resume(): void {
    if (this.ctx.state === "suspended") {
      this.ctx.resume().catch(() => {});
    }
  }

  enqueue(b64: string, format: string | undefined, meta: { utt?: number; words?: number }): void {
    if (!b64) return;
    this.queue.push({
      b64,
      fmt: (format ?? "mp3").toLowerCase(),
      utt: meta.utt ?? null,
      words: meta.words ?? 0,
    });
    void this.drain();
  }

  /**
   * Instant, total silence. With interrupted=true it returns the played
   * fraction of each utterance so the backend keeps only the words that were
   * actually heard. Queued (never started) utterances report 0.
   */
  stopAll(interrupted: boolean): PlayedReport[] {
    let played: PlayedReport[] = [];
    if (interrupted) {
      try {
        const now = this.ctx.currentTime;
        const byUtt = new Map<number, { played: number; total: number }>();
        for (const meta of this.sources.values()) {
          if (meta.utt === null) continue;
          const el = byUtt.get(meta.utt) ?? { played: 0, total: 0 };
          el.played += Math.max(0, Math.min(now - meta.start, meta.dur));
          el.total += meta.dur;
          byUtt.set(meta.utt, el);
        }
        for (const job of this.queue) {
          if (job.utt !== null && !byUtt.has(job.utt)) {
            byUtt.set(job.utt, { played: 0, total: 1 });
          }
        }
        played = [...byUtt.entries()].map(([utt, el]) => ({
          utt,
          frac: el.total > 0 ? Math.min(1, el.played / el.total) : 0,
        }));
      } catch {
        played = []; // report nothing rather than something wrong
      }
    }
    this.queue = [];
    this.pendingPcm = null;
    for (const src of this.sources.keys()) {
      try { src.onended = null; src.stop(); } catch { /* already stopped */ }
      try { src.disconnect(); } catch { /* already disconnected */ }
    }
    this.sources.clear();
    this.nextTime = 0;
    this.setPlaying(false);
    return played;
  }

  private setPlaying(v: boolean): void {
    if (this._playing === v) return;
    this._playing = v;
    this.listener?.(v);
  }

  private async drain(): Promise<void> {
    if (this.draining) return;
    this.draining = true;
    try {
      while (this.queue.length > 0) {
        const job = this.queue.shift()!;
        try {
          await this.decodeAndSchedule(job);
        } catch (e) {
          console.error("[player] decode failed:", e);
        }
      }
    } finally {
      this.draining = false;
      this.updateIdle();
    }
  }

  private async decodeAndSchedule(job: Job): Promise<void> {
    const bytes = b64ToBytes(job.b64);
    let buffer: AudioBuffer;

    if (job.fmt === "pcm") {
      // Reassemble float32 frames that were split across chunk boundaries.
      let all = bytes;
      if (this.pendingPcm && this.pendingPcm.length) {
        all = new Uint8Array(this.pendingPcm.length + bytes.length);
        all.set(this.pendingPcm, 0);
        all.set(bytes, this.pendingPcm.length);
        this.pendingPcm = null;
      }
      const rem = all.length % 4;
      if (rem) {
        this.pendingPcm = all.slice(all.length - rem);
        all = all.slice(0, all.length - rem);
      }
      const n = Math.floor(all.length / 4);
      if (n === 0) return;
      const floats = new Float32Array(n);
      const dv = new DataView(all.buffer, all.byteOffset, all.byteLength);
      for (let i = 0; i < n; i++) floats[i] = dv.getFloat32(i * 4, true);
      buffer = this.ctx.createBuffer(1, n, PCM_SAMPLE_RATE);
      buffer.copyToChannel(floats, 0);
    } else {
      const ab = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
      buffer = await this.ctx.decodeAudioData(ab);
    }

    this.schedule(buffer, job);
  }

  private schedule(buffer: AudioBuffer, job: Job): void {
    const src = this.ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(this.master);
    // Back-to-back scheduling; resync to "now" after silence or barge-in.
    const startAt = Math.max(this.nextTime, this.ctx.currentTime + 0.015);
    src.start(startAt);
    this.nextTime = startAt + buffer.duration;
    this.sources.set(src, { utt: job.utt, start: startAt, dur: buffer.duration });
    this.setPlaying(true);
    src.onended = () => {
      this.sources.delete(src);
      this.updateIdle();
    };
  }

  private updateIdle(): void {
    if (this.sources.size === 0 && this.queue.length === 0 && !this.draining) {
      this.setPlaying(false);
      this.nextTime = 0; // the next utterance must not inherit a stale offset
    }
  }
}
