/**
 * VoicePlayer — premium gapless TTS playback for ADA via Web Audio API.
 *
 * Replaces the old <audio> element chain (which left audible gaps between
 * sentence chunks and could only play mp3 blobs). Supports two formats:
 *
 *   - "pcm": raw float32 little-endian mono samples @24 kHz (Voxtral
 *     streaming). Fragments are decoded and scheduled on a gapless timeline,
 *     so her speech flows continuously even though chunks arrive mid-word.
 *   - "mp3"/"wav"/anything else: decoded via decodeAudioData.
 *
 * Barge-in: stopAll() halts every scheduled source instantly.
 */
const PCM_SAMPLE_RATE = 24000;

function b64ToBytes(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

export class VoicePlayer {
  constructor(onPlayingChange) {
    this.ctx = null;
    this.master = null;
    this.nextTime = 0;          // audio-context time the next buffer starts at
    this.sources = new Set();
    this.queue = [];
    this.draining = false;
    this.pendingPcm = null;     // leftover bytes from a split float32 frame
    this.playing = false;
    this._onPlayingChange = onPlayingChange || null;
  }

  _setPlaying(v) {
    if (this.playing !== v) {
      this.playing = v;
      if (this._onPlayingChange) this._onPlayingChange(v);
    }
  }

  _ensureCtx() {
    if (!this.ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return null;
      this.ctx = new AC();
      this.master = this.ctx.createGain();
      this.master.gain.value = 1.0;
      this.master.connect(this.ctx.destination);
    }
    if (this.ctx.state === 'suspended') {
      this.ctx.resume().catch(() => {});
    }
    return this.ctx;
  }

  /** Queues one base64-encoded chunk from the tts_audio socket event.
   *  meta {utt, words} tags playback anchoring: per-utterance scheduled
   *  durations let stopAll() report exactly how far each utterance played. */
  enqueueChunk(b64Audio, format, meta) {
    if (!b64Audio) return;
    this.queue.push({ b64: b64Audio, fmt: (format || 'mp3').toLowerCase(),
                      utt: meta && meta.utt !== undefined ? meta.utt : null,
                      words: (meta && meta.words) || 0 });
    this._drain();
  }

  async _drain() {
    if (this.draining) return;
    this.draining = true;
    try {
      while (this.queue.length > 0) {
        const job = this.queue.shift();
        try {
          await this._decodeAndSchedule(job);
        } catch (e) {
          console.error('[VoicePlayer] decode failed:', e);
        }
      }
    } finally {
      this.draining = false;
      this._updateIdle();
    }
  }

  async _decodeAndSchedule(job) {
    const ctx = this._ensureCtx();
    if (!ctx) return;
    const bytes = b64ToBytes(job.b64);

    let buffer;
    if (job.fmt === 'pcm') {
      // Reassemble float32 frames across chunk boundaries.
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
      for (let i = 0; i < n; i++) floats[i] = dv.getFloat32(i * 4, true); // LE
      buffer = ctx.createBuffer(1, n, PCM_SAMPLE_RATE);
      buffer.copyToChannel(floats, 0);
    } else {
      const ab = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
      buffer = await ctx.decodeAudioData(ab);
    }

    this._schedule(buffer, job);
  }

  _schedule(buffer, job) {
    const ctx = this.ctx;
    const src = ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(this.master);
    // Gapless: schedule back-to-back; resync to "now" after silence/barge-in.
    const startAt = Math.max(this.nextTime, ctx.currentTime + 0.015);
    src.start(startAt);
    this.nextTime = startAt + buffer.duration;
    // Playback anchoring ledger: which utterance owns this span.
    src._utt = job && job.utt !== undefined ? job.utt : null;
    src._uttStart = startAt;
    src._uttDur = buffer.duration;
    this.sources.add(src);
    this._setPlaying(true);
    src.onended = () => {
      this.sources.delete(src);
      this._updateIdle();
    };
  }

  _updateIdle() {
    if (this.sources.size === 0 && this.queue.length === 0 && !this.draining) {
      this._setPlaying(false);
      // Resync timeline so the next utterance doesn't inherit stale offsets.
      if (this.ctx) {
        this.nextTime = 0;
      }
    }
  }

  get isBusy() {
    return this.sources.size > 0 || this.queue.length > 0;
  }

  /** Barge-in: instant, total silence.
   *  When interrupted=true, returns per-utterance played fractions
   *  [{utt, frac}] so the backend keeps only words Sir actually heard
   *  (playback anchoring). Non-interrupt stops return []. */
  stopAll(interrupted) {
    let played = [];
    if (interrupted && this.ctx) {
      try {
        const now = this.ctx.currentTime;
        const byUtt = {};
        for (const src of this.sources) {
          if (src._utt === null || src._utt === undefined) continue;
          const el = byUtt[src._utt] || (byUtt[src._utt] = { played: 0, total: 0 });
          const span = Math.max(0, Math.min(now - (src._uttStart || now), src._uttDur || 0));
          el.played += span;
          el.total += (src._uttDur || 0);
        }
        // Queued (never started) jobs count as unplayed — reported as 0 so
        // the backend explicitly drops them instead of estimating.
        for (const job of this.queue) {
          if (job.utt !== null && job.utt !== undefined && !byUtt[job.utt]) {
            byUtt[job.utt] = { played: 0, total: 1 };
          }
        }
        played = Object.keys(byUtt).map((k) => ({
          utt: Number(k),
          frac: byUtt[k].total > 0 ? Math.min(1, byUtt[k].played / byUtt[k].total) : 0,
        }));
      } catch (e) { /* report nothing rather than wrong */ }
    }
    this.queue.length = 0;
    this.pendingPcm = null;
    for (const src of Array.from(this.sources)) {
      try { src.onended = null; src.stop(); } catch (e) { /* already stopped */ }
      try { src.disconnect(); } catch (e) {}
    }
    this.sources.clear();
    this.nextTime = 0;
    this._setPlaying(false);
    return played;
  }

  dispose() {
    this.stopAll();
    if (this.ctx) {
      try { this.ctx.close(); } catch (e) {}
      this.ctx = null;
    }
  }
}
