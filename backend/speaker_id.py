"""Feature 7 — Speaker & visitor awareness.

Voice identification is a *convenience signal*, never authentication: voice
can be deepfaked, so no auth path may read voiceprints (asserted in tests).
The explicit user rule: an unknown voice is never silently fingerprinted for
enrollment — ADA ASKS out loud ("I don't recognize that voice — who are you?")
and enrollment happens only when the answer confirms consent. Per-person
delete is available by voice ("forget that voice" / "forget my voice").

Design:
  * Zero local ML models, ever. MFCC fingerprinting is numpy-only DSP.
  * No raw audio is stored — only L2-normalized MFCC centroids in SQLite.
  * The hyperscalers are EXITING voice biometrics (Azure retired Sep 2025,
    AWS Voice ID end-of-support May 2026), so there is deliberately no
    dependency on them. Identification is CLOUD-ONLY via pyannoteAI whenever
    the PYANNOTE_API_KEY env var is set (owner's rule: cloud keys only) — a
    cloud failure degrades loudly to "visitor", never silently to another
    method, so key/quota problems stay visible instead of being masked.
  * With no key configured, a keyless MFCC fallback (pure numpy DSP — not a
    model, zero downloads, nothing leaves the machine) keeps the feature
    alive; it is a stopgap, not a peer of the cloud path.
  * The pyannote path has never been exercised against the live API (no key
    in CI); treat it as unverified until a keyed end-to-end run passes.

Pipeline (sound-sentinel style, research Feature 7):
  loudness normalize -> pre-emphasis -> VAD trim -> fixed 4s window ->
  MFCC (40 mel bands, drop C0) -> mean-pool -> L2-normalize ->
  cosine similarity vs enrolled centroids. MATCH_THRESHOLD = 0.85,
  deliberately conservative: on doubt the turn is tagged "visitor".

Public surface:
  fingerprint(pcm_16k_mono) -> np.ndarray | None
  VoiceprintStore            SQLite voiceprint registry (centroids only)
  identify_turn(...)         one interface: cloud if keyed, else local MFCC
  SpeakerManager             per-session turn processing + the voice-ask flow
  list_known_voices()        registry readable by the brain (names only)
  record_activity / recent_non_owner_active  feeds F8's speaker_present()
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import tempfile
import time
from collections import deque
from pathlib import Path

import numpy as np

# ── DSP constants ────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
FRAME_LEN = int(0.025 * SAMPLE_RATE)   # 25 ms
FRAME_HOP = int(0.010 * SAMPLE_RATE)   # 10 ms
N_FFT = 512
N_MELS = 40
N_DCT = N_MELS + 1                     # keep coeffs 1..40 (drop C0) -> 40 dims
PRE_EMPH = 0.97
FIXED_WINDOW_S = 4.0
MIN_VOICED_S = 2.0                     # shorter turns are skipped, never tagged
MATCH_THRESHOLD = 0.85                 # conservative: prefer "visitor" on doubt
# Ask dedup: a re-utterance of an already-asked unknown voice (cosine this
# high) is the same person -> never nag. Slightly looser than the match
# threshold so same-voice variation reliably dedups.
ASK_DEDUP_THRESHOLD = 0.80

VOICE_ASK_LINE = "I don't recognize that voice \u2014 who are you?"
FORGET_PHRASES = ("forget that voice", "forget my voice", "forget this voice")

# Labels returned by identify(): "owner:<name>" | "known:<name>" | "visitor"


# ── numpy-only MFCC ──────────────────────────────────────────────────────────

_mel_fb: np.ndarray | None = None
_dct_mat: np.ndarray | None = None


def _mel_filterbank() -> np.ndarray:
    """40 triangular mel filters over 0..8kHz, built once (deterministic)."""
    global _mel_fb
    if _mel_fb is not None:
        return _mel_fb
    mel_min = 2595.0 * np.log10(1.0)
    mel_max = 2595.0 * np.log10(1.0 + (SAMPLE_RATE / 2) / 700.0)
    mel_pts = np.linspace(mel_min, mel_max, N_MELS + 2)
    hz_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
    bins = np.floor((N_FFT + 1) * hz_pts / SAMPLE_RATE).astype(int)
    fb = np.zeros((N_MELS, N_FFT // 2 + 1))
    for m in range(1, N_MELS + 1):
        lo, mid, hi = bins[m - 1], bins[m], bins[m + 1]
        for k in range(lo, mid):
            fb[m - 1, k] = (k - lo) / max(mid - lo, 1)
        for k in range(mid, hi):
            fb[m - 1, k] = (hi - k) / max(hi - mid, 1)
    _mel_fb = fb
    return fb


def _dct_matrix() -> np.ndarray:
    """Type-II DCT basis, orthonormal, built once (deterministic)."""
    global _dct_mat
    if _dct_mat is not None:
        return _dct_mat
    n = np.arange(N_MELS)[None, :]
    k = np.arange(N_DCT)[:, None]
    m = np.cos(np.pi / N_MELS * (n + 0.5) * k) * np.sqrt(2.0 / N_MELS)
    m[0] /= np.sqrt(2.0)
    _dct_mat = m
    return m


def _frames(x: np.ndarray, length: int, hop: int) -> np.ndarray:
    n = max(1, 1 + (len(x) - length) // hop)
    idx = np.arange(length)[None, :] + (np.arange(n) * hop)[:, None]
    return x[np.clip(idx, 0, len(x) - 1)]


def fingerprint(pcm_16k_mono) -> np.ndarray | None:
    """MFCC voice fingerprint.

    Deterministic: the same input bytes always produce the identical vector,
    so cosine(fingerprint(x), fingerprint(x)) == 1.0. Returns None when the
    audio is silent or holds less than MIN_VOICED_S of voiced audio.
    """
    x = np.asarray(pcm_16k_mono, dtype=np.float64).ravel()
    if x.size == 0:
        return None
    # 1. loudness normalize (peak; log-mel + dropped C0 is level-invariant too)
    peak = float(np.max(np.abs(x)))
    if peak < 1e-9:
        return None
    x = x / peak
    # 2. pre-emphasis
    x = np.concatenate(([x[0]], x[1:] - PRE_EMPH * x[:-1]))
    # 3. VAD trim: frame RMS vs a fraction of the loudest frame
    fr = _frames(x, FRAME_LEN, FRAME_HOP)
    rms = np.sqrt(np.mean(fr ** 2, axis=1) + 1e-12)
    thr = max(float(np.max(rms)) * 0.08, 1e-4)
    voiced = rms >= thr
    if not bool(np.any(voiced)):
        return None
    first = int(np.argmax(voiced))
    last = int(len(voiced) - 1 - np.argmax(voiced[::-1]))
    s0 = first * FRAME_HOP
    s1 = min(len(x), last * FRAME_HOP + FRAME_LEN)
    if (s1 - s0) / SAMPLE_RATE < MIN_VOICED_S:
        return None
    # 4. fixed 4 s window (pad short turns with zeros)
    want = int(FIXED_WINDOW_S * SAMPLE_RATE)
    seg = x[s0:s1][:want]
    if len(seg) < want:
        seg = np.pad(seg, (0, want - len(seg)))
    # 5. MFCC: hamming -> FFT -> mel -> log -> DCT, drop C0
    win = np.hamming(FRAME_LEN)
    spec = np.abs(np.fft.rfft(_frames(seg, FRAME_LEN, FRAME_HOP) * win,
                              n=N_FFT, axis=1)) ** 2
    logmel = np.log10(np.maximum(spec @ _mel_filterbank().T, 1e-10))
    mfcc = (logmel @ _dct_matrix().T)[:, 1:]   # drop C0 -> (frames, 40)
    # 6. mean-pool over time, L2-normalize
    pooled = mfcc.mean(axis=0)
    nrm = float(np.linalg.norm(pooled))
    if nrm < 1e-12:
        return None
    return (pooled / nrm).astype(np.float64)


def decode_to_pcm16(audio_bytes: bytes, timeout: float = 15.0) -> np.ndarray | None:
    """Decode an uploaded clip (webm/opus/wav/…) to 16 kHz mono float PCM.

    Uses the system ffmpeg binary via subprocess; no Python audio dependency.
    Returns None on any failure — callers must skip speaker-ID, never crash.
    """
    if not audio_bytes:
        return None
    try:
        p = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", "pipe:0",
             "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "pipe:1"],
            input=audio_bytes, capture_output=True, timeout=timeout)
    except Exception:
        return None
    if p.returncode != 0 or not p.stdout:
        return None
    return np.frombuffer(p.stdout, dtype=np.int16).astype(np.float64) / 32768.0


# ── SQLite voiceprint store (centroids only, never raw audio) ────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS voiceprints(
    person_id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    centroid_blob BLOB NOT NULL,
    enrolled_ts REAL NOT NULL,
    consent_event_ref TEXT,
    updated_ts REAL NOT NULL,
    is_owner INTEGER NOT NULL DEFAULT 0,
    cloud_voiceprint_id TEXT
)"""


class VoiceprintStore:
    """Local registry of enrolled voice centroids.

    Default is in-memory: production passes an explicit path (ada_kernel/
    voiceprints.db); nothing here may ever drop a stray DB into the repo.
    *.db is gitignored — the DB is runtime state, not committed.
    """

    def __init__(self, db_path: str | None = None):
        db_path = db_path or os.getenv("ADA_VOICEPRINT_DB") or ":memory:"
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_SCHEMA)
        # tolerate DBs created by older schemas
        for col, ddl in (("is_owner", "INTEGER NOT NULL DEFAULT 0"),
                         ("cloud_voiceprint_id", "TEXT")):
            try:
                self._conn.execute(f"ALTER TABLE voiceprints ADD COLUMN {col} {ddl}")
            except sqlite3.OperationalError:
                pass
        self._conn.commit()

    @staticmethod
    def _slug(name: str) -> str:
        s = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
        return s or "person"

    def enroll(self, name: str, centroid: np.ndarray,
               consent_event_ref: str | None = None,
               owner: bool = False) -> str:
        """Enroll (or re-enroll) a voice. Returns the label for this person.

        Enrollment is ONLY ever called from the voice-ask flow — i.e. after
        the person answered the spoken ask — never silently.
        """
        name = " ".join(str(name).split())
        if not name:
            raise ValueError("enroll needs a name")
        centroid = np.asarray(centroid, dtype=np.float64).ravel()
        if centroid.size == 0:
            raise ValueError("enroll needs a non-empty centroid")
        now = time.time()
        pid = self._slug(name)
        row = self._conn.execute(
            "SELECT person_id FROM voiceprints WHERE name = ?", (name,)).fetchone()
        if row:
            pid = row["person_id"]
            self._conn.execute(
                "UPDATE voiceprints SET centroid_blob = ?, updated_ts = ?,"
                " consent_event_ref = COALESCE(?, consent_event_ref),"
                " is_owner = MAX(is_owner, ?) WHERE person_id = ?",
                (centroid.tobytes(), now, consent_event_ref, int(owner), pid))
        else:
            self._conn.execute(
                "INSERT INTO voiceprints(person_id, name, centroid_blob,"
                " enrolled_ts, consent_event_ref, updated_ts, is_owner)"
                " VALUES(?,?,?,?,?,?,?)",
                (pid, name, centroid.tobytes(), now, consent_event_ref,
                 now, int(owner)))
        self._conn.commit()
        return f"{'owner' if owner else 'known'}:{name}"

    def identify(self, centroid: np.ndarray) -> tuple[str, float]:
        """Best-match label + cosine score. Conservative: doubt -> "visitor"."""
        centroid = np.asarray(centroid, dtype=np.float64).ravel()
        rows = self._conn.execute(
            "SELECT name, centroid_blob, is_owner FROM voiceprints").fetchall()
        best_label, best_score = "visitor", 0.0
        for r in rows:
            c = np.frombuffer(r["centroid_blob"], dtype=np.float64)
            if c.size != centroid.size:
                continue
            score = float(np.dot(centroid, c))
            if score > best_score:
                role = "owner" if r["is_owner"] else "known"
                best_label, best_score = f"{role}:{r['name']}", score
        if best_score < MATCH_THRESHOLD:
            return "visitor", best_score
        return best_label, best_score

    def forget(self, name: str) -> bool:
        """Hard-delete one person's voiceprint row. Returns True if one existed."""
        cur = self._conn.execute("DELETE FROM voiceprints WHERE name = ?",
                                 (" ".join(str(name).split()),))
        self._conn.commit()
        return cur.rowcount > 0

    def get(self, name: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM voiceprints WHERE name = ?",
            (" ".join(str(name).split()),)).fetchone()
        return dict(row) if row else None

    def list_known_voices(self) -> list[dict]:
        """Registry the brain can read: names only, no audio, no vectors."""
        rows = self._conn.execute(
            "SELECT person_id, name, enrolled_ts, consent_event_ref, is_owner"
            " FROM voiceprints ORDER BY enrolled_ts").fetchall()
        return [{"person_id": r["person_id"], "name": r["name"],
                 "role": "owner" if r["is_owner"] else "known",
                 "enrolled_ts": r["enrolled_ts"],
                 "consent_event_ref": r["consent_event_ref"]}
                for r in rows]

    def owner_name(self) -> str | None:
        row = self._conn.execute(
            "SELECT name FROM voiceprints WHERE is_owner = 1 LIMIT 1").fetchone()
        return row["name"] if row else None


# ── optional pyannoteAI cloud path (keyed only, best-effort) ────────────────

_PYANNOTE_IDENTIFY_URL = "https://api.pyannote.ai/v2/identify"
_PYANNOTE_VOICEPRINTS_URL = "https://api.pyannote.ai/v2/voiceprints"


def _pyannote_enroll(audio_bytes: bytes, filename: str, name: str,
                     api_key: str, timeout: float = 30.0) -> str | None:
    """Create a pyannote voiceprint from an enrollment turn's raw audio.

    Raw audio is sent only at enrollment time, with consent, and only to
    pyannoteAI — it is never stored locally (DB keeps centroids only).
    """
    import httpx
    headers = {"Authorization": f"Bearer {api_key}"}
    files = {"file": (filename or "enroll.webm", audio_bytes, "audio/webm")}
    data = {"label": name}
    with httpx.Client(timeout=timeout) as c:
        r = c.post(_PYANNOTE_VOICEPRINTS_URL, headers=headers,
                   files=files, data=data)
        r.raise_for_status()
        body = r.json()
    vp = (body.get("voiceprint") or body.get("voicePrint") or body)
    vid = vp.get("id") if isinstance(vp, dict) else None
    return str(vid) if vid else None


def _pyannote_identify_best(audio_bytes: bytes, filename: str,
                            store: VoiceprintStore, api_key: str,
                            timeout: float = 30.0) -> tuple[str | None, float]:
    """Identify via pyannoteAI against previously cloud-enrolled voiceprints.

    Returns (None, 0.0) when nothing can be matched; raises on transport
    failure so the caller can fall back to the local MFCC path.
    """
    import httpx
    known = [v for v in store.list_known_voices()]
    cloud_ids = [store.get(v["name"]) or {} for v in known]
    pairs = [(v["name"], store.get(v["name"]) or {})
             for v in known]
    ids = [p["cloud_voiceprint_id"] for _, p in pairs
           if p.get("cloud_voiceprint_id")]
    if not ids:
        return None, 0.0
    headers = {"Authorization": f"Bearer {api_key}"}
    files = {"file": (filename or "turn.webm", audio_bytes, "audio/webm")}
    data = {"voiceprints": json.dumps(ids)}
    with httpx.Client(timeout=timeout) as c:
        r = c.post(_PYANNOTE_IDENTIFY_URL, headers=headers,
                   files=files, data=data)
        r.raise_for_status()
        body = r.json()
        job_id = (body.get("identificationJobId") or body.get("jobId")
                  or body.get("id"))
        if not job_id:
            return None, 0.0
        status, out = "", None
        for _ in range(60):
            g = c.get(f"{_PYANNOTE_IDENTIFY_URL}/{job_id}", headers=headers)
            g.raise_for_status()
            st = g.json()
            status = st.get("status", "")
            if status == "succeeded":
                out = st.get("output") or st.get("result")
                break
            if status in ("failed", "cancelled", "error"):
                raise RuntimeError(f"pyannote identify {status}")
            time.sleep(0.5)
    if not out:
        return None, 0.0
    # output: list of {speaker: <voiceprint id>, start, end}; take the
    # label with the most attributed time.
    totals: dict[str, float] = {}
    segs = out if isinstance(out, list) else out.get("segments", [])
    for s in segs if isinstance(segs, list) else []:
        sp = s.get("speaker") or s.get("label")
        dur = float(s.get("end", 0) or 0) - float(s.get("start", 0) or 0)
        if sp:
            totals[str(sp)] = totals.get(str(sp), 0.0) + max(dur, 0.0)
    if not totals:
        return None, 0.0
    winner = max(totals, key=totals.get)
    for name, p in pairs:
        if p.get("cloud_voiceprint_id") == winner:
            role = "owner" if (store.get(name) or {}).get("is_owner") else "known"
            return f"{role}:{name}", 0.9
    return "visitor", 0.0


def identify_turn(pcm=None, *, store: VoiceprintStore,
                  audio_bytes: bytes | None = None,
                  filename: str | None = None) -> tuple[str, float]:
    """One interface for turn identification.

    Cloud-only via pyannoteAI when PYANNOTE_API_KEY is set (owner's rule):
    a cloud failure degrades loudly to "visitor" — never silently to MFCC —
    so key/quota problems stay visible. With no key, the keyless MFCC
    fallback (pure DSP, not a model) answers. Never hardcodes keys — the
    key comes only from the environment.
    """
    key = os.environ.get("PYANNOTE_API_KEY")
    if key:
        if not audio_bytes:
            return "visitor", 0.0
        try:
            label, score = _pyannote_identify_best(audio_bytes, filename or "",
                                                   store, key)
            return (label, score) if label else ("visitor", 0.0)
        except Exception as e:
            print(f"[SPEAKER] cloud speaker-ID failed ({type(e).__name__}):"
                  " check PYANNOTE_API_KEY / quota; tagging visitor")
            return "visitor", 0.0
    if pcm is None:
        return "visitor", 0.0
    fp = fingerprint(pcm)
    if fp is None:
        return "visitor", 0.0
    return store.identify(fp)


# ── activity log (feeds F8's speaker_present hook) ───────────────────────────

_ACTIVITY: deque = deque(maxlen=256)   # (ts, label); process-local only


def record_activity(label: str, ts: float | None = None) -> None:
    _ACTIVITY.append((ts if ts is not None else time.time(), str(label)))


def recent_non_owner_active(window_minutes: float = 5.0,
                            now: float | None = None) -> bool:
    """True when a non-owner voice was active inside the window.

    Feeds the proactivity "don't interrupt human–human conversation" rail.
    Anything that is not the owner — a known person or an unknown visitor —
    counts, because either way a human is in the room.
    """
    now = now if now is not None else time.time()
    cutoff = now - window_minutes * 60.0
    return any(ts >= cutoff and not label.startswith("owner:")
               for ts, label in _ACTIVITY)


def _reset_activity() -> None:
    _ACTIVITY.clear()


# ── the voice-ask flow ───────────────────────────────────────────────────────

_OWNER_RE = re.compile(
    r"\b(that'?s me|it'?s me|this is me|i'?m the (owner|boss)|^yes\b|^yeah\b|^yep\b|^yup\b)",
    re.IGNORECASE)
_YASH_RE = re.compile(r"\byash\b", re.IGNORECASE)
_NAME_RE = re.compile(
    r"\b(?:i'?m|my name is|call me|this is)\s+([A-Za-z][A-Za-z .'\-]{1,29})",
    re.IGNORECASE)
_FORGET_RE = re.compile(r"\bforget\s+(?:that|my|this)\s+voice\b", re.IGNORECASE)
_FORGET_NAMED_RE = re.compile(
    r"\bforget\s+(?!that\b|my\b|this\b)([A-Za-z][A-Za-z .'\-]{1,29})'?s?\s+voice\b",
    re.IGNORECASE)


def _timeline_emit(source: str, etype: str, payload: dict):
    """Best-effort F4 timeline write; never breaks the voice path."""
    try:
        from perception import emit as _emit
        return _emit(source, etype, payload)
    except Exception:
        return None


class SpeakerManager:
    """Per-session turn processing + the human-like voice-ask flow.

    Contract with the client (Electron): /stt returns {"text", "speaker",
    "voice_ask"?}. When "voice_ask" is present the client SPEAKS it
    (voice-direct, never a popup) and routes the answering turn through
    /stt again; process_turn() detects the pending ask and enrolls.
    """

    def __init__(self, store: VoiceprintStore, owner_hint: str | None = None):
        self.store = store
        self.owner_hint = owner_hint
        # Unknown voices already asked about this session (centroids, not
        # hashes: a re-utterance of the same voice must dedup to "no nag").
        self._asked: list[np.ndarray] = []
        self._pending: dict | None = None      # {"centroid":..., "raw_audio":...}
        self.last_enrolled: str | None = None

    def _already_asked(self, fp: np.ndarray) -> bool:
        return any(float(np.dot(fp, prev)) >= ASK_DEDUP_THRESHOLD
                   for prev in self._asked)

    # ── public ──
    def process_turn(self, audio_bytes: bytes, filename: str | None = None,
                     transcript_text: str = "") -> dict:
        """Identify one STT turn; drive the ask/enroll/forget flow.

        Returns {"speaker_label", "speaker_score", "voice_ask" (or None),
        "enrolled" (or None), "forgot" (or None)}. Never raises.
        """
        out = {"speaker_label": "visitor", "speaker_score": 0.0,
               "voice_ask": None, "enrolled": None, "forgot": None}
        try:
            return self._process(audio_bytes, filename, transcript_text or "", out)
        except Exception as e:
            print(f"[SPEAKER] turn skipped: {type(e).__name__}: {str(e)[:120]}")
            return out

    def get_pending_ask(self) -> str | None:
        """Flag the brain reads: the spoken ask line, if one is due."""
        return VOICE_ASK_LINE if self._pending else None

    # ── internals ──
    def _process(self, audio_bytes, filename, text, out) -> dict:
        # 1. decode + identify (skip silently if too little voiced audio)
        pcm = decode_to_pcm16(audio_bytes or b"") if audio_bytes else None
        label, score = identify_turn(pcm, store=self.store,
                                     audio_bytes=audio_bytes,
                                     filename=filename or "turn.webm")
        out["speaker_label"], out["speaker_score"] = label, score
        record_activity(label)

        # 2. voice forget command (works any time, even as the ask's answer —
        #    an explicit delete is never an enrollment answer)
        if text and _FORGET_RE.search(text):
            name = self._forget_target(text)
            if name and self.store.forget(name):
                out["forgot"] = name
                _timeline_emit("speaker", "voice.forgotten",
                               {"name": name, "via": "voice"})
                if self.last_enrolled == name:
                    self.last_enrolled = None
            return out

        # 3. answering the pending voice ask? (the answer turn's OWN centroid
        #    is not what we enroll — we enroll the unknown voice we asked about)
        if self._pending and text.strip():
            self._answer_pending(text, out)
            return out

        # 4. unknown voice + never asked about it this session -> the ask.
        #    Same-voice re-utterances dedup via cosine: ask once, never nag.
        if label == "visitor" and pcm is not None:
            fp = fingerprint(pcm)
            if fp is not None and not self._already_asked(fp):
                self._asked.append(fp)
                self._pending = {
                    "centroid": fp,
                    # RAM only, cleared on answer; the DB never holds raw
                    # audio, centroids only.
                    "raw_audio": audio_bytes}
                out["voice_ask"] = VOICE_ASK_LINE
                _timeline_emit("speaker", "visitor.present",
                               {"score": round(float(score), 3)})
        return out

    def _answer_pending(self, text: str, out: dict) -> None:
        pending = self._pending
        self._pending = None
        centroid = pending["centroid"]
        low = text.strip().lower()
        seq_ref = f"voice.ask.answered@{int(time.time())}"
        # owner confirmation: "yes, that's me" / "it's me" / mentions Yash
        if _OWNER_RE.search(text) or _YASH_RE.search(text):
            name = self.owner_hint or self.store.owner_name() or "Yash"
            label = self.store.enroll(name, centroid,
                                      consent_event_ref=seq_ref, owner=True)
            self.last_enrolled = name
            out.update(speaker_label=label, speaker_score=1.0, enrolled=name)
            _timeline_emit("speaker", "visitor.enrolled",
                           {"name": name, "role": "owner",
                            "consent_event_ref": seq_ref})
            self._maybe_cloud_enroll(name, centroid, pending)
            return
        # named introduction: "I'm Maya" / "my name is Maya"
        m = _NAME_RE.search(text)
        if m:
            name = " ".join(m.group(1).split()).title()
            # never let an introduction claim the owner's name slot silently
            label = self.store.enroll(name, centroid,
                                      consent_event_ref=seq_ref, owner=False)
            self.last_enrolled = name
            out.update(speaker_label=label, speaker_score=1.0, enrolled=name)
            _timeline_emit("speaker", "visitor.enrolled",
                           {"name": name, "role": "known",
                            "consent_event_ref": seq_ref})
            self._maybe_cloud_enroll(name, centroid, pending)
            return
        # any other answer: leave them a visitor, never enroll, never nag again
        _timeline_emit("speaker", "visitor.present",
                       {"note": "ask declined or unrecognized answer"})
        _ = low  # (answer text stays in the STT transcript, not here)

    def _forget_target(self, text: str) -> str | None:
        m = _FORGET_NAMED_RE.search(text)
        if m:
            return " ".join(m.group(1).split()).title()
        # "forget that voice": drop the most recently enrolled non-owner
        if self.last_enrolled:
            row = self.store.get(self.last_enrolled)
            if row and not row.get("is_owner"):
                return self.last_enrolled
        known = [v for v in self.store.list_known_voices()
                 if v["role"] != "owner"]
        return known[-1]["name"] if known else None

    def _maybe_cloud_enroll(self, name: str, centroid: np.ndarray,
                            pending: dict) -> None:
        """pyannote voiceprint creation; the local row stays as the people
        registry (names, consent refs, owner flags) and as keyless-fallback
        data. When keyed, identification uses the cloud voiceprint id."""
        key = os.environ.get("PYANNOTE_API_KEY")
        raw = pending.get("raw_audio")
        if not (key and raw):
            return
        try:
            vid = _pyannote_enroll(raw, "enroll.webm", name, key)
            if vid:
                self.store._conn.execute(
                    "UPDATE voiceprints SET cloud_voiceprint_id = ?"
                    " WHERE name = ?", (vid, name))
                self.store._conn.commit()
        except Exception as e:
            print(f"[SPEAKER] cloud enroll skipped: {type(e).__name__}")

    # ── registry for the brain ──
    def list_known_voices(self) -> list[dict]:
        return self.store.list_known_voices()


# ── process-wide singleton (what /stt uses) ─────────────────────────────────

_MANAGER: SpeakerManager | None = None


def _owner_hint_from_soul() -> str | None:
    """Owner's first name from ada_kernel/soul/humans/*.md (never hardcoded)."""
    try:
        root = Path(__file__).resolve().parent.parent
        humans = root / "ada_kernel" / "soul" / "humans"
        for md in sorted(humans.glob("*.md")):
            return md.stem.replace("_", " ").strip() or None
    except Exception:
        pass
    return None


def default_db_path() -> str:
    if os.getenv("ADA_VOICEPRINT_DB"):
        return os.environ["ADA_VOICEPRINT_DB"]
    root = Path(__file__).resolve().parent.parent
    return str(root / "ada_kernel" / "voiceprints.db")


def get_manager() -> SpeakerManager:
    """Process-wide SpeakerManager (persistent SQLite, per-session ask set)."""
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = SpeakerManager(VoiceprintStore(default_db_path()),
                                  owner_hint=_owner_hint_from_soul())
    return _MANAGER


def reset_manager() -> None:
    """Tests only."""
    global _MANAGER
    _MANAGER = None
