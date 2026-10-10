"""
Realtime Text-to-Speech for ADA — Mistral Voxtral ("Jane", female British)
with true streaming, humanized input, and instant barge-in word tracking.

Pipeline:
  brain deltas -> SpeechStreamer -> sentence/clause split -> utterance batches
  -> speech_normalizer (humanized input) -> Mistral Voxtral STREAMING TTS
  (pcm chunks) -> audio streamed to the frontend while the brain generates.

Single-voice doctrine:
  Ada speaks with ONE voice only — Jane. No edge-tts, no a4f, no foreign
  fallback voices. If the API hiccups she retries intelligently (param-shape
  probe, guardrail-wording soften) and as an absolute last resort speaks a
  short apology line through the same voice rather than degrading to a
  different-sounding engine.

Model note:
  voxtral-mini-tts-latest is Mistral's ONLY text-to-speech model (the
  expressive voice from their playground demos). voxtral-small/mini without
  the -tts suffix are transcription models, not speech generators.

Why the voice sounds human:
  Voxtral steers emotion IMPLICITLY from the text itself — expressive writing
  in, expressive speech out. Every utterance passes through
  speech_normalizer.prepare_spoken_text(): markdown/emojis stripped, numbers
  verbalized to words, abbreviations spelled out. The brain's system prompt
  writes the expressiveness; this layer guarantees nothing robotic leaks in.

Barge-in: when Sir starts speaking again, cancel() fires instantly and records
EXACTLY which word of which sentence she was cut at — fed back into her next
turn so she knows where she left off.
"""
import os
import re
import sys
import time
import json as _json
import asyncio
import base64
from dotenv import load_dotenv

load_dotenv(override=True)

try:
    from speech_normalizer import prepare_spoken_text
except ImportError:  # pragma: no cover - keeps TTS alive if normalizer is missing
    def prepare_spoken_text(t):
        return t

_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')
_CLAUSE_SPLIT = re.compile(r'(?<=[,;:])\s+')
_SENTINEL = object()

# ─── Latency tuning ──────────────────────────────────────────────────────────
# EARLY_START_CHARS: speak the first clause before the full sentence arrives.
# 90 -> 50 cuts ~200-400ms off time-to-first-word (slightly choppier opener).
# PIPELINE: while utterance N is playing, utterance N+1's stream is already
# running server-side, so the gapless queue never starves on long replies.
EARLY_START_CHARS = int(os.environ.get("ADA_TTS_EARLY_START", "50"))
EARLY_START_MIN_CLAUSE = 20
PIPELINE_ENABLED = os.environ.get("ADA_TTS_PIPELINE", "1") != "0"
STREAM_STALL_TIMEOUT = float(os.environ.get("ADA_TTS_STALL_TIMEOUT", "15"))

# ─── Mistral Voxtral ─────────────────────────────────────────────────────────
VOXTRAL_MODEL = os.environ.get("ADA_TTS_MODEL", "voxtral-mini-tts-latest")
VOXTRAL_SAMPLE_RATE = 24000          # pcm: raw float32 LE mono samples/sec
VOXTRAL_MAX_WORDS = 290              # docs: keep prompts under 300 words
_vox_lock = asyncio.Lock()
_http = None        # warm sync httpx client
_ahttp = None       # warm async httpx client

# THE PLAYGROUND SECRET, DECODED:
# Mistral's Studio "Select an emotion" panel is NOT a text instruction — every
# preset voice is the SAME speaker performing a DIFFERENT emotion (reference
# clips; see the Voxtral TTS paper: emotion is steered by swapping the voice
# prompt). Jane ships as 9 emotion performances, all verified live:
#   neutral, sarcasm, confused, shameful, sad, jealousy, frustrated,
#   curious, confident.
# Ada picks the matching performance per utterance -> real human pacing,
# sudden slowdowns, gaps, and emotional texture. No param faking.

JANE_EMOTION_VOICES = {
    # auto_emotion label  ->  Jane's performed emotion (voice slug)
    "neutral":    "gb_jane_neutral",
    "calm":       "gb_jane_neutral",      # Clear, Measured
    "serious":    "gb_jane_neutral",
    "warm":       "gb_jane_neutral",
    "grateful":   "gb_jane_neutral",
    "tender":     "gb_jane_shameful",     # Quiet, soft — closest to tender
    "thoughtful": "gb_jane_confused",     # Hesitant, pondering pace
    "confused":   "gb_jane_confused",
    "worried":    "gb_jane_confused",     # Uncertain energy
    "ashamed":    "gb_jane_shameful",     # Remorseful
    "sad":        "gb_jane_sad",          # Soft, subdued
    "jealous":    "gb_jane_jealousy",     # Bitter, strained
    "frustrated": "gb_jane_frustrated",   # Tense, clipped
    "curious":    "gb_jane_curious",      # Inquisitive lift
    "surprised":  "gb_jane_curious",
    "confident":  "gb_jane_confident",    # Assured, poised
    "happy":      "gb_jane_confident",
    "cheerful":   "gb_jane_confident",
    "excited":    "gb_jane_confident",
    "sarcastic":  "gb_jane_sarcasm",      # Dry, wry — her signature
    "playful":    "gb_jane_sarcasm",
}

# Her signature voice when no emotion is detected (dry-wit British female,
# literally built for JARVIS-style banter).
VOXTRAL_FEMALE_VOICE = "gb_jane_sarcasm"

# Runtime-discovered request shape: "voice_id" (current API) or "voice"
# (legacy). Probed once, then cached for process lifetime.
_voice_param_mode = None
# Slugs proven missing from the catalog (404) — never retried this session.
_dead_voices = set()


def resolve_jane_voice(emotion) -> str:
    """Maps an emotion label to Jane's matching performed-emotion voice."""
    slug = JANE_EMOTION_VOICES.get((emotion or "").lower())
    if not slug or slug in _dead_voices:
        return VOXTRAL_FEMALE_VOICE
    return slug


def _warm_http():
    global _http
    if _http is None:
        import httpx
        try:
            _http = httpx.Client(http2=True, timeout=httpx.Timeout(60.0, connect=8.0))
        except Exception:   # h2 missing -> HTTP/1.1 fallback
            _http = httpx.Client(timeout=httpx.Timeout(60.0, connect=8.0))
    return _http


async def _warm_ahttp():
    global _ahttp
    if _ahttp is None:
        import httpx
        try:
            _ahttp = httpx.AsyncClient(http2=True, timeout=httpx.Timeout(120.0, connect=8.0))
        except Exception:
            _ahttp = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=8.0))
    return _ahttp


def VOXTRAL_PARAM_KEY() -> str:
    return "voice_id" if _voice_param_mode != "legacy" else "voice"


def _base_payload(text: str, response_format: str = "mp3", voice: str = None) -> dict:
    return {"model": VOXTRAL_MODEL, "input": text,
            "response_format": response_format,
            VOXTRAL_PARAM_KEY(): voice or VOXTRAL_FEMALE_VOICE}


def _looks_like_param_error(status: int, body: str) -> bool:
    if status not in (400, 404, 422):
        return False
    low = body.lower()
    return any(k in low for k in ("voice_id", '"voice"', "unknown field",
                                  "extra fields", "unrecognized", "invalid voice"))


def _is_voice_not_found(status: int, body: str, voice: str) -> bool:
    """Mistral removed/renamed a preset slug -> blacklist it, fall back."""
    if status != 404:
        return False
    low = (body or "").lower()
    return ("voice" in low and "not found" in low) or (voice and voice.lower() in low)


def _probe_legacy_after_failure() -> bool:
    """After a param-shape failure, flip to the legacy 'voice' key."""
    global _voice_param_mode
    if _voice_param_mode == "legacy":
        return False
    _voice_param_mode = "legacy"
    print("[TTS] [VOXTRAL] Falling back to legacy 'voice' parameter.")
    return True


def _is_guardrail_block(status: int, body: str) -> bool:
    low = (body or "").lower()
    return status == 403 and ("guardrail" in low or "moderation" in low
                              or "blocked" in low)


# ─── Guardrail dodge ─────────────────────────────────────────────────────────
# Mistral's TTS moderation occasionally false-positives on innocuous words
# (observed: "failed" or "cutting me off" in casual engineering banter ->
# HTTP 403). When that happens we soften ONLY the risky wording and retry —
# meaning stays identical, delivery stays on Jane's voice.

_GUARDRAIL_SWAPS = [
    # conversational phrases first (longest match wins via ordering)
    (re.compile(r"\bcutting (me|him|her|us|them) off\b", re.I), r"interrupting \1"),
    (re.compile(r"\bcut (me|him|her|us|them) off\b", re.I), r"interrupted \1"),
    (re.compile(r"\bcuts (me|him|her|us|them) off\b", re.I), r"interrupts \1"),
    (re.compile(r"\bchopping (me|him|her|us|them) off\b", re.I), r"interrupting \1"),
    # single words
    (re.compile(r"\bfailed\b", re.I), "did not succeed"),
    (re.compile(r"\bfails\b", re.I), "does not succeed"),
    (re.compile(r"\bfailing\b", re.I), "not succeeding"),
    (re.compile(r"\bfailures?\b", re.I), "setbacks"),
    (re.compile(r"\bdead\b", re.I), "not responding"),
    (re.compile(r"\bkilled\b", re.I), "stopped"),
    (re.compile(r"\bkill\b", re.I), "stop"),
    (re.compile(r"\bdestroyed\b", re.I), "badly damaged"),
    (re.compile(r"\bexplode[d]?\b", re.I), "broke apart"),
    (re.compile(r"\bcrashed\b", re.I), "went down"),
]


def _guardrail_soften(text: str) -> str:
    """Pass 1: single risky words. Pass 2 (aggressive): phrase rewrites used
    when pass 1 was already applied and still blocked."""
    for rx, repl in _GUARDRAIL_SWAPS:
        text = rx.sub(repl, text)
    return text


def _guardrail_soften_aggressive(text: str) -> str:
    """Second resort: also neutralize remaining violent/metaphor verbs."""
    text = _guardrail_soften(text)
    extra = [
        (re.compile(r"\bhate\b", re.I), "dislike"),
        (re.compile(r"\bstab\b|\bstabbing\b", re.I), "poke"),
        (re.compile(r"\bshoot\b|\bshooting\b", re.I), "launch"),
        (re.compile(r"\bwar\b", re.I), "conflict"),
        (re.compile(r"\battack\b", re.I), "assault-on-problem"),
    ]
    for rx, repl in extra:
        text = rx.sub(repl, text)
    return text


# ─── Core Voxtral calls ──────────────────────────────────────────────────────

def _voxtral_sync(text: str, emotion=None) -> bytes:
    """Mistral Voxtral TTS (blocking) — Jane via warm connection. Returns MP3.
    Picks Jane's emotion performance from the label; handles param-shape
    probing, missing-voice fallback, and guardrail-soften retries."""
    global _voice_param_mode
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        raise RuntimeError("MISTRAL_API_KEY not set")
    client = _warm_http()
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    voice = resolve_jane_voice(emotion)
    payload = _base_payload(text, voice=voice)
    resp = client.post("https://api.mistral.ai/v1/audio/speech", headers=headers, json=payload)

    # Emotion voice missing from catalog -> blacklist slug, retry her signature.
    if _is_voice_not_found(resp.status_code, resp.text, voice) and voice != VOXTRAL_FEMALE_VOICE:
        _dead_voices.add(voice)
        print(f"[TTS] [VOXTRAL] voice '{voice}' not in catalog -> falling back to {VOXTRAL_FEMALE_VOICE}.")
        voice = VOXTRAL_FEMALE_VOICE
        payload = _base_payload(text, voice=voice)
        resp = client.post("https://api.mistral.ai/v1/audio/speech", headers=headers, json=payload)

    if resp.status_code != 200 and _looks_like_param_error(resp.status_code, resp.text):
        if _probe_legacy_after_failure():
            payload = _base_payload(text, voice=voice)
            resp = client.post("https://api.mistral.ai/v1/audio/speech", headers=headers, json=payload)

    softened_level = 0
    while _is_guardrail_block(resp.status_code, resp.text) and softened_level < 2:
        softened_level += 1
        candidate = (_guardrail_soften if softened_level == 1 else _guardrail_soften_aggressive)(text)
        if candidate == text:
            break
        print(f"[TTS] [VOXTRAL] guardrail false-positive (pass {softened_level}) "
              f"-> softening wording, retrying.")
        payload = _base_payload(candidate, voice=voice)
        resp = client.post("https://api.mistral.ai/v1/audio/speech", headers=headers, json=payload)
        if resp.status_code == 200:
            text = candidate   # remember what actually got spoken

    if resp.status_code != 200:
        raise RuntimeError(f"[voxtral] HTTP {resp.status_code}: {resp.text[:140]}")
    data = resp.json().get("audio_data", "")
    if not data:
        raise RuntimeError("[voxtral] empty audio_data")
    if _voice_param_mode is None:
        _voice_param_mode = "voice_id"
    return base64.b64decode(data)


async def voxtral_stream(text: str, emotion=None, _soften_level: int = 0):
    """Streams Voxtral TTS as raw float32 LE PCM chunks (lowest latency path).
    Resolves Jane's emotion performance from the label, then delegates."""
    voice = VOXTRAL_FEMALE_VOICE if emotion == VOXTRAL_FEMALE_VOICE \
        else resolve_jane_voice(emotion)
    async for chunk in _voxtral_stream_voice(text, voice, emotion, _soften_level):
        yield chunk


async def _voxtral_stream_voice(text: str, voice: str, emotion=None, _soften_level: int = 0):
    """Streaming core with an explicit voice slug. Handles param-shape probing,
    missing-voice fallback, and up to two guardrail-soften retries."""
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        raise RuntimeError("MISTRAL_API_KEY not set")
    http = await _warm_ahttp()
    payload = _base_payload(text, response_format="pcm", voice=voice)
    payload["stream"] = True
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "Accept": "text/event-stream"}
    global _voice_param_mode
    first_event_seen = False
    async with http.stream("POST", "https://api.mistral.ai/v1/audio/speech",
                           headers=headers, json=payload) as resp:
        if resp.status_code != 200:
            body = (await resp.aread()).decode("utf-8", "ignore")[:300]
            if (_is_voice_not_found(resp.status_code, body, voice)
                    and voice != VOXTRAL_FEMALE_VOICE):
                _dead_voices.add(voice)
                print(f"[TTS] [VOXTRAL] voice '{voice}' not in catalog -> {VOXTRAL_FEMALE_VOICE}.")
                async for chunk in _voxtral_stream_voice(text, VOXTRAL_FEMALE_VOICE,
                                                         emotion, _soften_level):
                    yield chunk
                return
            if _looks_like_param_error(resp.status_code, body) and _probe_legacy_after_failure():
                async for chunk in _voxtral_stream_voice(text, voice, emotion, _soften_level):
                    yield chunk
                return
            if _is_guardrail_block(resp.status_code, body) and _soften_level < 2:
                softener = _guardrail_soften if _soften_level == 0 else _guardrail_soften_aggressive
                candidate = softener(text)
                if candidate != text:
                    print(f"[TTS] [VOXTRAL] guardrail false-positive in stream "
                          f"(pass {_soften_level + 1}) -> softening, retrying.")
                    async for chunk in _voxtral_stream_voice(candidate, voice,
                                                             emotion, _soften_level + 1):
                        yield chunk
                    return
            raise RuntimeError(f"[voxtral-stream] HTTP {resp.status_code}: {body[:140]}")
        if _voice_param_mode is None:
            _voice_param_mode = "voice_id"
        buf = b""
        aiter = resp.aiter_bytes()
        while True:
            # Inactivity watchdog: a stalled stream (rate-limit freeze, dead
            # socket) must never hang her voice — abort to the sync fallback.
            try:
                raw = await asyncio.wait_for(aiter.__anext__(), timeout=STREAM_STALL_TIMEOUT)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                raise RuntimeError(
                    f"[voxtral-stream] no data for {STREAM_STALL_TIMEOUT}s — aborting to fallback.")
            buf += raw
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line or line.startswith(b":") or line.startswith(b"event:"):
                    continue
                if line.startswith(b"data:"):
                    data_str = line[5:].strip()
                    if not data_str or data_str == b"[DONE]":
                        continue
                    try:
                        evt = _json.loads(data_str)
                    except Exception:
                        continue
                    audio_b64 = evt.get("audio_data") or (evt.get("data") or {}).get("audio_data")
                    if not audio_b64:
                        continue
                    first_event_seen = True
                    pcm = base64.b64decode(audio_b64)
                    if pcm:
                        yield pcm
        if not first_event_seen:
            raise RuntimeError("[voxtral-stream] stream ended with no audio events")


# ─── AUTO-EMOTION ENGINE ─────────────────────────────────────────────────────
# Scores every sentence against lexicons + punctuation signals and picks the
# most natural emotion — like a voice actor reading the line. With Voxtral the
# label doesn't map to API params (implicit steering reads the text itself);
# it is kept for logging, future voice-profile routing, and UI hints.

_EMOTION_LEXICON = {
    "excited": ("wow", "amazing", "incredible", "awesome", "fantastic", "brilliant",
                "let's go", "excellent", "perfect", "stunning", "insane", "unreal"),
    "happy": ("great", "nice", "love", "lovely", "good news", "yay", "haha", "hehe",
              "wonderful", "glad", "delight", "smile", "congrat"),
    "playful": ("tease", "kidding", "just kidding", "maybe i did", "whoops", "oopsie",
                "cheeky", "sneaky", "you caught me", "you do enjoy", "shall i start",
                "or are we", "improvising", "caught you"),
    "sarcastic": ("obviously", "oh yes", "sure thing", "totally", "as always",
                  "how convenient", "what a surprise", "famously", "suits you",
                  "before breakfast", "honestly, sir", "bold of you"),
    "sad": ("sorry", "unfortunately", "afraid", "sadly", "regret", "alas",
            "didn't make it", "lost"),
    "worried": ("careful", "warning", "risk", "might fail", "watch out", "danger",
                "concerned", "not sure this will", "could break"),
    "confident": ("guaranteed", "definitely", "no doubt", "will work", "certain",
                  "safety factor", "passes", "verified", "confirmed"),
    "calm": ("certainly", "as you wish", "right away", "understood", "of course, sir",
             "very good", "at once"),
    "curious": ("interesting", "let me check", "hmm", "i wonder", "shall we see",
                "what if", "why does"),
    "surprised": ("wait", "whoa", "really?", "no way", "unexpected", "actually,"),
    "frustrated": ("not again", "for the last time", "still broken", "keeps happening",
                   "third time", "refuses to", "ugh"),
    "jealous": ("must be nice", "wish i had", "some of us", "luckiest"),
    "ashamed": ("my mistake", "my fault", "i should have", "i was wrong", "apologies"),
    "confused": ("odd", "strange", "puzzling", "doesn't make sense", "baffling",
                 "unclear", "peculiar", "that's new"),
}

_PUNCT_RULES = [
    ("excited", re.compile(r"!{1,}")),
    ("surprised", re.compile(r"\bwait\b|\bwhoa\b|\?!", re.I)),
    ("happy", re.compile(r"(?<!\!)!(?:\s|$)")),   # single ! not already excited
    ("curious", re.compile(r"\?\s*$")),
]

_PRONOUN_SOFT = re.compile(r"\b(sir|please|thank you)\b", re.I)


def _auto_emotion_scored(text: str, default: str = "calm") -> tuple:
    """Scores a sentence; returns (label, best_score) so callers can demand
    a strong signal before switching voices mid-turn."""
    if not text:
        return default, 0.0
    low = text.lower()
    scores = {e: 0.0 for e in _EMOTION_LEXICON}
    for emotion, words in _EMOTION_LEXICON.items():
        for w in words:
            if w in low:
                scores[emotion] += 2.0
    for emotion, rx in _PUNCT_RULES:
        if rx.search(text):
            scores[emotion] += 1.5
    for tok in re.findall(r"\b[A-Z]{4,}\b", text):
        if tok.isupper() and len(tok) >= 4:
            scores["excited"] += 1.2
            break
    if _PRONOUN_SOFT.search(text):
        scores["calm"] += 0.6
        scores["confident"] += 0.3
    if "..." in text or "—" in text:
        scores["calm"] += 0.5
    best = max(scores, key=scores.get)
    return best, scores[best]


def auto_emotion(text: str, default: str = "calm") -> str:
    """Analyzes a sentence and returns the most fitting emotion."""
    best, score = _auto_emotion_scored(text, default)
    if score < 1.0:
        return default
    return best


# ─── Voice health + resilience ─────────────────────────────────────────────
# Jane stays the ONLY voice (single-voice doctrine: Voxtral or nothing spoken).
# What changes: the layer now fails FAST and stays informative instead of going
# silent or stacking latency during an outage.
#
#   - Circuit breaker: after TTS_MAX_CONSECUTIVE_FAILURES back-to-back failures
#     the network path is skipped for TTS_COOLDOWN_S seconds. The caller gets
#     b"" immediately and the already-delivered TEXT reply remains on screen,
#     so ADA is never mute — she is readable while her voice recovers.
#   - No double-latency outages: the last-resort apology retry is skipped while
#     the breaker is open (retrying a dead endpoint twice is what made outages
#     feel like hangs).
#   - ADA_TTS_MODE=text: skips synthesis entirely (silent + readable mode).
#   - tts_health() exposes the state for /status and logs.
TTS_MAX_CONSECUTIVE_FAILURES = int(os.environ.get("ADA_TTS_MAX_FAILURES", "3"))
TTS_COOLDOWN_S = float(os.environ.get("ADA_TTS_COOLDOWN_S", "60"))
TTS_MODE = os.environ.get("ADA_TTS_MODE", "voice").strip().lower()  # voice | text

_tts_consec_failures = 0
_tts_disabled_until = 0.0


def tts_health() -> dict:
    """Snapshot of the voice layer for /status and logs. No I/O."""
    now = time.time()
    return {
        "voice": "jane",
        "engine": "voxtral",
        "mode": TTS_MODE,
        "consecutive_failures": _tts_consec_failures,
        "breaker_open": now < _tts_disabled_until,
        "breaker_retry_in_s": round(max(0.0, _tts_disabled_until - now), 1),
    }


def _tts_breaker_open() -> bool:
    return time.time() < _tts_disabled_until


def _record_tts_success() -> None:
    global _tts_consec_failures, _tts_disabled_until
    _tts_consec_failures = 0
    _tts_disabled_until = 0.0


def _record_tts_failure(reason: str) -> None:
    global _tts_consec_failures, _tts_disabled_until
    _tts_consec_failures += 1
    print(f"[TTS] failure #{_tts_consec_failures}/{TTS_MAX_CONSECUTIVE_FAILURES} ({reason[:90]})")
    if _tts_consec_failures >= TTS_MAX_CONSECUTIVE_FAILURES:
        _tts_disabled_until = time.time() + TTS_COOLDOWN_S
        print(f"[TTS] breaker OPEN for {TTS_COOLDOWN_S:.0f}s — text replies continue, voice retries after cooldown.")


def _tts_muted() -> bool:
    """True when synthesis should be skipped without touching the network."""
    if TTS_MODE == "text":
        return True
    if _tts_breaker_open():
        return True
    return False


# ─── Public synthesize ───────────────────────────────────────────────────────

_LAST_RESORT_LINE = ("I'm having a small hiccup reaching my voice right now, "
                     "Sir. Give me a moment.")


async def synthesize(text: str, voice: str = "female", emotion=None,
                     allow_last_resort: bool = True):
    """Text -> (mp3_bytes, 'mp3'). Jane-only: normalized input -> Voxtral sync.
    On total failure, speaks a short apology IN THE SAME VOICE so ADA never
    goes silent and never sounds like someone else.

    Resilience (breaker): while the breaker is open this returns b"" immediately
    — the text reply is already on screen via transcription, so skipping the
    dead network path is what keeps her responsive instead of hanging."""
    if not text or not text.strip():
        return b"", "mp3"
    if _tts_muted():
        return b"", "mp3"
    spoken = prepare_spoken_text(text)
    if not spoken.strip():
        return b"", "mp3"
    try:
        async with _vox_lock:
            out = await asyncio.to_thread(_voxtral_sync, spoken, emotion), "mp3"
        _record_tts_success()
        return out
    except Exception as e:
        _record_tts_failure(f"Voxtral sync failed: {e}")
    if allow_last_resort and not _tts_breaker_open():
        try:
            async with _vox_lock:
                out = (await asyncio.to_thread(_voxtral_sync, _LAST_RESORT_LINE, None)), "mp3"
            _record_tts_success()
            return out
        except Exception as e:
            _record_tts_failure(f"last-resort line failed too: {e}")
    return b"", "mp3"


async def stream_reply(text: str, on_audio, voice: str = "female"):
    """Simple helper: speak a whole block of text through the streaming path.
    on_audio(bytes, format) receives audio fragments as they are produced."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    for sentence in sentences:
        try:
            emo = auto_emotion(sentence)
            spoke = False
            async for frag in _safe_stream(sentence, emo):
                await _call(on_audio, frag, "pcm")
                spoke = True
            if not spoke:
                audio, fmt = await synthesize(sentence, voice, emo)
                if audio:
                    await _call(on_audio, audio, fmt)
        except Exception as e:
            print(f"[TTS] Chunk failed ({str(e)[:100]}).")


async def _call(cb, audio, fmt):
    result = cb(audio, fmt)
    if asyncio.iscoroutine(result):
        await result


def audio_to_b64(audio_bytes: bytes) -> str:
    return base64.b64encode(audio_bytes).decode("utf-8")


async def _safe_stream(text: str, emotion=None):
    """Yields pcm fragments from Voxtral streaming; falls back to the sync mp3
    path (yielded as one fragment) if streaming produces nothing. While the
    breaker is open the network is skipped entirely so a dead endpoint can
    never stall speech delivery."""
    spoken = prepare_spoken_text(text)
    if not spoken.strip():
        return
    if _tts_muted():
        return
    try:
        got = False
        async for frag in voxtral_stream(spoken, emotion):
            got = True
            yield frag
        if got:
            _record_tts_success()
            return
        print("[TTS] stream produced no audio; falling back to sync.")
    except Exception as e:
        _record_tts_failure(f"stream failed: {e}")
    audio, _fmt = await synthesize(text, "female", emotion)
    if audio:
        yield audio


# ─── Realtime streamer with barge-in word tracking ──────────────────────────

class SpeechStreamer:
    """Streams speech utterance-by-utterance while the brain generates.

    Premium delivery:
      - Sentences are batched into small utterances (~2 sentences) so prosody
        flows naturally across sentence boundaries — but the FIRST utterance
        goes out alone so her first word lands fast.
      - Each utterance uses Voxtral STREAMING (pcm): audio fragments hit the
        frontend well before the sentence finishes generating.

    Barge-in intelligence:
      - Tracks every emitted utterance (text + timestamp + bytes received).
      - cancel() computes EXACTLY which word was being spoken when interrupted,
        storing it in .interrupted_at for the next turn's context.
    """

    MP3_BYTES_PER_SEC = 6000        # ~48kbps mp3 (sync fallback chunks)
    PCM_BYTES_PER_SEC = VOXTRAL_SAMPLE_RATE * 4   # float32 LE mono

    MAX_BATCH_CHARS = 240           # ~2 sentences; keeps prosody continuous
    MAX_BATCH_UTTERANCES = 2

    def __init__(self, on_audio, voice: str = "female", max_sentence_chars: int = 200):
        self.on_audio = on_audio
        self.voice = voice
        self.max_sentence_chars = max_sentence_chars
        self._buffer = ""
        self._queue = asyncio.Queue()
        self._worker = None
        self._cancelled = False
        self.emotion = None                  # explicit tag emotion (turn-level)
        self._explicit_emotion = False
        self.interrupted_at = None           # dict when barged-in mid-speech
        self._emitted = []                   # [{text, ts, nbytes, rate}]
        self._queued_texts = []
        self._gen = 0                        # generation guard for stale emits
        self._batch = []                     # pending sentences awaiting batching
        self._batch_len = 0
        self._first_of_turn = True
        self._utterance_idx = 0            # counts spoken utterances this turn
        self._pumps = set()                  # active _StreamPump tasks
        self._sentinel_seen = False
        # Playback anchoring (PACE-lite): frontend-reported per-utterance
        # played fractions {utt_idx: 0.0-1.0}. Preferred over the time-based
        # estimate below — emission time is NOT playback time (network +
        # frontend queue delay). Empty = voice path without reports (HTTP).
        self._played_frac: dict = {}

    def _emotion_for(self, text: str) -> str:
        """Per-utterance Mistral performance pick, chosen INSTANTLY before speaking.

        The brain's [emotion:X] tag is the turn default — and always the voice
        of the FIRST utterance, which the tag was written for. Any LATER
        utterance whose own words strongly signal a different feeling (a real
        lexicon hit, score >= 2.0) takes its own matching Jane performance
        instead of staying locked. Weak/no signal falls back to the tag.
        No tag at all -> pure per-sentence auto pick.
        """
        auto, score = _auto_emotion_scored(text)
        if self._explicit_emotion:
            if self._utterance_idx > 1 and score >= 2.0 and auto != self.emotion:
                return auto
            return self.emotion
        return auto

    def _start_pump(self, gen, text):
        """Opens the Voxtral stream for an utterance in the background so its
        first fragments are ready the moment the previous utterance finishes."""
        self._utterance_idx += 1
        pump = _StreamPump(self, gen, text)
        pump.entry["idx"] = self._utterance_idx
        pump.entry["words"] = len(text.split())
        self._pumps.add(pump)
        pump.start()
        return pump

    def _pop_next(self):
        """Non-blocking queue pop for prefetching; handles the sentinel."""
        if self._queue.empty():
            return None
        item = self._queue.get_nowait()
        if item[0] is _SENTINEL:
            self._sentinel_seen = True
            return None
        return item

    def reset(self):
        self._cancelled = False
        self._explicit_emotion = False
        self.emotion = None
        self.interrupted_at = None
        self._emitted.clear()
        self._played_frac.clear()
        self._queued_texts.clear()
        self._batch.clear()
        self._batch_len = 0
        self._first_of_turn = True
        self._utterance_idx = 0
        self._gen += 1

    def set_emotion(self, emotion):
        """Explicit [emotion:X] tag from the brain overrides auto-detection."""
        if emotion:
            self.emotion = emotion
            self._explicit_emotion = True

    def cancel(self):
        """Instant stop. Before wiping state, snapshot WHERE she was cut off."""
        self._flush_batch(cancel_only=True)
        if not self._cancelled:
            self._capture_interruption()
        self._cancelled = True
        self._buffer = ""
        self._queued_texts.clear()
        for pump in list(self._pumps):
            pump.abort()
        self._pumps.clear()
        try:
            while True:
                self._queue.get_nowait()
        except asyncio.QueueEmpty:
            pass

    def note_played(self, utt_idx, frac) -> None:
        """Frontend playback report: utterance idx was played to `frac`.

        Called from the `tts_played` socket event on barge-in. Clamped,
        monotonic per utterance (only the freshest report counts).
        """
        try:
            idx, f = int(utt_idx), max(0.0, min(1.0, float(frac)))
        except Exception:
            return
        prev = self._played_frac.get(idx, -1.0)
        if f > prev:
            self._played_frac[idx] = f

    def heard_text(self) -> str:
        """Words Sir actually heard, across all emitted utterances.

        Fully-played utterances contribute everything; the interrupted one
        contributes its played prefix (reported frac preferred, time-estimate
        fallback). This is the playback boundary — history must never keep
        more than this after a barge-in.
        """
        bits = []
        for e in self._emitted:
            words = (e.get("text") or "").split()
            if not words:
                continue
            idx = e.get("idx")
            frac = self._played_frac.get(idx, None) if idx is not None else None
            if frac is None:
                # No report: all but the last utterance count as fully played;
                # the last (interrupted) one falls back to the time estimate.
                if e is self._emitted[-1]:
                    elapsed = time.time() - e.get("ts", time.time())
                    est_dur = max(0.4, e.get("nbytes", 0) / max(1, e.get(
                        "rate", self.PCM_BYTES_PER_SEC)))
                    frac = min(0.98, max(0.0, elapsed / est_dur))
                else:
                    frac = 1.0
            n = int(frac * len(words))
            if n > 0:
                bits.append(" ".join(words[:n]))
        return " ".join(bits).strip()

    def _capture_interruption(self):
        last = self._emitted[-1] if self._emitted else None
        if last:
            elapsed = time.time() - last["ts"]
            est_dur = max(0.4, last["nbytes"] / max(1, last.get("rate", self.PCM_BYTES_PER_SEC)))
            frac = min(0.98, max(0.0, elapsed / est_dur))
            words = last["text"].split()
            wi = min(len(words) - 1, int(frac * len(words)))
            self.interrupted_at = {
                "was_speaking": True,
                "sentence": last["text"],
                "word_index": wi,
                "word": words[wi] if words else "",
                "words_left_in_sentence": len(words) - wi - 1,
                "chunks_pending": len(self._queued_texts),
            }
        elif self._queued_texts:
            self.interrupted_at = {
                "was_speaking": False,
                "sentence": self._queued_texts[0],
                "word_index": 0,
                "word": "",
                "words_left_in_sentence": 0,
                "chunks_pending": len(self._queued_texts),
            }
        if self.interrupted_at:
            try:
                self.interrupted_at["heard_text"] = self.heard_text()
                self.interrupted_at["anchored"] = bool(self._played_frac)
            except Exception:
                self.interrupted_at["heard_text"] = ""
                self.interrupted_at["anchored"] = False
            print(f"[TTS] [BARGE-IN] Cut off at word #{self.interrupted_at['word_index'] + 1} "
                  f"('{self.interrupted_at['word']}') of: \"{self.interrupted_at['sentence'][:70]}\"")

    def _extract_sentences(self) -> list:
        """Complete sentences; ALSO splits long first clauses so speech starts fast."""
        out = []
        buf = self._buffer
        while True:
            m = None
            for idx, ch in enumerate(buf):
                if ch in ".!?" and (idx + 1 >= len(buf) or buf[idx + 1].isspace()):
                    m = idx
                    break
            if m is None:
                # Early-start: speak the FIRST clause as soon as the buffer is
                # long enough — her first word lands before the sentence ends.
                if len(buf) > EARLY_START_CHARS:
                    cm = _CLAUSE_SPLIT.search(buf)
                    if cm and cm.start() > EARLY_START_MIN_CLAUSE:
                        clause = buf[:cm.start() + 1].strip()
                        buf = buf[cm.start() + 1:].lstrip()
                        if clause:
                            out.append(clause)
                        self._buffer = buf
                        continue
                break
            candidate = buf[: m + 1]
            if re.fullmatch(r"\d{1,4}[.!?]", candidate):
                buf = buf[m + 1:].lstrip()
                continue
            sentence = candidate.strip()
            buf = buf[m + 1:].lstrip()
            if sentence:
                out.append(sentence)
        self._buffer = buf
        return out

    def _flush_batch(self, force: bool = False, cancel_only: bool = False):
        """Queues batched sentences as one utterance when big enough."""
        if not self._batch:
            return
        enough = (self._batch_len >= self.MAX_BATCH_CHARS or
                  len(self._batch) >= self.MAX_BATCH_UTTERANCES)
        if cancel_only or force or enough:
            text = " ".join(self._batch).strip()
            self._batch.clear()
            self._batch_len = 0
            if text and not cancel_only:
                self._queued_texts.append(text)
                self._queue.put_nowait((self._gen, text))

    def start(self):
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run_worker())

    def push(self, delta: str):
        if not delta or self._cancelled:
            return
        self.start()
        self._buffer += delta
        for sentence in self._extract_sentences():
            # First sentence of a turn flies out ALONE (fastest possible
            # first word); later ones batch for flowing prosody.
            if self._first_of_turn:
                self._first_of_turn = False
                self._queued_texts.append(sentence)
                self._queue.put_nowait((self._gen, sentence))
            else:
                self._batch.append(sentence)
                self._batch_len += len(sentence)
                self._flush_batch()

    async def finish(self):
        self._flush_batch(force=True)
        text = self._buffer.strip()
        self._buffer = ""
        if not self._cancelled and text:
            self.start()
            self._queued_texts.append(text)
            self._queue.put_nowait((self._gen, text))
        if self._worker is not None and not self._worker.done():
            self._queue.put_nowait((_SENTINEL, None))
            try:
                await asyncio.wait_for(asyncio.shield(self._worker), timeout=180)
            except (asyncio.TimeoutError, Exception):
                pass

    async def _run_worker(self):
        """Pipelined delivery: while utterance N's fragments are being emitted,
        utterance N+1's Voxtral stream is already open server-side — the gapless
        queue never starves mid-reply."""
        while True:
            try:
                item = await self._queue.get()
            except asyncio.CancelledError:
                return
            if item[0] is _SENTINEL:
                return
            gen, text = item
            if self._cancelled or gen != self._gen:
                continue
            self._sentinel_seen = False

            pump = self._start_pump(gen, text)
            prefetch = None
            if PIPELINE_ENABLED:
                nxt = self._pop_next()
                if nxt:
                    prefetch = self._start_pump(*nxt)

            while pump is not None:
                spoke = await self._emit_pump(pump, gen)
                if self._cancelled or gen != self._gen:
                    return
                if not spoke:
                    await self._speak_last_resort(gen)
                pump = prefetch
                prefetch = None
                if pump is not None:
                    another = self._pop_next()
                    if another:
                        prefetch = self._start_pump(*another)

            if self._sentinel_seen:
                return

    async def _emit_pump(self, pump, gen) -> bool:
        """Drains a pump's fragments to the frontend. Returns True if any audio
        was delivered."""
        entry = pump.entry
        spoke = False
        while True:
            frag = await pump.q.get()
            if frag is pump.DONE:
                break
            if self._cancelled or gen != self._gen:
                return spoke
            entry["nbytes"] += len(frag)
            spoke = True
            try:
                # Utterance tagging (playback anchoring): the frontend reports
                # per-utt played fractions on barge-in, so history keeps only
                # words Sir actually heard. Extra kwargs — plain on_audio
                # callbacks that take (bytes, format) keep working.
                result = self.on_audio(frag, "pcm", utt=entry.get("idx"),
                                       words=entry.get("words", 0))
                if asyncio.iscoroutine(result):
                    await result
            except TypeError:
                # legacy on_audio(bytes, format) — retry without tags
                try:
                    result = self.on_audio(frag, "pcm")
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    print(f"[TTS] on_audio failed ({str(e)[:90]}).")
            except Exception as e:
                print(f"[TTS] on_audio failed ({str(e)[:90]}).")
        return spoke

    async def _speak_last_resort(self, gen):
        """Total synthesis failure: apologize in her own voice — never mute,
        never a different assistant."""
        try:
            audio, fmt = await synthesize(_LAST_RESORT_LINE, self.voice, None,
                                          allow_last_resort=False)
            if audio and not self._cancelled and gen == self._gen:
                self._emitted.append({"text": _LAST_RESORT_LINE, "ts": time.time(),
                                      "nbytes": len(audio), "rate": self.MP3_BYTES_PER_SEC})
                result = self.on_audio(audio, fmt)
                if asyncio.iscoroutine(result):
                    await result
        except Exception as e:
            print(f"[TTS] last-resort failed ({str(e)[:90]}).")


class _StreamPump:
    """Consumes one utterance's Voxtral stream into an internal queue so it can
    run concurrently with playback of the previous utterance (pipelining)."""

    DONE = object()

    def __init__(self, streamer, gen, text):
        self.s = streamer
        self.gen = gen
        self.text = text
        self.emotion = streamer._emotion_for(text)
        self.entry = {"text": text, "ts": time.time(), "nbytes": 0,
                      "rate": SpeechStreamer.PCM_BYTES_PER_SEC}
        self.q = asyncio.Queue()
        self.task = None

    def start(self):
        self.s._emitted.append(self.entry)
        self.task = asyncio.create_task(self._run())

    async def _run(self):
        try:
            async for frag in _safe_stream(self.text, self.emotion):
                await self.q.put(frag)
        except Exception as e:
            print(f"[TTS] pump failed ({str(e)[:90]}).")
        finally:
            await self.q.put(self.DONE)
            self.s._pumps.discard(self)

    def abort(self):
        if self.task and not self.task.done():
            self.task.cancel()
