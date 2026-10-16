"""Voice-latency shootout: preset voice vs cloned voice on Mistral Voxtral.

Compares, for the SAME text:
  A. preset voice_id      (what ADA uses today, e.g. gb_jane_sarcasm)
  B. zero-shot clone      (ref_audio base64 sent with every request)
  C. saved custom voice   (create once from the recording -> voice_id -> delete)

Measures time-to-first-audio (streaming SSE) and total request time.

Usage:
    MISTRAL_API_KEY=... python3 tests/test_tts_voice_latency.py <recording> ["text to speak"]

The recording: 3-25 seconds, single speaker, clean, no background noise.
wav or mp3. Only clone a voice you have the right to use.

Needs: httpx (in requirements.txt).
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time

import httpx

API = "https://api.mistral.ai/v1/audio/speech"
VOICES_API = "https://api.mistral.ai/v1/audio/voices"
PRESET_VOICE = os.environ.get("ADA_LATENCY_PRESET_VOICE", "gb_jane_sarcasm")
TEXT = ("The market just did something interesting, and I think you should "
        "see this before everyone else catches on.")


def _headers(key: str) -> dict:
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _measure_stream(key: str, payload: dict, tag: str) -> dict:
    """POST with stream=true; return ttfa_s and total_s."""
    t0 = time.perf_counter()
    ttfa = None
    chunks = 0
    with httpx.stream("POST", API, headers=_headers(key),
                      json={**payload, "stream": True, "response_format": "mp3"},
                      timeout=90.0) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            try:
                evt = json.loads(line[5:].strip())
            except Exception:
                continue
            data = evt.get("data", evt)
            if data.get("audio_data"):
                chunks += 1
                if ttfa is None:
                    ttfa = time.perf_counter() - t0
            if evt.get("type") == "speech.audio.done" or \
               data.get("type") == "speech.audio.done":
                break
    total = time.perf_counter() - t0
    return {"tag": tag, "ttfa_s": round(ttfa, 3) if ttfa else None,
            "total_s": round(total, 3), "chunks": chunks}


def main() -> int:
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        print("MISTRAL_API_KEY not set", file=sys.stderr)
        return 2
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    rec_path = sys.argv[1]
    text = sys.argv[2] if len(sys.argv) > 2 else TEXT
    raw = open(rec_path, "rb").read()
    ref_b64 = base64.b64encode(raw).decode()
    print(f"recording: {rec_path} ({len(raw)/1024:.1f} KiB)  "
          f"text: {len(text)} chars\n")

    results = []
    # A. preset voice
    results.append(_measure_stream(
        key, {"input": text, "voice_id": PRESET_VOICE}, "A preset voice_id"))
    # B. zero-shot clone (ref audio sent every request)
    results.append(_measure_stream(
        key, {"input": text, "ref_audio": ref_b64}, "B zero-shot ref_audio"))
    # C. saved custom voice: create -> use -> delete
    try:
        with httpx.Client(timeout=60.0) as c:
            r = c.post(VOICES_API, headers=_headers(key), json={
                "name": "latency-test-voice",
                "sample_audio": ref_b64,
                "sample_filename": os.path.basename(rec_path),
            })
            r.raise_for_status()
            vid = r.json().get("id")
        if vid:
            results.append(_measure_stream(
                key, {"input": text, "voice_id": vid}, "C saved voice_id"))
            with httpx.Client(timeout=30.0) as c:
                c.delete(f"{VOICES_API}/{vid}", headers=_headers(key))
            print("(saved test voice deleted)\n")
    except Exception as e:
        print(f"C skipped: {type(e).__name__}: {e}\n")

    print(f"{'mode':<22}{'TTFA':>10}{'total':>10}{'chunks':>8}")
    for r in results:
        print(f"{r['tag']:<22}{r['ttfa_s'] or '-':>10}{r['total_s']:>10}{r['chunks']:>8}")
    a = next((r for r in results if r["tag"].startswith("A")), None)
    b = next((r for r in results if r["tag"].startswith("B")), None)
    c = next((r for r in results if r["tag"].startswith("C")), None)
    print()
    if a and b and a["ttfa_s"] and b["ttfa_s"]:
        d = b["ttfa_s"] - a["ttfa_s"]
        print(f"Zero-shot clone vs preset TTFA: {'+' if d >= 0 else ''}{d:.3f}s "
              f"({'slower' if d > 0 else 'faster'})")
    if a and c and a["ttfa_s"] and c["ttfa_s"]:
        d = c["ttfa_s"] - a["ttfa_s"]
        print(f"Saved custom voice vs preset TTFA: {'+' if d >= 0 else ''}{d:.3f}s "
              f"({'slower' if d > 0 else 'faster'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
