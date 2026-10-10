"""
Live TTS probe — verifies the full Voxtral pipeline end-to-end.

Synthesizes 3 expressive lines (the exact kind of humanized input the new
pipeline produces) and writes audio files to a temp dir. Prints which voice
parameter mode worked and per-line timing so latency is visible.

Usage:  python backend/_tts_probe.py
Requires: MISTRAL_API_KEY in .env, network access.
"""
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(override=True)

import tts_agent  # noqa: E402

LINES = [
    ("calm", "[emotion:warm] Good evening, Sir. All systems are online, and I am fully at your service."),
    ("excited", "Oh, that's wonderful! The new design passed every single test at ninety-eight percent efficiency!"),
    ("playful", "A Ferrari gearbox before breakfast? Honestly, Sir, ambition suits you."),
]

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "tts_probe")


async def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # Voxtral is the only engine — there is no engine selector to report.
    print(f"[probe] engine=voxtral model={tts_agent.VOXTRAL_MODEL} "
          f"voice={tts_agent.VOXTRAL_FEMALE_VOICE}")
    if not os.getenv("MISTRAL_API_KEY"):
        print("[probe] MISTRAL_API_KEY missing - cannot run live probe.")
        return 1

    for i, (emo, line) in enumerate(LINES):
        t0 = time.time()
        first = {"at": None}
        chunks = 0
        total = 0

        async def on_frag(audio, fmt):
            nonlocal chunks, total
            if first["at"] is None:
                first["at"] = time.time() - t0
                print(f"       first fragment ({fmt}) after {first['at']*1000:.0f} ms")
            chunks += 1
            total += len(audio)

        try:
            async for frag in tts_agent._safe_stream(line, emo):
                await on_frag(frag, "pcm")
            dur = time.time() - t0
            print(f"[{i+1}/3] {emo}: streamed {chunks} fragments, {total} bytes "
                  f"in {dur:.2f}s")
        except Exception as e:
            print(f"[{i+1}/3] streaming failed: {e}")
            continue

        # Also produce an mp3 artifact for listening via the sync path.
        audio, fmt = await tts_agent.synthesize(line, "female", emo)
        path = os.path.join(OUT_DIR, f"line_{i+1}_{emo}.mp3")
        with open(path, "wb") as f:
            f.write(audio)
        print(f"       mp3 saved -> {path} ({len(audio)} bytes)")

    print(f"\n[probe] voice param mode in use: {tts_agent.VOXTRAL_PARAM_KEY()}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
