"""
Listening test — hear ADA's new voice right now.

Every line goes through the FULL production path exactly once:
humanized input -> speech_normalizer -> Voxtral 'Jane' -> mp3.
No fallback engines exist anymore, so what you hear is what Ada says.

Usage:  python backend/_tts_listen.py [--no-play]
"""
import asyncio
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(override=True)

import tts_agent  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "tts_listen")

# Expressive, human-written input — exactly what the new pipeline feeds Jane.
# Each line's emotion maps to one of Jane's 9 vocal performances (the same
# ones behind Mistral Studio's "Select an emotion" buttons).
LINES = [
    ("boot_warm", "[emotion:warm] Good evening, Sir. All systems are online, and I have been looking forward to this."),
    ("excited", "Oh, that's wonderful! The gearbox passed every single test at ninety-eight point five percent efficiency!"),
    ("sarcastic", "A Ferrari gearbox before breakfast? Honestly, Sir, ambition suits you."),
    ("worried", "Careful with this one. The stress simulation shows a safety factor of one point three. I would not fly it yet."),
    ("sad", "I'm afraid the print did not succeed this time. The housing warped near the bearing seat... we'll get it next round."),
    ("curious", "Now that is interesting. Why would the resonance spike at exactly that frequency, I wonder..."),
    ("confident", "Consider it done, Sir. The design is verified, the tolerances are clean, and it will work."),
    ("frustrated", "The compiler rejected it a third time. This build system genuinely refuses to cooperate tonight."),
    ("playful", "You do enjoy interrupting me at the good part, Sir. Shall I start over, or are we improvising?"),
]


async def main():
    play = "--no-play" not in sys.argv
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[listen] model={tts_agent.VOXTRAL_MODEL} voice={tts_agent.VOXTRAL_FEMALE_VOICE} "
          f"(Jane-only pipeline — no fallback engines)")

    files = []
    for i, (name, raw_line) in enumerate(LINES):
        # Production parity: strip emotion tags like ada.py does before speaking,
        # then synthesize ONCE through the same entry point ADA uses.
        line = tts_agent.prepare_spoken_text(raw_line)
        emo = tts_agent.auto_emotion(line)
        voice = tts_agent.resolve_jane_voice(emo)
        t0 = time.time()
        audio, fmt = await tts_agent.synthesize(line, "female", emo)
        dt = time.time() - t0
        if not audio:
            print(f"[{i+1}/{len(LINES)}] {name}: FAILED (no audio) — check logs above")
            continue
        path = os.path.join(OUT_DIR, f"{i+1}_{name}.mp3")
        with open(path, "wb") as f:
            f.write(audio)
        files.append((name, path))
        print(f"[{i+1}/{len(LINES)}] {name} ({emo} -> {voice}, {len(audio)} bytes, {dt:.1f}s)")
        await asyncio.sleep(0.4)   # be gentle with the free tier

    if not files:
        print("[listen] nothing synthesized.")
        return 1

    if play and sys.platform == "darwin":
        print("\n[listen] playing aloud...")
        for name, p in files:
            print(f"   >> {name}")
            subprocess.run(["afplay", p], check=False)
        print("[listen] done.")
    elif play:
        print("[listen] auto-play only wired for macOS; open files manually:")
        for _, p in files:
            print("   ", p)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
