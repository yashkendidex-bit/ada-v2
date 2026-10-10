"""
TTS latency benchmark — BEFORE vs AFTER, measured in one run.

Simulates Ada's real conversation path through SpeechStreamer:
  short ack  -> measures time-to-FIRST-fragment (opener latency)
  long reply -> measures first-fragment AND the worst gap BETWEEN utterances
                (pipeline benefit)

Modes:
  legacy   : 90-char early-start, pipeline OFF (the old behavior)
  optimized: 50-char early-start, pipelined prefetch (current defaults)

Usage:  .venv/bin/python backend/_tts_speedtest.py [--rounds 3]
"""
import argparse
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(override=True)

import tts_agent  # noqa: E402

SHORT = ("[emotion:warm] On it, Sir. Give me a moment.")
# First sentence is deliberately long (no period for ~100 chars) so the
# early-start threshold is the deciding factor for when her voice begins.
LONG = ("The gearbox analysis is complete, and the news is good across every single "
        "metric we tracked overnight. Torque capacity is up twelve percent while the "
        "assembly lost two hundred grams. The bearing seats passed every tolerance "
        "check with room to spare. There is one thing you should review though. The "
        "third gear shows minor stress concentration near the root. I would widen the "
        "fillet by half a millimeter before we print it.")

LLM_CHARS_PER_SEC = 60   # realistic brain streaming cadence


async def push_like_llm(streamer, text):
    """Feeds text in small deltas at brain speed so early-start thresholds
    are exercised the way production exercises them."""
    step = 8
    for i in range(0, len(text), step):
        streamer.push(text[i:i + step])
        await asyncio.sleep(step / LLM_CHARS_PER_SEC)
    streamer.push(" ")


class Recorder:
    def __init__(self):
        self.t0 = time.time()      # set at test start, before push
        self.first = None
        self.last_ts = None
        self.gaps = []
        self.fragments = 0

    async def on_audio(self, audio, fmt):
        now = time.time()
        self.fragments += 1
        if self.first is None:
            self.first = now
        elif self.last_ts is not None:
            gap = now - self.last_ts
            if gap > 0.25:               # only real inter-utterance stalls count
                self.gaps.append(gap)
        self.last_ts = now


async def run_mode(mode: str, rounds: int):
    tts_agent.EARLY_START_CHARS = 90 if mode == "legacy" else 50
    tts_agent.PIPELINE_ENABLED = mode == "optimized"

    results = {"short_ttfa": [], "long_ttfa": [], "long_max_gap": []}
    for r in range(rounds):
        # short ack
        rec = Recorder()
        s = tts_agent.SpeechStreamer(on_audio=rec.on_audio)
        s.reset()
        s.push(SHORT)
        await s.finish()
        if s._worker:
            try:
                await asyncio.wait_for(s._worker, timeout=90)
            except Exception:
                pass
        if rec.first:
            results["short_ttfa"].append(rec.first - rec.t0)

        await asyncio.sleep(0.4)

        # long reply, fed at LLM cadence (exercises early-start threshold)
        rec = Recorder()
        s = tts_agent.SpeechStreamer(on_audio=rec.on_audio)
        s.reset()
        await push_like_llm(s, LONG)
        await s.finish()
        if s._worker:
            try:
                await asyncio.wait_for(s._worker, timeout=120)
            except Exception:
                pass
        if rec.first:
            results["long_ttfa"].append(rec.first - rec.t0)
        if rec.gaps:
            results["long_max_gap"].append(max(rec.gaps))

        await asyncio.sleep(0.4)
    return results


def med(xs):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    print(f"[speedtest] rounds={args.rounds} model={tts_agent.VOXTRAL_MODEL}")
    print(f"[speedtest] warm-up request (TLS + auth)...")
    t0 = time.time()
    async for _ in tts_agent.voxtral_stream("Hello.", None):
        break
    print(f"[speedtest] warm complete in {time.time()-t0:.2f}s\n")

    rows = {}
    for mode in ("legacy", "optimized"):
        print(f"[speedtest] running mode: {mode} ...")
        rows[mode] = await run_mode(mode, args.rounds)

    lo, opt = rows["legacy"], rows["optimized"]
    print("\n================ RESULTS (median) ================")
    print(f"{'metric':28} {'legacy':>10} {'optimized':>10} {'delta':>10}")
    for key, label in (("short_ttfa", "short reply: first audio"),
                       ("long_ttfa", "long reply: first audio"),
                       ("long_max_gap", "long reply: worst gap")):
        a, b = med(lo[key]), med(opt[key])
        delta = (b - a) / a * 100 if a == a and b == b and a else float("nan")
        print(f"{label:28} {a*1000:9.0f}ms {b*1000:9.0f}ms {delta:+9.1f}%")
    print("==================================================")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
