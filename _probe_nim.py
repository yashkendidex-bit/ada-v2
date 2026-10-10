"""Two candidate fixes for the NVIDIA gateway dropping long CAD requests.

Established facts:
  * 27-token prompt  -> deepseek-v4-flash-0731 answers in 172s
  * 6,419-token CAD prompt -> connection killed at exactly ~271s, whatever max_tokens is
  * kimi-k3 -> same, at ~270s

A consistent ~270s kill is a server-side timeout, not a client one. Two ways round it:

  A. STREAM the reply. A silent connection gets reaped; one that is emitting tokens
     should not. This is the most likely fix.
  B. SHRINK the prompt. `brain.system_prompt(example=False)` drops the worked example,
     which is most of the 25,527 characters.

Test both, measure, report honestly.

    cd backend && ../.venv/bin/python ../_probe_nim.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODELS = ["deepseek-ai/deepseek-v4-flash-0731", "moonshotai/kimi-k3"]


def key() -> str:
    for line in open(Path(__file__).resolve().parent / ".env"):
        line = line.strip()
        if line.startswith("NVIDIA_API_KEY="):
            return line.split("=", 1)[1].strip()
    return ""


def build_messages(example: bool):
    import cad_v3
    cad_v3.bootstrap()
    from cad_v3 import brain
    sp = brain.system_prompt(example=example)
    return [
        {"role": "system", "content": sp},
        {"role": "user", "content":
            "Design this as an assembly spec:\n\na Ferrari V8 engine\n\n"
            "Use as many parts as the machine really needs. Every part must touch "
            "something. Return only JSON."},
    ]


async def stream_probe(model: str, messages: list, label: str):
    import httpx
    t = time.time()
    chunks, thinking, first = 0, 0, None
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0)) as c:
            async with c.stream("POST", URL,
                                headers={"Authorization": f"Bearer {key()}",
                                         "Content-Type": "application/json"},
                                json={"model": model, "messages": messages,
                                      "max_tokens": 32000, "temperature": 0.2,
                                      "stream": True}) as r:
                if r.status_code != 200:
                    body = (await r.aread())[:150]
                    print(f"  FAIL {label} {model[:34]:34s} HTTP {r.status_code} {body}")
                    return
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        obj = json.loads(payload)
                    except Exception:
                        continue
                    delta = (obj.get("choices") or [{}])[0].get("delta") or {}
                    if delta.get("content"):
                        chunks += 1
                        if first is None:
                            first = time.time() - t
                    if delta.get("reasoning_content"):
                        thinking += 1
        dt = time.time() - t
        print(f"  OK   {label} {model[:34]:34s} {dt:6.1f}s "
              f"content_chunks={chunks} thinking_chunks={thinking} "
              f"first_token={first and round(first, 1)}s")
    except Exception as e:
        print(f"  ERR  {label} {model[:34]:34s} {time.time() - t:6.1f}s "
              f"{type(e).__name__}: {str(e)[:70]}")


async def main() -> int:
    full = build_messages(example=True)
    small = build_messages(example=False)
    print(f"prompt sizes: full={sum(len(m['content']) for m in full):,} chars, "
          f"no-example={sum(len(m['content']) for m in small):,} chars")
    print()
    print("A. STREAMING, full prompt")
    for m in MODELS:
        await stream_probe(m, full, "stream-full")
    print()
    print("B. STREAMING, no-example prompt")
    for m in MODELS:
        await stream_probe(m, small, "stream-small")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
