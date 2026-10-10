"""Probe: does the Mistral embedding API work, and what does it score?

    python memory/_probe_mistral.py

Reads MISTRAL_API_KEY at runtime. Prints no credential material.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from memory.embeddings import _read_api_key  # noqa: E402


def main() -> int:
    key = _read_api_key()
    print(f"key found: {bool(key)} ({len(key)} chars)")
    if not key:
        return 1

    import httpx
    print("calling https://api.mistral.ai/v1/embeddings ...")
    try:
        r = httpx.post(
            "https://api.mistral.ai/v1/embeddings",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": "mistral-embed",
                  "input": ["the bracket cracked at the motor mount",
                            "what broke on the rig?"]},
            timeout=40.0,
        )
    except Exception as e:
        print(f"REQUEST FAILED: {type(e).__name__}: {e}")
        return 2

    print(f"HTTP {r.status_code}")
    if r.status_code != 200:
        print("body:", r.text[:300])
        return 3

    data = r.json().get("data") or []
    embs = [np.asarray(d["embedding"], dtype=np.float32)
            for d in sorted(data, key=lambda x: x.get("index", 0))]
    print(f"vectors: {len(embs)}   dimension: {embs[0].size}")
    a, b = embs
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    print(f"cosine(related pair): {cos:.4f}")
    print("usage:", r.json().get("usage"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
