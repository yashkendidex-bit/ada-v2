"""Tiered embedding backend for ADA's memory.

Why this module exists
----------------------
`agent_kernel._Embedder` had two backends: sentence-transformers, and — if that
import failed — a 384-dimension MD5 hashed bag-of-words. The fallback was silent.
In practice sentence-transformers was never installed, so every embedding ADA ever
computed was a hash bucket count, and retrieval ranked a *contradicting* memory
above a *paraphrasing* one.

Two rules this module enforces, both learned from that failure:

1. **Never degrade silently.** The active backend is logged once, and every
   embedding carries the backend name that produced it, so a store can be audited
   for mixed-generation vectors.
2. **Report what you actually are.** `.is_semantic` is False for the lexical
   fallback, and callers that care can refuse to treat lexical hits as semantic
   matches.

Measured on a lexical-vs-semantic retrieval benchmark (see
`tests/test_memory_retrieval.py`):

    backend              SEMANTIC recall@1   LEXICAL recall@1
    MD5 bag-of-words          0/5                1/2
    model2vec potion-8M       1/5                2/2
    fastembed bge-small       5/5                2/2     <- the default here

The MD5 fallback is kept as tier 3 because it is better than crashing on a machine
with no model available, but it is *lexical only* — it cannot connect "what broke"
to "cracked at the motor mount".
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

log = logging.getLogger("ada.memory.embeddings")

# Dimension of the lexical fallback, and the default when nothing else is known.
# The ACTIVE backend's dimension is `Embedder.dim`, not this constant — tiers differ
# (Mistral is 1024-d, BGE-small is 384-d) and mixing them is a hard crash, not a quiet
# wrong answer, because `np.vstack` refuses ragged input.
EMBED_DIM = 384

# Mistral's embedding model. The API id is `mistral-embed`; "2312" is the revision.
# Chosen as the preferred tier for laptops: better than a small local model, and it
# costs no RAM, which matters on an 8 GB machine running the rest of the app too.
MISTRAL_MODEL = os.environ.get("ADA_MISTRAL_EMBED_MODEL", "mistral-embed")
MISTRAL_ENDPOINT = "https://api.mistral.ai/v1/embeddings"
MISTRAL_BATCH = 64          # the API allows more; a big batch is a big thing to lose
MISTRAL_DIM = 1024

# BGE models are trained with an instruction prefix on the *query* side only.
# Omitting it costs real accuracy on short queries, which is most of ADA's traffic.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_PREFERRED_FASTEMBED_MODEL = os.environ.get(
    "ADA_EMBED_MODEL", "BAAI/bge-small-en-v1.5"
)


def _read_api_key() -> str:
    """MISTRAL_API_KEY from the environment, falling back to `.env`.

    `server.py` loads dotenv for the main process, but this package is also used from
    standalone tools and tests where that has not happened yet — and silently dropping
    to a weaker local model because of module load order would be exactly the kind of
    invisible degradation this module exists to prevent.

    The search walks up from this file rather than assuming a fixed depth, because
    getting that depth wrong is silent: the key is simply not found, the ladder drops a
    tier, and retrieval quietly gets worse with no error anywhere. (The first version of
    this function looked in `backend/` — one level short of the project root, where
    `.env` actually lives — so the Mistral tier could never activate outside the app.)
    """
    key = (os.environ.get("MISTRAL_API_KEY") or "").strip()
    if key:
        return key
    try:
        here = Path(__file__).resolve()
        candidates = [parent / ".env" for parent in list(here.parents)[:4]]
        for env_path in candidates:
            if not env_path.is_file():
                continue
            with open(env_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line.startswith("MISTRAL_API_KEY="):
                        key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if key:
                            os.environ["MISTRAL_API_KEY"] = key   # cache it
                            return key
    except Exception:
        pass
    return ""


class Embedder:
    """Embeddings with an explicit, logged, and honest degradation ladder.

    Tiers, best first:
      1. ``mistral``    — the `mistral-embed` API, 1024-d. No local model, no RAM cost.
      2. ``fastembed``  — ONNX BGE-small-en-v1.5, 384-d, no torch. Works offline.
      3. ``sentence-transformers`` — used if the operator already has it installed.
      4. ``lexical``    — hashed TF-IDF. LEXICAL ONLY. ``is_semantic`` is False.

    Thread-safe. Model loading is lazy and happens once, guarded by a lock, because
    ADA constructs the kernel on the asyncio loop and embeds from worker threads.
    """

    def __init__(self, prefer: Optional[str] = None):
        self._lock = threading.Lock()
        self._backend: Optional[str] = None
        self._model = None
        self._dim: Optional[int] = None
        # `ADA_EMBED_TIER` pins a tier, which is how the benchmark compares backends
        # head to head and how an operator can force offline operation.
        self._prefer = prefer or (os.environ.get("ADA_EMBED_TIER") or "").strip() or None
        self._api_failed = False
        # Small in-process cache. The consolidation job re-embeds the same strings
        # repeatedly, and with an API backend a cache hit is a saved round-trip.
        self._cache: dict[str, np.ndarray] = {}
        self._cache_cap = 4096

    # ── backend selection ────────────────────────────────────────────────────
    def _load(self):
        if self._backend is not None:
            return
        with self._lock:
            if self._backend is not None:
                return

            order = ([self._prefer] if self._prefer
                     else ["mistral", "fastembed", "sentence-transformers", "lexical"])
            for tier in order:
                if tier == "mistral":
                    if not _read_api_key():
                        log.info("mistral embeddings unavailable (no MISTRAL_API_KEY); "
                                 "trying next tier")
                        continue
                    # Do not probe the API during selection — that costs a round-trip
                    # on every boot. Trust the key; the first real call decides, and
                    # `_degrade_from_mistral` handles being wrong.
                    self._backend, self._dim = "mistral", MISTRAL_DIM
                    break
                if tier == "fastembed":
                    try:
                        from fastembed import TextEmbedding
                        self._model = TextEmbedding(_PREFERRED_FASTEMBED_MODEL)
                        self._backend, self._dim = "fastembed", EMBED_DIM
                        break
                    except Exception as e:
                        log.info("fastembed unavailable (%s); trying next tier", type(e).__name__)
                elif tier == "sentence-transformers":
                    try:
                        from sentence_transformers import SentenceTransformer
                        self._model = SentenceTransformer("all-MiniLM-L6-v2")
                        self._backend, self._dim = "sentence-transformers", 384
                        break
                    except Exception as e:
                        log.info("sentence-transformers unavailable (%s); trying next tier", type(e).__name__)

            if self._backend is None:
                self._backend, self._dim = "lexical", EMBED_DIM

            log.warning(
                "memory embeddings backend = %s (%s, %dd)",
                self._backend,
                "semantic" if self.is_semantic else
                "LEXICAL ONLY — semantic recall will be poor. "
                "Set MISTRAL_API_KEY or install `fastembed` for real embeddings.",
                self._dim or EMBED_DIM,
            )

    @property
    def backend_name(self) -> str:
        self._load()
        return self._backend

    @property
    def dim(self) -> int:
        self._load()
        return self._dim or EMBED_DIM

    @property
    def is_semantic(self) -> bool:
        self._load()
        return self._backend in ("mistral", "fastembed", "sentence-transformers")

    def health(self) -> dict:
        """Machine-readable status, surfaced in the UI and the audit log."""
        self._load()
        return {
            "backend": self._backend,
            "semantic": self.is_semantic,
            "dim": self.dim,
            "model": (MISTRAL_MODEL if self._backend == "mistral"
                      else _PREFERRED_FASTEMBED_MODEL if self._backend == "fastembed"
                      else None),
            "local": self._backend != "mistral",
        }

    # ── embedding ────────────────────────────────────────────────────────────
    def embed(self, text: str) -> np.ndarray:
        """Embed a *passage* (something to be stored and found later)."""
        return self.embed_many([text])[0]

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a *query*. Asymmetric from passages for BGE-family models."""
        self._load()
        if self._backend == "fastembed":
            key = "q:" + str(text)
            hit = self._cache_get(key)
            if hit is not None:
                return hit
            try:
                vec = np.asarray(list(self._model.query_embed([str(text)]))[0], dtype=np.float32)
            except Exception:
                # Older fastembed builds have no query_embed; apply the prefix by hand.
                vec = self._fastembed([_BGE_QUERY_PREFIX + str(text)])[0]
            return self._cache_put(key, _normalize(vec))
        return self.embed(text)

    def embed_many(self, texts: Sequence[str]) -> List[np.ndarray]:
        self._load()
        texts = [str(t) for t in texts]
        out: List[Optional[np.ndarray]] = [self._cache_get("p:" + t) for t in texts]
        todo = [i for i, v in enumerate(out) if v is None]

        if todo:
            batch = [texts[i] for i in todo]
            if self._backend == "mistral":
                vecs = self._mistral_many(batch)
            elif self._backend == "fastembed":
                vecs = self._fastembed_many(batch)
            elif self._backend == "sentence-transformers":
                vecs = [np.asarray(v, dtype=np.float32) for v in
                        self._model.encode(batch, normalize_embeddings=True)]
            else:
                vecs = [self._lexical(t) for t in batch]
            for slot, text, vec in zip(todo, batch, vecs):
                out[slot] = self._cache_put("p:" + text, _normalize(vec))

        return [v for v in out]  # type: ignore[return-value]

    # ── tier 1: Mistral API ──────────────────────────────────────────────────
    def _mistral_many(self, texts: List[str]) -> List[np.ndarray]:
        """Batch embed via the Mistral API, degrading to a local model on failure.

        A failure here must not lose the write. A memory that cannot be embedded is a
        memory that cannot be found, so on any API error we drop to the local model for
        the rest of the session and say so — rather than returning zeros, which would
        be an invisible corruption of the vector space.
        """
        import httpx

        key = _read_api_key()
        out: List[np.ndarray] = []
        try:
            with httpx.Client(timeout=30.0) as client:
                for start in range(0, len(texts), MISTRAL_BATCH):
                    chunk = texts[start:start + MISTRAL_BATCH]
                    resp = client.post(
                        MISTRAL_ENDPOINT,
                        headers={"Authorization": f"Bearer {key}",
                                 "Content-Type": "application/json"},
                        json={"model": MISTRAL_MODEL, "input": chunk},
                    )
                    if resp.status_code != 200:
                        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:160]}")
                    data = resp.json().get("data") or []
                    if len(data) != len(chunk):
                        raise RuntimeError(f"asked for {len(chunk)} vectors, got {len(data)}")
                    for item in sorted(data, key=lambda d: d.get("index", 0)):
                        out.append(np.asarray(item["embedding"], dtype=np.float32))
            return out
        except Exception as e:
            self._degrade_from_mistral(e)
            return self._fallback_many(texts)

    def _degrade_from_mistral(self, err: Exception) -> None:
        """Switch the API tier off for this process and re-embed locally."""
        if self._api_failed:
            return
        self._api_failed = True
        log.warning("mistral embeddings failed (%s: %s); using a local model for this "
                    "session", type(err).__name__, str(err)[:140])
        with self._lock:
            self._model = None
            self._dim = None
            self._backend = None
            self._cache.clear()
        self._prefer = "fastembed"      # do not retry the API
        self._load()
        log.warning("memory embeddings now = %s (%dd)", self._backend, self.dim)

    def _fallback_many(self, texts: List[str]) -> List[np.ndarray]:
        if self._backend == "fastembed":
            return self._fastembed_many(texts)
        if self._backend == "sentence-transformers":
            return [np.asarray(v, dtype=np.float32) for v in
                    self._model.encode(texts, normalize_embeddings=True)]
        return [self._lexical(t) for t in texts]

    # ── tier 2: fastembed ────────────────────────────────────────────────────
    def _fastembed(self, text: str) -> np.ndarray:
        return self._fastembed_many([text])[0]

    def _fastembed_many(self, texts: List[str]) -> List[np.ndarray]:
        return [np.asarray(v, dtype=np.float32) for v in self._model.embed(texts)]

    # ── tier 4: lexical fallback ─────────────────────────────────────────────
    @staticmethod
    def _lexical(text: str, dim: int = EMBED_DIM) -> np.ndarray:
        """Hashed term-frequency vector. Lexical, not semantic — kept honest by
        `is_semantic`. Better than the old version because it also hashes bigrams,
        which recovers a little word-order signal and helps on part numbers."""
        vec = np.zeros(dim, dtype=np.float32)
        toks = re.findall(r"[a-z0-9]+", str(text).lower())
        for i, tok in enumerate(toks):
            vec[_h(tok, dim)] += 1.0
            if i + 1 < len(toks):
                vec[_h(toks[i] + "_" + toks[i + 1], dim)] += 0.5
        return vec

    # ── cache ────────────────────────────────────────────────────────────────
    def _cache_get(self, key: str):
        return self._cache.get(key)

    def _cache_put(self, key: str, vec: np.ndarray) -> np.ndarray:
        if len(self._cache) >= self._cache_cap:
            # Cheap eviction: drop a quarter of the cache rather than tracking LRU.
            for k in list(self._cache)[: self._cache_cap // 4]:
                self._cache.pop(k, None)
        self._cache[key] = vec
        return vec


def _h(token: str, dim: int) -> int:
    return int(hashlib.md5(token.encode()).hexdigest()[:8], 16) % dim


def _normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalise so cosine similarity is a plain dot product."""
    vec = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(vec))
    return vec / n if n > 0 else vec


def cosine(a, b) -> float:
    """Cosine similarity that tolerates raw bytes, lists, or ndarrays."""
    va = _as_vec(a)
    vb = _as_vec(b)
    if va is None or vb is None or va.size != vb.size:
        return 0.0
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def _as_vec(v) -> Optional[np.ndarray]:
    if v is None:
        return None
    if isinstance(v, np.ndarray):
        return v.astype(np.float32, copy=False)
    if isinstance(v, (bytes, bytearray)):
        return np.frombuffer(v, dtype=np.float32)
    try:
        return np.asarray(v, dtype=np.float32)
    except Exception:
        return None


def to_blob(vec: np.ndarray) -> bytes:
    """Serialise as raw float32 — half the size of JSON and no parse cost."""
    return np.asarray(vec, dtype=np.float32).tobytes()


def from_blob(blob) -> Optional[np.ndarray]:
    """Read either the new float32 BLOB or the legacy JSON-text encoding.

    The existing store wrote `json.dumps(list)` into a BLOB column, so old rows
    are text. Accepting both means no migration is needed to read history.
    """
    if blob is None:
        return None
    if isinstance(blob, memoryview):
        blob = bytes(blob)
    if isinstance(blob, (bytes, bytearray)):
        # Legacy JSON rows start with '[' (0x5b).
        if blob[:1] == b"[":
            import json
            try:
                return np.asarray(json.loads(blob.decode("utf-8")), dtype=np.float32)
            except Exception:
                return None
        try:
            return np.frombuffer(blob, dtype=np.float32).copy()
        except Exception:
            return None
    return _as_vec(blob)


_shared: Optional[Embedder] = None
_shared_lock = threading.Lock()


def shared() -> Embedder:
    """Process-wide embedder. Loading a transformer twice wastes ~100 MB."""
    global _shared
    if _shared is None:
        with _shared_lock:
            if _shared is None:
                _shared = Embedder()
    return _shared
