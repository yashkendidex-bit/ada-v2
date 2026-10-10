"""Tests for the embedding tier ladder.

The Mistral tier cannot be live-tested from a sandbox that blocks reading the API key
(which is the correct behaviour — a credential broker should refuse to hand a secret to
code that then POSTs it somewhere). So the API is stubbed here and the *contract* is
tested: which tier is chosen, how a 1024-d vector is stored and retrieved, and that a
failure degrades loudly instead of corrupting the vector space.

    .venv/bin/python -m pytest tests/test_memory_embeddings.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import memory.embeddings as emb_mod  # noqa: E402
from memory.embeddings import Embedder, from_blob, to_blob  # noqa: E402


class _FakeResponse:
    def __init__(self, payload, status=200):
        self.status_code = status
        self._payload = payload
        self.text = "stub"

    def json(self):
        return self._payload


class _FakeClient:
    """Stands in for httpx.Client, returning deterministic 1024-d vectors."""

    calls: list = []

    def __init__(self, *a, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, headers=None, json=None, **kw):
        _FakeClient.calls.append({"url": url, "headers": headers, "json": json})
        # Deterministic, text-derived vectors so ranking is reproducible.
        data = []
        for i, text in enumerate(json.get("input", [])):
            seed = sum(ord(c) for c in text) % 997
            rng = np.random.default_rng(seed)
            vec = rng.normal(size=1024).astype(np.float32)
            data.append({"index": i, "embedding": vec.tolist()})
        return _FakeResponse({"data": data, "usage": {"prompt_tokens": 7}})


@pytest.fixture()
def mistral(monkeypatch):
    """Force the Mistral tier with a stubbed HTTP client."""
    _FakeClient.calls = []
    monkeypatch.setattr(emb_mod, "_read_api_key", lambda: "stub-key")
    import httpx
    monkeypatch.setattr(httpx, "Client", _FakeClient)
    return Embedder(prefer="mistral")


def test_mistral_tier_is_selected_and_reports_1024d(mistral):
    h = mistral.health()
    assert h["backend"] == "mistral"
    assert h["semantic"] is True
    assert h["dim"] == 1024
    assert h["model"] == emb_mod.MISTRAL_MODEL
    assert h["local"] is False, "the API tier is not local — the UI should know"


def test_mistral_returns_vectors_of_the_declared_dimension(mistral):
    vec = mistral.embed("the bracket cracked at the motor mount")
    assert isinstance(vec, np.ndarray)
    assert vec.size == 1024
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5, "must be L2-normalised"


def test_mistral_sends_the_documented_request_shape(mistral):
    mistral.embed_many(["alpha", "beta"])
    assert len(_FakeClient.calls) == 1, "two short texts should be one batched call"
    call = _FakeClient.calls[0]
    assert call["url"] == emb_mod.MISTRAL_ENDPOINT
    assert call["headers"]["Authorization"] == "Bearer stub-key"
    assert call["json"]["model"] == emb_mod.MISTRAL_MODEL
    assert call["json"]["input"] == ["alpha", "beta"]


def test_mistral_batches_large_inputs(mistral):
    texts = [f"memory number {i}" for i in range(emb_mod.MISTRAL_BATCH * 2 + 1)]
    vecs = mistral.embed_many(texts)
    assert len(vecs) == len(texts)
    assert len(_FakeClient.calls) == 3, "should chunk at MISTRAL_BATCH"


def test_mistral_caches_repeated_text(mistral):
    mistral.embed("the same string twice")
    mistral.embed("the same string twice")
    assert len(_FakeClient.calls) == 1, "a cache hit must not cost a round-trip"


def test_api_failure_degrades_loudly_not_silently(monkeypatch, caplog):
    """A failure must fall back to a local model AND say so.

    The alternative — returning zeros — would be an invisible corruption of the vector
    space, which is the exact class of bug this subsystem was built to eliminate.
    """
    monkeypatch.setattr(emb_mod, "_read_api_key", lambda: "stub-key")

    class _Broken(_FakeClient):
        def post(self, *a, **kw):
            return _FakeResponse({}, status=401)

    import httpx
    monkeypatch.setattr(httpx, "Client", _Broken)

    e = Embedder(prefer="mistral")
    with caplog.at_level("WARNING"):
        vec = e.embed("this should fall back")

    assert e.backend_name != "mistral", "the API tier must be switched off after a failure"
    assert e.is_semantic, "and it must land on another *semantic* tier, not lexical"
    assert vec.size == e.dim
    assert vec.size > 0, "never return an empty vector"
    assert any("mistral embeddings failed" in r.message for r in caplog.records), \
        "the degradation must be announced"


def test_lexical_tier_admits_it_is_not_semantic(monkeypatch):
    """Tier 4 is the honest fallback: it works, and it says what it is."""
    monkeypatch.setattr(emb_mod, "_read_api_key", lambda: "")
    e = Embedder(prefer="lexical")
    h = e.health()
    assert h["backend"] == "lexical"
    assert h["semantic"] is False, "the UI and callers must be able to tell"
    assert e.embed("anything").size == emb_mod.EMBED_DIM


def test_blob_roundtrip_across_dimensions():
    """1024-d and 384-d vectors must both survive the store, and stay distinguishable."""
    for dim in (384, 1024):
        v = np.random.default_rng(dim).normal(size=dim).astype(np.float32)
        assert np.allclose(from_blob(to_blob(v)), v)
    a, b = from_blob(to_blob(np.zeros(384, dtype=np.float32))), \
           from_blob(to_blob(np.zeros(1024, dtype=np.float32)))
    assert a.size != b.size, "mixed-generation vectors must be separable, not silently equal"


def test_retriever_skips_mismatched_dimensions(tmp_path):
    """A store mid-migration holds two dimensions. Querying must skip the stale rows
    rather than crashing — np.vstack on ragged input raises, which would take down
    every query over one leftover row."""
    from memory.retrieve import Retriever
    from memory.store import Store

    store = Store(tmp_path / "mixed.db")
    now = 1_700_000_000.0
    for i, dim in enumerate((384, 1024, 1024)):
        vec = np.random.default_rng(dim + i).normal(size=dim).astype(np.float32)
        store.execute(
            "INSERT INTO recall_entries(kind, content, ts, ts_event, embedding,"
            " embed_backend, stability, difficulty) VALUES(?,?,?,?,?,?,?,?)",
            ("conversation", f"memory {i}", now - i, now - i, to_blob(vec),
             f"model-{dim}", 1.0, 5.0))

    r = Retriever(store, Embedder(prefer="lexical"))   # 384-d
    mat, ids = r._index(expect_dim=384)
    assert mat.shape[1] == 384
    assert len(ids) == 1, "only the matching-dimension row should be indexed"
    store.close()
