"""
Tests for the multi-provider fallback layer (NVIDIA NIM / Groq / Mistral).
"""
import os
import asyncio
import pytest
import providers


@pytest.fixture(autouse=True)
def fake_keys(monkeypatch):
    """Provide fake keys so providers look available."""
    monkeypatch.setenv("NVIDIA_API_KEY", "fake-nvidia")
    monkeypatch.setenv("GROQ_API_KEY", "fake-groq")
    monkeypatch.setenv("MISTRAL_API_KEY", "fake-mistral")
    providers.reset_health()
    yield
    providers.reset_health()


class TestProviderRegistry:
    def test_available_without_key(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY")
        assert providers.available("groq") is False
        assert providers.available("nvidia") is True

    def test_unknown_provider_unavailable(self):
        assert providers.available("openai") is False

    def test_all_known_providers_registered(self):
        # openrouter is the primary brain provider (ada.BRAIN_PROVIDER) —
        # the registry must include every provider the brain can route to.
        assert set(providers.PROVIDERS) == {"nvidia", "groq", "mistral", "openrouter"}


class TestFallbackChain:
    def test_uses_first_successful_provider(self, monkeypatch):
        calls = []

        async def fake_chat(provider, model, messages, **kw):
            calls.append((provider, model))
            if provider == "nvidia":
                raise RuntimeError("nvidia down")
            return f"code from {model}"

        monkeypatch.setattr(providers, "chat", fake_chat)
        text = asyncio.run(providers.chat_with_fallback(
            [("nvidia", "n1"), ("groq", "g1"), ("mistral", "m1")],
            [{"role": "user", "content": "hi"}],
        ))
        assert text == "code from g1"
        assert calls == [("nvidia", "n1"), ("groq", "g1")]

    def test_raises_when_all_fail(self, monkeypatch):
        async def fake_chat(provider, model, messages, **kw):
            raise RuntimeError(f"{provider} failed")

        monkeypatch.setattr(providers, "chat", fake_chat)
        with pytest.raises(RuntimeError):
            asyncio.run(providers.chat_with_fallback(
                [("nvidia", "n1"), ("groq", "g1")],
                [{"role": "user", "content": "hi"}],
            ))

    def test_skips_provider_without_key(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY")
        calls = []

        async def fake_chat(provider, model, messages, **kw):
            calls.append(provider)
            return "ok"

        monkeypatch.setattr(providers, "chat", fake_chat)
        text = asyncio.run(providers.chat_with_fallback(
            [("groq", "g1"), ("mistral", "m1")],
            [{"role": "user", "content": "hi"}],
        ))
        assert text == "ok"
        assert calls == ["mistral"]


class TestHealthTracking:
    def test_provider_disabled_after_repeated_failures(self, monkeypatch):
        async def fake_chat(provider, model, messages, **kw):
            raise RuntimeError("boom")

        monkeypatch.setattr(providers, "chat", fake_chat)
        for _ in range(providers.MAX_CONSECUTIVE_FAILURES):
            with pytest.raises(RuntimeError):
                asyncio.run(providers.chat_with_fallback([("groq", "g1")], []))
        assert providers.available("groq") is False

    def test_reset_health_reenables(self, monkeypatch):
        async def fake_chat(provider, model, messages, **kw):
            raise RuntimeError("boom")

        monkeypatch.setattr(providers, "chat", fake_chat)
        for _ in range(providers.MAX_CONSECUTIVE_FAILURES):
            with pytest.raises(RuntimeError):
                asyncio.run(providers.chat_with_fallback([("groq", "g1")], []))
        assert providers.available("groq") is False
        providers.reset_health()
        assert providers.available("groq") is True
