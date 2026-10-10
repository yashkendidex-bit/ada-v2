import os
import asyncio
import json
import httpx
from dotenv import load_dotenv

load_dotenv(override=True)

# OpenAI-compatible provider registry ("body parts" of ADA).
# Each entry: base_url (OpenAI /chat/completions style), env key name.
PROVIDERS = {
    "nvidia": {"base_url": "https://integrate.api.nvidia.com/v1", "env": "NVIDIA_API_KEY"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "env": "GROQ_API_KEY"},
    "mistral": {"base_url": "https://api.mistral.ai/v1", "env": "MISTRAL_API_KEY"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "env": "OPENROUTER_API_KEY",
                   "headers": {"HTTP-Referer": "https://github.com/ada", "X-Title": "ADA"}},
}

# The CAD brain: Layer 2 writes a SPEC, so this chain is asked for long, exact JSON
# rather than conversation.
# 2026-09-26: flash LEADS (verified live today), full 5.3 catches behind it —
# the full model hung with zero response on 3 attempts (90s/180s/240s) while
# flash on the same endpoint answered instantly. Same family, one swap to flip
# the order back once the full model responds again.
# Override without editing code:  ADA_CAD_MODEL="openrouter:qwen/qwen3-coder"
#
# Dead entries removed 2026-09-24 (verified, not transient):
#   openrouter stealth/ox-alpha       -> HTTP 404, testing period ended
#   nvidia deepseek-v4-flash-0731     -> HTTP 410, end of life 2026-09-21
#   nvidia moonshotai/kimi-k3         -> timeout even at 180s (replaced by glm-5.3)
#   nvidia stepfun-ai/step-3.7-flash  -> HTTP 410, end of life 2026-08-28
CAD_CHAIN = [
    ("nvidia", "z-ai/glm-5.3-flash"),
    ("nvidia", "z-ai/glm-5.3"),
]

# NOTE: the old AGENTIC_CHAIN (nvidia kimi-k3 / muse-glimmer-30b, "reserved for
# future tasks") was removed 2026-09-26 — nothing ever called it, and Glimmer is
# now the primary brain directly. Do not re-add speculative chains.


def cad_chain() -> list[tuple[str, str]]:
    """CAD_CHAIN, with ADA_CAD_MODEL='provider:model[,provider:model]' taking the lead.

    Read at call time, not import time, so a shell that exports the variable after
    `providers` is already imported still gets the model it asked for.
    """
    raw = os.getenv("ADA_CAD_MODEL", "").strip()
    if not raw:
        return list(CAD_CHAIN)
    head = []
    for item in raw.split(","):
        provider, _, model = item.strip().partition(":")
        if provider in PROVIDERS and model:
            head.append((provider, model))
    return head + [pair for pair in CAD_CHAIN if pair not in head]

# Consecutive failures before a provider is temporarily disabled.
MAX_CONSECUTIVE_FAILURES = 3

# Wait this long before the single retry of a 429 that carries no Retry-After and no
# suggested delay — an upstream-saturated model is usually free again in seconds.
DEFAULT_429_BACKOFF = 6.0

_health = {name: 0 for name in PROVIDERS}  # consecutive failure count
_lock = asyncio.Lock()


def available(provider: str) -> bool:
    """True if the provider has an API key configured and is not unhealthy."""
    if provider not in PROVIDERS:
        return False
    if not os.getenv(PROVIDERS[provider]["env"]):
        return False
    return _health[provider] < MAX_CONSECUTIVE_FAILURES


def reset_health():
    """Re-enable all providers (e.g. after a transient outage)."""
    for name in PROVIDERS:
        _health[name] = 0


async def _record_failure(provider: str):
    async with _lock:
        _health[provider] += 1
        print(f"[PROVIDERS] '{provider}' failure #{_health[provider]}/{MAX_CONSECUTIVE_FAILURES} - "
              f"{'DISABLED until reset_health()' if _health[provider] >= MAX_CONSECUTIVE_FAILURES else 'still usable'}")


async def _record_success(provider: str):
    async with _lock:
        _health[provider] = 0


def _extract_content(content):
    """Normalizes an assistant message's 'content' to plain text. Some providers
    (Mistral) return a list of content parts incl. nested 'thinking' blocks."""
    if isinstance(content, list):
        out = []
        for c in content:
            if isinstance(c, dict):
                if c.get("text"):
                    out.append(c["text"])
                elif c.get("thinking"):
                    out.append(_extract_content(c["thinking"]))
            elif c:
                out.append(str(c))
        return "".join(out)
    return content or ""


def _looks_like_an_answer(text: str) -> bool:
    """Is this salvaged thinking actually a deliverable, or just a model musing?

    Deliberately structural rather than clever. Every caller on this transport asks
    for JSON (a spec, a patch array) or for fenced Python (a batch of skills), so a
    balanced brace/bracket pair or a closed code fence is the whole test. Prose that
    merely *discusses* a spec has neither, and prose is what a truncated reasoning
    dump leaves behind.
    """
    body = (text or "").strip()
    if len(body) < 200:
        return False
    if body.count("```") >= 2:
        return True
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = body.find(opener), body.rfind(closer)
        if 0 <= start < end:
            return True
    return False


async def chat(provider: str, model: str, messages, temperature=0.7, max_tokens=8192, timeout=180.0, reasoning_effort=None) -> str:
    """Single OpenAI-compatible chat completion call. Returns assistant text."""
    msg = await chat_message(provider, model, messages, temperature=temperature, max_tokens=max_tokens, timeout=timeout, reasoning_effort=reasoning_effort)
    content = _extract_content(msg.get("content"))
    if not content:
        raise RuntimeError(f"[{provider}] empty response from {model}")
    return content


async def chat_message(provider: str, model: str, messages, tools=None, temperature=0.7, max_tokens=8192, timeout=180.0, reasoning_effort=None) -> dict:
    """Single OpenAI-compatible chat completion call. Returns the raw assistant
    message dict (content + tool_calls) so callers can drive tool-use loops."""
    return await _chat_message_impl(provider, model, messages, tools=tools, temperature=temperature,
                                    max_tokens=max_tokens, timeout=timeout, on_delta=None,
                                    reasoning_effort=reasoning_effort)


async def chat_message_stream(provider: str, model: str, messages, tools=None, on_delta=None,
                              temperature=0.7, max_tokens=8192, timeout=300.0, reasoning_effort=None) -> dict:
    """Streaming chat completion (SSE). `on_delta(text)` is called for every text
    token as it arrives (for realtime TTS). Returns the assembled message dict."""
    return await _chat_message_impl(provider, model, messages, tools=tools, temperature=temperature,
                                    max_tokens=max_tokens, timeout=timeout, on_delta=on_delta,
                                    reasoning_effort=reasoning_effort)


async def _chat_message_impl(provider: str, model: str, messages, tools=None, temperature=0.7,
                             max_tokens=8192, timeout=180.0, on_delta=None, reasoning_effort=None) -> dict:
    """Core implementation: one OpenAI-compatible chat completion call.
    With on_delta set, streams the response (SSE) and feeds text tokens live."""
    if not available(provider):
        raise RuntimeError(f"Provider '{provider}' unavailable (missing key or unhealthy)")

    cfg = PROVIDERS[provider]
    headers = {"Authorization": f"Bearer {os.getenv(cfg['env'])}", "Content-Type": "application/json"}
    headers.update(cfg.get("headers") or {})
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    if tools:
        payload["tools"] = tools
    if on_delta is not None:
        payload["stream"] = True

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            if on_delta is None:
                resp = await client.post(f"{cfg['base_url']}/chat/completions", headers=headers, json=payload)
                if resp.status_code in (429, 529):
                    # Transient: rate limit (TPM/RPM) or an overloaded upstream.
                    # Small waits are worth it; LONG waits are not — Ada runs a
                    # provider CHAIN, and 25s of silence is worse than moving on.
                    # Cap: sleep at most 5s, then fail fast to the next provider.
                    delay = await _retry_delay(resp)
                    if delay is None or delay > 5.0:
                        raise RuntimeError(f"[{provider}] HTTP {resp.status_code} (rate-limited, failing fast to next provider)")
                    print(f"[PROVIDERS] '{provider}' {resp.status_code}; retrying in {delay:.1f}s")
                    await asyncio.sleep(delay)
                    resp = await client.post(f"{cfg['base_url']}/chat/completions", headers=headers, json=payload)
                if resp.status_code != 200:
                    raise RuntimeError(f"[{provider}] HTTP {resp.status_code}: {resp.text[:300]}")
                try:
                    choice = resp.json()["choices"][0]
                    msg = choice["message"]
                except (KeyError, IndexError) as e:
                    raise RuntimeError(f"[{provider}] malformed response") from e
                if not msg.get("content") and not msg.get("tool_calls"):
                    # A reasoning model that spent its whole budget thinking returns an
                    # empty `content` with the thinking in a side field. The answer is
                    # often in there, so salvage it rather than calling the call dead —
                    # and if there is nothing to salvage, say *why* it was empty.
                    #
                    # But only if it is plausibly an ANSWER. Accepting any non-empty
                    # scrap turns a clean "empty reply -> try the next provider" into
                    # "a paragraph of thinking -> success", and the caller then fails to
                    # parse it with no fallback left. A live run lost four synthesised
                    # skills to 421 characters of preamble that way. Every caller on
                    # this transport wants either JSON or fenced code, so demand one of
                    # those structural markers and otherwise let the chain move on.
                    reason = choice.get("finish_reason") or "no finish_reason"
                    for key in ("reasoning_content", "reasoning", "thinking"):
                        salvaged = _extract_content(msg.get(key))
                        if not salvaged or not salvaged.strip():
                            continue
                        if not _looks_like_an_answer(salvaged):
                            print(f"[PROVIDERS] '{provider}/{model}' left only "
                                  f"{len(salvaged)} chars of '{key}' with no JSON or "
                                  "code block in it; treating as empty and moving on.")
                            continue
                        print(f"[PROVIDERS] '{provider}/{model}' put its answer in "
                              f"'{key}' ({len(salvaged)} chars); using it.")
                        msg["content"] = salvaged
                        return msg
                    usage = resp.json().get("usage") or {}
                    raise RuntimeError(
                        f"[{provider}] empty response from {model} "
                        f"(finish_reason={reason}, max_tokens={max_tokens}, "
                        f"completion_tokens={usage.get('completion_tokens', '?')}) — "
                        f"{'raise max_tokens: it ran out of budget mid-answer' if reason == 'length' else 'no content and no tool call'}")
                return msg

            # Streaming path (SSE)
            async def _stream_once():
                async with client.stream("POST", f"{cfg['base_url']}/chat/completions", headers=headers, json=payload) as resp:
                    if resp.status_code in (429, 529):
                        delay = await _retry_delay(resp)
                        if delay is None or delay > 5.0:
                            raise RuntimeError(f"[{provider}] HTTP {resp.status_code} (rate-limited, failing fast to next provider)")
                        print(f"[PROVIDERS] '{provider}' rate-limited; retrying in {delay:.1f}s")
                        await asyncio.sleep(delay)
                        return None
                    if resp.status_code != 200:
                        body = (await resp.aread()).decode("utf-8", "replace")[:300]
                        raise RuntimeError(f"[{provider}] HTTP {resp.status_code}: {body}")
                    return await _drain_stream(resp, on_delta)

            full_text = None
            for _attempt in range(2):
                result = await _stream_once()
                if result is not None:
                    full_text, tool_calls = result
                    break
            if full_text is None:
                raise RuntimeError(f"[{provider}] request failed after retry")
    except httpx.HTTPError as e:
        raise RuntimeError(f"[{provider}] request failed: {e}") from e

    msg = {"content": full_text or None, "tool_calls": tool_calls or None}
    if not msg["content"] and not msg["tool_calls"]:
        raise RuntimeError(f"[{provider}] empty response from {model}")
    return msg


async def _retry_delay(resp) -> float | None:
    """How long to wait before retrying a 429, or None to give up.

    Three sources, most authoritative first: the standard `Retry-After` header, Groq's
    "Please try again in 1.5s" prose, and — for a 429 that carries no hint at all, which
    is what OpenRouter sends when an upstream provider is saturated — a short fixed
    backoff. Giving up on a hintless 429 used to drop the primary model on the first
    hiccup and spend the request on a fallback instead.
    """
    after = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
    if after:
        try:
            return min(max(float(after), 0.5), 30.0)
        except ValueError:
            pass
    try:
        body = resp.json()
    except Exception:
        return DEFAULT_429_BACKOFF
    msg = body.get("error", {}).get("message", "")
    import re
    m = re.search(r"Please try again in ([\d.]+)s", msg)
    if m:
        return min(float(m.group(1)) + 0.5, 30.0)
    return DEFAULT_429_BACKOFF


async def _drain_stream(resp, on_delta) -> tuple[str, list]:
    """Consumes an SSE stream, accumulating content + tool calls and calling on_delta live.
    Returns (full_text, tool_calls)."""
    full_text = ""
    tool_calls = []
    async for line in resp.aiter_lines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        if not chunk.get("choices"):
            continue
        delta = chunk["choices"][0].get("delta", {})
        if delta.get("content"):
            full_text += delta["content"]
            if on_delta:
                result = on_delta(delta["content"])
                if asyncio.iscoroutine(result):
                    await result
        for tc in delta.get("tool_calls") or []:
            idx = tc.get("index", len(tool_calls))
            while len(tool_calls) <= idx:
                tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
            if tc.get("id"):
                tool_calls[idx]["id"] = tc["id"]
            if tc.get("function"):
                if tc["function"].get("name"):
                    tool_calls[idx]["function"]["name"] += tc["function"]["name"]
                if tc["function"].get("arguments"):
                    tool_calls[idx]["function"]["arguments"] += tc["function"]["arguments"]
    return full_text, tool_calls


async def transcribe_audio(audio_bytes: bytes, filename: str = "audio.webm") -> str:
    """Speech-to-text via Groq Whisper (whisper-large-v3-turbo)."""
    cfg = PROVIDERS["groq"]
    headers = {"Authorization": f"Bearer {os.getenv(cfg['env'])}"}
    files = {"file": (filename, audio_bytes, "audio/webm")}
    data = {"model": "whisper-large-v3-turbo"}
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{cfg['base_url']}/audio/transcriptions",
                                 headers=headers, data=data, files=files)
    if resp.status_code != 200:
        raise RuntimeError(f"[groq] STT HTTP {resp.status_code}: {resp.text[:200]}")
    text = (resp.json().get("text") or "").strip()
    if not text:
        raise RuntimeError("[groq] STT returned empty text")
    return text


async def chat_with_fallback(chain, messages, **kwargs) -> str:
    """Tries (provider, model) pairs in order; returns first non-empty text.

    chain: list of tuples like [("groq", "openai/gpt-oss-120b"), ("nvidia", "z-ai/glm-5.3-flash")]
    Health tracking lives here: success/failure is measured per attempt.
    """
    last_error = None
    for provider, model in chain:
        if not available(provider):
            print(f"[PROVIDERS] Skipping '{provider}/{model}' (missing key or unhealthy).")
            continue
        try:
            text = await chat(provider, model, messages, **kwargs)
            await _record_success(provider)
            print(f"[PROVIDERS] OK '{provider}/{model}' returned {len(text)} chars.")
            return text
        except Exception as e:
            await _record_failure(provider)
            last_error = e
            print(f"[PROVIDERS] '{provider}/{model}' failed: {str(e)[:150]}")
            continue
    if last_error:
        raise last_error
    raise RuntimeError("No provider available for fallback chain")
