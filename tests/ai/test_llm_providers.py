"""
Unit and integration tests for Project AARNA LLM providers and Hybrid Router.
Tests GeminiLLMProvider, OllamaLLMProvider, HybridLLMRouter, LLMQuotaGuard,
and secret hygiene.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from src.ai.providers.llm_base import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMError,
    LLMModelNotFoundError,
    LLMQuotaExhaustedError,
    LLMRateLimitError,
    LLMQuotaGuard,
    redact_secrets,
)
from src.ai.providers.llm_gemini import GeminiLLMProvider
from src.ai.providers.llm_ollama import OllamaLLMProvider
from src.ai.providers.llm_router import HybridLLMRouter
from src.ui.workers import ChatWorker


# ============================================================================
# 1. Secret Hygiene & Redaction Tests
# ============================================================================

def test_secret_redaction_masks_api_keys() -> None:
    raw = "Failed with key AIzaSyFakeApiKey1234567890 and Bearer 1234567890abcdef"
    sanitized = redact_secrets(raw)
    assert "AIzaSyFakeApiKey1234567890" not in sanitized
    assert "AIza***" in sanitized
    assert "Bearer ***" in sanitized


def test_secret_redaction_with_explicit_secret() -> None:
    custom_secret = "my_super_secret_test_token"
    raw = f"Traceback containing {custom_secret} in connection URL"
    sanitized = redact_secrets(raw, extra_secrets=[custom_secret])
    assert custom_secret not in sanitized
    assert "***REDACTED***" in sanitized


def test_gemini_provider_repr_never_leaks_key() -> None:
    secret_key = "AIzaSyTestingKeyLeakage999"
    provider = GeminiLLMProvider(api_key=secret_key)
    rep = repr(provider)
    assert secret_key not in rep


# ============================================================================
# 2. Rate Limit & Quota Guard Tests
# ============================================================================

def test_quota_guard_permits_under_limit() -> None:
    guard = LLMQuotaGuard(max_rpm=3, window_seconds=60.0)
    assert guard.can_request()[0] is True
    guard.record_request()
    guard.record_request()
    assert guard.can_request()[0] is True
    guard.record_request()
    # Now at limit
    allowed, wait_sec = guard.can_request()
    assert allowed is False
    assert wait_sec > 0.0


def test_quota_guard_raises_quota_exhausted() -> None:
    guard = LLMQuotaGuard(max_rpm=2, window_seconds=60.0)
    guard.check_and_record()
    guard.check_and_record()
    with pytest.raises(LLMQuotaExhaustedError) as exc_info:
        guard.check_and_record()
    assert "free tier guard" in str(exc_info.value)


# ============================================================================
# 3. Gemini Message Normalization Tests
# ============================================================================

def test_gemini_message_normalization_merges_consecutive_roles() -> None:
    provider = GeminiLLMProvider(api_key="test-key")
    messages = [
        {"role": "system", "content": "You are Aarna."},
        {"role": "user", "content": "Hello"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I am great."},
        {"role": "assistant", "content": "How can I help you today?"},
    ]

    sys_inst, contents = provider._prepare_contents_and_system(messages)
    assert sys_inst == "You are Aarna."
    assert len(contents) == 2
    assert contents[0].role == "user"
    assert "Hello\n\nHow are you?" in contents[0].parts[0].text
    assert contents[1].role == "model"
    assert "I am great.\n\nHow can I help you today?" in contents[1].parts[0].text


def test_gemini_message_normalization_prepends_user_if_starting_with_model() -> None:
    provider = GeminiLLMProvider(api_key="test-key")
    messages = [
        {"role": "assistant", "content": "Previous assistant greeting"},
        {"role": "user", "content": "Follow-up question"},
    ]

    _, contents = provider._prepare_contents_and_system(messages)
    assert len(contents) == 3
    assert contents[0].role == "user"
    assert contents[1].role == "model"
    assert contents[2].role == "user"


# ============================================================================
# 4. Gemini Provider Streaming & Fallback Logic Tests
# ============================================================================

def test_gemini_provider_unconfigured_raises_auth() -> None:
    provider = GeminiLLMProvider(api_key="")
    with pytest.raises(LLMAuthenticationError):
        list(provider.chat_stream([{"role": "user", "content": "hi"}]))


def test_gemini_provider_cooldown_fails_fast() -> None:
    provider = GeminiLLMProvider(api_key="valid-test-key")
    provider._cooldown_until = time.time() + 30.0
    with pytest.raises(LLMRateLimitError) as exc_info:
        list(provider.chat_stream([{"role": "user", "content": "hi"}]))
    assert "cooling down" in str(exc_info.value)


def test_gemini_provider_automatic_model_fallback_on_404(mocker) -> None:
    mock_client = MagicMock()
    # First candidate (2.5) throws 404
    mock_chunk = MagicMock()
    mock_chunk.text = "Hello from fallback model!"

    def mock_stream(model, contents, config):
        if model == "gemini-2.5-flash":
            exc = LLMModelNotFoundError("404 NOT_FOUND: model retired")
            setattr(exc, "code", 404)
            raise exc
        return iter([mock_chunk])

    mock_client.models.generate_content_stream.side_effect = mock_stream

    provider = GeminiLLMProvider(
        api_key="test-key",
        model="gemini-2.5-flash",
        fallback_model="gemini-3.5-flash",
    )
    provider._client = mock_client

    tokens = list(provider.chat_stream([{"role": "user", "content": "Hello"}]))
    assert tokens == ["Hello from fallback model!"]
    assert provider.model_name == "gemini-3.5-flash"


def test_gemini_cancellation_halts_stream(mocker) -> None:
    mock_client = MagicMock()
    c1, c2, c3 = MagicMock(), MagicMock(), MagicMock()
    c1.text, c2.text, c3.text = "One ", "Two ", "Three "
    mock_client.models.generate_content_stream.return_value = iter([c1, c2, c3])

    provider = GeminiLLMProvider(api_key="test-key")
    provider._client = mock_client

    cancelled = False

    def is_cancelled():
        return cancelled

    gen = provider.chat_stream(
        [{"role": "user", "content": "Count"}],
        cancellation_check=is_cancelled,
    )
    first = next(gen)
    assert first == "One "
    cancelled = True
    remaining = list(gen)
    assert remaining == []


# ============================================================================
# 5. Hybrid Router Failover Tests
# ============================================================================

def test_hybrid_router_mode_selection() -> None:
    router = HybridLLMRouter(mode="ollama")
    assert router.mode == "ollama"
    router.mode = "gemini"
    assert router.mode == "gemini"
    router.mode = "auto"
    assert router.mode == "auto"

    with pytest.raises(ValueError):
        router.mode = "invalid_mode"


def test_hybrid_router_falls_back_to_ollama_on_gemini_failure() -> None:
    mock_gemini = MagicMock()
    mock_gemini.is_available.return_value = True
    mock_gemini.is_cooling_down = False
    mock_gemini.chat_stream.side_effect = LLMConnectionError("Network timeout", provider="gemini")

    mock_ollama = MagicMock()
    mock_ollama.model_name = "qwen3.5:4b"
    mock_ollama.chat_stream.return_value = iter(["Fallback ", "response ", "from ", "Ollama."])

    fallback_events = []

    def on_fallback(f, t, reason):
        fallback_events.append((f, t, reason))

    router = HybridLLMRouter(
        gemini_provider=mock_gemini,
        ollama_provider=mock_ollama,
        mode="auto",
        enable_local_fallback=True,
        on_fallback=on_fallback,
    )

    tokens = list(router.chat_stream([{"role": "user", "content": "hi"}]))
    assert "".join(tokens) == "Fallback response from Ollama."
    assert router.last_provider_used == "ollama"
    assert len(fallback_events) == 1
    assert fallback_events[0][0] == "gemini"
    assert fallback_events[0][1] == "ollama"
    assert "Network timeout" in fallback_events[0][2]


def test_hybrid_router_strict_gemini_mode_does_not_fall_back() -> None:
    mock_gemini = MagicMock()
    mock_gemini.is_available.return_value = True
    mock_gemini.is_cooling_down = False
    mock_gemini.chat_stream.side_effect = LLMRateLimitError("429 Too Many Requests", provider="gemini")

    mock_ollama = MagicMock()
    router = HybridLLMRouter(
        gemini_provider=mock_gemini,
        ollama_provider=mock_ollama,
        mode="gemini",
        enable_local_fallback=True,  # Strict gemini mode should override this
    )

    with pytest.raises(LLMRateLimitError):
        list(router.chat_stream([{"role": "user", "content": "hi"}]))
    mock_ollama.chat_stream.assert_not_called()


def test_hybrid_router_diagnostics_structure() -> None:
    mock_gemini = MagicMock()
    mock_gemini.is_available.return_value = True
    mock_gemini.is_cooling_down = False
    mock_gemini.model_name = "gemini-3.5-flash"
    mock_gemini.check_connection.return_value = (True, "Gemini Online")

    mock_ollama = MagicMock()
    mock_ollama.model_name = "qwen3.5:4b"
    mock_ollama.check_connection.return_value = (True, "Ollama Online")

    router = HybridLLMRouter(
        gemini_provider=mock_gemini,
        ollama_provider=mock_ollama,
        mode="auto",
    )

    diag = router.get_diagnostics()
    assert diag["mode"] == "auto"
    assert diag["active_provider"] == "gemini"
    assert diag["active_model"] == "gemini-3.5-flash"
    assert diag["gemini_available"] is True
    assert diag["ollama_available"] is True
    assert "₹0" in diag["cost_profile"]


# ============================================================================
# 6. ChatWorker Integration Tests
# ============================================================================

def test_chat_worker_emits_chunks_and_ttft(qapp) -> None:
    mock_client = MagicMock()
    mock_client.last_provider_used = "gemini"
    mock_client.last_model_used = "gemini-3.5-flash"
    mock_client.chat_stream.return_value = iter(["Hello ", "world", "!"])

    worker = ChatWorker(
        client=mock_client,
        messages=[{"role": "user", "content": "Hi"}],
        model="gemini-3.5-flash",
    )

    received_chunks = []
    received_ttft = []
    finished_response = []
    provider_used = []

    worker.chunk_received.connect(received_chunks.append)
    worker.first_token_received.connect(received_ttft.append)
    worker.finished_response.connect(finished_response.append)
    worker.provider_used.connect(lambda p, m: provider_used.append((p, m)))

    worker.run()

    assert "".join(received_chunks) == "Hello world!"
    assert len(received_ttft) == 1
    assert received_ttft[0] >= 0.0
    assert finished_response == ["Hello world!"]
    assert provider_used == [("gemini", "gemini-3.5-flash")]


def test_chat_worker_cancellation_aborts_cleanly(qapp) -> None:
    mock_client = MagicMock()

    def slow_stream(**kwargs):
        yield "Chunk1 "
        worker.cancel()
        yield "Chunk2 "

    mock_client.chat_stream.side_effect = slow_stream

    worker = ChatWorker(
        client=mock_client,
        messages=[{"role": "user", "content": "Hi"}],
        model="qwen3.5:4b",
    )

    received = []
    finished = []
    worker.chunk_received.connect(received.append)
    worker.finished_response.connect(finished.append)

    worker.run()

    assert received == ["Chunk1 "]
    assert finished == []  # finished_response must NOT be emitted on cancellation
