"""
Comprehensive Test Suite for NVIDIA NIM Free Endpoints and Multi-Tier Cloud LLM Routing in Project AARNA.

Verifies:
1. Configuration & secure environment key handling (NVIDIA_API_KEY).
2. Model endpoint mapping:
   - nvidia/nemotron-3.5-super-vl-120b-a12b (Primary Visual/Coding Expert)
   - nvidia/nemotron-3-ultra-550b-a55b (Automation Fallback)
   - deepseek-ai/deepseek-v4.1-flash (Secondary Cloud Tier Fallback)
3. Secret redaction and key masking hygiene.
4. Multimodal base64 data URL conversion for NVIDIA OpenAI-compatible format.
5. Streaming token generation via OpenAI SDK and HTTP SSE fallback.
6. Advanced Multimodal Routing: routing code/reasoning vision queries to Nemotron 3.5 Super VL.
7. Resilient Multi-Tier Fallback:
   Tier 1 (Gemini Flash) -> Tier 2 (NVIDIA DeepSeek v4.1 Flash) -> Tier 3 (Ollama Local).
"""

from __future__ import annotations

import base64
import io
import time
from unittest.mock import MagicMock, patch

import pytest

from src.ai.prompts import AARNA_SYSTEM_PROMPT
from src.ai.providers.llm_base import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMRateLimitError,
    redact_secrets,
)
from src.ai.providers.llm_nvidia import NvidiaNIMProvider
from src.ai.providers.llm_router import HybridLLMRouter
from src.core.config import AppConfig


# ============================================================================
# 1. Configuration & Key Mapping Tests
# ============================================================================

def test_nvidia_config_defaults() -> None:
    """Verify default model strings and base URL in AppConfig."""
    config = AppConfig()
    assert config.nvidia_api_key == ""
    assert config.nvidia_base_url == "https://integrate.api.nvidia.com/v1"
    assert config.nvidia_vision_model == "nvidia/nemotron-3.5-super-vl-120b-a12b"
    assert config.nvidia_automation_model == "nvidia/nemotron-3-ultra-550b-a55b"
    assert config.nvidia_fallback_model == "deepseek-ai/deepseek-v4.1-flash"
    assert config.nvidia_timeout_seconds == 30.0
    assert config.nvidia_max_retries == 2
    assert config.nvidia_rate_limit_rpm == 40


def test_nvidia_config_load_from_environment(monkeypatch) -> None:
    """Verify loading custom NVIDIA NIM settings from environment variables."""
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-testkey1234567890abcdef")
    monkeypatch.setenv("NVIDIA_BASE_URL", "https://custom.nvidia.nim/v1")
    monkeypatch.setenv("NVIDIA_VISION_MODEL", "nvidia/nemotron-3.5-super-vl-120b-a12b")
    monkeypatch.setenv("NVIDIA_AUTOMATION_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
    monkeypatch.setenv("NVIDIA_FALLBACK_MODEL", "deepseek-ai/deepseek-v4.1-flash")
    monkeypatch.setenv("NVIDIA_TIMEOUT_SECONDS", "45.0")
    monkeypatch.setenv("NVIDIA_RATE_LIMIT_RPM", "50")

    config = AppConfig.load()
    assert config.nvidia_api_key == "nvapi-testkey1234567890abcdef"
    assert config.nvidia_base_url == "https://custom.nvidia.nim/v1"
    assert config.nvidia_vision_model == "nvidia/nemotron-3.5-super-vl-120b-a12b"
    assert config.nvidia_automation_model == "nvidia/nemotron-3-ultra-550b-a55b"
    assert config.nvidia_fallback_model == "deepseek-ai/deepseek-v4.1-flash"
    assert config.nvidia_timeout_seconds == 45.0
    assert config.nvidia_rate_limit_rpm == 50


# ============================================================================
# 2. Secret Hygiene & Redaction Tests
# ============================================================================

def test_nvidia_secret_redaction_masks_nvapi_keys() -> None:
    """Ensure nvapi- keys are automatically masked in strings and logs."""
    raw = "Failed with key nvapi-abc1234567890_defghij and Bearer 1234567890"
    sanitized = redact_secrets(raw)
    assert "nvapi-abc1234567890_defghij" not in sanitized
    assert "nvapi-***" in sanitized


def test_nvidia_provider_repr_masks_api_key() -> None:
    """Ensure NvidiaNIMProvider __repr__ never leaks raw API credentials."""
    secret_key = "nvapi-supersecretkey999888777"
    provider = NvidiaNIMProvider(api_key=secret_key)
    rep = repr(provider)
    assert secret_key not in rep
    assert "nvapi-s***" in rep


# ============================================================================
# 3. Multimodal Payload Formatting (Base64 Data URLs)
# ============================================================================

def test_nvidia_multimodal_message_preparation() -> None:
    """Verify raw image bytes are correctly encoded into data:image/jpeg;base64 URL format."""
    provider = NvidiaNIMProvider(api_key="nvapi-dummykey")
    dummy_image = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01"

    messages = [
        {"role": "user", "content": "Ye code screen dekho aur bug batao."}
    ]

    out_messages = provider._prepare_openai_messages(messages, image_bytes=dummy_image)

    # First message should be system prompt
    assert out_messages[0]["role"] == "system"
    assert out_messages[0]["content"] == AARNA_SYSTEM_PROMPT

    # Second message should be multimodal user message
    user_msg = out_messages[1]
    assert user_msg["role"] == "user"
    assert isinstance(user_msg["content"], list)
    assert user_msg["content"][0]["type"] == "text"
    assert user_msg["content"][0]["text"] == "Ye code screen dekho aur bug batao."

    img_part = user_msg["content"][1]
    assert img_part["type"] == "image_url"
    expected_b64 = base64.b64encode(dummy_image).decode("utf-8")
    assert img_part["image_url"]["url"] == f"data:image/jpeg;base64,{expected_b64}"


# ============================================================================
# 4. Streaming Token Generation Tests (SDK & SSE Fallback)
# ============================================================================

def test_nvidia_streaming_via_sdk_client() -> None:
    """Verify streaming token delivery when official OpenAI SDK client is used."""
    provider = NvidiaNIMProvider(api_key="nvapi-testkey")

    mock_chunk1 = MagicMock()
    mock_chunk1.choices = [MagicMock(delta=MagicMock(content="Namaste! "))]
    mock_chunk2 = MagicMock()
    mock_chunk2.choices = [MagicMock(delta=MagicMock(content="Main AARNA hoon."))]

    mock_sdk = MagicMock()
    mock_sdk.chat.completions.create.return_value = [mock_chunk1, mock_chunk2]

    with patch.object(provider, "_get_openai_client", return_value=mock_sdk):
        stream = provider.chat_stream([{"role": "user", "content": "Hello"}])
        tokens = list(stream)

    assert tokens == ["Namaste! ", "Main AARNA hoon."]
    mock_sdk.chat.completions.create.assert_called_once()
    kwargs = mock_sdk.chat.completions.create.call_args[1]
    assert kwargs["model"] == "nvidia/nemotron-3.5-super-vl-120b-a12b"
    assert kwargs["stream"] is True


def test_nvidia_streaming_via_requests_sse_fallback() -> None:
    """Verify streaming token delivery via standard requests SSE fallback."""
    provider = NvidiaNIMProvider(api_key="nvapi-testkey")

    sse_lines = [
        b'data: {"choices": [{"delta": {"content": "Screen "}}]}\n\n',
        b'data: {"choices": [{"delta": {"content": "analysis "}}]}\n\n',
        b'data: {"choices": [{"delta": {"content": "done."}}]}\n\n',
        b'data: [DONE]\n\n',
    ]

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_lines.return_value = iter(sse_lines)
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None

    # Force fallback by mocking _get_openai_client to return None
    with patch.object(provider, "_get_openai_client", return_value=None):
        with patch("requests.post", return_value=mock_resp):
            stream = provider.chat_stream([{"role": "user", "content": "Inspect screen"}])
            tokens = list(stream)

    assert "".join(tokens) == "Screen analysis done."


def test_nvidia_rate_limit_429_sets_cooldown() -> None:
    """Verify 429 response triggers 60s cooldown and raises LLMRateLimitError."""
    provider = NvidiaNIMProvider(api_key="nvapi-testkey")

    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.text = "Too Many Requests (Rate Limit Exceeded)"
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None

    with patch.object(provider, "_get_openai_client", return_value=None):
        with patch("requests.post", return_value=mock_resp):
            with pytest.raises(LLMRateLimitError) as exc_info:
                list(provider.chat_stream([{"role": "user", "content": "Hi"}]))
            assert "429" in str(exc_info.value)
            assert provider.is_cooling_down is True


# ============================================================================
# 5. Advanced Multimodal Routing Layer Tests
# ============================================================================

def test_hybrid_router_multimodal_code_routes_to_nemotron_vl() -> None:
    """
    Verify that when image_bytes are present and prompt involves code/syntax debugging,
    router dynamically routes to NVIDIA NIM Nemotron 3.5 Super VL.
    """
    mock_gemini = MagicMock()
    mock_gemini.is_available.return_value = True
    mock_gemini.is_cooling_down = False

    mock_nvidia = MagicMock()
    mock_nvidia.is_available.return_value = True
    mock_nvidia.is_cooling_down = False
    mock_nvidia.vision_model = "nvidia/nemotron-3.5-super-vl-120b-a12b"
    mock_nvidia.chat_stream.return_value = iter(["Nemotron: ", "Syntax error on line 42."])

    mock_ollama = MagicMock()

    router = HybridLLMRouter(
        gemini_provider=mock_gemini,
        nvidia_provider=mock_nvidia,
        ollama_provider=mock_ollama,
        mode="auto",
    )

    dummy_image = b"\xff\xd8\xff\xe0"
    messages = [{"role": "user", "content": "Ye code screen dekho aur syntax error batao."}]

    tokens = list(router.chat_stream(messages, image_bytes=dummy_image))

    assert "".join(tokens) == "Nemotron: Syntax error on line 42."
    assert router.last_provider_used == "nvidia"
    assert router.last_model_used == "nvidia/nemotron-3.5-super-vl-120b-a12b"
    mock_nvidia.chat_stream.assert_called_once()
    mock_gemini.chat_stream.assert_not_called()


def test_hybrid_router_multimodal_nemotron_failure_falls_over_to_gemini() -> None:
    """Verify that if Nemotron VL fails, it gracefully falls over to Gemini Flash."""
    mock_gemini = MagicMock()
    mock_gemini.is_available.return_value = True
    mock_gemini.is_cooling_down = False
    mock_gemini.model_name = "gemini-2.5-flash"
    mock_gemini.chat_stream.return_value = iter(["Gemini visual fallback response."])

    mock_nvidia = MagicMock()
    mock_nvidia.is_available.return_value = True
    mock_nvidia.is_cooling_down = False
    mock_nvidia.vision_model = "nvidia/nemotron-3.5-super-vl-120b-a12b"
    mock_nvidia.chat_stream.side_effect = LLMConnectionError("NVIDIA NIM 503 Service Unavailable")

    fallback_events = []
    router = HybridLLMRouter(
        gemini_provider=mock_gemini,
        nvidia_provider=mock_nvidia,
        mode="auto",
        on_fallback=lambda f, t, r: fallback_events.append((f, t, r)),
    )

    dummy_image = b"\xff\xd8\xff\xe0"
    messages = [{"role": "user", "content": "Ye python traceback debug karo"}]

    tokens = list(router.chat_stream(messages, image_bytes=dummy_image))

    assert "".join(tokens) == "Gemini visual fallback response."
    assert router.last_provider_used == "gemini"
    assert len(fallback_events) == 1
    assert fallback_events[0][0] == "nvidia"
    assert fallback_events[0][1] == "gemini"


# ============================================================================
# 6. Resilient Multi-Tier Fallback Lifeline Tests
# ============================================================================

def test_multi_tier_fallback_gemini_to_nvidia_deepseek() -> None:
    """
    Tier 1 (Gemini 429 Rate Limit) -> Tier 2 (NVIDIA DeepSeek v4.1 Flash).
    Verifies that Gemini failure redirects to NVIDIA NIM before touching Ollama.
    """
    mock_gemini = MagicMock()
    mock_gemini.is_available.return_value = True
    mock_gemini.is_cooling_down = False
    mock_gemini.chat_stream.side_effect = LLMRateLimitError("429 Too Many Requests", provider="gemini")

    mock_nvidia = MagicMock()
    mock_nvidia.is_available.return_value = True
    mock_nvidia.is_cooling_down = False
    mock_nvidia.fallback_model = "deepseek-ai/deepseek-v4.1-flash"
    mock_nvidia.chat_stream.return_value = iter(["NVIDIA DeepSeek: Haan Abhi, main sun rahi hoon!"])

    mock_ollama = MagicMock()

    fallback_events = []
    router = HybridLLMRouter(
        gemini_provider=mock_gemini,
        nvidia_provider=mock_nvidia,
        ollama_provider=mock_ollama,
        mode="auto",
        on_fallback=lambda f, t, r: fallback_events.append((f, t, r)),
    )

    tokens = list(router.chat_stream([{"role": "user", "content": "Kaise ho Aarna?"}]))

    assert "".join(tokens) == "NVIDIA DeepSeek: Haan Abhi, main sun rahi hoon!"
    assert router.last_provider_used == "nvidia"
    assert router.last_model_used == "deepseek-ai/deepseek-v4.1-flash"
    assert len(fallback_events) == 1
    assert fallback_events[0][0] == "gemini"
    assert fallback_events[0][1] == "nvidia"
    assert "429" in fallback_events[0][2]
    mock_ollama.chat_stream.assert_not_called()


def test_multi_tier_fallback_gemini_and_nvidia_to_ollama() -> None:
    """
    Tier 1 (Gemini Error) -> Tier 2 (NVIDIA Error) -> Tier 3 (Ollama Local Backup).
    Verifies full graceful degradation all the way to local offline neural model.
    """
    mock_gemini = MagicMock()
    mock_gemini.is_available.return_value = True
    mock_gemini.is_cooling_down = False
    mock_gemini.chat_stream.side_effect = LLMConnectionError("Gemini network unreachable")

    mock_nvidia = MagicMock()
    mock_nvidia.is_available.return_value = True
    mock_nvidia.is_cooling_down = False
    mock_nvidia.fallback_model = "deepseek-ai/deepseek-v4.1-flash"
    mock_nvidia.chat_stream.side_effect = LLMRateLimitError("NVIDIA 429 quota reached")

    mock_ollama = MagicMock()
    mock_ollama.model_name = "qwen3.5:4b"
    mock_ollama.chat_stream.return_value = iter(["Ollama: Local offline response."])

    fallback_events = []
    router = HybridLLMRouter(
        gemini_provider=mock_gemini,
        nvidia_provider=mock_nvidia,
        ollama_provider=mock_ollama,
        mode="auto",
        on_fallback=lambda f, t, r: fallback_events.append((f, t, r)),
    )

    tokens = list(router.chat_stream([{"role": "user", "content": "Help me with this problem"}]))

    assert "".join(tokens) == "Ollama: Local offline response."
    assert router.last_provider_used == "ollama"
    assert len(fallback_events) == 2
    assert fallback_events[0] == ("gemini", "nvidia", "Gemini error: Gemini network unreachable")
    assert fallback_events[1][0] == "nvidia"
    assert fallback_events[1][1] == "ollama"
    mock_ollama.chat_stream.assert_called_once()


# ============================================================================
# 7. Safe Key-less Bypass Mechanism Tests
# ============================================================================

@pytest.mark.parametrize(
    "placeholder_key",
    [
        "",
        "none",
        "null",
        "your_key_here",
        "your_nvidia_api_key_here",
        "placeholder",
        "nvapi-your-key-here",
    ],
)
def test_nvidia_placeholder_key_triggers_safe_bypass(placeholder_key: str) -> None:
    """Verify that placeholder or empty NVIDIA keys do not crash initialization or probe."""
    provider = NvidiaNIMProvider(api_key=placeholder_key)
    assert provider.is_available() is False

    is_online, status_msg = provider.check_connection()
    assert is_online is False
    assert "Not configured" in status_msg or "bypass active" in status_msg


def test_openrouter_placeholder_key_in_config(monkeypatch) -> None:
    """Verify AppConfig safely treats placeholder OPENROUTER_API_KEY as empty string."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "your_openrouter_api_key_here")
    config = AppConfig.load()
    assert config.openrouter_api_key == ""


def test_router_keyless_bypass_routes_multimodal_to_gemini() -> None:
    """
    When NVIDIA keys are unconfigured (or placeholder), router must seamlessly route
    multimodal coding/reasoning queries directly to Tier 1 Gemini without error.
    """
    mock_gemini = MagicMock()
    mock_gemini.is_available.return_value = True
    mock_gemini.is_cooling_down = False
    mock_gemini.model_name = "gemini-2.5-flash"
    mock_gemini.chat_stream.return_value = iter(["Gemini handled vision query via keyless bypass."])

    keyless_nvidia = NvidiaNIMProvider(api_key="your_nvidia_api_key_here")
    assert keyless_nvidia.is_available() is False

    router = HybridLLMRouter(
        gemini_provider=mock_gemini,
        nvidia_provider=keyless_nvidia,
        mode="auto",
    )

    dummy_image = b"\xff\xd8\xff\xe0"
    messages = [{"role": "user", "content": "Ye python code syntax error dekho"}]

    tokens = list(router.chat_stream(messages, image_bytes=dummy_image))
    assert "".join(tokens) == "Gemini handled vision query via keyless bypass."
    assert router.last_provider_used == "gemini"


def test_router_keyless_bypass_gemini_failure_drops_straight_to_ollama() -> None:
    """
    When Gemini fails and NVIDIA is unconfigured/placeholder, router skips NVIDIA
    entirely and falls back directly to local offline Ollama.
    """
    mock_gemini = MagicMock()
    mock_gemini.is_available.return_value = True
    mock_gemini.is_cooling_down = False
    mock_gemini.chat_stream.side_effect = LLMRateLimitError("429 Gemini rate limit", provider="gemini")

    keyless_nvidia = NvidiaNIMProvider(api_key="")
    mock_ollama = MagicMock()
    mock_ollama.model_name = "qwen3.5:4b"
    mock_ollama.chat_stream.return_value = iter(["Ollama local fallback."])

    fallback_events = []
    router = HybridLLMRouter(
        gemini_provider=mock_gemini,
        nvidia_provider=keyless_nvidia,
        ollama_provider=mock_ollama,
        mode="auto",
        on_fallback=lambda f, t, r: fallback_events.append((f, t, r)),
    )

    tokens = list(router.chat_stream([{"role": "user", "content": "Hello"}]))
    assert "".join(tokens) == "Ollama local fallback."
    assert router.last_provider_used == "ollama"
    assert len(fallback_events) == 1
    assert fallback_events[0][0] == "gemini"
    assert fallback_events[0][1] == "ollama"
