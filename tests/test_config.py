"""Tests for application configuration."""

import os
import pytest
from src.core.config import AppConfig


def test_default_config():
    """Verify default configuration values including AARNA wake-word decisions and H2 defaults."""
    config = AppConfig()
    assert config.app_name == "AARNA"
    assert config.app_version == "0.1.0"
    assert config.wake_word == "Aarna"
    assert config.wake_word_activation_phrase == "Aarna!"
    assert config.require_wake_word_prefix is False
    assert config.ollama_base_url == "http://localhost:11434"
    assert config.default_model == "qwen3.5:4b"
    assert config.temperature == 0.7
    assert config.request_timeout == 60.0
    assert config.enable_thinking is False
    assert config.max_context_messages == 20
    assert config.tts_engine == "piper"
    assert config.tts_voice == "en_US-lessac-medium"
    assert config.tts_sample_rate == 22050
    assert config.tts_use_gpu is False
    assert config.voice_auto_submit is True
    assert config.voice_barge_in is True


def test_load_from_environment(monkeypatch):
    """Verify configuration overrides from environment variables."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://192.168.1.100:11434/")
    monkeypatch.setenv("DEFAULT_MODEL", "qwen2.5:3b")
    monkeypatch.setenv("TEMPERATURE", "0.2")
    monkeypatch.setenv("REQUEST_TIMEOUT", "120")
    monkeypatch.setenv("MAX_CONTEXT_MESSAGES", "35")
    monkeypatch.setenv("WAKE_WORD", "Aarna")
    monkeypatch.setenv("WAKE_WORD_ACTIVATION_PHRASE", "Aarna!")
    monkeypatch.setenv("REQUIRE_WAKE_WORD_PREFIX", "false")
    monkeypatch.setenv("TTS_ENGINE", "piper")
    monkeypatch.setenv("TTS_VOICE", "en_US-amy-medium")
    monkeypatch.setenv("TTS_SPEED", "1.1")
    monkeypatch.setenv("TTS_VOLUME", "0.9")
    monkeypatch.setenv("ENABLE_THINKING", "true")

    config = AppConfig.load()
    assert config.ollama_base_url == "http://192.168.1.100:11434"
    assert config.default_model == "qwen2.5:3b"
    assert config.temperature == 0.2
    assert config.request_timeout == 120.0
    assert config.max_context_messages == 35
    assert config.wake_word == "Aarna"
    assert config.wake_word_activation_phrase == "Aarna!"
    assert config.require_wake_word_prefix is False
    assert config.tts_engine == "piper"
    assert config.tts_voice == "en_US-amy-medium"
    assert config.tts_speed == 1.1
    assert config.tts_volume == 0.9
    assert config.enable_thinking is True


def test_invalid_environment_fallbacks(monkeypatch):
    """Verify safe fallbacks when environment variables contain invalid values."""
    monkeypatch.setenv("TEMPERATURE", "not-a-float")
    monkeypatch.setenv("REQUEST_TIMEOUT", "invalid")
    monkeypatch.setenv("MAX_CONTEXT_MESSAGES", "bad-int")
    monkeypatch.setenv("TTS_SAMPLE_RATE", "not-an-int")
    monkeypatch.setenv("TTS_SPEED", "not-a-float")

    config = AppConfig.load()
    assert config.temperature == 0.7
    assert config.request_timeout == 60.0
    assert config.enable_thinking is False
    assert config.max_context_messages == 20
    assert config.tts_sample_rate == 22050
    assert config.tts_speed == 1.0
