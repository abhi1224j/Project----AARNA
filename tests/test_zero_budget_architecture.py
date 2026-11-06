"""
Automated unit tests for AARNA Zero-Budget Architecture Governance (Milestone H1).
Enforces the 12 mandatory rules: strict ₹0 budget, local-first default,
Piper TTS evaluation default, and verified-free cloud gating.
"""

import os
import pytest
from pathlib import Path

from src.core.config import AppConfig
from src.voice.config import VoiceConfig


def test_zero_budget_defaults():
    """Verify that zero-budget governance invariants are active by default."""
    config = AppConfig()
    assert config.zero_budget_enforced is True
    assert config.allow_cloud_services is False
    assert config.require_zero_cost is True
    assert config.auto_local_fallback is True
    assert config.tts_engine == "piper"


def test_zero_budget_env_load(monkeypatch):
    """Verify that loading config from env preserves zero-budget constraints."""
    monkeypatch.setenv("ALLOW_CLOUD_SERVICES", "true")
    monkeypatch.setenv("TTS_ENGINE", "piper")

    config = AppConfig.load()
    assert config.zero_budget_enforced is True
    assert config.allow_cloud_services is True
    assert config.require_zero_cost is True
    assert config.auto_local_fallback is True
    assert config.tts_engine == "piper"


def test_voice_subsystem_retains_faster_whisper_offline():
    """Verify that voice subsystem continues using Faster-Whisper offline."""
    v_cfg = VoiceConfig()
    assert v_cfg.engine == "faster_whisper"
    assert v_cfg.model_size in ("tiny", "base", "small")
    assert v_cfg.compute_device == "cpu"
    assert v_cfg.compute_type == "int8"
    assert "whisper" in str(v_cfg.models_dir).lower()


def test_zero_budget_architecture_document_exists():
    """Verify that the formal H1 Zero-Budget Architecture document is published."""
    doc_path = Path("docs/ZERO_BUDGET_HYBRID_ARCHITECTURE.md")
    assert doc_path.is_file()
    content = doc_path.read_text(encoding="utf-8")
    assert "₹0 Lifetime Budget Charter" in content
    assert "Piper TTS" in content
    assert "LiveKit" in content
    assert "H1–H6" in content or "H1-H6" in content
