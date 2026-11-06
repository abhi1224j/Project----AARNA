"""Unit tests for VoiceConfig validation and environment parsing."""

from pathlib import Path
import pytest

from src.voice.config import (
    TRANSCRIPTION_SAMPLE_RATE,
    VoiceConfig,
)


def test_voice_config_defaults():
    cfg = VoiceConfig()
    assert cfg.enabled is True
    assert cfg.engine == "faster_whisper"
    assert cfg.model_size == "small"
    assert cfg.language_mode == "auto"
    assert cfg.compute_device == "cpu"
    assert cfg.compute_type == "int8"
    assert cfg.target_sample_rate == TRANSCRIPTION_SAMPLE_RATE
    assert cfg.preferred_device_index is None
    assert cfg.vad_enabled is True
    assert cfg.vad_auto_stop_on_silence is True
    assert cfg.silence_timeout_seconds == 1.6
    assert cfg.auto_submit_transcript is False
    assert cfg.barge_in_enabled is False
    assert cfg.log_transcripts is False


def test_voice_config_compute_matrix_validation():
    # Valid CPU compute types
    VoiceConfig(compute_device="cpu", compute_type="int8")
    VoiceConfig(compute_device="cpu", compute_type="float32")
    VoiceConfig(compute_device="cpu", compute_type="int8_float32")

    # Invalid CPU compute type: float16 is not supported on CPU in CTranslate2
    with pytest.raises(ValueError, match="Unsupported compute type 'float16' for CPU"):
        VoiceConfig(compute_device="cpu", compute_type="float16")

    # Valid CUDA compute types
    VoiceConfig(compute_device="cuda", compute_type="float16")
    VoiceConfig(compute_device="cuda", compute_type="int8")

    # Invalid CUDA compute type
    with pytest.raises(ValueError, match="Unsupported compute type 'bogus' for CUDA"):
        VoiceConfig(compute_device="cuda", compute_type="bogus")


def test_voice_config_sample_rate_enforcement():
    # Only 16000 Hz is permitted for Whisper transcription
    with pytest.raises(ValueError, match="Whisper requires strictly 16000 Hz"):
        VoiceConfig(target_sample_rate=44100)

    with pytest.raises(ValueError, match="Whisper requires strictly 16000 Hz"):
        VoiceConfig(target_sample_rate=8000)


def test_voice_config_boundary_validation():
    # Negative device index
    with pytest.raises(ValueError, match="preferred_device_index must be >= 0"):
        VoiceConfig(preferred_device_index=-1)

    # Valid device index
    cfg = VoiceConfig(preferred_device_index=0)
    assert cfg.preferred_device_index == 0

    # Chunk duration boundaries
    with pytest.raises(ValueError, match="chunk_duration_ms"):
        VoiceConfig(chunk_duration_ms=5)

    with pytest.raises(ValueError, match="chunk_duration_ms"):
        VoiceConfig(chunk_duration_ms=2000)

    # Beam size boundaries
    with pytest.raises(ValueError, match="beam_size"):
        VoiceConfig(beam_size=0)

    with pytest.raises(ValueError, match="beam_size"):
        VoiceConfig(beam_size=15)

    # VAD threshold boundaries
    with pytest.raises(ValueError, match="vad_energy_threshold_db"):
        VoiceConfig(vad_energy_threshold_db=10.0)

    with pytest.raises(ValueError, match="vad_energy_threshold_db"):
        VoiceConfig(vad_energy_threshold_db=-95.0)

    # Silence timeout boundaries
    with pytest.raises(ValueError, match="silence_timeout_seconds"):
        VoiceConfig(silence_timeout_seconds=0.1)

    # Engine validation (mock engine is removed)
    with pytest.raises(ValueError, match="Unsupported voice engine"):
        VoiceConfig(engine="mock")


def test_voice_config_immutability():
    cfg = VoiceConfig()
    with pytest.raises(Exception):
        cfg.model_size = "tiny"  # type: ignore

    updated = cfg.with_updates(model_size="tiny", compute_device="cpu", compute_type="int8")
    assert updated.model_size == "tiny"
    assert cfg.model_size == "small"


def test_voice_config_load_from_env(monkeypatch):
    monkeypatch.setenv("VOICE_INPUT_ENABLED", "false")
    monkeypatch.setenv("VOICE_MODEL_SIZE", "small")
    monkeypatch.setenv("VOICE_LANGUAGE_MODE", "hi")
    monkeypatch.setenv("VOICE_COMPUTE_DEVICE", "cpu")
    monkeypatch.setenv("VOICE_COMPUTE_TYPE", "int8")
    monkeypatch.setenv("VOICE_DEVICE_INDEX", "2")
    monkeypatch.setenv("VOICE_BEAM_SIZE", "5")
    monkeypatch.setenv("VOICE_MAX_RECORDING_SECONDS", "45.0")
    monkeypatch.setenv("VOICE_SILENCE_TIMEOUT_SECONDS", "3.0")

    cfg = VoiceConfig.load_from_env()
    assert cfg.enabled is False
    assert cfg.model_size == "small"
    assert cfg.language_mode == "hi"
    assert cfg.compute_device == "cpu"
    assert cfg.compute_type == "int8"
    assert cfg.preferred_device_index == 2
    assert cfg.beam_size == 5
    assert cfg.max_recording_duration_seconds == 45.0
    assert cfg.silence_timeout_seconds == 3.0
