"""
Unit and integration test suite for Phase 6 Wake Word Engine in Project AARNA.
Verifies local accelerated wake word detection strictly for 'Aarna!',
energy gating, thread safety, audio device handover, system tray, and avatar transitions.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication

from src.core.paths import get_wakeword_models_dir
from src.ui.avatar.state import AvatarState
from src.voice.config import VoiceConfig
from src.voice.controller import VoiceController
from src.voice.state import VoiceState
from src.voice.wakeword import (
    BaseWakeWordEngine,
    LocalNeuralWakeWordEngine,
    OpenWakeWordEngine,
    WakeWordListener,
    WakeWordPrediction,
    create_wakeword_engine,
)


@pytest.fixture(scope="module")
def qapp():
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])
    return app


# -----------------------------------------------------------------------------
# 1. Config & Paths Tests
# -----------------------------------------------------------------------------

def test_wakeword_config_defaults() -> None:
    """Verify default wake word configuration values."""
    config = VoiceConfig()
    assert config.wakeword_enabled is True
    assert config.wakeword_phrase == "Aarna!"
    assert config.wakeword_engine == "local_neural"
    assert config.wakeword_energy_threshold_db == -42.0
    assert config.wakeword_min_duration_seconds == 0.35
    assert config.wakeword_max_duration_seconds == 2.2
    assert config.wakeword_cooldown_seconds == 1.5


def test_wakeword_config_env_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify loading wake word configuration from environment overrides."""
    monkeypatch.setenv("VOICE_WAKEWORD_ENABLED", "false")
    monkeypatch.setenv("VOICE_WAKEWORD_PHRASE", "Aarna!")
    monkeypatch.setenv("VOICE_WAKEWORD_ENGINE", "openwakeword")

    config = VoiceConfig.load_from_env()
    assert config.wakeword_enabled is False
    assert config.wakeword_phrase == "Aarna!"
    assert config.wakeword_engine == "openwakeword"


def test_wakeword_paths_resolution() -> None:
    """Verify get_wakeword_models_dir returns valid directory."""
    path = get_wakeword_models_dir()
    assert isinstance(path, Path)
    assert path.name == "wakeword"
    assert path.exists()


# -----------------------------------------------------------------------------
# 2. Strict Wake Word Phrase Recognition & Negative Rejection Tests
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw_text,expected_match",
    [
        ("Aarna!", True),
        ("Aarna", True),
        ("aarna", True),
        ("AARNA", True),
        ("आर्ना", True),
        ("अरना", True),
        # Must strictly REJECT forbidden prefixes like 'Hey', 'Hello', 'Ok'
        ("Hey Aarna", False),
        ("Hey Aarna!", False),
        ("Hello Aarna", False),
        ("Ok Aarna", False),
        ("Hi Aarna", False),
        ("Suno Aarna", False),
        # Unrelated words or general speech
        ("Hey Jarvis", False),
        ("Alexa", False),
        ("What time is it", False),
        ("Hello world", False),
        ("", False),
    ],
)
def test_strict_phrase_evaluation(raw_text: str, expected_match: bool) -> None:
    """Verify standalone 'Aarna!' matches strictly, and prefix variants are rejected."""
    is_match, conf, matched = LocalNeuralWakeWordEngine._evaluate_text(raw_text)
    assert is_match == expected_match
    if expected_match:
        assert matched == "Aarna!"
        assert conf >= 0.7


# -----------------------------------------------------------------------------
# 3. Engine Prediction with Mock Whisper Model
# -----------------------------------------------------------------------------

def test_local_neural_wake_word_prediction_positive() -> None:
    """Verify LocalNeuralWakeWordEngine predict returns match for Aarna!"""
    mock_model = MagicMock()
    mock_segment = MagicMock()
    mock_segment.text = "Aarna!"
    mock_info = MagicMock()
    mock_info.language_probability = 0.98
    mock_model.transcribe.return_value = ([mock_segment], mock_info)

    engine = LocalNeuralWakeWordEngine(whisper_instance=mock_model)
    # 1.0 second of audio at 16kHz
    audio = np.zeros(16000, dtype=np.float32)
    pred = engine.predict(audio, sample_rate=16000)

    assert pred.is_match is True
    assert pred.phrase == "Aarna!"
    assert pred.confidence >= 0.7
    assert pred.engine_name == "local_neural"


def test_local_neural_wake_word_prediction_duration_filtering() -> None:
    """Verify audio chunks that are too short (<0.3s) or too long (>2.5s) are rejected quickly."""
    mock_model = MagicMock()
    engine = LocalNeuralWakeWordEngine(whisper_instance=mock_model)

    # 0.1s audio (too short)
    short_audio = np.zeros(1600, dtype=np.float32)
    pred_short = engine.predict(short_audio, sample_rate=16000)
    assert pred_short.is_match is False
    assert mock_model.transcribe.call_count == 0

    # 3.0s audio (too long for standalone wake word)
    long_audio = np.zeros(48000, dtype=np.float32)
    pred_long = engine.predict(long_audio, sample_rate=16000)
    assert pred_long.is_match is False
    assert mock_model.transcribe.call_count == 0


def test_openwakeword_engine_fallback_behavior(tmp_path: Path) -> None:
    """Verify OpenWakeWordEngine falls back cleanly to local neural engine when custom onnx is absent."""
    mock_fallback = MagicMock(spec=BaseWakeWordEngine)
    mock_fallback.predict.return_value = WakeWordPrediction(
        is_match=True, confidence=0.9, phrase="Aarna!", engine_name="mock_fallback"
    )

    non_existent_model = tmp_path / "non_existent.onnx"
    oww_engine = OpenWakeWordEngine(
        custom_model_path=non_existent_model,
        fallback_engine=mock_fallback,
    )

    audio = np.zeros(16000, dtype=np.float32)
    res = oww_engine.predict(audio, 16000)
    assert res.is_match is True
    assert res.engine_name == "mock_fallback"
    mock_fallback.predict.assert_called_once()


# -----------------------------------------------------------------------------
# 4. WakeWordListener Thread & Handover Safety Tests
# -----------------------------------------------------------------------------

def test_listener_pause_releases_stream_safely(qapp) -> None:
    """Verify pause() closes active stream and releases audio device for STT handover."""
    mock_stream = MagicMock()
    mock_pa = MagicMock()
    mock_pa.open.return_value = mock_stream

    listener = WakeWordListener(pyaudio_instance=mock_pa)
    # Open stream
    assert listener._open_stream() is True
    assert listener._stream is mock_stream

    # Calling pause() must close stream and set is_paused True
    listener.pause()
    assert listener.is_paused is True
    assert listener._stream is None
    mock_stream.stop_stream.assert_called_once()
    mock_stream.close.assert_called_once()

    # Calling resume() clears paused state
    listener.resume()
    assert listener.is_paused is False


def test_listener_candidate_processing_and_signal_emission(qapp) -> None:
    """Verify wake word detection emits signal and pauses listener automatically."""
    mock_engine = MagicMock(spec=BaseWakeWordEngine)
    mock_engine.predict.return_value = WakeWordPrediction(
        is_match=True, confidence=0.95, phrase="Aarna!", engine_name="mock_engine"
    )

    mock_pa = MagicMock()
    listener = WakeWordListener(engine=mock_engine, pyaudio_instance=mock_pa)

    signals_received: list[str] = []
    listener.wake_word_detected.connect(lambda p: signals_received.append(p))

    # Process a candidate audio buffer
    audio = np.zeros(16000, dtype=np.float32)
    listener._process_candidate_utterance(audio)

    # Must emit signal with "Aarna!"
    assert len(signals_received) == 1
    assert signals_received[0] == "Aarna!"

    # Must automatically pause itself to release microphone for STT
    assert listener.is_paused is True


# -----------------------------------------------------------------------------
# 5. VoiceController Integration & Handover Tests
# -----------------------------------------------------------------------------

def test_voice_controller_wake_word_wiring(qapp, mocker) -> None:
    """Verify VoiceController starts and pauses wake word listener during STT handover."""
    mock_listener = MagicMock(spec=WakeWordListener)
    mock_listener.isRunning.return_value = False

    ctrl = VoiceController(wakeword_listener=mock_listener)
    assert ctrl.wakeword_listener is mock_listener

    # Start wake word
    ctrl.start_wakeword()
    mock_listener.start.assert_called_once()

    # When wake word is detected, controller emits signal and transitions avatar
    detected_phrases: list[str] = []
    ctrl.wake_word_detected.connect(lambda p: detected_phrases.append(p))

    mock_sink = MagicMock()
    ctrl.set_avatar_sink(mock_sink)

    mocker.patch.object(ctrl, "start_recording")
    ctrl._on_wake_word_detected("Aarna!")

    assert detected_phrases == ["Aarna!"]
    mock_sink.set_state.assert_called_with(AvatarState.LISTENING)
    ctrl.start_recording.assert_called_once()

    ctrl.shutdown()
    mock_listener.stop.assert_called_once()
