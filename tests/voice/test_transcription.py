"""
Unit tests for FasterWhisperTranscriptionEngine (Phase C5).
Tests input validation, duration limits, cooperative cancellation,
language modes (auto, en, hi), task="transcribe", structured result formatting,
and error handling.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest

from src.voice.config import VoiceConfig
from src.voice.exceptions import (
    EmptyRecordingError,
    RecordingTooLongError,
    RecordingTooShortError,
    SpeechModelError,
    TranscriptionError,
    VoiceCancelledError,
)
from src.voice.models import ModelAvailabilityState, ModelStatusInfo, ProcessedAudio
from src.voice.transcription import FasterWhisperTranscriptionEngine


def _make_dummy_processed_audio(duration_s: float = 2.0, sample_rate: int = 16000) -> ProcessedAudio:
    num_samples = int(duration_s * sample_rate)
    samples = np.zeros(num_samples, dtype=np.float32)
    return ProcessedAudio(
        samples=samples,
        sample_rate=sample_rate,
        duration_seconds=duration_s,
        original_sample_rate=sample_rate,
        original_channels=1,
        peak_amplitude=0.2,
        rms_db=-30.0,
        was_resampled=False,
        was_clipped=False,
    )


class DummyWhisperSegment:
    def __init__(self, start: float, end: float, text: str, avg_logprob: float = -0.2, no_speech_prob: float = 0.01):
        self.start = start
        self.end = end
        self.text = text
        self.avg_logprob = avg_logprob
        self.no_speech_prob = no_speech_prob


class DummyTranscriptionInfo:
    def __init__(self, language: str = "en", language_probability: float = 0.98):
        self.language = language
        self.language_probability = language_probability
        self.all_language_probs = [("en", 0.98), ("hi", 0.01)]
        self.compression_ratio_threshold = 2.4


def test_transcribe_rejects_empty_audio():
    engine = FasterWhisperTranscriptionEngine()
    empty_audio = ProcessedAudio(
        samples=np.array([], dtype=np.float32),
        sample_rate=16000,
        duration_seconds=0.0,
        original_sample_rate=16000,
        original_channels=1,
        peak_amplitude=0.0,
        rms_db=-100.0,
        was_resampled=False,
        was_clipped=False,
    )
    with pytest.raises(EmptyRecordingError):
        engine.transcribe(empty_audio)


def test_transcribe_rejects_audio_too_short():
    cfg = VoiceConfig(min_recording_duration_seconds=0.5)
    engine = FasterWhisperTranscriptionEngine(config=cfg)
    short_audio = _make_dummy_processed_audio(duration_s=0.2)
    with pytest.raises(RecordingTooShortError):
        engine.transcribe(short_audio)


def test_transcribe_rejects_audio_too_long():
    cfg = VoiceConfig(max_recording_duration_seconds=10.0)
    engine = FasterWhisperTranscriptionEngine(config=cfg)
    long_audio = _make_dummy_processed_audio(duration_s=15.0)
    with pytest.raises(RecordingTooLongError):
        engine.transcribe(long_audio)


def test_transcribe_cancellation_before_inference():
    engine = FasterWhisperTranscriptionEngine()
    audio = _make_dummy_processed_audio(duration_s=2.0)
    with pytest.raises(VoiceCancelledError):
        engine.transcribe(audio, cancel_check=lambda: True)


def test_transcribe_cancellation_during_segment_loop():
    mock_mgr = MagicMock()
    mock_mgr.is_loaded = True
    mock_model = MagicMock()
    mock_mgr.active_model = mock_model

    def segment_generator():
        yield DummyWhisperSegment(0.0, 1.0, "Hello")
        # Trigger cancellation after first segment
        nonlocal cancelled
        cancelled = True
        yield DummyWhisperSegment(1.0, 2.0, "world")

    cancelled = False
    mock_model.transcribe.return_value = (segment_generator(), DummyTranscriptionInfo())

    engine = FasterWhisperTranscriptionEngine(model_manager=mock_mgr)
    audio = _make_dummy_processed_audio(duration_s=2.0)

    with pytest.raises(VoiceCancelledError, match="cancelled during token decoding"):
        engine.transcribe(audio, cancel_check=lambda: cancelled)


def test_transcribe_lazy_loading_success(tmp_path: Path):
    cfg = VoiceConfig(lazy_load_model=True, model_size="tiny")
    mock_mgr = MagicMock()
    mock_mgr.is_loaded = False

    def fake_load(*args, **kwargs):
        mock_mgr.is_loaded = True
        mock_mgr.active_model = MagicMock()
        mock_mgr.active_model.transcribe.return_value = (
            [DummyWhisperSegment(0.0, 1.5, "Testing lazy load")],
            DummyTranscriptionInfo("en", 0.99),
        )

    mock_mgr.load_model.side_effect = fake_load
    mock_mgr.get_model_status.return_value = ModelStatusInfo(
        model_size="tiny",
        repo_id="Systran/faster-whisper-tiny",
        state=ModelAvailabilityState.READY,
        local_dir=tmp_path / "tiny",
        compute_device="cpu",
        compute_type="int8",
    )

    engine = FasterWhisperTranscriptionEngine(config=cfg, model_manager=mock_mgr)
    audio = _make_dummy_processed_audio(duration_s=2.0)

    result = engine.transcribe(audio)
    assert mock_mgr.load_model.called
    assert result.text == "Testing lazy load"
    assert result.detected_language == "en"


def test_transcribe_not_loaded_and_lazy_load_disabled_raises():
    cfg = VoiceConfig(lazy_load_model=False, model_size="tiny")
    mock_mgr = MagicMock()
    mock_mgr.is_loaded = False

    engine = FasterWhisperTranscriptionEngine(config=cfg, model_manager=mock_mgr)
    audio = _make_dummy_processed_audio(duration_s=2.0)

    with pytest.raises(SpeechModelError, match="not resident in memory"):
        engine.transcribe(audio)


@pytest.mark.parametrize("requested_lang,expected_whisper_lang", [
    ("auto", None),
    ("en", "en"),
    ("hi", "hi"),
    ("unknown_lang", None),
])
def test_transcribe_language_modes_and_task(tmp_path: Path, requested_lang, expected_whisper_lang):
    mock_mgr = MagicMock()
    mock_mgr.is_loaded = True
    mock_model = MagicMock()
    mock_mgr.active_model = mock_model
    mock_mgr.get_model_status.return_value = ModelStatusInfo(
        model_size="tiny",
        repo_id="Systran/faster-whisper-tiny",
        state=ModelAvailabilityState.READY,
        local_dir=tmp_path / "tiny",
        compute_device="cpu",
        compute_type="int8",
    )

    mock_model.transcribe.return_value = (
        [DummyWhisperSegment(0.0, 1.5, "Namaste")],
        DummyTranscriptionInfo("hi", 0.95),
    )

    engine = FasterWhisperTranscriptionEngine(model_manager=mock_mgr)
    audio = _make_dummy_processed_audio(duration_s=2.0)

    res = engine.transcribe(audio, language_mode=requested_lang)
    assert res.text == "Namaste"
    assert res.detected_language == "hi"

    # Verify model.transcribe arguments: strictly transcribe, never translate
    call_kwargs = mock_model.transcribe.call_args.kwargs
    assert call_kwargs["language"] == expected_whisper_lang
    assert call_kwargs["task"] == "transcribe"
    assert call_kwargs["vad_filter"] is True
    assert call_kwargs["repetition_penalty"] == 1.2
    assert call_kwargs["no_repeat_ngram_size"] == 3
    assert call_kwargs["condition_on_previous_text"] is False


def test_transcribe_inference_failure_raises_transcription_error():
    mock_mgr = MagicMock()
    mock_mgr.is_loaded = True
    mock_model = MagicMock()
    mock_mgr.active_model = mock_model

    mock_model.transcribe.side_effect = RuntimeError("CUDA/CTranslate2 internal decoding error")

    engine = FasterWhisperTranscriptionEngine(model_manager=mock_mgr)
    audio = _make_dummy_processed_audio(duration_s=2.0)

    with pytest.raises(TranscriptionError, match="Faster-Whisper inference failure"):
        engine.transcribe(audio)


def test_transcribe_rejects_hallucinations_and_repeated_digits(tmp_path: Path):
    """Verify degenerate loops like '1 1 1' or subtitle artifacts produce empty text."""
    mock_mgr = MagicMock()
    mock_mgr.is_loaded = True
    mock_model = MagicMock()
    mock_mgr.active_model = mock_model
    mock_mgr.get_model_status.return_value = ModelStatusInfo(
        model_size="small",
        repo_id="Systran/faster-whisper-small",
        state=ModelAvailabilityState.READY,
        local_dir=tmp_path / "small",
        compute_device="cpu",
        compute_type="int8",
    )

    engine = FasterWhisperTranscriptionEngine(model_manager=mock_mgr)
    audio = _make_dummy_processed_audio(duration_s=2.0)

    # 1. Repeated digit hallucination: '1 1 1'
    mock_model.transcribe.return_value = (
        [DummyWhisperSegment(0.0, 1.5, "1 1 1")],
        DummyTranscriptionInfo("en", 0.5),
    )
    res = engine.transcribe(audio)
    assert res.text == ""

    # 2. Known silence artifact: 'Thank you for watching'
    mock_model.transcribe.return_value = (
        [DummyWhisperSegment(0.0, 1.5, "Thank you for watching.")],
        DummyTranscriptionInfo("en", 0.4),
    )
    res2 = engine.transcribe(audio)
    assert res2.text == ""


def test_transcribe_filters_no_speech_and_low_confidence_segments(tmp_path: Path):
    """Verify segments with high no_speech_prob or low avg_logprob are excluded."""
    mock_mgr = MagicMock()
    mock_mgr.is_loaded = True
    mock_model = MagicMock()
    mock_mgr.active_model = mock_model
    mock_mgr.get_model_status.return_value = ModelStatusInfo(
        model_size="small",
        repo_id="Systran/faster-whisper-small",
        state=ModelAvailabilityState.READY,
        local_dir=tmp_path / "small",
        compute_device="cpu",
        compute_type="int8",
    )

    engine = FasterWhisperTranscriptionEngine(model_manager=mock_mgr)
    audio = _make_dummy_processed_audio(duration_s=3.0)

    # First segment is genuine speech, second segment is pure silence noise
    mock_model.transcribe.return_value = (
        [
            DummyWhisperSegment(0.0, 1.0, "Hello Aarna", avg_logprob=-0.2, no_speech_prob=0.01),
            DummyWhisperSegment(1.0, 2.5, "1 2 3", avg_logprob=-1.4, no_speech_prob=0.85),
        ],
        DummyTranscriptionInfo("en", 0.95),
    )
    res = engine.transcribe(audio)
    assert res.text == "Hello Aarna"
