"""Unit tests for domain models and data structures."""

from pathlib import Path
import numpy as np
import pytest

from src.voice.models import (
    AudioDeviceInfo,
    ModelAvailabilityState,
    ModelStatusInfo,
    ProcessedAudio,
    RawAudioBuffer,
    TranscriptionResult,
    TranscriptionSegment,
)


def test_audio_device_info_formatting():
    # Float sample rate (typical PortAudio return)
    dev_float = AudioDeviceInfo(
        index=1,
        name="Microphone Array",
        host_api_name="MME",
        max_input_channels=2,
        default_samplerate=44100.0,
        is_default=True,
    )
    assert dev_float.formatted_sample_rate == "44.1 kHz"
    assert dev_float.display_label == "[Default] Microphone Array (44.1 kHz)"
    # Must encode cleanly to ASCII and cp1252 without UnicodeEncodeError
    dev_float.display_label.encode("cp1252")

    # Integer kHz rate
    dev_16k = AudioDeviceInfo(
        index=2,
        name="USB Mic",
        host_api_name="WASAPI",
        max_input_channels=1,
        default_samplerate=16000.0,
        is_default=False,
    )
    assert dev_16k.formatted_sample_rate == "16 kHz"
    assert dev_16k.display_label == "USB Mic (16 kHz)"

    # Invalid rate must raise ValueError
    with pytest.raises(ValueError, match="default_samplerate must be positive"):
        AudioDeviceInfo(
            index=3,
            name="Bad Mic",
            host_api_name="MME",
            max_input_channels=1,
            default_samplerate=0.0,
        )


def test_raw_audio_buffer_best_effort_clear():
    # Writable array
    data = np.ones(1600, dtype=np.float32)
    buf = RawAudioBuffer(
        samples=data,
        sample_rate=16000,
        channels=1,
        duration_seconds=0.1,
    )
    assert buf.num_frames == 1600
    buf.clear_sensitive_memory()
    assert buf.num_frames == 0
    assert buf.duration_seconds == 0.0
    # Original array was zeroed in-place
    assert np.all(data == 0.0)

    # Read-only array should not crash clear_sensitive_memory
    ro_data = np.ones(800, dtype=np.float32)
    ro_data.setflags(write=False)
    buf_ro = RawAudioBuffer(
        samples=ro_data,
        sample_rate=16000,
        channels=1,
        duration_seconds=0.05,
    )
    buf_ro.clear_sensitive_memory()
    assert buf_ro.num_frames == 0
    assert buf_ro.duration_seconds == 0.0


def test_processed_audio_clear():
    data = np.ones(3200, dtype=np.float32)
    from src.voice.models import VadAnalysisResult, VadSpeechSegment
    vad = VadAnalysisResult(
        has_speech=True,
        total_duration_seconds=0.2,
        speech_duration_seconds=0.15,
        leading_silence_seconds=0.02,
        trailing_silence_seconds=0.03,
        peak_rms_db=-12.0,
        segments=[VadSpeechSegment(0.02, 0.17)],
    )
    proc = ProcessedAudio(
        samples=data,
        sample_rate=16000,
        duration_seconds=0.2,
        original_sample_rate=44100,
        original_channels=2,
        peak_amplitude=1.0,
        rms_db=-10.0,
        was_resampled=True,
        was_clipped=False,
        vad_result=vad,
    )
    assert proc.vad_result is not None
    proc.clear_sensitive_memory()
    assert proc.samples.size == 0
    assert proc.duration_seconds == 0.0
    assert proc.vad_result is None
    assert np.all(data == 0.0)


def test_transcription_result_caller_dict_isolation():
    caller_dict = {"vad_trimmed": True, "beam_fallbacks": 0}
    res = TranscriptionResult(
        text="Hello Aarna",
        detected_language="en",
        language_probability=0.98,
        processing_duration_seconds=0.5,
        audio_duration_seconds=1.0,
        model_identifier="base",
        _diagnostics=caller_dict,
    )
    # Mutating caller dictionary after construction must not mutate res.diagnostics
    caller_dict["vad_trimmed"] = False
    caller_dict["injected"] = "tampered"
    assert res.diagnostics["vad_trimmed"] is True
    assert "injected" not in res.diagnostics


def test_transcription_result_immutability():
    seg = TranscriptionSegment(start_seconds=0.0, end_seconds=1.0, text="Hello Aarna")
    res = TranscriptionResult(
        text="Hello Aarna",
        detected_language="en",
        language_probability=0.98,
        processing_duration_seconds=0.5,
        audio_duration_seconds=1.0,
        model_identifier="base",
        segments=(seg,),
        _diagnostics={"vad_trimmed": True},
    )

    assert res.is_empty is False
    assert res.realtime_factor == 0.5
    assert res.diagnostics["vad_trimmed"] is True

    # Attempting to mutate diagnostics mapping must fail
    with pytest.raises((TypeError, Exception)):
        res.diagnostics["new_key"] = "illegal"  # type: ignore

    # Attempting to mutate dataclass fields must fail
    with pytest.raises(Exception):
        res.text = "Tampered"  # type: ignore


def test_model_status_info():
    info = ModelStatusInfo(
        model_size="base",
        repo_id="Systran/faster-whisper-base",
        state=ModelAvailabilityState.READY,
        local_dir=Path("models/whisper/base"),
        size_bytes=145 * 1024 * 1024,
    )
    assert info.is_ready is True
    assert info.size_mb == 145.0
