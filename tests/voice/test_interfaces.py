"""Unit tests for abstract interface contracts in src.voice.interfaces."""

from typing import Callable, Optional
import numpy as np
import pytest

from src.voice.interfaces import AudioCaptureBackend, TranscriptionBackend
from src.voice.models import (
    AudioDeviceInfo,
    ModelAvailabilityState,
    ModelStatusInfo,
    ProcessedAudio,
    RawAudioBuffer,
    TranscriptionResult,
)


def test_audio_capture_backend_contract():
    # Abstract class cannot be instantiated directly
    with pytest.raises(TypeError):
        AudioCaptureBackend()  # type: ignore

    # Concrete test fake implementing the contract
    class FakeAudioCapture(AudioCaptureBackend):
        def __init__(self):
            self._recording = False

        def list_input_devices(self) -> list[AudioDeviceInfo]:
            return [
                AudioDeviceInfo(0, "Mock Mic", "MME", 1, 16000.0, True),
            ]

        def open_stream(self, device_index=None, sample_rate=16000, channels=1) -> AudioDeviceInfo:
            return self.list_input_devices()[0]

        def start_stream(self) -> None:
            self._recording = True

        def stop_stream(self) -> RawAudioBuffer:
            self._recording = False
            return RawAudioBuffer(
                samples=np.zeros(1600, dtype=np.float32),
                sample_rate=16000,
                channels=1,
                duration_seconds=0.1,
            )

        def cancel_stream(self) -> None:
            self._recording = False

        def close(self) -> None:
            self._recording = False

        def get_current_peak_level(self) -> float:
            return 0.25

        @property
        def is_recording(self) -> bool:
            return self._recording

    fake = FakeAudioCapture()
    devs = fake.list_input_devices()
    assert len(devs) == 1
    assert devs[0].name == "Mock Mic"

    fake.open_stream(0)
    fake.start_stream()
    assert fake.is_recording is True
    assert fake.get_current_peak_level() == 0.25

    buf = fake.stop_stream()
    assert buf.num_frames == 1600
    assert fake.is_recording is False

    fake.close()


def test_transcription_backend_contract():
    # Abstract class cannot be instantiated directly
    with pytest.raises(TypeError):
        TranscriptionBackend()  # type: ignore

    # Concrete test fake implementing the contract
    class FakeTranscriptionEngine(TranscriptionBackend):
        def __init__(self):
            self._loaded = False

        def get_model_status(self) -> ModelStatusInfo:
            return ModelStatusInfo(
                model_size="base",
                repo_id="Systran/faster-whisper-base",
                state=ModelAvailabilityState.READY if self._loaded else ModelAvailabilityState.INSTALLED_NOT_LOADED,
                local_dir=None,  # type: ignore
            )

        def load_model(self) -> ModelStatusInfo:
            self._loaded = True
            return self.get_model_status()

        def unload_model(self) -> None:
            self._loaded = False

        @property
        def is_loaded(self) -> bool:
            return self._loaded

        def transcribe(
            self,
            audio: ProcessedAudio,
            language_mode: str = "auto",
            cancel_check: Optional[Callable[[], bool]] = None,
        ) -> TranscriptionResult:
            return TranscriptionResult(
                text="Test transcription",
                detected_language="en",
                language_probability=0.99,
                processing_duration_seconds=0.1,
                audio_duration_seconds=audio.duration_seconds,
                model_identifier="mock-base",
            )

    fake = FakeTranscriptionEngine()
    assert fake.is_loaded is False
    status = fake.load_model()
    assert status.is_ready is True
    assert fake.is_loaded is True

    proc = ProcessedAudio(
        samples=np.zeros(1600, dtype=np.float32),
        sample_rate=16000,
        duration_seconds=0.1,
        original_sample_rate=16000,
        original_channels=1,
        peak_amplitude=0.0,
        rms_db=-90.0,
        was_resampled=False,
        was_clipped=False,
    )

    result = fake.transcribe(proc)
    assert result.text == "Test transcription"
    assert result.detected_language == "en"

    fake.unload_model()
    assert fake.is_loaded is False


def test_no_speculative_tts_or_wake_word_interfaces():
    # Ensure interfaces.py does not define speculative TTS or Wake-word implementations
    import src.voice.interfaces as iface
    assert not hasattr(iface, "FutureTTSOutputInterface")
    assert not hasattr(iface, "FutureWakeWordListenerInterface")
    assert not hasattr(iface, "BargeInInterruptionHook")
