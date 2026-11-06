"""Unit tests for SoundDeviceAudioCapture with mocked PortAudio stream."""

from unittest.mock import MagicMock
import numpy as np
import pytest

from src.voice.audio.capture import SoundDeviceAudioCapture
from src.voice.audio.devices import AudioDeviceManager
from src.voice.exceptions import (
    AudioCaptureError,
    EmptyRecordingError,
    VoiceErrorCategory,
)


class MockInputStream:
    """Mock for sounddevice.InputStream."""

    def __init__(self, samplerate, channels, device, dtype, callback):
        self.samplerate = samplerate
        self.channels = channels
        self.device = device
        self.dtype = dtype
        self.callback = callback
        self.is_active = False
        self.is_closed = False

    def start(self):
        self.is_active = True

    def stop(self):
        self.is_active = False

    def close(self):
        self.is_active = False
        self.is_closed = True


class MockSoundDeviceForCapture:
    """Mock sounddevice module for capture testing."""

    def __init__(self):
        self._devices = [
            {"name": "Default Test Mic", "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 16000.0, "hostapi": 0}
        ]
        self._hostapis = [{"name": "WASAPI", "devices": [0]}]
        self.default = MagicMock()
        self.default.device = (0, 0)
        self.active_stream = None
        self.fail_on_open = False
        self.fail_on_start = False

    def query_devices(self):
        return self._devices

    def query_hostapis(self):
        return self._hostapis

    def check_input_settings(self, device, samplerate, channels):
        return True

    def InputStream(self, samplerate, channels, device, dtype, callback):
        if self.fail_on_open:
            raise RuntimeError("Hardware open failure")
        stream = MockInputStream(samplerate, channels, device, dtype, callback)
        if self.fail_on_start:
            stream.start = MagicMock(side_effect=RuntimeError("Hardware start failure"))
        self.active_stream = stream
        return stream


def test_capture_lifecycle_success():
    mock_sd = MockSoundDeviceForCapture()
    capture = SoundDeviceAudioCapture(sd_module=mock_sd)

    dev = capture.open_stream(sample_rate=16000, channels=1)
    assert dev.name == "Default Test Mic"
    assert capture._stream is not None

    capture.start_stream()
    assert capture.is_recording is True

    # Simulate callback delivering two chunks of audio
    chunk1 = np.ones((480, 1), dtype=np.float32) * 0.2
    chunk2 = np.ones((480, 1), dtype=np.float32) * 0.4
    capture._audio_callback(chunk1, 480, None, None)
    capture._audio_callback(chunk2, 480, None, None)

    # Check peak level polling
    peak = capture.get_current_peak_level()
    assert peak >= 0.35

    buf = capture.stop_stream()
    assert capture.is_recording is False
    assert buf.sample_rate == 16000
    assert buf.channels == 1
    assert buf.samples.shape == (960, 1)
    assert buf.duration_seconds == pytest.approx(960 / 16000.0)

    capture.close()
    assert capture._stream is None


def test_capture_max_duration_bound():
    mock_sd = MockSoundDeviceForCapture()
    # Set max duration to 0.1 seconds (1600 frames at 16kHz)
    capture = SoundDeviceAudioCapture(sd_module=mock_sd, max_recording_duration_seconds=0.1)
    capture.open_stream(sample_rate=16000, channels=1)
    capture.start_stream()

    # Deliver 1000 frames (OK)
    capture._audio_callback(np.ones((1000, 1), dtype=np.float32), 1000, None, None)
    assert capture._stopped_by_max_duration is False

    # Deliver another 1000 frames (exceeds 1600 frames limit)
    capture._audio_callback(np.ones((1000, 1), dtype=np.float32), 1000, None, None)
    assert capture._stopped_by_max_duration is True

    buf = capture.stop_stream()
    assert buf.stopped_by_max_duration is True
    # Buffer should only hold the first chunk (1000 frames), not the overflowing second chunk
    assert buf.samples.shape[0] == 1000


def test_capture_cancel_clears_memory():
    mock_sd = MockSoundDeviceForCapture()
    capture = SoundDeviceAudioCapture(sd_module=mock_sd)
    capture.open_stream()
    capture.start_stream()

    # Deliver audio chunk
    chunk = np.ones((1600, 1), dtype=np.float32) * 0.5
    capture._audio_callback(chunk, 1600, None, None)

    capture.cancel_stream()
    assert capture.is_recording is False
    assert len(capture._chunks) == 0
    assert capture.get_current_peak_level() == 0.0


def test_capture_empty_recording_raises():
    mock_sd = MockSoundDeviceForCapture()
    capture = SoundDeviceAudioCapture(sd_module=mock_sd)
    capture.open_stream()
    capture.start_stream()

    # Stop without any chunks delivered
    with pytest.raises(EmptyRecordingError) as exc_info:
        capture.stop_stream()

    assert exc_info.value.category == VoiceErrorCategory.EMPTY_RECORDING


def test_capture_open_failure_raises_typed_error():
    mock_sd = MockSoundDeviceForCapture()
    mock_sd.fail_on_open = True
    capture = SoundDeviceAudioCapture(sd_module=mock_sd)

    with pytest.raises(AudioCaptureError) as exc_info:
        capture.open_stream()

    assert exc_info.value.category == VoiceErrorCategory.DEVICE_UNAVAILABLE


def test_capture_concurrent_start_prevented():
    mock_sd = MockSoundDeviceForCapture()
    capture = SoundDeviceAudioCapture(sd_module=mock_sd)
    capture.open_stream()
    capture.start_stream()

    with pytest.raises(AudioCaptureError) as exc_info:
        capture.start_stream()

    assert exc_info.value.category == VoiceErrorCategory.DEVICE_BUSY
    assert "already actively recording" in str(exc_info.value)
    capture.cancel_stream()
    capture.close()


def test_capture_preroll_rolling_buffer():
    """Verify rolling preroll buffer bounds pre-speech audio, then locks on speech onset."""
    mock_sd = MockSoundDeviceForCapture()
    # 0.1s preroll at 16kHz = 1600 frames window
    capture = SoundDeviceAudioCapture(sd_module=mock_sd, preroll_duration_seconds=0.1)
    capture.open_stream(sample_rate=16000, channels=1)
    capture.start_stream()

    # Deliver 5 chunks of 800 frames each (4000 frames total) before speech onset
    for _ in range(5):
        chunk = np.ones((800, 1), dtype=np.float32) * 0.1
        capture._audio_callback(chunk, 800, None, None)

    # Chunks should have been trimmed so frames are bounded near 1600 (+ at most 1 chunk)
    assert capture._total_frames <= 2400

    # User starts speaking -> notify speech onset to lock preroll
    capture.notify_speech_onset()

    # Deliver 2 more chunks of 800 frames each
    for _ in range(2):
        chunk = np.ones((800, 1), dtype=np.float32) * 0.4
        capture._audio_callback(chunk, 800, None, None)

    pre_stop_frames = capture._total_frames
    buf = capture.stop_stream()
    assert buf.samples.shape[0] == pre_stop_frames
    assert buf.samples.shape[0] >= 3200
    capture.close()

