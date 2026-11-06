"""Unit tests for Qt Worker threads (AudioCaptureWorker, DeviceDiscoveryWorker)."""

import os
import sys
import time
from unittest.mock import MagicMock
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from src.voice.audio.processing import AudioProcessor
from src.voice.exceptions import AudioCaptureError, VoiceErrorCategory
from src.voice.interfaces import AudioCaptureBackend
from src.voice.models import AudioDeviceInfo, ProcessedAudio, RawAudioBuffer
from src.voice.workers import AudioCaptureWorker, DeviceDiscoveryWorker


@pytest.fixture(scope="session")
def qapp():
    """Ensure QApplication instance exists for Qt worker tests."""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def wait_for_worker(worker, qapp, timeout_ms=2000):
    """Wait for QThread completion while actively pumping the Qt event loop."""
    start = time.monotonic()
    while worker.isRunning() and (time.monotonic() - start) * 1000 < timeout_ms:
        qapp.processEvents()
        time.sleep(0.01)
    worker.wait(timeout_ms)
    # Drain any remaining queued signal deliveries
    for _ in range(5):
        qapp.processEvents()
        time.sleep(0.005)


class MockAudioBackend(AudioCaptureBackend):
    """Mock backend implementing AudioCaptureBackend interface."""

    def __init__(self, raise_on_open: bool = False):
        self.raise_on_open = raise_on_open
        self._is_recording = False
        self.peak_level = 0.5
        self.cancel_called = False
        self.stop_called = False
        self.close_called = False

    def list_input_devices(self) -> list[AudioDeviceInfo]:
        return [
            AudioDeviceInfo(index=0, name="Mock Mic", host_api_name="WASAPI", max_input_channels=1, default_samplerate=16000.0, is_default=True)
        ]

    def open_stream(self, device_index=None, sample_rate=16000, channels=1) -> AudioDeviceInfo:
        if self.raise_on_open:
            raise AudioCaptureError(
                "Hardware failed to open stream",
                category=VoiceErrorCategory.DEVICE_UNAVAILABLE,
            )
        return self.list_input_devices()[0]

    def start_stream(self) -> None:
        self._is_recording = True

    def stop_stream(self) -> RawAudioBuffer:
        self._is_recording = False
        self.stop_called = True
        return RawAudioBuffer(
            samples=np.ones(1600, dtype=np.float32) * 0.2,
            sample_rate=16000,
            channels=1,
            duration_seconds=0.1,
            device_name="Mock Mic",
        )

    def cancel_stream(self) -> None:
        self._is_recording = False
        self.cancel_called = True

    def close(self) -> None:
        self.close_called = True

    def get_current_peak_level(self) -> float:
        return self.peak_level

    @property
    def is_recording(self) -> bool:
        return self._is_recording


def test_audio_capture_worker_lifecycle(qapp):
    backend = MockAudioBackend()
    proc = AudioProcessor()
    worker = AudioCaptureWorker(capture_backend=backend, processor=proc)

    started_events = []
    level_events = []
    finished_events = []

    worker.started_capture.connect(lambda name: started_events.append(name))
    worker.level_updated.connect(lambda lvl: level_events.append(lvl))
    worker.finished_capture.connect(lambda res: finished_events.append(res))

    worker.start()

    # Wait briefly for capture to start
    for _ in range(25):
        if len(started_events) > 0 and len(level_events) > 0:
            break
        qapp.processEvents()
        time.sleep(0.02)

    assert len(started_events) == 1
    assert started_events[0] == "Mock Mic"
    assert len(level_events) > 0

    # Stop capture
    worker.request_stop()
    wait_for_worker(worker, qapp)

    assert len(finished_events) == 1
    result = finished_events[0]
    assert isinstance(result, ProcessedAudio)
    assert result.sample_rate == 16000
    assert backend.stop_called is True
    assert backend.close_called is True


def test_audio_capture_worker_cancel(qapp):
    backend = MockAudioBackend()
    worker = AudioCaptureWorker(capture_backend=backend)

    cancelled_events = []
    worker.cancelled_capture.connect(lambda: cancelled_events.append(True))

    worker.start()

    for _ in range(25):
        if backend.is_recording:
            break
        qapp.processEvents()
        time.sleep(0.02)

    worker.request_cancel()
    wait_for_worker(worker, qapp)

    assert len(cancelled_events) == 1
    assert backend.cancel_called is True
    assert backend.close_called is True


def test_audio_capture_worker_error(qapp):
    backend = MockAudioBackend(raise_on_open=True)
    worker = AudioCaptureWorker(capture_backend=backend)

    error_events = []
    worker.error_occurred.connect(lambda cat, msg, hint: error_events.append((cat, msg, hint)))

    worker.start()
    wait_for_worker(worker, qapp)

    assert len(error_events) == 1
    cat, msg, hint = error_events[0]
    assert cat == VoiceErrorCategory.DEVICE_UNAVAILABLE.value
    assert "Hardware failed to open stream" not in msg  # sanitized
    assert len(hint) > 0


def test_device_discovery_worker(qapp):
    mock_mgr = MagicMock()
    mock_devs = [
        AudioDeviceInfo(index=0, name="Built-in Mic", host_api_name="WASAPI", max_input_channels=2, default_samplerate=48000.0, is_default=True)
    ]
    mock_mgr.list_input_devices.return_value = mock_devs

    worker = DeviceDiscoveryWorker(device_manager=mock_mgr)
    discovered = []
    worker.devices_discovered.connect(lambda d: discovered.append(d))

    worker.start()
    wait_for_worker(worker, qapp)

    assert len(discovered) == 1
    assert len(discovered[0]) == 1
    assert discovered[0][0].name == "Built-in Mic"


def test_audio_capture_worker_speech_onset_and_auto_stop(qapp):
    """Verify AudioCaptureWorker detects speech onset and auto-stops after silence."""
    backend = MockAudioBackend()
    # Mock dynamic peak progression: starts with speech (peak=0.3 -> -10dB), then silence (peak=0.0001 -> -80dB)
    peaks = [0.3] * 10 + [0.0001] * 20
    call_count = 0

    def mock_peak():
        nonlocal call_count
        val = peaks[min(call_count, len(peaks) - 1)]
        call_count += 1
        return val

    backend.get_current_peak_level = mock_peak

    worker = AudioCaptureWorker(
        capture_backend=backend,
        vad_auto_stop=True,
        silence_timeout_seconds=0.3,
        min_speech_duration_ms=66,
    )

    onset_events = []
    silence_events = []
    finished_events = []

    worker.speech_onset.connect(lambda: onset_events.append(True))
    worker.silence_timeout_reached.connect(lambda: silence_events.append(True))
    worker.finished_capture.connect(lambda obj: finished_events.append(obj))

    worker.start()
    wait_for_worker(worker, qapp, timeout_ms=3000)

    assert len(onset_events) >= 1
    assert len(silence_events) >= 1
    assert len(finished_events) == 1


def test_audio_capture_worker_rejects_transient_noise_spike(qapp):
    """Verify transient clicks or clatter (< min_speech_duration_ms) do not trigger speech onset."""
    backend = MockAudioBackend()
    # 2 transient frames of loud peak (66ms), followed by quiet frames
    # min_speech_duration_ms = 150 requires ~5 frames sustained
    peaks = [0.4, 0.4] + [0.001] * 20
    idx = 0

    def mock_peak():
        nonlocal idx
        v = peaks[min(idx, len(peaks) - 1)]
        idx += 1
        return v

    backend.get_current_peak_level = mock_peak

    worker = AudioCaptureWorker(
        capture_backend=backend,
        vad_auto_stop=True,
        silence_timeout_seconds=0.2,
        min_speech_duration_ms=150,
    )

    onset_events = []
    worker.speech_onset.connect(lambda: onset_events.append(True))

    worker.start()
    # Let it process for ~300ms
    for _ in range(15):
        qapp.processEvents()
        time.sleep(0.02)

    worker.request_stop()
    wait_for_worker(worker, qapp, timeout_ms=2000)

    # Transient noise should NOT trigger speech onset
    assert len(onset_events) == 0


def test_audio_capture_worker_adapts_to_steady_noise_floor(qapp):
    """Verify worker adapts to ambient noise floor so background hum does not trigger speech."""
    backend = MockAudioBackend()
    # Continuous noise floor at peak ~0.008 (-42 dB), then genuine speech at peak ~0.3 (-10 dB)
    peaks = [0.008] * 15 + [0.3] * 8 + [0.001] * 15
    idx = 0

    def mock_peak():
        nonlocal idx
        v = peaks[min(idx, len(peaks) - 1)]
        idx += 1
        return v

    backend.get_current_peak_level = mock_peak

    worker = AudioCaptureWorker(
        capture_backend=backend,
        vad_auto_stop=True,
        silence_timeout_seconds=0.25,
        min_speech_duration_ms=99,
        energy_threshold_db=-45.0,
    )

    onset_events = []
    worker.speech_onset.connect(lambda: onset_events.append(True))

    worker.start()
    wait_for_worker(worker, qapp, timeout_ms=3000)

    # Speech onset should trigger when the loud utterance arrives
    assert len(onset_events) == 1


def test_audio_capture_worker_acoustic_echo_suppression_while_speaking(qapp):
    """Verify louder threshold is required when assistant is speaking aloud to avoid acoustic feedback."""
    backend = MockAudioBackend()
    is_speaking = True

    # Moderate speaker audio leakage: peak = 0.015 (~ -36 dB)
    # When speaking, threshold requires > -32 dB and SNR +10 dB
    peaks = [0.015] * 12 + [0.25] * 8 + [0.001] * 15
    idx = 0

    def mock_peak():
        nonlocal idx
        v = peaks[min(idx, len(peaks) - 1)]
        idx += 1
        return v

    backend.get_current_peak_level = mock_peak

    worker = AudioCaptureWorker(
        capture_backend=backend,
        vad_auto_stop=True,
        silence_timeout_seconds=0.25,
        min_speech_duration_ms=99,
        is_speaking_check=lambda: is_speaking,
    )

    onset_events = []
    worker.speech_onset.connect(lambda: onset_events.append(True))

    worker.start()

    # During speaker playback alone, 0.015 shouldn't trigger
    for _ in range(8):
        qapp.processEvents()
        time.sleep(0.02)
    assert len(onset_events) == 0

    # Continue run: the loud interruption (0.25) arrives and triggers
    wait_for_worker(worker, qapp, timeout_ms=3000)
    assert len(onset_events) == 1
