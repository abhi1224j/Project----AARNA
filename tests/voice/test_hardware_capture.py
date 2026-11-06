"""Hardware-level audio integration tests requiring physical microphone access."""

import time
import pytest
import sounddevice as sd

from src.voice.audio.capture import SoundDeviceAudioCapture
from src.voice.audio.devices import AudioDeviceManager
from src.voice.audio.processing import AudioProcessor
from src.voice.audio.vad import VoiceActivityDetector
from src.voice.models import ProcessedAudio, RawAudioBuffer


@pytest.mark.hardware
def test_real_hardware_device_enumeration():
    """Verify device discovery against real host operating system audio endpoints."""
    mgr = AudioDeviceManager()
    devices = mgr.list_input_devices()

    if not devices:
        pytest.skip("No physical microphone available on this host.")

    assert len(devices) >= 1
    default_dev = mgr.get_default_input_device()
    assert default_dev is not None
    assert default_dev.max_input_channels >= 1
    assert default_dev.default_samplerate > 0


@pytest.mark.hardware
def test_real_hardware_capture_and_processing_pipeline():
    """Test short live audio capture and processing on the physical default microphone."""
    mgr = AudioDeviceManager()
    devices = mgr.list_input_devices()
    if not devices:
        pytest.skip("No physical microphone available on this host.")

    capture = SoundDeviceAudioCapture(device_manager=mgr)
    processor = AudioProcessor()
    vad = VoiceActivityDetector()

    try:
        # Open live stream
        dev_info = capture.open_stream(sample_rate=16000, channels=1)
        assert dev_info is not None

        # Start live stream
        capture.start_stream()
        assert capture.is_recording is True

        # Capture live frames for 250 milliseconds
        time.sleep(0.25)
        peak = capture.get_current_peak_level()
        assert 0.0 <= peak <= 1.0

        # Stop and retrieve buffer
        raw_buffer = capture.stop_stream()
        assert isinstance(raw_buffer, RawAudioBuffer)
        assert raw_buffer.samples.shape[0] > 0
        assert raw_buffer.duration_seconds >= 0.15

        # Process buffer through full pipeline
        processed = processor.process_raw_buffer(raw_buffer, vad=vad)
        assert isinstance(processed, ProcessedAudio)
        assert processed.sample_rate == 16000
        assert processed.samples.dtype == "float32"
        assert processed.samples.ndim == 1
        assert processed.duration_seconds >= 0.15
        assert processed.vad_result is not None
    finally:
        capture.close()
