"""Unit tests for AudioDeviceManager with mocked sounddevice hardware."""

import pytest
from unittest.mock import MagicMock

from src.voice.audio.devices import AudioDeviceManager, is_loopback_device, is_virtual_mapper
from src.voice.exceptions import (
    AudioDeviceError,
    NoMicrophoneDetectedError,
    UnsupportedSampleRateError,
    VoiceErrorCategory,
)
from src.voice.models import AudioDeviceInfo


class MockSoundDevice:
    """Mock sounddevice module for deterministic unit testing."""

    def __init__(self, devices=None, hostapis=None, default_device=(1, 3)):
        self._devices = devices if devices is not None else [
            {"name": "Speakers (Realtek Audio)", "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 48000.0, "hostapi": 0},
            {"name": "Microphone Array (AMD Audio Device)", "max_input_channels": 2, "max_output_channels": 0, "default_samplerate": 48000.0, "hostapi": 0},
            {"name": "USB Headset Mic", "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 16000.0, "hostapi": 1},
        ]
        self._hostapis = hostapis if hostapis is not None else [
            {"name": "Windows WASAPI", "devices": [0, 1], "default_input_device": 1},
            {"name": "Windows DirectSound", "devices": [2], "default_input_device": 2},
        ]
        self.default = MagicMock()
        self.default.device = default_device
        self._supported_rates = {
            (1, 16000, 1): True,
            (1, 48000, 1): True,
            (1, 44100, 1): True,
            (2, 16000, 1): True,
        }

    def query_devices(self):
        return self._devices

    def query_hostapis(self):
        return self._hostapis

    def check_input_settings(self, device, samplerate, channels):
        key = (device, samplerate, channels)
        if self._supported_rates.get(key, False):
            return
        raise RuntimeError(f"Invalid settings for device {device}: {samplerate}Hz {channels}ch")


def test_list_input_devices_filters_output_only():
    mock_sd = MockSoundDevice()
    mgr = AudioDeviceManager(sd_module=mock_sd)
    devices = mgr.list_input_devices()

    # Index 0 is Speakers (max_input_channels=0), so must be excluded
    assert len(devices) == 2
    device_names = [d.name for d in devices]
    assert "Speakers (Realtek Audio)" not in device_names
    assert "Microphone Array (AMD Audio Device)" in device_names
    assert "USB Headset Mic" in device_names


def test_list_input_devices_default_placed_first():
    mock_sd = MockSoundDevice(default_device=(1, 3))
    mgr = AudioDeviceManager(sd_module=mock_sd)
    devices = mgr.list_input_devices()

    assert len(devices) == 2
    assert devices[0].index == 1
    assert devices[0].is_default is True
    assert devices[1].index == 2
    assert devices[1].is_default is False


def test_get_default_input_device_success():
    mock_sd = MockSoundDevice(default_device=(1, 3))
    mgr = AudioDeviceManager(sd_module=mock_sd)
    default_dev = mgr.get_default_input_device()

    assert default_dev.index == 1
    assert "Microphone Array" in default_dev.name
    assert default_dev.is_default is True


def test_get_default_input_device_no_microphone_error():
    # Only output devices
    mock_sd = MockSoundDevice(devices=[
        {"name": "Speakers", "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 48000.0, "hostapi": 0}
    ])
    mgr = AudioDeviceManager(sd_module=mock_sd)

    with pytest.raises(NoMicrophoneDetectedError) as exc_info:
        mgr.get_default_input_device()

    assert exc_info.value.category == VoiceErrorCategory.NO_MICROPHONE
    assert "No microphone" in exc_info.value.user_message


def test_resolve_device_by_index():
    mock_sd = MockSoundDevice()
    mgr = AudioDeviceManager(sd_module=mock_sd)

    dev = mgr.resolve_device(device_index=2)
    assert dev.index == 2
    assert dev.name == "USB Headset Mic"


def test_resolve_device_by_invalid_index_strict_privacy_default():
    mock_sd = MockSoundDevice()
    mgr = AudioDeviceManager(sd_module=mock_sd)

    # By default, fallback_to_default is False to prevent silent recording source switching
    with pytest.raises(AudioDeviceError) as exc_info:
        mgr.resolve_device(device_index=99)

    assert exc_info.value.category == VoiceErrorCategory.DEVICE_UNAVAILABLE
    assert "no longer connected" in exc_info.value.user_message


def test_resolve_device_by_invalid_index_explicit_fallback():
    mock_sd = MockSoundDevice()
    mgr = AudioDeviceManager(sd_module=mock_sd)

    # Only when explicitly requested does it fall back to default
    dev = mgr.resolve_device(device_index=99, fallback_to_default=True)
    assert dev.index == 1
    assert dev.is_default is True


def test_resolve_device_by_name():
    mock_sd = MockSoundDevice()
    mgr = AudioDeviceManager(sd_module=mock_sd)

    dev = mgr.resolve_device(device_name="headset")
    assert dev.index == 2
    assert "USB Headset" in dev.name


def test_resolve_device_by_nonexistent_name_strict_privacy_default():
    mock_sd = MockSoundDevice()
    mgr = AudioDeviceManager(sd_module=mock_sd)

    # Strict privacy default: missing preferred device raises error, never silently switches mic
    with pytest.raises(AudioDeviceError) as exc_info:
        mgr.resolve_device(device_name="Studio Mic Pro")

    assert exc_info.value.category == VoiceErrorCategory.DEVICE_UNAVAILABLE
    assert "not found" in exc_info.value.user_message


def test_resolve_device_by_nonexistent_name_explicit_fallback():
    mock_sd = MockSoundDevice()
    mgr = AudioDeviceManager(sd_module=mock_sd)

    dev = mgr.resolve_device(device_name="Studio Mic Pro", fallback_to_default=True)
    assert dev.index == 1
    assert dev.is_default is True


def test_resolve_device_system_default_mode():
    mock_sd = MockSoundDevice()
    mgr = AudioDeviceManager(sd_module=mock_sd)

    # When no explicit device is specified, returns system default
    dev = mgr.resolve_device()
    assert dev.index == 1
    assert dev.is_default is True


def test_resolve_device_prefers_wasapi_on_name_collision():
    # Two devices with identical names under MME and WASAPI
    mock_sd = MockSoundDevice(
        devices=[
            {"name": "Microphone Array", "max_input_channels": 2, "max_output_channels": 0, "default_samplerate": 44100.0, "hostapi": 0},
            {"name": "Microphone Array", "max_input_channels": 2, "max_output_channels": 0, "default_samplerate": 48000.0, "hostapi": 1},
        ],
        hostapis=[
            {"name": "MME", "default_input_device": 0},
            {"name": "Windows WASAPI", "default_input_device": 1},
        ],
        default_device=(1, 0),
    )
    mgr = AudioDeviceManager(sd_module=mock_sd)
    dev = mgr.resolve_device(device_name="Microphone Array")
    assert dev.index == 1
    assert "WASAPI" in dev.host_api_name


def test_virtual_mapper_deprioritized_over_physical_mic():
    mock_sd = MockSoundDevice(
        devices=[
            {"name": "Microsoft Sound Mapper - Input", "max_input_channels": 2, "max_output_channels": 0, "default_samplerate": 44100.0, "hostapi": 0},
            {"name": "Microphone Array (AMD Audio Device)", "max_input_channels": 2, "max_output_channels": 0, "default_samplerate": 48000.0, "hostapi": 1},
        ],
        hostapis=[
            {"name": "MME", "default_input_device": 0},
            {"name": "Windows WASAPI", "default_input_device": 1},
        ],
        default_device=(1, 0),
    )
    mgr = AudioDeviceManager(sd_module=mock_sd)
    default_dev = mgr.get_default_input_device()
    assert default_dev.name == "Microphone Array (AMD Audio Device)"
    assert default_dev.index == 1


def test_wasapi_hostapi_dynamic_index_resolution():
    # WASAPI at index 2 (middle)
    mock_sd1 = MockSoundDevice(
        devices=[
            {"name": "MME Mic", "max_input_channels": 1, "default_samplerate": 44100.0, "hostapi": 0},
            {"name": "DS Mic", "max_input_channels": 1, "default_samplerate": 44100.0, "hostapi": 1},
            {"name": "WASAPI Mic", "max_input_channels": 1, "default_samplerate": 48000.0, "hostapi": 2},
        ],
        hostapis=[
            {"name": "MME", "default_input_device": 0},
            {"name": "Windows DirectSound", "default_input_device": 1},
            {"name": "Windows WASAPI", "default_input_device": 2},
        ],
        default_device=(0, 0),
    )
    mgr1 = AudioDeviceManager(sd_module=mock_sd1)
    dev1 = mgr1.get_default_input_device()
    assert dev1.index == 2
    assert dev1.name == "WASAPI Mic"

    # WASAPI at index 0 (first)
    mock_sd2 = MockSoundDevice(
        devices=[
            {"name": "WASAPI Mic First", "max_input_channels": 1, "default_samplerate": 48000.0, "hostapi": 0},
            {"name": "Other Mic", "max_input_channels": 1, "default_samplerate": 44100.0, "hostapi": 1},
        ],
        hostapis=[
            {"name": "Windows WASAPI", "default_input_device": 0},
            {"name": "MME", "default_input_device": 1},
        ],
        default_device=(1, 0),
    )
    mgr2 = AudioDeviceManager(sd_module=mock_sd2)
    dev2 = mgr2.get_default_input_device()
    assert dev2.index == 0
    assert dev2.name == "WASAPI Mic First"


def test_wasapi_absent_clean_fallback():
    # Non-Windows host (ALSA / PulseAudio) where WASAPI is absent
    mock_sd = MockSoundDevice(
        devices=[
            {"name": "default", "max_input_channels": 2, "default_samplerate": 48000.0, "hostapi": 0},
            {"name": "pulse", "max_input_channels": 2, "default_samplerate": 48000.0, "hostapi": 1},
        ],
        hostapis=[
            {"name": "ALSA", "default_input_device": 0},
            {"name": "PulseAudio", "default_input_device": 1},
        ],
        default_device=(0, 0),
    )
    mgr = AudioDeviceManager(sd_module=mock_sd)
    dev = mgr.get_default_input_device()
    assert dev.index == 0
    assert dev.name == "default"


def test_wasapi_invalid_default_device_falls_back():
    # WASAPI has default_input_device = -1 (no device) or out of bounds
    mock_sd = MockSoundDevice(
        devices=[
            {"name": "Fallback Mic", "max_input_channels": 1, "default_samplerate": 44100.0, "hostapi": 0},
        ],
        hostapis=[
            {"name": "MME", "default_input_device": 0},
            {"name": "Windows WASAPI", "default_input_device": -1},
        ],
        default_device=(0, 0),
    )
    mgr = AudioDeviceManager(sd_module=mock_sd)
    dev = mgr.get_default_input_device()
    assert dev.index == 0
    assert dev.name == "Fallback Mic"


def test_is_virtual_mapper_and_loopback_helpers():
    assert is_virtual_mapper("Microsoft Sound Mapper - Input") is True
    assert is_virtual_mapper("Primary Sound Capture Driver") is True
    assert is_virtual_mapper("Microphone Array (Realtek)") is False

    assert is_loopback_device("Stereo Mix (Realtek)") is True
    assert is_loopback_device("What U Hear") is True
    assert is_loopback_device("USB Microphone") is False


def test_check_device_supported():
    mock_sd = MockSoundDevice()
    mgr = AudioDeviceManager(sd_module=mock_sd)

    assert mgr.check_device_supported(device_index=1, sample_rate=16000, channels=1) is True
    assert mgr.check_device_supported(device_index=1, sample_rate=96000, channels=1) is False


def test_probe_best_sample_rate():
    mock_sd = MockSoundDevice()
    mgr = AudioDeviceManager(sd_module=mock_sd)

    # 16000 is supported on device 1, so it should be returned first
    best = mgr.probe_best_sample_rate(device_index=1, channels=1)
    assert best == 16000


def test_probe_best_sample_rate_unsupported_raises():
    mock_sd = MockSoundDevice()
    mgr = AudioDeviceManager(sd_module=mock_sd)

    with pytest.raises(UnsupportedSampleRateError) as exc_info:
        mgr.probe_best_sample_rate(device_index=99, channels=1)

    assert exc_info.value.category == VoiceErrorCategory.UNSUPPORTED_FORMAT
