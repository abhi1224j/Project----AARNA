"""Unit tests for VoiceError hierarchy and message sanitization."""

import pytest

from src.voice.exceptions import (
    AudioCaptureError,
    AudioDeviceError,
    InvalidVoiceStateTransitionError,
    SpeechModelError,
    TranscriptionError,
    VoiceCancelledError,
    VoiceError,
    VoiceErrorCategory,
)


def test_voice_error_sanitization():
    # When technical details are passed, user_message must remain clean and safe
    raw_hw_error = "PortAudioError -9999: Unanticipated host error at line 42 in pa_win_wasapi.c"
    err = AudioDeviceError(raw_hw_error)

    # str(err) and technical_details retain internal info for logging
    assert str(err) == raw_hw_error
    assert err.technical_details == raw_hw_error

    # user_message must NOT leak raw technical exception string to UI
    assert raw_hw_error not in err.user_message
    assert "microphone is currently unavailable" in err.user_message
    assert "Check your microphone connection" in err.recovery_hint


def test_voice_error_explicit_user_message_override():
    custom_msg = "Please plug in your USB microphone."
    custom_hint = "Look for the blue LED indicator on the microphone."
    err = AudioDeviceError(
        "Internal I/O failure",
        user_message=custom_msg,
        recovery_hint=custom_hint,
    )
    assert err.user_message == custom_msg
    assert err.recovery_hint == custom_hint
    assert err.technical_details == "Internal I/O failure"


def test_voice_error_subclasses_categories():
    assert AudioDeviceError().category == VoiceErrorCategory.DEVICE_UNAVAILABLE
    assert AudioCaptureError().category == VoiceErrorCategory.CAPTURE_FAILURE
    assert SpeechModelError().category == VoiceErrorCategory.MODEL_LOAD_FAILED
    assert TranscriptionError().category == VoiceErrorCategory.TRANSCRIPTION_FAILED
    assert InvalidVoiceStateTransitionError().category == VoiceErrorCategory.INVALID_STATE
    assert VoiceCancelledError().category == VoiceErrorCategory.CANCELLED


def test_no_references_to_nonexistent_cli_or_buttons():
    # Verify recovery hints do not mention nonexistent CLI commands or UI buttons
    for cat in VoiceErrorCategory:
        err = VoiceError("Test internal", category=cat)
        assert "setup_models" not in err.recovery_hint
        assert "Setup Voice Model" not in err.recovery_hint
        assert "python -m" not in err.recovery_hint
