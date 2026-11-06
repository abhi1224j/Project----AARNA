"""
Integration and orchestration unit tests for VoiceController (Phase C6).
Tests complete push-to-talk lifecycle, state-machine transitions,
avatar feedback arbitration, LLM generation locking, review confirmation/discard,
and graceful error recovery.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication

from src.ui.avatar.state import AvatarState
from src.voice.config import VoiceConfig
from src.voice.controller import VoiceController
from src.voice.exceptions import VoiceErrorCategory
from src.voice.models import (
    ModelAvailabilityState,
    ModelStatusInfo,
    ProcessedAudio,
    TranscriptionResult,
    TranscriptionSegment,
)
from src.voice.state import VoiceState, VoiceStateMachine


@pytest.fixture(scope="module")
def qapp():
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])
    return app


def _make_dummy_processed_audio() -> ProcessedAudio:
    return ProcessedAudio(
        samples=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        duration_seconds=1.0,
        original_sample_rate=16000,
        original_channels=1,
        peak_amplitude=0.3,
        rms_db=-25.0,
        was_resampled=False,
        was_clipped=False,
    )


def _make_dummy_transcription_result(
    text: str = "Hello Aarna",
    detected_language: str = "en",
) -> TranscriptionResult:
    return TranscriptionResult(
        text=text,
        detected_language=detected_language,
        language_probability=0.99,
        processing_duration_seconds=0.2,
        audio_duration_seconds=1.0,
        model_identifier="faster-whisper-tiny",
        compute_device="cpu",
        compute_type="int8",
        requested_language_mode="auto",
        segments=(
            TranscriptionSegment(0.0, 1.0, text, avg_logprob=-0.1, no_speech_prob=0.01),
        ),
    )


def test_controller_initialization_defaults(qapp):
    ctrl = VoiceController()
    assert ctrl.current_state == VoiceState.READY
    assert ctrl.pending_review_text == ""
    assert not ctrl.is_recording
    assert not ctrl.is_transcribing
    ctrl.shutdown()


def test_controller_disabled_initial_state(qapp):
    cfg = VoiceConfig(enabled=False)
    ctrl = VoiceController(config=cfg)
    assert ctrl.current_state == VoiceState.DISABLED
    ctrl.shutdown()


def test_controller_push_to_talk_happy_path(qapp):
    avatar_mock = MagicMock()
    mock_engine = MagicMock()
    dummy_result = _make_dummy_transcription_result("Testing push to talk")
    mock_engine.transcribe.return_value = dummy_result

    ctrl = VoiceController(avatar_sink=avatar_mock, transcription_engine=mock_engine)

    state_transitions = []
    ctrl.state_changed.connect(state_transitions.append)

    review_signals = []
    ctrl.transcript_ready_for_review.connect(review_signals.append)

    # 1. Start recording
    with pytest.MonkeyPatch.context() as mp:
        mock_capture_worker_cls = MagicMock()
        mock_capture_worker_instance = MagicMock()
        mock_capture_worker_cls.return_value = mock_capture_worker_instance
        mp.setattr("src.voice.controller.AudioCaptureWorker", mock_capture_worker_cls)

        mock_transcription_worker_cls = MagicMock()
        mock_transcription_worker_instance = MagicMock()
        mock_transcription_worker_cls.return_value = mock_transcription_worker_instance
        mp.setattr("src.voice.controller.TranscriptionWorker", mock_transcription_worker_cls)

        ctrl.start_recording()
        assert ctrl.current_state == VoiceState.REQUESTING_DEVICE
        assert avatar_mock.set_state.called
        assert avatar_mock.set_state.call_args[0][0] == AvatarState.LISTENING

        # Simulate worker emitting capture started
        ctrl._on_capture_started("Mock Microphone")
        assert ctrl.current_state == VoiceState.LISTENING

        # 2. Stop recording and simulate capture output
        ctrl.stop_recording()
        assert ctrl.current_state == VoiceState.STOPPING

        dummy_audio = _make_dummy_processed_audio()
        ctrl._on_capture_finished(dummy_audio)
        assert ctrl.current_state == VoiceState.TRANSCRIBING
        assert avatar_mock.set_state.call_args[0][0] == AvatarState.THINKING

        # 3. Simulate transcription worker completion
        ctrl._on_transcription_finished(dummy_result)
        assert ctrl.current_state == VoiceState.REVIEW
        assert avatar_mock.set_state.call_args[0][0] == AvatarState.IDLE
        assert ctrl.pending_review_text == "Testing push to talk"
        assert review_signals == ["Testing push to talk"]

        # 4. User confirms transcript
        confirmed = ctrl.confirm_review()
        assert confirmed == "Testing push to talk"
        assert ctrl.pending_review_text == ""
        assert ctrl.current_state == VoiceState.READY

    ctrl.shutdown()


def test_controller_discard_review(qapp):
    ctrl = VoiceController()
    dummy_result = _make_dummy_transcription_result("Discard this")

    # Force controller into review state
    ctrl.state_machine.transition(VoiceState.REQUESTING_DEVICE)
    ctrl.state_machine.transition(VoiceState.LISTENING)
    ctrl.state_machine.transition(VoiceState.STOPPING)
    ctrl.state_machine.transition(VoiceState.PROCESSING_AUDIO)
    ctrl.state_machine.transition(VoiceState.TRANSCRIBING)
    ctrl._on_transcription_finished(dummy_result)

    assert ctrl.current_state == VoiceState.REVIEW
    assert ctrl.pending_review_text == "Discard this"

    ctrl.discard_review()
    assert ctrl.pending_review_text == ""
    assert ctrl.current_state == VoiceState.READY
    ctrl.shutdown()


def test_controller_empty_transcript_handling(qapp):
    ctrl = VoiceController()
    empty_result = _make_dummy_transcription_result("")

    ctrl.state_machine.transition(VoiceState.REQUESTING_DEVICE)
    ctrl.state_machine.transition(VoiceState.LISTENING)
    ctrl.state_machine.transition(VoiceState.STOPPING)
    ctrl.state_machine.transition(VoiceState.PROCESSING_AUDIO)
    ctrl.state_machine.transition(VoiceState.TRANSCRIBING)

    review_signals = []
    ctrl.transcript_ready_for_review.connect(review_signals.append)

    ctrl._on_transcription_finished(empty_result)
    # Should cleanly pass through REVIEW and return to READY without review prompt
    assert ctrl.current_state == VoiceState.READY
    assert len(review_signals) == 0
    ctrl.shutdown()


def test_controller_llm_generating_blocks_recording(qapp):
    llm_active = True
    errors = []
    ctrl = VoiceController(is_llm_generating=lambda: llm_active)
    ctrl.error_occurred.connect(lambda cat, msg, hint: errors.append((cat, msg, hint)))

    ctrl.start_recording()
    assert ctrl.current_state == VoiceState.READY
    assert len(errors) == 1
    assert errors[0][0] == VoiceErrorCategory.DEVICE_BUSY.value
    assert "Cannot record while AARNA is speaking" in errors[0][1]
    ctrl.shutdown()


def test_controller_llm_generating_preserves_avatar_thinking(qapp):
    llm_active = True
    avatar_mock = MagicMock()
    ctrl = VoiceController(
        avatar_sink=avatar_mock,
        is_llm_generating=lambda: llm_active,
    )

    # When LLM is generating, VoiceController._update_avatar_state(AvatarState.IDLE) should be suppressed
    avatar_mock.reset_mock()
    ctrl._update_avatar_state(AvatarState.IDLE)
    avatar_mock.set_state.assert_not_called()

    # But LISTENING and THINKING should still be forwarded
    ctrl._update_avatar_state(AvatarState.LISTENING)
    avatar_mock.set_state.assert_called_with(AvatarState.LISTENING)

    ctrl.shutdown()


def test_controller_cancel_recording_recovers_to_ready(qapp):
    ctrl = VoiceController()
    ctrl.state_machine.transition(VoiceState.REQUESTING_DEVICE)
    ctrl.state_machine.transition(VoiceState.LISTENING)

    ctrl.cancel_recording()
    assert ctrl.current_state == VoiceState.READY
    assert ctrl.pending_review_text == ""
    ctrl.shutdown()


def test_controller_capture_error_handling(qapp):
    ctrl = VoiceController()
    errors = []
    ctrl.error_occurred.connect(lambda cat, msg, hint: errors.append((cat, msg, hint)))

    ctrl.state_machine.transition(VoiceState.REQUESTING_DEVICE)
    ctrl.state_machine.transition(VoiceState.LISTENING)

    ctrl._on_capture_error(
        VoiceErrorCategory.DEVICE_DISCONNECTED.value,
        "Microphone was unplugged.",
        "Reconnect microphone.",
    )

    assert ctrl.current_state == VoiceState.ERROR
    assert len(errors) == 1
    assert errors[0][0] == VoiceErrorCategory.DEVICE_DISCONNECTED.value

    # Start recording again safely recovers to ready before transitioning
    with pytest.MonkeyPatch.context() as mp:
        mock_worker_cls = MagicMock()
        mock_worker_cls.return_value = MagicMock()
        mp.setattr("src.voice.controller.AudioCaptureWorker", mock_worker_cls)
        ctrl.start_recording()
        assert ctrl.current_state == VoiceState.REQUESTING_DEVICE

    ctrl.shutdown()


def test_controller_transcription_error_handling(qapp):
    ctrl = VoiceController()
    errors = []
    ctrl.error_occurred.connect(lambda cat, msg, hint: errors.append((cat, msg, hint)))

    ctrl.state_machine.transition(VoiceState.REQUESTING_DEVICE)
    ctrl.state_machine.transition(VoiceState.LISTENING)
    ctrl.state_machine.transition(VoiceState.STOPPING)
    ctrl.state_machine.transition(VoiceState.PROCESSING_AUDIO)
    ctrl.state_machine.transition(VoiceState.TRANSCRIBING)

    ctrl._on_transcription_error(
        VoiceErrorCategory.TRANSCRIPTION_FAILED.value,
        "Inference failed",
        "Try again",
    )

    assert ctrl.current_state == VoiceState.ERROR
    assert len(errors) == 1
    ctrl.shutdown()


def test_controller_auto_submit_workflow(qapp):
    """Verify auto-submit workflow bypasses manual review and transitions directly to READY."""
    cfg = VoiceConfig(auto_submit_transcript=True)
    ctrl = VoiceController(config=cfg)

    auto_submits = []
    ctrl.transcript_auto_submitted.connect(lambda txt, lang: auto_submits.append((txt, lang)))

    ctrl.state_machine.transition(VoiceState.REQUESTING_DEVICE)
    ctrl.state_machine.transition(VoiceState.LISTENING)
    ctrl.state_machine.transition(VoiceState.STOPPING)
    ctrl.state_machine.transition(VoiceState.PROCESSING_AUDIO)
    ctrl.state_machine.transition(VoiceState.TRANSCRIBING)

    dummy_result = _make_dummy_transcription_result("नमस्ते आरना", detected_language="hi")
    ctrl._on_transcription_finished(dummy_result)

    assert ctrl.current_state == VoiceState.READY
    assert len(auto_submits) == 1
    assert auto_submits[0] == ("नमस्ते आरना", "hi")
    ctrl.shutdown()


def test_controller_barge_in_and_speech_onset(qapp):
    """Verify speech onset signal forwards properly and barge-in permits recording during LLM gen."""
    cfg = VoiceConfig(barge_in_enabled=True)
    llm_active = True
    ctrl = VoiceController(config=cfg, is_llm_generating=lambda: llm_active)

    onset_signals = []
    ctrl.speech_onset_detected.connect(lambda: onset_signals.append(True))

    with pytest.MonkeyPatch.context() as mp:
        mock_worker_cls = MagicMock()
        mock_worker_instance = MagicMock()
        mock_worker_cls.return_value = mock_worker_instance
        mp.setattr("src.voice.controller.AudioCaptureWorker", mock_worker_cls)

        # barge_in_enabled=True allows recording startup while LLM is generating
        ctrl.start_recording()
        assert ctrl.current_state == VoiceState.REQUESTING_DEVICE

        # Worker detects speech onset
        ctrl._on_speech_onset()
        assert len(onset_signals) == 1

    ctrl.shutdown()


def test_controller_dynamic_stt_provider_switching(qapp, monkeypatch):
    """Test switching STT provider dynamically between Groq and local without silent changes."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_dummykey1234567890abcdef1234567890abcdef12345678")
    cfg = VoiceConfig(stt_provider="groq", cloud_stt_enabled=True, cloud_consent_granted=True)
    ctrl = VoiceController(config=cfg)

    provider_events = []
    ctrl.stt_provider_changed.connect(lambda p: provider_events.append(p))

    # Initial state should be Groq
    assert ctrl.config.stt_provider == "groq"

    # Switch to local
    ctrl.set_stt_provider("local")
    assert ctrl.config.stt_provider == "local"
    assert ctrl.config.cloud_stt_enabled is False
    assert len(provider_events) == 1
    assert provider_events[0] == "local"

    # Switch back to groq
    ctrl.set_stt_provider("groq")
    assert ctrl.config.stt_provider == "groq"
    assert ctrl.config.cloud_stt_enabled is True
    assert len(provider_events) == 2
    assert provider_events[1] == "groq"

    # Invalid provider rejected
    with pytest.raises(ValueError):
        ctrl.set_stt_provider("invalid_provider")

    ctrl.shutdown()


def test_controller_dynamic_conversation_mode_switching(qapp):
    """Test switching conversation mode between Hands-Free and Push-to-Talk."""
    cfg = VoiceConfig(conversation_mode="push_to_talk")
    ctrl = VoiceController(config=cfg)

    mode_events = []
    ctrl.conversation_mode_changed.connect(lambda m: mode_events.append(m))

    assert ctrl.is_hands_free is False

    # Switch to hands-free
    with pytest.MonkeyPatch.context() as mp:
        mock_worker = MagicMock()
        mp.setattr("src.voice.controller.AudioCaptureWorker", MagicMock(return_value=mock_worker))
        ctrl.set_conversation_mode("hands_free")

        assert ctrl.is_hands_free is True
        assert len(mode_events) == 1
        assert mode_events[0] == "hands_free"

        # Switch back to push-to-talk
        ctrl.set_conversation_mode("push_to_talk")
        assert ctrl.is_hands_free is False
        assert len(mode_events) == 2
        assert mode_events[1] == "push_to_talk"

    ctrl.shutdown()


def test_controller_hands_free_pause_resume(qapp):
    """Test pause, resume, and toggle of hands-free continuous listening."""
    cfg = VoiceConfig(conversation_mode="hands_free")
    ctrl = VoiceController(config=cfg)

    pause_events = []
    ctrl.hands_free_paused_changed.connect(lambda p: pause_events.append(p))

    assert ctrl.is_hands_free is True
    assert ctrl.is_hands_free_paused is False

    # Pause hands-free
    ctrl.pause_hands_free()
    assert ctrl.is_hands_free_paused is True
    assert len(pause_events) == 1
    assert pause_events[0] is True

    # Resume hands-free
    with pytest.MonkeyPatch.context() as mp:
        mock_worker = MagicMock()
        mp.setattr("src.voice.controller.AudioCaptureWorker", MagicMock(return_value=mock_worker))
        ctrl.resume_hands_free()
        assert ctrl.is_hands_free_paused is False
        assert len(pause_events) == 2
        assert pause_events[1] is False

        # Toggle hands-free
        ctrl.toggle_hands_free_pause()
        assert ctrl.is_hands_free_paused is True
        ctrl.toggle_hands_free_pause()
        assert ctrl.is_hands_free_paused is False

    ctrl.shutdown()


def test_controller_capture_error_pauses_hands_free(qapp):
    """Test that a capture error in hands-free mode pauses hands-free to avoid error loops."""
    cfg = VoiceConfig(conversation_mode="hands_free")
    ctrl = VoiceController(config=cfg)

    pause_events = []
    ctrl.hands_free_paused_changed.connect(lambda p: pause_events.append(p))

    # Simulate hardware error during capture
    ctrl._on_capture_error(
        category=VoiceErrorCategory.DEVICE_UNAVAILABLE.value,
        msg="Microphone disconnected",
        hint="Plug in microphone",
    )

    # Hands-free should automatically pause
    assert ctrl.is_hands_free_paused is True
    assert len(pause_events) == 1
    assert pause_events[0] is True

    ctrl.shutdown()
