"""Unit tests for VoiceStateMachine lifecycle, lock-free callbacks, and transitions."""

import threading
import pytest

from src.voice.exceptions import InvalidVoiceStateTransitionError
from src.voice.state import (
    VALID_VOICE_STATE_TRANSITIONS,
    VoiceState,
    VoiceStateMachine,
)


def test_state_machine_happy_path():
    sm = VoiceStateMachine(initial_state=VoiceState.READY)
    assert sm.state == VoiceState.READY
    assert sm.is_busy is False
    assert sm.is_recording is False

    # Start push-to-talk
    sm.transition(VoiceState.REQUESTING_DEVICE)
    assert sm.state == VoiceState.REQUESTING_DEVICE
    assert sm.is_recording is True
    assert sm.is_busy is True

    # Stream opened
    sm.transition(VoiceState.LISTENING)
    assert sm.state == VoiceState.LISTENING
    assert sm.is_recording is True

    # User clicks stop
    sm.transition(VoiceState.STOPPING)
    assert sm.state == VoiceState.STOPPING
    assert sm.is_recording is False

    # Audio preprocessing
    sm.transition(VoiceState.PROCESSING_AUDIO)
    assert sm.state == VoiceState.PROCESSING_AUDIO

    # Local Whisper STT
    sm.transition(VoiceState.TRANSCRIBING)
    assert sm.state == VoiceState.TRANSCRIBING

    # Review in UI
    sm.transition(VoiceState.REVIEW)
    assert sm.state == VoiceState.REVIEW
    assert sm.is_recording is False
    assert sm.is_busy is False

    # User confirms transcript
    sm.transition(VoiceState.SUBMITTING)
    assert sm.state == VoiceState.SUBMITTING
    assert sm.is_busy is True

    # Submission completed -> back to READY
    sm.transition(VoiceState.READY)
    assert sm.state == VoiceState.READY
    assert sm.is_busy is False


def test_review_retranscribe_retry():
    # User in REVIEW changes language mode (e.g. Auto -> Hindi) and clicks Re-Transcribe
    sm = VoiceStateMachine(initial_state=VoiceState.REVIEW)
    assert sm.can_transition_to(VoiceState.TRANSCRIBING) is True

    sm.transition(VoiceState.TRANSCRIBING)
    assert sm.state == VoiceState.TRANSCRIBING

    sm.transition(VoiceState.REVIEW)
    assert sm.state == VoiceState.REVIEW


def test_review_discard_and_cancel():
    # User discards transcript from review
    sm = VoiceStateMachine(initial_state=VoiceState.REVIEW)
    assert sm.can_transition_to(VoiceState.CANCELLED) is True

    sm.transition(VoiceState.CANCELLED)
    assert sm.state == VoiceState.CANCELLED

    sm.transition(VoiceState.READY)
    assert sm.state == VoiceState.READY


def test_invalid_transition_raises_typed_error():
    sm = VoiceStateMachine(initial_state=VoiceState.READY)

    # Unknown string state
    with pytest.raises(InvalidVoiceStateTransitionError) as exc_info:
        sm.transition("unknown_state")
    assert "Invalid voice input state" in exc_info.value.user_message

    # Illegal transition (skipping REQUESTING_DEVICE directly to LISTENING)
    with pytest.raises(InvalidVoiceStateTransitionError) as exc_info:
        sm.transition(VoiceState.LISTENING)
    assert "Cannot perform action while voice input is in 'ready' state" in exc_info.value.user_message


def test_callback_executed_outside_lock():
    # Verify that listener callbacks are executed outside the lock, preventing deadlocks
    sm = VoiceStateMachine(initial_state=VoiceState.REQUESTING_DEVICE)
    lock_acquired_in_callback = False

    def on_transition(prev, current, reason):
        nonlocal lock_acquired_in_callback
        # Another thread should be able to acquire sm._lock while callback is running
        def try_lock():
            nonlocal lock_acquired_in_callback
            if sm._lock.acquire(timeout=0.5):
                lock_acquired_in_callback = True
                sm._lock.release()

        t = threading.Thread(target=try_lock)
        t.start()
        t.join()

    sm._on_transition = on_transition
    sm.transition(VoiceState.LISTENING)
    assert lock_acquired_in_callback is True


def test_callback_exception_does_not_corrupt_state():
    sm = VoiceStateMachine(initial_state=VoiceState.READY)

    def failing_callback(prev, curr, reason):
        raise RuntimeError("Simulated UI listener failure")

    sm._on_transition = failing_callback

    # Transition should succeed and commit state despite listener exception
    sm.transition(VoiceState.REQUESTING_DEVICE)
    assert sm.state == VoiceState.REQUESTING_DEVICE


def test_resource_busy_check_prevents_ready_transition():
    worker_running = True

    def is_resource_busy():
        return worker_running

    sm = VoiceStateMachine(
        initial_state=VoiceState.SUBMITTING,
        is_resource_busy_check=is_resource_busy,
    )

    # While worker is running, cannot transition to READY
    assert sm.can_transition_to(VoiceState.READY) is False
    with pytest.raises(InvalidVoiceStateTransitionError, match="background audio or worker resources remain active"):
        sm.transition(VoiceState.READY)

    # Once worker completes, transition succeeds
    worker_running = False
    assert sm.can_transition_to(VoiceState.READY) is True
    sm.transition(VoiceState.READY)
    assert sm.state == VoiceState.READY


def test_state_machine_shutdown():
    sm = VoiceStateMachine(initial_state=VoiceState.LISTENING)
    sm.shutdown(reason="window_closed")
    assert sm.state == VoiceState.DISABLED
    assert sm.can_transition_to(VoiceState.READY) is True


def test_state_machine_history_bounded():
    sm = VoiceStateMachine(initial_state=VoiceState.READY, max_history=10)
    for _ in range(25):
        sm.transition(VoiceState.REQUESTING_DEVICE)
        sm.transition(VoiceState.CANCELLED)
        sm.transition(VoiceState.READY)

    assert len(sm.history) <= 10


def test_recover_to_ready():
    # From ERROR state
    sm = VoiceStateMachine(initial_state=VoiceState.ERROR)
    sm.recover_to_ready()
    assert sm.state == VoiceState.READY

    # From LISTENING state (steps through CANCELLED -> READY)
    sm2 = VoiceStateMachine(initial_state=VoiceState.LISTENING)
    sm2.recover_to_ready()
    assert sm2.state == VoiceState.READY
    assert VoiceState.CANCELLED in sm2.history
