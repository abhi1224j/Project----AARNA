"""Automated GUI lifecycle and interaction tests for AarnaMainWindow."""

import os
import sys
from pathlib import Path
import pytest

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from src.core.config import AppConfig
from src.ui.main_window import AarnaMainWindow


@pytest.fixture(scope="session")
def qapp():
    """Ensure a single QApplication instance exists for the test session."""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def test_main_window_lifecycle_and_offline_mode(qapp):
    """Test full window initialization, empty inputs, offline messaging, and clear."""
    config = AppConfig(default_model="qwen2.5:3b", wake_word="Aarna")
    window = AarnaMainWindow(config=config)

    # Initial UI state assertions
    assert "AARNA" in window.windowTitle()
    assert window.config.wake_word == "Aarna"
    assert window.config.require_wake_word_prefix is False
    assert window.is_connected is False

    # Check welcome bubble was added
    initial_bubbles = window.chat_layout.count()
    assert initial_bubbles == 2  # Welcome bubble + trailing stretch

    # Test empty message does not add bubbles
    window.input_edit.setPlainText("   ")
    window.transmit_message()
    assert window.chat_layout.count() == initial_bubbles

    # Test offline message handling: adds user message + informative offline bubble
    window.input_edit.setPlainText("Hello AARNA")
    window.transmit_message()
    assert window.chat_layout.count() == initial_bubbles + 2  # User bubble + Assistant offline bubble

    # Test clear conversation
    window.clear_conversation()
    assert window.chat_layout.count() == 2  # Reset back to single cleared acknowledgment bubble + stretch

    # Clean shutdown
    window.close()
    qapp.processEvents()
    if window.status_worker and window.status_worker.isRunning():
        window.status_worker.wait(4000)


def test_gui_time_queries_resolution(qapp):
    """Test deterministic ground-truth clock resolution directly in the UI."""
    config = AppConfig(default_model="qwen2.5:3b", wake_word="Aarna")
    window = AarnaMainWindow(config=config)

    # 1. Ask for time in Chennai
    window.input_edit.setPlainText("What time is it in Chennai?")
    window.transmit_message()

    # The last assistant message bubble should contain Chennai and current time
    history = window.conversation.get_history()
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].content == "What time is it in Chennai?"
    assert history[1].role == "assistant"
    assert "Chennai" in history[1].content
    assert "India Standard Time" in history[1].content or "IST" in history[1].content or "UTC+05:30" in history[1].content

    # 2. Ask "What is today's date?"
    window.input_edit.setPlainText("What is today's date?")
    window.transmit_message()
    history = window.conversation.get_history()
    assert len(history) == 4
    assert "Today is" in history[3].content

    # 3. Ask "What time is it?"
    window.input_edit.setPlainText("What time is it?")
    window.transmit_message()
    history = window.conversation.get_history()
    assert len(history) == 6
    assert "The current local time is" in history[5].content

    window.close()
    qapp.processEvents()
    if window.status_worker and window.status_worker.isRunning():
        window.status_worker.wait(4000)


def test_gui_tts_integration_and_avatar_speaking_sync(qapp):
    """Test TTS speech triggering, speech cancellation, and avatar SPEAKING/IDLE transitions."""
    from unittest.mock import MagicMock
    from src.ui.avatar.state import AvatarState

    config = AppConfig(default_model="qwen3.5:4b", wake_word="Aarna")
    window = AarnaMainWindow(config=config, enable_webengine=False)

    # Verify TTS engine and player initialized
    assert hasattr(window, "tts_engine")
    assert hasattr(window, "audio_player")

    # Mock audio_player to avoid actual sound card output in automated tests
    mock_player = MagicMock()
    mock_player.is_playing = False
    window.audio_player = mock_player

    # 1. Test speech started transition
    window._on_speech_started()
    assert window.avatar_widget.current_state == AvatarState.SPEAKING
    assert "speaking" in window.footer_status.text().lower()

    # 2. Test speech finished transition
    window._on_speech_finished()
    assert window.avatar_widget.current_state == AvatarState.IDLE
    assert "ready" in window.footer_status.text().lower()

    # 3. Test speech cancellation when user transmits new message or stops
    window._on_speech_started()
    assert window.avatar_widget.current_state == AvatarState.SPEAKING
    window._cancel_active_speech()
    assert window.avatar_widget.current_state == AvatarState.IDLE
    assert mock_player.stop.called

    window.close()
    qapp.processEvents()
    if window.status_worker and window.status_worker.isRunning():
        window.status_worker.wait(4000)


def test_gui_voice_autosubmit_and_barge_in(qapp):
    """Test voice transcript auto-submission and VAD speech onset barge-in interruption in GUI."""
    from unittest.mock import MagicMock
    from src.ui.avatar.state import AvatarState

    config = AppConfig(default_model="qwen3.5:4b", wake_word="Aarna")
    window = AarnaMainWindow(config=config, enable_webengine=False)

    mock_player = MagicMock()
    mock_player.is_playing = False
    window.audio_player = mock_player

    # 1. Test auto-submit workflow
    window._on_voice_transcript_auto_submitted("What time is it in Chennai?", "en")
    history = window.conversation.get_history()
    assert len(history) == 2
    assert "Chennai" in history[1].content

    # 2. Test barge-in: AARNA is speaking, user speech onset interrupts playback
    window._on_speech_started()
    assert window.avatar_widget.current_state == AvatarState.SPEAKING

    # User starts speaking
    window._on_user_speech_onset()
    assert window.avatar_widget.current_state == AvatarState.IDLE
    assert "barge-in" in window.footer_status.text().lower()

    window.close()
    qapp.processEvents()
    if window.status_worker and window.status_worker.isRunning():
        window.status_worker.wait(4000)


def test_gui_streaming_response_speech_forwarding(qapp):
    """Test streaming chunk forwarding to the streaming speech queue."""
    from unittest.mock import MagicMock

    config = AppConfig(default_model="qwen3.5:4b", wake_word="Aarna")
    window = AarnaMainWindow(config=config, enable_webengine=False)

    mock_queue = MagicMock()
    mock_queue.isRunning.return_value = True
    window.streaming_speech = mock_queue

    # Simulate chunk received during LLM generation
    window.active_stream_bubble = MagicMock()
    window._on_chunk_received("Namaste! ")
    assert mock_queue.push_chunk.called
    assert mock_queue.push_chunk.call_args[0][0] == "Namaste! "

    # Simulate finish response
    window._on_finished_response("Namaste! Kaise hain aap?")
    assert mock_queue.finish_stream.called

    window.close()
    qapp.processEvents()
    if window.status_worker and window.status_worker.isRunning():
        window.status_worker.wait(4000)


def test_gui_synchronized_speech_text_reveal(qapp):
    """Test lockstep synchronization between incremental speech and text reveal."""
    from unittest.mock import MagicMock
    from src.ui.main_window import MessageBubble

    config = AppConfig(default_model="qwen3.5:4b", wake_word="Aarna")
    window = AarnaMainWindow(config=config, enable_webengine=False)

    mock_queue = MagicMock()
    mock_queue.isRunning.return_value = True
    window.streaming_speech = mock_queue

    # 1. Start streaming response
    bubble = MessageBubble(role="assistant", content="Thinking...")
    window.active_stream_bubble = bubble
    window._pending_full_response = "Arey, kya hua? Thak gaye ho ya aaj bas mood nahi hai?"

    # 2. While TTS is active, LLM chunk arriving does NOT dump prematurely into bubble
    window._on_chunk_received("Arey, kya hua? ")
    assert bubble.body_label.text() == "Thinking..."

    # 3. Audio begins for first clause -> bubble updates to first clause
    window._on_sentence_started("Arey, kya hua?")
    assert bubble.body_label.text() == "Arey, kya hua?"

    # 4. Audio begins for second clause -> bubble updates to accumulated clauses
    window._on_sentence_started("Thak gaye ho ya aaj bas mood nahi hai?")
    assert bubble.body_label.text() == "Arey, kya hua? Thak gaye ho ya aaj bas mood nahi hai?"

    # 5. Speech finishes -> finalizes bubble
    window._on_speech_finished()
    assert window.active_stream_bubble is None
    assert bubble.body_label.text() == "Arey, kya hua? Thak gaye ho ya aaj bas mood nahi hai?"

    # 6. Test cancellation / barge-in preservation
    bubble2 = MessageBubble(role="assistant", content="Thinking...")
    window.active_stream_bubble = bubble2
    window._spoken_text = "Arey, kya hua?"
    window._pending_full_response = "Arey, kya hua? Thak gaye ho ya aaj bas mood nahi hai?"
    window._on_speech_cancelled()
    assert bubble2.body_label.text() == "Arey, kya hua? —"
    assert window.active_stream_bubble is None

    window.close()
    qapp.processEvents()
    if window.status_worker and window.status_worker.isRunning():
        window.status_worker.wait(4000)
