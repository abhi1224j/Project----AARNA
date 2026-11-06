"""
Automated unit and integration tests for Phase 13 Cyberpunk Glassmorphism Dashboard
and Sound-Reactive Lip Sync Engine in Project AARNA.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QGridLayout, QProgressBar, QPlainTextEdit

from src.core.config import AppConfig
from src.ui.avatar.state import AvatarState
from src.ui.main_window import AarnaMainWindow


@pytest.fixture(scope="session")
def qapp():
    """Ensure QApplication instance in headless/offscreen mode for GUI tests."""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def test_cyberpunk_glassmorphism_dashboard_layout(qapp):
    """Verify structured QGridLayout wrapper and the 5 frosted-glass panel cards."""
    config = AppConfig(default_model="qwen2.5:3b", wake_word="Aarna")
    window = AarnaMainWindow(config=config, enable_webengine=False)

    # 1. Verify Grid Layout Wrapper
    assert hasattr(window, "dashboard_grid")
    assert isinstance(window.dashboard_grid, QGridLayout)

    # 2. Verify Left Telemetry Board
    assert hasattr(window, "telemetry_left_panel")
    assert window.telemetry_left_panel.objectName() == "telemetry_left_panel"

    # 3. Verify Center Avatar Stage Card
    assert hasattr(window, "avatar_stage_card")
    assert window.avatar_stage_card.objectName() == "avatar_stage_card"
    assert hasattr(window, "avatar_widget")
    assert window.avatar_widget is not None

    # 4. Verify Center-Right AI Chat Terminal Panel
    assert hasattr(window, "chat_terminal_panel")
    assert window.chat_terminal_panel.objectName() == "chat_terminal_panel"
    assert hasattr(window, "chat_pane")
    assert hasattr(window, "chat_layout")
    assert hasattr(window, "input_edit")

    # 5. Verify Right GPU & Routing Telemetry Panel
    assert hasattr(window, "telemetry_right_panel")
    assert window.telemetry_right_panel.objectName() == "telemetry_right_panel"

    # 6. Verify Lower Scrolling Operations Log Feed Panel
    assert hasattr(window, "ops_log_panel")
    assert window.ops_log_panel.objectName() == "ops_log_panel"
    assert hasattr(window, "ops_log_terminal")
    assert isinstance(window.ops_log_terminal, QPlainTextEdit)
    assert window.ops_log_terminal.isReadOnly() is True

    window.close()
    qapp.processEvents()
    if window.status_worker and window.status_worker.isRunning():
        window.status_worker.wait(3000)


def test_hardware_telemetry_dials_and_update(qapp):
    """Verify hardware gauges (CPU, RAM, GPU, VRAM) and periodic telemetry updates."""
    config = AppConfig(default_model="qwen2.5:3b", wake_word="Aarna")
    window = AarnaMainWindow(config=config, enable_webengine=False)

    # Verify telemetry bars exist
    assert hasattr(window, "cpu_bar")
    assert isinstance(window.cpu_bar, QProgressBar)
    assert hasattr(window, "ram_bar")
    assert isinstance(window.ram_bar, QProgressBar)
    assert hasattr(window, "gpu_bar")
    assert isinstance(window.gpu_bar, QProgressBar)
    assert hasattr(window, "vram_bar")
    assert isinstance(window.vram_bar, QProgressBar)
    assert hasattr(window, "voice_level_bar")
    assert isinstance(window.voice_level_bar, QProgressBar)

    # Execute telemetry refresh cycle
    window._update_hardware_telemetry()
    assert 0 <= window.cpu_bar.value() <= 100
    assert 0 <= window.ram_bar.value() <= 100
    assert 0 <= window.gpu_bar.value() <= 100
    assert 0 <= window.vram_bar.value() <= 100

    assert len(window.cpu_val_label.text()) > 0
    assert len(window.ram_val_label.text()) > 0
    assert len(window.gpu_val_label.text()) > 0

    window.close()
    qapp.processEvents()
    if window.status_worker and window.status_worker.isRunning():
        window.status_worker.wait(3000)


def test_scrolling_operations_log_feed(qapp):
    """Verify logging of background automation paths, file edits, and system terminal streams."""
    config = AppConfig(default_model="qwen2.5:3b", wake_word="Aarna")
    window = AarnaMainWindow(config=config, enable_webengine=False)

    initial_text = window.ops_log_terminal.toPlainText()
    assert "SYSTEM" in initial_text
    assert "AVATAR" in initial_text

    # Log automation path
    window.log_operation("OS_AGENT", "Dispatched PowerShell launch for VS Code")
    window.log_operation("VISION", "Screen captured to in-memory buffer (io.BytesIO)")
    window.log_operation("REMINDERS", "SQLite reminder item checked against due timestamp")

    updated_text = window.ops_log_terminal.toPlainText()
    assert "[OS_AGENT] Dispatched PowerShell launch for VS Code" in updated_text
    assert "[VISION] Screen captured to in-memory buffer" in updated_text
    assert "[REMINDERS] SQLite reminder item checked" in updated_text

    window.close()
    qapp.processEvents()
    if window.status_worker and window.status_worker.isRunning():
        window.status_worker.wait(3000)


def test_sound_reactive_lip_sync_amplitude_routing(qapp):
    """Verify live audio amplitude routing to avatar morphs and voice level meter."""
    config = AppConfig(default_model="qwen2.5:3b", wake_word="Aarna")
    window = AarnaMainWindow(config=config, enable_webengine=False)

    # High amplitude during loud vocalization
    window._on_audio_level(0.85)
    assert window.avatar_widget.current_audio_level == pytest.approx(0.85, abs=0.01)
    assert window.voice_level_bar.value() == 85
    assert "85.0%" in window.voice_level_val.text()

    # Mid amplitude
    window._on_audio_level(0.42)
    assert window.avatar_widget.current_audio_level == pytest.approx(0.42, abs=0.01)
    assert window.voice_level_bar.value() == 42

    # Silence
    window._on_audio_level(0.0)
    assert window.avatar_widget.current_audio_level == 0.0
    assert window.voice_level_bar.value() == 0
    assert "Silent" in window.voice_level_val.text()

    window.close()
    qapp.processEvents()
    if window.status_worker and window.status_worker.isRunning():
        window.status_worker.wait(3000)


def test_speech_finish_and_interruption_neutral_pose_reset(qapp):
    """Verify jaw and mouth vertex meshes return to neutral pose immediately upon silence/barge-in."""
    config = AppConfig(default_model="qwen2.5:3b", wake_word="Aarna")
    window = AarnaMainWindow(config=config, enable_webengine=False)

    mock_player = MagicMock()
    mock_player.is_playing = False
    window.audio_player = mock_player

    # 1. Speech finishes cleanly -> level snaps to 0.0
    window._on_speech_started()
    window._on_audio_level(0.68)
    assert window.avatar_widget.current_audio_level > 0.0
    window._on_speech_finished()
    assert window.avatar_widget.current_audio_level == 0.0
    assert window.voice_level_bar.value() == 0

    # 2. Active speech cancelled -> level snaps to 0.0
    window._on_speech_started()
    window._on_audio_level(0.74)
    assert window.avatar_widget.current_audio_level > 0.0
    window._cancel_active_speech()
    assert window.avatar_widget.current_audio_level == 0.0
    assert window.voice_level_bar.value() == 0

    # 3. User barge-in speech onset -> level snaps to 0.0
    window._on_speech_started()
    window._on_audio_level(0.82)
    assert window.avatar_widget.current_audio_level > 0.0
    window._on_user_speech_onset()
    assert window.avatar_widget.current_audio_level == 0.0
    assert window.voice_level_bar.value() == 0

    window.close()
    qapp.processEvents()
    if window.status_worker and window.status_worker.isRunning():
        window.status_worker.wait(3000)


def test_avatar_show_hide_dashboard_card_visibility(qapp):
    """Verify avatar show/hide toggle toggles avatar_stage_card in the dashboard grid."""
    config = AppConfig(default_model="qwen2.5:3b", wake_word="Aarna")
    window = AarnaMainWindow(config=config, enable_webengine=False)
    window.show()

    assert window.is_avatar_visible is True
    assert not window.avatar_stage_card.isHidden()

    # Hide avatar
    window.set_avatar_visible(False)
    assert window.is_avatar_visible is False
    assert window.avatar_stage_card.isVisible() is False
    assert window.toggle_avatar_btn.text() == "Show Avatar"

    # Show avatar
    window.set_avatar_visible(True)
    assert window.is_avatar_visible is True
    assert window.avatar_stage_card.isVisible() is True
    assert window.toggle_avatar_btn.text() == "Hide Avatar"

    window.close()
    qapp.processEvents()
    if window.status_worker and window.status_worker.isRunning():
        window.status_worker.wait(3000)
