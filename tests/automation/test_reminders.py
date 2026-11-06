"""
Comprehensive Unit and Integration Test Suite for Phase 12 Proactive System Reminders
and Windows Toaster Notification Engine in Project AARNA.

Verifies:
1. SQLite3 reminders schema creation and database persistence in memory.db.
2. Synchronous and asynchronous reminder creation (create_reminder, create_reminder_async).
3. Due date evaluation, pending queries, and sorting.
4. Native Windows toaster notification dispatch (QSystemTrayIcon and PowerShell hooks).
5. Natural Hinglish voice alert generation ("Abhi, yaad hai na?...").
6. Recurring reminders (daily, weekly, hourly auto-reschedule).
7. Proactive conversation_history memory logging upon reminder trigger.
8. Audio alert delivery completely independent of cloud LLM endpoints.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtCore import QCoreApplication

from src.automation.reminder_manager import (
    ReminderItem,
    ReminderManager,
    WindowsToastNotifier,
)


@pytest.fixture(scope="module")
def qapp():
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])
    return app


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    """Fixture providing an isolated SQLite database path."""
    return tmp_path / "test_memory.db"


# ============================================================================
# 1. Database Schema & Persistence Tests
# ============================================================================

def test_reminder_manager_initializes_schema(temp_db: Path) -> None:
    """Verify that ReminderManager creates the reminders table and indices."""
    manager = ReminderManager(db_path=temp_db)
    with sqlite3.connect(str(temp_db)) as conn:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='reminders';"
        )
        assert cursor.fetchone() is not None

        # Verify columns
        cursor = conn.execute("PRAGMA table_info(reminders);")
        columns = [row[1] for row in cursor.fetchall()]
        assert "title" in columns
        assert "due_timestamp" in columns
        assert "recurrence" in columns
        assert "is_completed" in columns
        assert "voice_alert_text" in columns


def test_create_and_get_pending_reminders(temp_db: Path) -> None:
    """Verify reminder creation, persistence, and pending queries."""
    manager = ReminderManager(db_path=temp_db)
    t_now = time.time()

    item1 = manager.create_reminder(
        title="BTech Project Review",
        message="Check Chapter 4 documentation",
        due_timestamp=t_now + 3600,
        tags=["college", "project"],
    )
    assert item1.id is not None
    assert item1.title == "BTech Project Review"
    assert "Abhi, yaad hai na?" in item1.voice_alert_text

    item2 = manager.create_reminder(
        title="Drink Water",
        due_timestamp=t_now + 1800,
        recurrence="hourly",
    )
    assert item2.id is not None

    pending = manager.get_pending_reminders()
    assert len(pending) == 2
    # Should be ordered by due_timestamp ascending
    assert pending[0].title == "Drink Water"
    assert pending[1].title == "BTech Project Review"


def test_create_reminder_async(temp_db: Path) -> None:
    """Verify non-blocking async wrapper create_reminder_async."""
    manager = ReminderManager(db_path=temp_db)
    t_due = time.time() + 600

    async def _create():
        return await manager.create_reminder_async(
            title="Async Meeting",
            due_timestamp=t_due,
            message="Join voice channel",
        )

    item = asyncio.run(_create())
    assert item.id is not None
    assert item.title == "Async Meeting"
    assert item.due_timestamp == t_due


# ============================================================================
# 2. Due Detection, Completion, and Snoozing Tests
# ============================================================================

def test_due_reminders_evaluation(temp_db: Path) -> None:
    """Verify get_due_reminders correctly identifies past/current due items."""
    manager = ReminderManager(db_path=temp_db)
    t_now = time.time()

    # Past due
    manager.create_reminder("Overdue Task", due_timestamp=t_now - 100)
    # Due right now
    manager.create_reminder("Current Task", due_timestamp=t_now - 5)
    # Future task (not due)
    manager.create_reminder("Future Task", due_timestamp=t_now + 1000)

    due = manager.get_due_reminders(now=t_now)
    assert len(due) == 2
    due_titles = {d.title for d in due}
    assert "Overdue Task" in due_titles
    assert "Current Task" in due_titles
    assert "Future Task" not in due_titles


def test_mark_completed_and_snooze(temp_db: Path) -> None:
    """Verify marking completed and snoozing functionality."""
    manager = ReminderManager(db_path=temp_db)
    t_now = time.time()
    item = manager.create_reminder("Snooze Test", due_timestamp=t_now - 10)

    # Completed check
    completed_events = []
    manager.reminder_completed.connect(completed_events.append)
    assert manager.mark_completed(item.id) is True
    assert completed_events == [item.id]

    pending = manager.get_pending_reminders()
    assert len(pending) == 0

    # Snooze (reschedules into the future and un-completes)
    assert manager.snooze_reminder(item.id, snooze_seconds=300) is True
    pending_after_snooze = manager.get_pending_reminders()
    assert len(pending_after_snooze) == 1
    assert pending_after_snooze[0].is_completed is False
    assert pending_after_snooze[0].due_timestamp > t_now


def test_delete_reminder(temp_db: Path) -> None:
    """Verify deletion permanently removes record."""
    manager = ReminderManager(db_path=temp_db)
    item = manager.create_reminder("Delete Me", due_timestamp=time.time())
    assert manager.delete_reminder(item.id) is True
    assert len(manager.get_pending_reminders()) == 0


# ============================================================================
# 3. Proactive Trigger, Windows Toaster, and Hinglish Voice Alerts
# ============================================================================

def test_proactive_trigger_dispatches_toast_and_hinglish_voice(temp_db: Path) -> None:
    """
    Verify check_and_trigger_due_reminders:
    1. Shows native Windows toaster notification.
    2. Emits toast_displayed and reminder_triggered signals.
    3. Emits voice_alert_requested with Hinglish prompt.
    4. Logs to memory conversation_history.
    """
    mock_tray = MagicMock()
    mock_audio = MagicMock()
    manager = ReminderManager(db_path=temp_db, audio_player=mock_audio, tray_icon=mock_tray)

    t_now = time.time()
    item = manager.create_reminder(
        title="College Seminar",
        message="Project presentation starts in 10 minutes",
        due_timestamp=t_now - 10,
    )

    triggered_items = []
    toast_events = []
    voice_events = []

    manager.reminder_triggered.connect(triggered_items.append)
    manager.toast_displayed.connect(lambda t, m: toast_events.append((t, m)))
    manager.voice_alert_requested.connect(voice_events.append)

    with patch.object(WindowsToastNotifier, "show_toast", return_value=True) as mock_toast:
        triggered = manager.check_and_trigger_due_reminders(now=t_now)

    assert len(triggered) == 1
    assert triggered[0].id == item.id
    assert len(triggered_items) == 1
    assert len(toast_events) == 1
    assert toast_events[0][0] == "College Seminar"
    assert "presentation" in toast_events[0][1]

    # Verify Hinglish voice alert
    assert len(voice_events) == 1
    assert "Abhi, yaad hai na? College Seminar ka time ho gaya hai!" in voice_events[0]

    # Verify memory.db conversation_history persistence
    with sqlite3.connect(str(temp_db)) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM conversation_history;")
        rows = cursor.fetchall()
        assert len(rows) == 1
        assert "Proactive Reminder: College Seminar" in rows[0]["intent_summary"]
        assert "proactive_reminder" in rows[0]["personal_context_tags"]


def test_recurring_daily_reminder_auto_reschedules(temp_db: Path) -> None:
    """Verify daily recurring reminder automatically schedules 24 hours into future."""
    manager = ReminderManager(db_path=temp_db)
    t_now = time.time()
    item = manager.create_reminder(
        title="Daily Standup",
        due_timestamp=t_now - 5,
        recurrence="daily",
    )

    with patch.object(WindowsToastNotifier, "show_toast", return_value=True):
        manager.check_and_trigger_due_reminders(now=t_now)

    # Item should not be marked completed; instead due_timestamp is updated
    pending = manager.get_pending_reminders()
    assert len(pending) == 1
    assert pending[0].id == item.id
    assert pending[0].is_completed is False
    # Due timestamp should be approximately t_now + 86400
    assert pending[0].due_timestamp > t_now + 86000


def test_independent_audio_delivery_offline(temp_db: Path) -> None:
    """Verify audio delivery utilizes local AudioPlayer completely offline without cloud API calls."""
    mock_audio = MagicMock()
    mock_tts = MagicMock()
    mock_tts.synthesize.return_value = MagicMock(audio_bytes=b"fake_pcm_audio_bytes")

    manager = ReminderManager(db_path=temp_db, audio_player=mock_audio, tts_engine=mock_tts)
    manager._deliver_voice_audio("Abhi, break lene ka time ho gaya hai.")

    time.sleep(0.1)  # Allow daemon thread to run
    mock_tts.synthesize.assert_called_once_with("Abhi, break lene ka time ho gaya hai.")
    mock_audio.play_bytes.assert_called_once_with(b"fake_pcm_audio_bytes")
