"""
Automated tests for System Time Service and ground-truth clock resolution.
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import patch
import pytest

from src.core.time_service import TimeService, SystemTimeInfo, SystemClockError
from src.ai.conversation import ConversationManager


def test_system_local_aware_time():
    """Verify that TimeService retrieves an aware datetime matching the OS clock."""
    info = TimeService.get_current_time()
    raw_now = datetime.now().astimezone()

    assert info.local_datetime.tzinfo is not None
    assert info.timezone_name == raw_now.tzname()
    assert info.utc_offset_str == raw_now.strftime("%z")[:3] + ":" + raw_now.strftime("%z")[3:]
    assert info.date_short == raw_now.strftime("%Y-%m-%d")
    assert info.day_of_week == raw_now.strftime("%A")


def test_ist_detection_and_chennai_equivalence():
    """Verify that an IST datetime correctly identifies Indian timezone equivalence."""
    # Create fixed mock IST datetime: UTC+05:30
    ist_tz = timezone(timedelta(hours=5, minutes=30), name="India Standard Time")
    mock_dt = datetime(2026, 11, 6, 3, 40, 0, tzinfo=ist_tz)

    info = TimeService.get_current_time(target_dt=mock_dt)
    assert info.is_indian_timezone is True
    assert info.time_12h == "03:40 AM"
    assert info.utc_offset_str == "+05:30"
    assert info.timezone_name == "India Standard Time"

    # Test "What time is it in Chennai?"
    res_chennai = TimeService.try_resolve_exact_time_query("What time is it in Chennai?", time_info=info)
    assert res_chennai is not None
    assert "Chennai is 03:40 AM" in res_chennai
    assert "India Standard Time" in res_chennai
    assert "matches your workstation's local time exactly" in res_chennai


def test_direct_time_query_resolution():
    """Verify deterministic resolution of 'What time is it?'."""
    mock_tz = timezone(timedelta(hours=5, minutes=30), name="India Standard Time")
    mock_dt = datetime(2026, 11, 6, 3, 45, 12, tzinfo=mock_tz)
    info = TimeService.get_current_time(target_dt=mock_dt)

    queries = [
        "What time is it?",
        "what time is it",
        "What's the time?",
        "current time",
        "Tell me the time",
        "what is the current time"
    ]
    for q in queries:
        ans = TimeService.try_resolve_exact_time_query(q, time_info=info)
        assert ans is not None, f"Failed for query: {q}"
        assert "03:45 AM" in ans
        assert "India Standard Time" in ans


def test_direct_date_query_resolution():
    """Verify deterministic resolution of 'What is today's date?'."""
    mock_tz = timezone(timedelta(hours=5, minutes=30), name="India Standard Time")
    mock_dt = datetime(2026, 11, 6, 3, 45, 12, tzinfo=mock_tz)
    info = TimeService.get_current_time(target_dt=mock_dt)

    queries = [
        "What is today's date?",
        "what's today's date",
        "Today's date",
        "What day is it today?",
        "which day is today"
    ]
    for q in queries:
        ans = TimeService.try_resolve_exact_time_query(q, time_info=info)
        assert ans is not None, f"Failed for query: {q}"
        assert "Friday, November 06, 2026" in ans or "Friday, November 6, 2026" in ans
        assert "Friday" in ans


def test_non_time_query_returns_none():
    """Verify that general questions are not intercepted by the time resolver."""
    queries = [
        "Who is Tony Stark?",
        "Calculate 25 * 4",
        "How do I sort a list in Python?",
        "Time travel in physics"
    ]
    for q in queries:
        assert TimeService.try_resolve_exact_time_query(q) is None


def test_conversation_injects_live_time_context():
    """Verify that ConversationManager injects live host clock into API system prompt."""
    conv = ConversationManager(system_prompt="Base System Prompt")
    payload = conv.get_messages_for_api()

    assert len(payload) == 1
    system_msg = payload[0]["content"]
    assert "Base System Prompt" in system_msg
    assert "[WORKSTATION GROUND TRUTH CLOCK]" in system_msg
    assert datetime.now().astimezone().strftime("%Y") in system_msg


def test_system_clock_failure_handling():
    """Verify graceful handling when system clock retrieval fails."""
    with patch("src.core.time_service.datetime") as mock_dt:
        mock_dt.now.side_effect = OSError("Hardware clock inaccessible")
        with pytest.raises(SystemClockError, match="Failed to retrieve workstation clock"):
            TimeService.get_current_time()
