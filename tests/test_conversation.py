"""Tests for conversation management and context maintenance."""

import pytest
from src.ai.conversation import ConversationManager, ChatMessage


def test_conversation_empty_init():
    """Verify clean initialization."""
    mgr = ConversationManager(system_prompt="Test System Prompt", max_history=10)
    assert len(mgr) == 0
    assert mgr.get_history() == []

    api_payload = mgr.get_messages_for_api()
    assert len(api_payload) == 1
    assert api_payload[0]["role"] == "system"
    assert "Test System Prompt" in api_payload[0]["content"]

    raw_payload = mgr.get_messages_for_api(include_time_context=False)
    assert raw_payload[0] == {"role": "system", "content": "Test System Prompt"}


def test_add_user_and_assistant_messages():
    """Verify adding messages and retrieving API payload."""
    mgr = ConversationManager(system_prompt="System", max_history=5)
    mgr.add_user_message("Hello AARNA")
    mgr.add_assistant_message("Greetings, sir.")

    assert len(mgr) == 2
    history = mgr.get_history()
    assert history[0].role == "user"
    assert history[0].content == "Hello AARNA"
    assert history[1].role == "assistant"
    assert history[1].content == "Greetings, sir."

    api_payload = mgr.get_messages_for_api()
    assert len(api_payload) == 3
    assert api_payload[0]["role"] == "system"
    assert "System" in api_payload[0]["content"]
    assert api_payload[1] == {"role": "user", "content": "Hello AARNA"}
    assert api_payload[2] == {"role": "assistant", "content": "Greetings, sir."}


def test_reject_empty_user_message():
    """Ensure empty or whitespace-only messages are rejected."""
    mgr = ConversationManager()
    with pytest.raises(ValueError, match="cannot be empty"):
        mgr.add_user_message("   ")

    with pytest.raises(ValueError, match="cannot be empty"):
        mgr.add_user_message("")


def test_context_window_trimming():
    """Verify that messages older than max_history are excluded from the API payload."""
    mgr = ConversationManager(system_prompt="Sys", max_history=4)
    for i in range(10):
        mgr.add_user_message(f"Query {i}")
        mgr.add_assistant_message(f"Answer {i}")

    assert len(mgr) == 20  # Full history is preserved

    payload = mgr.get_messages_for_api()
    # 1 system message + 4 most recent messages
    assert len(payload) == 5
    assert payload[0]["role"] == "system"
    assert payload[1]["content"] == "Query 8"
    assert payload[2]["content"] == "Answer 8"
    assert payload[3]["content"] == "Query 9"
    assert payload[4]["content"] == "Answer 9"


def test_clear_conversation():
    """Verify conversation reset."""
    mgr = ConversationManager()
    mgr.add_user_message("Test message")
    mgr.add_assistant_message("Test reply")
    assert len(mgr) == 2

    mgr.clear()
    assert len(mgr) == 0
    assert len(mgr.get_messages_for_api()) == 1  # Only system prompt
