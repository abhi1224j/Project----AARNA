"""Tests for Ollama HTTP Client."""

import json
from unittest.mock import MagicMock, patch
import requests
import pytest

from src.ai.client import (
    OllamaClient,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
    OllamaError,
)


def test_connection_online():
    """Verify online check when Ollama responds with 200."""
    client = OllamaClient("http://localhost:11434")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"version": "0.4.1"}

    with patch("requests.get", return_value=mock_resp):
        is_online, msg = client.check_connection()
        assert is_online is True
        assert "Online" in msg
        assert "0.4.1" in msg


def test_connection_offline():
    """Verify offline handling when connection is refused."""
    client = OllamaClient("http://localhost:11434")

    with patch("requests.get", side_effect=requests.exceptions.ConnectionError("Connection refused")):
        is_online, msg = client.check_connection()
        assert is_online is False
        assert "Offline" in msg


def test_list_models():
    """Verify parsing of available models from /api/tags."""
    client = OllamaClient("http://localhost:11434")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "models": [
            {"name": "qwen2.5:3b"},
            {"name": "llama3.2:3b"},
        ]
    }

    with patch("requests.get", return_value=mock_resp):
        models = client.list_models()
        assert models == ["qwen2.5:3b", "llama3.2:3b"]
        assert client.is_model_available("qwen2.5:3b") is True
        assert client.is_model_available("mistral") is False


def test_chat_streaming():
    """Verify chunked streaming generator and assembly."""
    client = OllamaClient("http://localhost:11434")

    lines = [
        json.dumps({"message": {"content": "Hello"}}).encode("utf-8"),
        json.dumps({"message": {"content": " sir."}}).encode("utf-8"),
    ]

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_lines.return_value = lines
    mock_resp.__enter__.return_value = mock_resp

    with patch("requests.post", return_value=mock_resp):
        chunks = list(client.chat_stream([{"role": "user", "content": "Hi"}], model="qwen2.5:3b"))
        assert chunks == ["Hello", " sir."]

        full = client.chat_complete([{"role": "user", "content": "Hi"}], model="qwen2.5:3b")
        assert full == "Hello sir."


def test_chat_model_not_found():
    """Verify OllamaModelNotFoundError raised on 404."""
    client = OllamaClient("http://localhost:11434")

    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.__enter__.return_value = mock_resp

    with patch("requests.post", return_value=mock_resp):
        with pytest.raises(OllamaModelNotFoundError, match="is not installed"):
            list(client.chat_stream([{"role": "user", "content": "Hi"}], model="missing-model"))


def test_chat_timeout():
    """Verify OllamaTimeoutError raised on network timeout."""
    client = OllamaClient("http://localhost:11434", timeout=5.0)

    with patch("requests.post", side_effect=requests.exceptions.Timeout("Timed out")):
        with pytest.raises(OllamaTimeoutError, match="timed out"):
            list(client.chat_stream([{"role": "user", "content": "Hi"}], model="qwen2.5:3b"))


def test_chat_streaming_think_parameter():
    """Verify think parameter is passed correctly in payload (default False vs True)."""
    client = OllamaClient("http://localhost:11434")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_lines.return_value = [json.dumps({"message": {"content": "ok"}}).encode("utf-8")]
    mock_resp.__enter__.return_value = mock_resp

    with patch("requests.post", return_value=mock_resp) as mock_post:
        # 1. Default think=False
        list(client.chat_stream([{"role": "user", "content": "Hi"}], model="qwen3.5:4b"))
        sent_payload = mock_post.call_args.kwargs["json"]
        assert sent_payload["think"] is False

        # 2. Explicit think=True
        list(client.chat_stream([{"role": "user", "content": "Hi"}], model="qwen3.5:4b", think=True))
        sent_payload_true = mock_post.call_args.kwargs["json"]
        assert sent_payload_true["think"] is True


def test_chat_streaming_filters_thinking_tokens():
    """Verify reasoning tokens are excluded from content stream and optionally dispatched to callback."""
    client = OllamaClient("http://localhost:11434")

    # Simulate Ollama streaming thinking tokens first, then content answer
    lines = [
        json.dumps({"message": {"thinking": "Thinking step 1: analyze input"}}).encode("utf-8"),
        json.dumps({"message": {"thinking": "\nThinking step 2: formulate answer"}}).encode("utf-8"),
        json.dumps({"message": {"content": "Hello! "}}).encode("utf-8"),
        json.dumps({"message": {"content": "How can I help you?"}}).encode("utf-8"),
    ]

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_lines.return_value = lines
    mock_resp.__enter__.return_value = mock_resp

    thinking_captured = []

    def on_think(chunk: str):
        thinking_captured.append(chunk)

    with patch("requests.post", return_value=mock_resp):
        # Stream with on_thinking_chunk callback
        content_chunks = list(
            client.chat_stream(
                [{"role": "user", "content": "Hi"}],
                model="qwen3.5:4b",
                think=True,
                on_thinking_chunk=on_think,
            )
        )

        # Content generator must contain strictly the answer, never thinking tokens
        assert content_chunks == ["Hello! ", "How can I help you?"]
        assert "".join(content_chunks) == "Hello! How can I help you?"

        # Thinking callback captured internal reasoning
        assert len(thinking_captured) == 2
        assert "Thinking step 1" in thinking_captured[0]
        assert "Thinking step 2" in thinking_captured[1]
