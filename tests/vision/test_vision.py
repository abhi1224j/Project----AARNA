"""
Comprehensive Unit and Integration Test Suite for Phase 11 Multimodal Vision Architecture
and Desktop Screen Analysis in Project AARNA.

Verifies:
1. In-memory screen capture directly into io.BytesIO with zero disk writes.
2. Pillow and synthetic fallback reliability.
3. Async screen capture via asyncio.to_thread.
4. Gemini 2.5 Flash multimodal payload preparation (types.Part.from_bytes) at ₹0 cost.
5. Multimodal streaming generation through GeminiLLMProvider and ChatWorker.
6. Three.js 3D avatar visual analysis glow without violating the 6-state AvatarState specification.
7. Vision query intent detection for both English and Hinglish voice/text triggers.
8. MainWindow hotkeys (Ctrl+Shift+S, F9) and UI vision trigger button.
"""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtCore import QCoreApplication
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QApplication

from src.ai.prompts import AARNA_SYSTEM_PROMPT
from src.ai.providers.llm_gemini import GeminiLLMProvider
from src.core.config import AppConfig
from src.ui.avatar.state import AVATAR_STATE_SPECS, AvatarState
from src.ui.avatar.widget import AarnaAvatarWidget
from src.ui.main_window import AarnaMainWindow
from src.ui.workers import ChatWorker
from src.vision.screen_capturer import (
    capture_screen_async,
    capture_screen_to_bytes,
)


@pytest.fixture(scope="session")
def qapp():
    import os
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# ============================================================================
# 1. In-Memory Screen Capture Tests (Zero Disk Footprint)
# ============================================================================

def test_screen_capturer_returns_valid_in_memory_bytes(qapp) -> None:
    """Verify screen capture returns io.BytesIO containing valid JPEG data with 0 disk writes."""
    buffer = capture_screen_to_bytes()
    assert isinstance(buffer, io.BytesIO)
    raw = buffer.getvalue()
    assert len(raw) > 0
    # Standard JPEG magic number is 0xFF, 0xD8, 0xFF
    assert raw.startswith(b"\xff\xd8\xff")


def test_screen_capturer_zero_disk_writes(tmp_path, monkeypatch) -> None:
    """Verify that screen capture never touches or creates any files on disk."""
    # Ensure current working directory has no new files created during capture
    initial_files = set(Path(".").iterdir())
    buffer = capture_screen_to_bytes()
    final_files = set(Path(".").iterdir())
    assert initial_files == final_files
    assert buffer.getvalue().startswith(b"\xff\xd8\xff")


def test_screen_capturer_pillow_fallback(monkeypatch) -> None:
    """Verify Pillow ImageGrab fallback works when Qt screen capture returns None."""
    with patch("PySide6.QtGui.QGuiApplication.primaryScreen", return_value=None):
        buffer = capture_screen_to_bytes()
        assert isinstance(buffer, io.BytesIO)
        assert len(buffer.getvalue()) > 0
        assert buffer.getvalue().startswith(b"\xff\xd8\xff")


def test_screen_capturer_synthetic_fallback() -> None:
    """Verify synthetic fallback works when both Qt and Pillow grab fail."""
    with patch("PySide6.QtGui.QGuiApplication.primaryScreen", return_value=None):
        with patch("PIL.ImageGrab.grab", side_effect=RuntimeError("Desktop grab unavailable")):
            buffer = capture_screen_to_bytes()
            assert isinstance(buffer, io.BytesIO)
            assert len(buffer.getvalue()) > 0
            assert buffer.getvalue().startswith(b"\xff\xd8\xff")


def test_screen_capturer_async(qapp) -> None:
    """Verify non-blocking asynchronous screen capture executes properly."""
    async def _run():
        buf = await capture_screen_async()
        return buf

    buffer = asyncio.run(_run())
    assert isinstance(buffer, io.BytesIO)
    assert len(buffer.getvalue()) > 0
    assert buffer.getvalue().startswith(b"\xff\xd8\xff")


# ============================================================================
# 2. Gemini Multimodal Payload Preparation & Free Tier Integration
# ============================================================================

def test_gemini_multimodal_payload_preparation() -> None:
    """Verify GeminiLLMProvider packages in-memory image bytes into types.Part.from_bytes."""
    provider = GeminiLLMProvider(api_key="AIzaSyDummyKeyForTestingPayloadOnly12345")
    dummy_image = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00"

    messages = [
        {"role": "user", "content": "What is visible on this screen?"}
    ]

    system_instruction, contents = provider._prepare_contents_and_system(
        messages=messages,
        image_bytes=dummy_image,
    )

    assert len(contents) == 1
    user_content = contents[0]
    assert user_content.role == "user"

    # Parts should contain both the text prompt and the image part
    parts = user_content.parts
    assert len(parts) == 2

    text_part = parts[0]
    assert text_part.text == "What is visible on this screen?"

    img_part = parts[1]
    assert hasattr(img_part, "inline_data")
    assert img_part.inline_data.mime_type == "image/jpeg"
    assert img_part.inline_data.data == dummy_image


def test_gemini_multimodal_stream_dispatch() -> None:
    """Verify GeminiLLMProvider.chat_stream forwards image payload to generate_content_stream."""
    provider = GeminiLLMProvider(api_key="AIzaSyDummyKeyForTestingPayloadOnly12345")
    dummy_image = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00"

    mock_chunk1 = MagicMock()
    mock_chunk1.text = "Maine aapka screen dekh liya hai. "
    mock_chunk2 = MagicMock()
    mock_chunk2.text = "Yahan VS Code me Python file open hai."

    mock_client = MagicMock()
    mock_client.models.generate_content_stream.return_value = [mock_chunk1, mock_chunk2]
    provider._client = mock_client

    stream = provider.chat_stream(
        messages=[{"role": "user", "content": "Screen dekho"}],
        image_bytes=dummy_image,
    )

    tokens = list(stream)
    assert len(tokens) == 2
    assert "Maine aapka screen dekh liya hai." in tokens[0]
    assert "VS Code" in tokens[1]
    mock_client.models.generate_content_stream.assert_called_once()


def test_gemini_analyze_image_stream() -> None:
    """Verify analyze_image_stream helper functions correctly."""
    provider = GeminiLLMProvider(api_key="AIzaSyDummyKeyForTestingPayloadOnly12345")
    dummy_image = b"\xff\xd8\xff\xe0\x00\x10JFIF"

    mock_chunk = MagicMock()
    mock_chunk.text = "Visual analysis complete."
    mock_client = MagicMock()
    mock_client.models.generate_content_stream.return_value = [mock_chunk]
    provider._client = mock_client

    stream = provider.analyze_image_stream(
        prompt="Explain this diagram",
        image_bytes=dummy_image,
    )
    tokens = list(stream)
    assert tokens == ["Visual analysis complete."]


# ============================================================================
# 3. Avatar Analysis Glow & State Machine Integrity Tests
# ============================================================================

def test_avatar_analysis_glow_lifecycle(qapp) -> None:
    """
    Verify avatar analysis glow sets distinct visual styling and badge text
    without breaking the Section C 6-state AvatarState specification.
    """
    cfg = AppConfig()
    widget = AarnaAvatarWidget(config=cfg, enable_webengine=False)

    # 1. Initial State
    assert widget.is_analyzing_visual is False
    assert widget.current_state == AvatarState.IDLE

    # 2. Activate Analysis Glow
    widget.set_analysis_glow(True)
    assert widget.is_analyzing_visual is True
    assert "VISUAL ANALYSIS" in widget.state_badge_label.text()
    assert "#c084fc" in widget.state_badge_label.styleSheet() or "#7c3aed" in widget.state_badge_label.styleSheet()

    # 3. Verify underlying state machine remains valid
    assert widget.current_state in AVATAR_STATE_SPECS
    assert set(AVATAR_STATE_SPECS.keys()) == {
        AvatarState.GREETING,
        AvatarState.IDLE,
        AvatarState.LISTENING,
        AvatarState.THINKING,
        AvatarState.SPEAKING,
        AvatarState.ERROR,
    }

    # 4. Deactivate Analysis Glow
    widget.set_analysis_glow(False)
    assert widget.is_analyzing_visual is False
    assert widget.state_badge_label.text() == AVATAR_STATE_SPECS[AvatarState.IDLE].badge_text


# ============================================================================
# 4. Hinglish & English Vision Intent Detection Tests
# ============================================================================

@pytest.mark.parametrize(
    "utterance,expected",
    [
        ("Aarna, ye dekho", True),
        ("Ye dekho", True),
        ("Screen dekho", True),
        ("Aarna screen dekh ke batao kya chal raha hai", True),
        ("Look at my screen", True),
        ("Can you look at the screen?", True),
        ("Please check this error on my screen", True),
        ("Ye error dekho kya keh raha hai", True),
        ("What is on my screen right now?", True),
        ("Screen analysis karo", True),
        ("What is the capital of India?", False),
        ("Open chrome browser", False),
        ("Kal ka weather kaisa hoga?", False),
        ("Play some music please", False),
        ("How are you feeling today?", False),
    ],
)
def test_vision_query_detection(utterance: str, expected: bool) -> None:
    """Verify regex intent matching correctly classifies vision-oriented queries."""
    assert AarnaMainWindow._is_vision_query(utterance) == expected


# ============================================================================
# 5. MainWindow Hotkeys, UI Button, and Async Worker Ingestion
# ============================================================================

def test_main_window_has_vision_shortcuts_and_button(qapp) -> None:
    """Verify MainWindow has Ctrl+Shift+S, F9, and 👁️ button configured."""
    window = AarnaMainWindow()
    assert hasattr(window, "vision_shortcut_primary")
    assert window.vision_shortcut_primary.key() == QKeySequence("Ctrl+Shift+S")

    assert hasattr(window, "vision_shortcut_f9")
    assert window.vision_shortcut_f9.key() == QKeySequence("F9")

    assert hasattr(window, "vision_btn")
    assert window.vision_btn.text() == "👁️"


def test_chat_worker_accepts_and_forwards_image_bytes(qapp) -> None:
    """Verify ChatWorker accepts image_bytes and passes them into client.chat_stream."""
    mock_router = MagicMock()
    mock_router.chat_stream.return_value = iter(["Screen analysis token."])

    dummy_image = b"\xff\xd8\xff\xe0\x00\x10JFIF"
    worker = ChatWorker(
        client=mock_router,
        messages=[{"role": "user", "content": "What is on my screen?"}],
        model="gemini-2.5-flash",
        image_bytes=dummy_image,
    )
    assert worker.image_bytes == dummy_image

    chunks = []
    worker.chunk_received.connect(chunks.append)
    worker.start()
    worker.wait(3000)

    mock_router.chat_stream.assert_called_once()
    call_kwargs = mock_router.chat_stream.call_args[1]
    assert call_kwargs.get("image_bytes") == dummy_image


def test_prompts_contain_multimodal_vision_guidelines() -> None:
    """Verify AARNA_SYSTEM_PROMPT includes explicit visual screen analysis guidelines."""
    assert "visual analysis" in AARNA_SYSTEM_PROMPT.lower()
    assert "code windows" in AARNA_SYSTEM_PROMPT.lower() or "terminal" in AARNA_SYSTEM_PROMPT.lower()
    assert "hinglish" in AARNA_SYSTEM_PROMPT.lower()
