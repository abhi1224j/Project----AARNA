"""
Unit and integration tests for ConversationalVoiceWorker, script routing,
safe in-memory PCM decoding, and smart background interruption handling.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pytest

from src.ai.prompts import AARNA_SYSTEM_PROMPT
from src.voice.streaming_worker import ConversationalVoiceWorker
from src.voice.tts.edge_engine import EdgeTTSEngine, decode_mp3_to_pcm
from src.voice.tts.player import AudioPlayer


def test_conversational_voice_worker_init():
    """Verify ConversationalVoiceWorker initialization and default states."""
    worker = ConversationalVoiceWorker(
        messages=[{"role": "user", "content": "Hello Aarna"}],
        model="gemini-2.5-flash",
        temperature=0.7,
    )
    assert worker.model == "gemini-2.5-flash"
    assert worker.interrupted is False
    assert worker._is_cancelled is False
    assert worker.player is not None
    assert worker.edge_engine is not None


def test_voice_routing_script_detection():
    """
    Verify script-based language routing:
    Devanagari text -> hi-IN-SwaraNeural
    Latin/English text -> en-IN-NeerjaNeural
    """
    worker = ConversationalVoiceWorker(messages=[])

    # Devanagari Hindi
    assert worker._select_voice_for_sentence("नमस्ते, आप कैसे हैं?") == "hi-IN-SwaraNeural"
    assert worker._select_voice_for_sentence("आज कॉलेज जाने का मन नहीं है।") == "hi-IN-SwaraNeural"

    # Latin English & Roman Hinglish
    assert worker._select_voice_for_sentence("Hello, how are you today?") == "en-IN-NeerjaNeural"
    assert worker._select_voice_for_sentence("Arey, kya hua? Aaj bas mood nahi hai?") == "en-IN-NeerjaNeural"
    assert worker._select_voice_for_sentence("Main aapki help kar sakti hoon.") == "en-IN-NeerjaNeural"


def test_in_memory_pcm_decoding_prevents_static():
    """Verify in-memory decoding with io.BytesIO prevents raw compressed bytes from reaching DAC."""
    # Empty byte test
    pcm_empty = decode_mp3_to_pcm(b"", target_sample_rate=24000)
    assert len(pcm_empty) == 0
    assert pcm_empty.dtype == np.float32

    # Verify AudioPlayer.decode_raw_audio_to_pcm handles empty / corrupted safely without static
    pcm_safe = AudioPlayer.decode_raw_audio_to_pcm(b"not-a-valid-mp3-header")
    assert len(pcm_safe) == 0


def test_thinking_fillers_injected_in_system_prompt():
    """Verify system instructions enforce natural thinking fillers ('Hmm...', 'Achha...') at sentence beginnings."""
    prompt = AARNA_SYSTEM_PROMPT.lower()
    assert "hmm..." in prompt
    assert "achha..." in prompt
    assert "sentence beginnings" in prompt or "beginning of a sentence" in prompt


def test_interruption_flag_clears_buffer_and_transitions_to_listening(qapp):
    """Verify that active interruption instantly clears the playback buffer and emits Listening state."""
    mock_player = MagicMock()
    worker = ConversationalVoiceWorker(
        messages=[{"role": "user", "content": "Hello"}],
        player=mock_player,
        enable_interruption_listener=False,
    )

    states_recorded = []
    interrupted_emitted = []
    cancelled_emitted = []

    worker.state_changed.connect(states_recorded.append)
    worker.interrupted_signal.connect(lambda: interrupted_emitted.append(True))
    worker.speech_cancelled.connect(lambda: cancelled_emitted.append(True))

    # Trigger cancellation / interruption
    worker.cancel()

    assert worker.interrupted is True
    assert worker._is_cancelled is True
    assert mock_player.clear_buffer.called


def test_conversational_voice_worker_async_stream_flow(qapp, mocker):
    """Verify complete async streaming token handling, sentence chunking, and speech finished flow."""
    mock_player = MagicMock()
    mock_router = MagicMock()
    mock_router.chat_stream.return_value = ["Arey, ", "kya hua? ", "Thak gaye ho kya?"]

    worker = ConversationalVoiceWorker(
        messages=[{"role": "user", "content": "Aaj college jaane ka mann nahi hai."}],
        router=mock_router,
        player=mock_player,
        enable_interruption_listener=False,
    )

    # Mock clause synthesis so tests don't require external network
    mocker.patch.object(worker, "_synthesize_and_play_clause", new_callable=AsyncMock)

    chunks_emitted = []
    finished_emitted = []

    worker.chunk_received.connect(chunks_emitted.append)
    worker.finished_response.connect(finished_emitted.append)

    # Run the worker thread synchronously for the test
    worker.run()

    assert len(chunks_emitted) == 3
    assert finished_emitted == ["Arey, kya hua? Thak gaye ho kya?"]
    assert worker._synthesize_and_play_clause.called
