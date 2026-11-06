"""
Unit and integration tests for Microsoft Edge Neural TTS, language detection,
HybridTTSEngine, and prefetching StreamingSpeechQueue in Project AARNA.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.voice.tts.edge_engine import EdgeTTSEngine, decode_mp3_to_pcm
from src.voice.tts.engine import TTSError, TTSSynthesisError
from src.voice.tts.hybrid_engine import HybridTTSEngine
from src.voice.tts.language import (
    detect_sentence_language,
    prepare_text_for_voice,
)
from src.voice.tts.player import AudioPlayer
from src.voice.tts.streaming import SentenceSplitter, StreamingSpeechQueue


# ============================================================================
# 1. Language Detection & Text Preparation Tests
# ============================================================================

def test_language_detection_english() -> None:
    samples = [
        "Hello! How can I help you today?",
        "Gravity is the fundamental force of attraction.",
        "The quick brown fox jumps over the lazy dog.",
        "Please open settings and check your system configuration.",
    ]
    for s in samples:
        assert detect_sentence_language(s) == "en", f"Failed for: {s}"


def test_language_detection_hindi_devanagari() -> None:
    samples = [
        "नमस्ते! मैं आरना हूँ, आपकी निजी सहायक।",
        "मैं आपके सभी सवालों के जवाब दे सकती हूँ।",
        "मौसम आज बहुत अच्छा है।",
    ]
    for s in samples:
        assert detect_sentence_language(s) == "hi", f"Failed for: {s}"


def test_language_detection_roman_hinglish() -> None:
    samples = [
        "Main aapke daily routine aur scheduling me help kar sakti hoon.",
        "Sure! Main abhi aapki files check karti hoon.",
        "Theek hai, hum ye task start karte hain.",
        "Aap mujhe bataiye ki aage kya karna hai?",
        "Bilkul, ye kaam zaroor ho jayega.",
    ]
    for s in samples:
        assert detect_sentence_language(s) == "hinglish", f"Failed for: {s}"


def test_prepare_text_for_voice_abbreviations_and_sanitization() -> None:
    raw = (
        "Hello! I am **AARNA**, an AI assistant for your PC.\n"
        "```python\nprint('code')\n```\n"
        "We can check the UI and API approx. now! 🎉"
    )
    cleaned = prepare_text_for_voice(raw)
    assert "**" not in cleaned
    assert "Aarna" in cleaned
    assert "A I" in cleaned
    assert "P C" in cleaned
    assert "U I" in cleaned
    assert "A P I" in cleaned
    assert "approximately" in cleaned
    assert "🎉" not in cleaned
    assert "code omitted" in cleaned


# ============================================================================
# 2. EdgeTTSEngine Voice Selection & MP3 Decoding Tests
# ============================================================================

def test_edge_engine_voice_selection() -> None:
    engine = EdgeTTSEngine()
    v_en, lang_en = engine.select_voice_for_text("Good morning, how are you today?")
    assert "Neerja" in v_en
    assert lang_en == "en"

    v_hi, lang_hi = engine.select_voice_for_text("नमस्ते, आपका स्वागत है।")
    assert "Swara" in v_hi
    assert lang_hi == "hi"

    v_hgl, lang_hgl = engine.select_voice_for_text("Main aapki help kar sakti hoon.")
    assert "Swara" in v_hgl
    assert lang_hgl == "hinglish"


def test_decode_mp3_to_pcm_empty() -> None:
    pcm = decode_mp3_to_pcm(b"")
    assert len(pcm) == 0
    assert pcm.dtype == np.float32


def test_edge_engine_cooldown_fails_fast() -> None:
    engine = EdgeTTSEngine()
    engine._cooldown_until = time.time() + 30.0
    with pytest.raises(TTSSynthesisError) as exc_info:
        engine.synthesize("Hello")
    assert "cooling down" in str(exc_info.value)


# ============================================================================
# 3. HybridTTSEngine Routing & Failover Tests
# ============================================================================

def test_hybrid_tts_mode_switching() -> None:
    engine = HybridTTSEngine(mode="edge")
    assert engine.mode == "edge"
    engine.mode = "piper"
    assert engine.mode == "piper"
    engine.mode = "auto"
    assert engine.mode == "auto"

    with pytest.raises(ValueError):
        engine.mode = "invalid_mode"


def test_hybrid_tts_falls_back_to_piper_on_edge_failure() -> None:
    mock_edge = MagicMock()
    mock_edge.is_cooling_down = False
    mock_edge.synthesize.side_effect = TTSSynthesisError("Connection reset by peer")

    mock_piper = MagicMock()
    mock_piper.config.model_name = "en_US-lessac-medium"
    mock_piper.config.sample_rate = 22050
    dummy_pcm = [np.zeros(2205, dtype=np.float32)]
    mock_piper.synthesize.return_value = dummy_pcm

    fallback_events = []
    def on_fallback(fe, te, reason):
        fallback_events.append((fe, te, reason))

    hybrid = HybridTTSEngine(
        edge_engine=mock_edge,
        piper_engine=mock_piper,
        mode="auto",
        enable_local_fallback=True,
        on_fallback=on_fallback,
    )

    chunks, sr = hybrid.synthesize("Hello world")
    assert len(chunks) == 1
    assert sr == 22050
    assert hybrid.last_engine_used == "piper"
    assert len(fallback_events) == 1
    assert fallback_events[0][0] == "edge"
    assert fallback_events[0][1] == "piper"
    assert "Connection reset" in fallback_events[0][2]


def test_hybrid_tts_piper_mode_bypasses_edge() -> None:
    mock_edge = MagicMock()
    mock_piper = MagicMock()
    mock_piper.config.model_name = "en_US-lessac-medium"
    mock_piper.config.sample_rate = 22050
    mock_piper.synthesize.return_value = [np.zeros(100, dtype=np.float32)]

    hybrid = HybridTTSEngine(
        edge_engine=mock_edge,
        piper_engine=mock_piper,
        mode="piper",
    )

    chunks, sr = hybrid.synthesize("Offline test")
    assert hybrid.last_engine_used == "piper"
    mock_edge.synthesize.assert_not_called()
    mock_piper.synthesize.assert_called_once_with("Offline test")


def test_hybrid_tts_diagnostics() -> None:
    mock_edge = MagicMock()
    mock_edge.check_connection.return_value = (True, "Edge Online")
    mock_piper = MagicMock()
    mock_piper.config.is_model_available.return_value = True

    hybrid = HybridTTSEngine(
        edge_engine=mock_edge,
        piper_engine=mock_piper,
        mode="auto",
    )

    diag = hybrid.get_diagnostics()
    assert diag["mode"] == "auto"
    assert diag["edge_available"] is True
    assert diag["piper_available"] is True
    assert "₹0" in diag["cost_profile"]


# ============================================================================
# 4. StreamingSpeechQueue Prefetching & Interruption Tests
# ============================================================================

def test_streaming_speech_queue_prefetching_and_playback(qapp) -> None:
    mock_engine = MagicMock()
    mock_player = MagicMock()
    mock_player.is_playing = False

    # Simulate fast mock synthesis returning 100 samples
    mock_engine.synthesize.return_value = ([np.zeros(100, dtype=np.float32)], 24000)
    mock_player.play_sync.return_value = True

    queue = StreamingSpeechQueue(engine=mock_engine, player=mock_player)

    started_sentences = []
    queue.sentence_started.connect(started_sentences.append)

    finished_called = []
    queue.speech_finished.connect(lambda: finished_called.append(True))

    queue.reset_stream()
    queue.start()

    # Push sentences
    queue.push_chunk("First sentence complete. ")
    queue.push_chunk("Second sentence complete. ")
    queue.finish_stream()

    queue.wait(3000)
    qapp.processEvents()

    assert mock_engine.synthesize.call_count == 2
    assert mock_player.play_sync.call_count == 2
    assert len(finished_called) == 1


def test_streaming_speech_queue_barge_in_cancellation(qapp) -> None:
    mock_engine = MagicMock()
    mock_player = MagicMock()

    mock_engine.synthesize.return_value = ([np.zeros(1000, dtype=np.float32)], 24000)

    def slow_play(*args, **kwargs):
        time.sleep(0.05)
        return False  # Aborted

    mock_player.play_sync.side_effect = slow_play

    queue = StreamingSpeechQueue(engine=mock_engine, player=mock_player)

    cancelled_called = []
    queue.speech_cancelled.connect(lambda: cancelled_called.append(True))

    queue.reset_stream()
    queue.start()

    queue.push_chunk("Sentence one. ")
    queue.push_chunk("Sentence two. ")

    time.sleep(0.02)
    # Trigger barge-in
    queue.cancel()

    queue.wait(2000)
    qapp.processEvents()

    assert mock_player.stop.called
    assert len(cancelled_called) == 1
