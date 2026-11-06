"""
Unit and integration tests for AARNA Local TTS Subsystem.
Tests PiperTTSEngine, clean_text_for_speech, AudioPlayer, TTSConfig, and TTSWorker.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from src.voice.tts.config import TTSConfig
from src.voice.tts.engine import (
    PiperTTSEngine,
    TTSModelNotFoundError,
    TTSSynthesisError,
    clean_text_for_speech,
)
from src.voice.tts.player import AudioPlaybackError, AudioPlayer
from src.voice.tts.streaming import SentenceSplitter, StreamingSpeechQueue
from src.voice.tts.worker import TTSWorker


@pytest.fixture(scope="session")
def qapp():
    """Ensure a QCoreApplication / QApplication instance exists for QObject tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_tts_config_defaults():
    """Verify default TTS configuration parameters."""
    cfg = TTSConfig()
    assert cfg.model_name == "en_US-lessac-medium"
    assert cfg.models_dir == Path("models/piper")
    assert cfg.sample_rate == 22050
    assert cfg.volume == 1.0
    assert cfg.speed == 1.0
    assert cfg.use_gpu is False
    assert cfg.resolved_model_path() == Path("models/piper/en_US-lessac-medium.onnx")
    assert cfg.resolved_config_path() == Path("models/piper/en_US-lessac-medium.onnx.json")


def test_tts_config_custom_paths(tmp_path):
    """Verify custom model and config paths."""
    m_path = tmp_path / "custom.onnx"
    c_path = tmp_path / "custom.onnx.json"
    cfg = TTSConfig(model_path=m_path, config_path=c_path)
    assert cfg.resolved_model_path() == m_path
    assert cfg.resolved_config_path() == c_path
    assert cfg.is_model_available() is False

    m_path.write_bytes(b"dummy onnx content")
    c_path.write_text("{}", encoding="utf-8")
    assert cfg.is_model_available() is True


def test_clean_text_for_speech():
    """Verify markdown sanitization for natural speech output."""
    raw = (
        "# Heading\n\n"
        "Hello **world**! This is *AARNA*.\n"
        "Here is a [documentation link](https://example.com) and `python inline code`.\n"
        "```python\nprint('hello')\n```\n"
        "- Bullet 1\n"
        "- Bullet 2\n"
        "Everything looks great! 😊🎉\n"
        "नमस्ते आप कैसे हैं?"
    )
    cleaned = clean_text_for_speech(raw)

    assert "#" not in cleaned
    assert "**" not in cleaned
    assert "*" not in cleaned
    assert "https://example.com" not in cleaned
    assert "documentation link" in cleaned
    assert "python inline code" in cleaned
    assert "code block omitted" in cleaned
    assert "Bullet 1" in cleaned
    assert "Bullet 2" in cleaned
    assert "😊" not in cleaned
    assert "नमस्ते आप कैसे हैं?" in cleaned


def test_clean_text_empty():
    """Verify empty or whitespace text returns empty string."""
    assert clean_text_for_speech("") == ""
    assert clean_text_for_speech("   \n\t  ") == ""


def test_piper_engine_missing_model(tmp_path):
    """Verify error raised when model weights or config are missing."""
    cfg = TTSConfig(
        model_path=tmp_path / "missing.onnx",
        config_path=tmp_path / "missing.onnx.json",
    )
    engine = PiperTTSEngine(cfg)
    assert engine.is_loaded is False

    with pytest.raises(TTSModelNotFoundError, match="model weights not found"):
        engine.load()


def test_piper_engine_mock_synthesis():
    """Verify synthesis flow and chunk extraction using mocked PiperVoice."""
    engine = PiperTTSEngine(TTSConfig())
    mock_voice = MagicMock()

    chunk1 = MagicMock()
    chunk1.audio_float_array = np.array([0.1, -0.1, 0.2], dtype=np.float32)
    chunk2 = MagicMock()
    chunk2.audio_float_array = np.array([0.3, -0.3, 0.4], dtype=np.float32)
    mock_voice.synthesize.return_value = [chunk1, chunk2]

    engine._voice = mock_voice
    engine._is_loaded = True

    chunks = engine.synthesize("Hello AARNA")
    assert len(chunks) == 2
    assert np.allclose(chunks[0], [0.1, -0.1, 0.2])
    assert np.allclose(chunks[1], [0.3, -0.3, 0.4])

    # Unload
    engine.unload()
    assert engine.is_loaded is False


def test_piper_engine_real_model_synthesis():
    """Verify end-to-end synthesis with the downloaded en_US-lessac-medium model if present."""
    cfg = TTSConfig()
    if not cfg.is_model_available():
        pytest.skip("Piper voice model not downloaded yet")

    engine = PiperTTSEngine(cfg)
    engine.load()
    assert engine.is_loaded is True

    chunks = engine.synthesize("AARNA voice synthesis test.")
    assert len(chunks) > 0
    assert isinstance(chunks[0], np.ndarray)
    assert chunks[0].dtype == np.float32
    assert len(chunks[0]) > 0

    engine.unload()
    assert engine.is_loaded is False


def test_audio_player_init():
    """Verify AudioPlayer initial state."""
    player = AudioPlayer(sample_rate=22050)
    assert player.is_playing is False
    assert player.sample_rate == 22050


def test_audio_player_empty_chunks():
    """Verify empty chunks list completes immediately."""
    player = AudioPlayer()
    res = player.play_sync([])
    assert res is True
    assert player.is_playing is False


def test_audio_player_stop():
    """Verify stop() sets cancellation and halts playback."""
    player = AudioPlayer()
    player.stop()
    assert player._stop_event.is_set()


@patch("sounddevice.OutputStream")
def test_audio_player_play_sync(mock_output_stream_cls):
    """Verify synchronous playback with mocked sounddevice stream."""
    mock_stream = MagicMock()
    mock_output_stream_cls.return_value = mock_stream

    player = AudioPlayer(sample_rate=22050, block_size=256)

    dummy_chunk = np.ones(512, dtype=np.float32) * 0.1
    levels_recorded = []
    started_called = []
    finished_called = []

    def _on_start():
        started_called.append(True)

    def _on_finish():
        finished_called.append(True)

    def _on_level(level):
        levels_recorded.append(level)

    res = player.play_sync(
        chunks=[dummy_chunk],
        sample_rate=22050,
        sentence_silence_seconds=0.0,
        on_start=_on_start,
        on_finish=_on_finish,
        on_level=_on_level,
    )

    assert res is True
    assert len(started_called) == 1
    assert len(finished_called) == 1
    assert len(levels_recorded) > 0
    assert mock_stream.start.called
    assert mock_stream.write.called
    assert mock_stream.stop.called


def test_tts_worker_lifecycle(qapp):
    """Verify TTSWorker signals and execution with mocked engine and player."""
    engine = PiperTTSEngine()
    engine._is_loaded = True
    mock_voice = MagicMock()
    chunk = MagicMock()
    chunk.audio_float_array = np.zeros(256, dtype=np.float32)
    mock_voice.synthesize.return_value = [chunk]
    engine._voice = mock_voice

    mock_player = MagicMock()
    mock_player.play_sync.return_value = True

    worker = TTSWorker("Hello test", engine=engine, player=mock_player)

    started_events = []
    finished_events = []

    worker.playback_started.connect(lambda: started_events.append(True))
    worker.playback_finished.connect(lambda: finished_events.append(True))

    worker.run()

    assert mock_player.play_sync.called
    assert len(finished_events) == 1


def test_tts_worker_cancellation(qapp):
    """Verify that cancelling TTSWorker emits playback_cancelled."""
    engine = PiperTTSEngine()
    mock_player = MagicMock()

    worker = TTSWorker("Hello test", engine=engine, player=mock_player)
    cancelled_events = []
    worker.playback_cancelled.connect(lambda: cancelled_events.append(True))

    worker.cancel()
    worker.run()

    assert len(cancelled_events) == 1


def test_sentence_splitter_english():
    """Verify sentence splitting on periods, question marks, exclamation marks, and clauses."""
    splitter = SentenceSplitter(min_clause_chars=30)

    # 1. Stream tokens incrementally
    tokens = ["Hello! ", "I am ", "Aarna. ", "How can ", "I help ", "you today? "]
    sents = []
    for t in tokens:
        sents.extend(splitter.push(t))
    sents.extend(splitter.flush())

    assert sents == ["Hello!", "I am Aarna.", "How can I help you today?"]

    # 2. Abbreviations and decimals are not falsely split
    splitter2 = SentenceSplitter()
    s2 = splitter2.push("Dr. Smith said pi is 3.14. It is true.")
    s2.extend(splitter2.flush())
    assert s2 == ["Dr. Smith said pi is 3.14.", "It is true."]


def test_sentence_splitter_hindi():
    """Verify sentence splitting on Devanagari purna viram and Hindi punctuation."""
    splitter = SentenceSplitter()
    s = splitter.push("नमस्ते! मैं आरना हूँ। आज का दिन शुभ हो।")
    s.extend(splitter.flush())
    assert s == ["नमस्ते!", "मैं आरना हूँ।", "आज का दिन शुभ हो।"]


def test_sentence_splitter_early_first_clause():
    """Verify early clause splitting on the first turn unit for immediate TTS playback."""
    splitter = SentenceSplitter()

    # Case 1: Early clause with comma >= 12 chars splits immediately
    chunks = ["Haha, samajh sakti hoon, ", "aaj koi interesting lecture nahi hai kya?"]
    sents = []
    for c in chunks:
        sents.extend(splitter.push(c))
    sents.extend(splitter.flush())
    assert sents == [
        "Haha, samajh sakti hoon,",
        "aaj koi interesting lecture nahi hai kya?",
    ]

    # Case 2: Very short comma (< 12 chars like "Hi,") waits for sentence terminator or full clause
    splitter.clear()
    sents2 = []
    sents2.extend(splitter.push("Hi, friend! "))
    sents2.extend(splitter.flush())
    assert sents2 == ["Hi, friend!"]



def _pump_events(qapp, count: int = 10):
    for _ in range(count):
        qapp.processEvents()
        import time
        time.sleep(0.005)


def test_streaming_speech_queue_lifecycle(qapp):
    """Verify StreamingSpeechQueue pushes, synthesizes, and plays chunks in FIFO sequence."""
    engine = PiperTTSEngine()
    engine._is_loaded = True
    mock_voice = MagicMock()
    chunk = MagicMock()
    chunk.audio_float_array = np.zeros(256, dtype=np.float32)
    mock_voice.synthesize.return_value = [chunk]
    engine._voice = mock_voice

    mock_player = MagicMock()
    def _mock_play_sync(*args, **kwargs):
        on_start = kwargs.get("on_start")
        if on_start:
            on_start()
        return True
    mock_player.play_sync.side_effect = _mock_play_sync
    mock_player.is_playing = False

    queue_worker = StreamingSpeechQueue(engine=engine, player=mock_player)

    started_events = []
    sentences_spoken = []
    finished_events = []

    queue_worker.speech_started.connect(lambda: started_events.append(True))
    queue_worker.sentence_started.connect(lambda s: sentences_spoken.append(s))
    queue_worker.speech_finished.connect(lambda: finished_events.append(True))

    queue_worker.start()

    # Stream text tokens into queue
    queue_worker.push_chunk("Hello! ")
    queue_worker.push_chunk("I am Aarna.")
    queue_worker.finish_stream()

    queue_worker.wait(3000)
    _pump_events(qapp)

    assert len(started_events) == 1
    assert len(sentences_spoken) == 2
    assert "Hello!" in sentences_spoken[0]
    assert "I am Aarna." in sentences_spoken[1]
    assert len(finished_events) == 1


def test_streaming_speech_queue_cancellation(qapp):
    """Verify cancelling StreamingSpeechQueue purges remaining queue and halts player."""
    engine = PiperTTSEngine()
    mock_player = MagicMock()

    queue_worker = StreamingSpeechQueue(engine=engine, player=mock_player)
    cancelled_events = []
    queue_worker.speech_cancelled.connect(lambda: cancelled_events.append(True))

    queue_worker.start()
    queue_worker.push_chunk("Long sentence that will be cancelled. ")
    queue_worker.cancel()
    queue_worker.finish_stream()

    queue_worker.wait(2000)
    _pump_events(qapp)

    assert mock_player.stop.called
    assert len(cancelled_events) == 1
