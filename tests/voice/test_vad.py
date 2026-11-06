"""Unit tests for VoiceActivityDetector (energy + ZCR hysteresis speech detection)."""

import numpy as np
import pytest

from src.voice.audio.vad import VoiceActivityDetector
from src.voice.models import VadAnalysisResult, VadSpeechSegment


def test_vad_empty_audio():
    vad = VoiceActivityDetector()
    res = vad.analyze_speech(np.zeros((0,), dtype=np.float32), sample_rate=16000)

    assert res.has_speech is False
    assert res.total_duration_seconds == 0.0
    assert res.speech_duration_seconds == 0.0
    assert len(res.segments) == 0


def test_vad_pure_silence():
    vad = VoiceActivityDetector(energy_threshold_db=-42.0)
    # 2 seconds of pure zero samples
    silence = np.zeros(32000, dtype=np.float32)
    res = vad.analyze_speech(silence, sample_rate=16000)

    assert res.has_speech is False
    assert res.total_duration_seconds == 2.0
    assert res.speech_duration_seconds == 0.0
    assert res.leading_silence_seconds == 2.0
    assert res.trailing_silence_seconds == 2.0
    assert len(res.segments) == 0


def test_vad_speech_burst_detection():
    vad = VoiceActivityDetector(
        energy_threshold_db=-30.0,
        frame_duration_ms=30,
        min_speech_duration_ms=90,
        silence_timeout_seconds=0.5,
        speech_pad_ms=50,
    )
    # 1s silence, 1s 1000Hz tone at -10 dBFS (~0.3 amplitude), 1s silence
    sr = 16000
    t_speech = np.linspace(0, 1.0, sr, endpoint=False)
    tone = (0.3 * np.sin(2 * np.pi * 1000 * t_speech)).astype(np.float32)

    samples = np.concatenate([
        np.zeros(sr, dtype=np.float32),  # 1.0s silence
        tone,                            # 1.0s speech
        np.zeros(sr, dtype=np.float32),  # 1.0s silence
    ])

    res = vad.analyze_speech(samples, sample_rate=sr)

    assert res.has_speech is True
    assert res.total_duration_seconds == 3.0
    assert len(res.segments) >= 1
    # Speech segment should start around 1.0s and end around 2.0s
    seg = res.segments[0]
    assert 0.8 <= seg.start_seconds <= 1.1
    assert 1.9 <= seg.end_seconds <= 2.2
    assert 0.8 <= res.leading_silence_seconds <= 1.1
    assert 0.8 <= res.trailing_silence_seconds <= 1.1


def test_vad_transient_click_rejection():
    vad = VoiceActivityDetector(
        energy_threshold_db=-30.0,
        frame_duration_ms=30,
        min_speech_duration_ms=180,  # Requires at least 6 consecutive 30ms frames
    )
    # A single click lasting 30 ms (1 frame) surrounded by silence
    sr = 16000
    click_len = 480  # 30 ms
    click = np.ones(click_len, dtype=np.float32) * 0.8

    samples = np.concatenate([
        np.zeros(sr, dtype=np.float32),
        click,
        np.zeros(sr, dtype=np.float32),
    ])

    res = vad.analyze_speech(samples, sample_rate=sr)
    # The transient click is too brief to meet min_speech_duration_ms (180ms)
    assert res.has_speech is False
    assert len(res.segments) == 0


def test_vad_frame_metrics():
    vad = VoiceActivityDetector()
    frame = np.sin(2 * np.pi * 500 * np.linspace(0, 0.03, 480, endpoint=False)).astype(np.float32) * 0.5
    rms_db, zcr = vad.compute_frame_metrics(frame)

    assert -10.0 <= rms_db <= -3.0
    assert 0.01 <= zcr <= 0.15
