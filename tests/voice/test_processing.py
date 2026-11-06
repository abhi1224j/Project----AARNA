"""Unit tests for AudioProcessor (anti-aliased resampling, downmix, normalization)."""

import numpy as np
import pytest

from src.voice.audio.processing import AudioProcessor
from src.voice.exceptions import AudioProcessingError, EmptyRecordingError
from src.voice.models import ProcessedAudio, RawAudioBuffer


def test_audio_processor_enforces_16k_target():
    with pytest.raises(ValueError):
        AudioProcessor(target_sample_rate=44100)
    proc = AudioProcessor(target_sample_rate=16000)
    assert proc.target_sample_rate == 16000


def test_process_raw_buffer_rejects_empty():
    proc = AudioProcessor()

    with pytest.raises(EmptyRecordingError):
        proc.process_raw_buffer(None)

    empty_buf = RawAudioBuffer(
        samples=np.zeros((0,), dtype=np.float32),
        sample_rate=16000,
        channels=1,
        duration_seconds=0.0,
    )
    with pytest.raises(EmptyRecordingError):
        proc.process_raw_buffer(empty_buf)


def test_process_raw_buffer_sanitizes_nan_inf():
    proc = AudioProcessor()
    # 0.5s of 16kHz audio with NaNs and Infs injected
    samples = np.array([0.1, np.nan, 0.2, np.inf, -np.inf, 0.3], dtype=np.float32)
    buf = RawAudioBuffer(
        samples=samples,
        sample_rate=16000,
        channels=1,
        duration_seconds=len(samples) / 16000.0,
    )
    processed = proc.process_raw_buffer(buf)

    assert not np.isnan(processed.samples).any()
    assert not np.isinf(processed.samples).any()
    assert processed.samples.dtype == np.float32


def test_convert_to_mono_multichannel():
    proc = AudioProcessor()

    # 1D passthrough
    s1d = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    mono1 = proc.convert_to_mono(s1d)
    assert mono1.ndim == 1
    assert np.allclose(mono1, s1d)

    # 2D stereo (frames, 2)
    s2d = np.array([[0.2, 0.4], [-0.6, -0.2]], dtype=np.float32)
    mono2 = proc.convert_to_mono(s2d)
    assert mono2.ndim == 1
    assert np.allclose(mono2, [0.3, -0.4])

    # 2D single channel (frames, 1)
    s_col = np.array([[0.5], [-0.5]], dtype=np.float32)
    mono3 = proc.convert_to_mono(s_col)
    assert mono3.ndim == 1
    assert np.allclose(mono3, [0.5, -0.5])


def test_dc_offset_removal():
    proc = AudioProcessor()
    # Signal with +0.5 DC bias
    raw_signal = np.ones(1600, dtype=np.float32) * 0.5
    buf = RawAudioBuffer(
        samples=raw_signal,
        sample_rate=16000,
        channels=1,
        duration_seconds=0.1,
    )
    processed = proc.process_raw_buffer(buf)
    # Mean of processed signal must be virtually 0.0
    assert abs(np.mean(processed.samples)) < 1e-6


def test_resampling_16k_passthrough():
    proc = AudioProcessor()
    samples = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000, endpoint=False)).astype(np.float32)
    buf = RawAudioBuffer(
        samples=samples,
        sample_rate=16000,
        channels=1,
        duration_seconds=1.0,
    )
    processed = proc.process_raw_buffer(buf)

    assert processed.was_resampled is False
    assert processed.sample_rate == 16000
    assert len(processed.samples) == 16000


def test_resampling_44k_downsample():
    proc = AudioProcessor()
    # 1.0 second of 44.1 kHz audio
    samples = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100, endpoint=False)).astype(np.float32)
    buf = RawAudioBuffer(
        samples=samples,
        sample_rate=44100,
        channels=1,
        duration_seconds=1.0,
    )
    processed = proc.process_raw_buffer(buf)

    assert processed.was_resampled is True
    assert processed.sample_rate == 16000
    assert len(processed.samples) == 16000
    assert abs(processed.duration_seconds - 1.0) < 0.01


def test_resampling_48k_downsample():
    proc = AudioProcessor()
    # 0.5 second of 48 kHz audio
    samples = np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, 24000, endpoint=False)).astype(np.float32)
    buf = RawAudioBuffer(
        samples=samples,
        sample_rate=48000,
        channels=1,
        duration_seconds=0.5,
    )
    processed = proc.process_raw_buffer(buf)

    assert processed.was_resampled is True
    assert processed.sample_rate == 16000
    assert len(processed.samples) == 8000


def test_resampling_8k_upsample():
    proc = AudioProcessor()
    # 0.5 second of 8 kHz audio
    samples = np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, 4000, endpoint=False)).astype(np.float32)
    buf = RawAudioBuffer(
        samples=samples,
        sample_rate=8000,
        channels=1,
        duration_seconds=0.5,
    )
    processed = proc.process_raw_buffer(buf)

    assert processed.was_resampled is True
    assert processed.sample_rate == 16000
    assert len(processed.samples) == 8000


def test_fir_anti_aliasing_attenuation():
    proc = AudioProcessor()
    # Generate a 12 kHz tone sampled at 44.1 kHz (above Nyquist of 16 kHz = 8 kHz)
    t = np.linspace(0, 0.2, int(44100 * 0.2), endpoint=False)
    high_freq_tone = (0.8 * np.sin(2 * np.pi * 12000 * t)).astype(np.float32)
    buf = RawAudioBuffer(
        samples=high_freq_tone,
        sample_rate=44100,
        channels=1,
        duration_seconds=0.2,
    )
    processed = proc.process_raw_buffer(buf)

    # 12 kHz tone must be significantly attenuated by the sinc-Hamming low-pass filter
    # Original peak is 0.8; resampled peak should be < 0.15 (attenuated by > 14 dB)
    assert processed.peak_amplitude < 0.15


def test_normalization_and_clipping():
    proc = AudioProcessor()
    # Blown out signal with peak 2.5
    samples = np.array([-2.5, 0.0, 1.8, -0.5, 0.2], dtype=np.float32)
    buf = RawAudioBuffer(
        samples=samples,
        sample_rate=16000,
        channels=1,
        duration_seconds=len(samples) / 16000.0,
    )
    processed = proc.process_raw_buffer(buf)

    assert processed.was_clipped is True
    assert processed.peak_amplitude <= 0.98 + 1e-4
    assert np.all(processed.samples <= 0.98 + 1e-4)
    assert np.all(processed.samples >= -0.98 - 1e-4)
