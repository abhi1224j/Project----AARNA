"""Tests for the cloud STT provider layer: script helpers, Groq provider, router, config."""

from __future__ import annotations

import io
import threading
import time
import wave
from types import SimpleNamespace
from typing import Any, Callable, Optional

import httpx
import numpy as np
import pytest

groq = pytest.importorskip("groq")

from src.voice.config import VoiceConfig
from src.voice.exceptions import (
    EmptyRecordingError,
    RecordingTooShortError,
    VoiceCancelledError,
)
from src.voice.models import ProcessedAudio, TranscriptionResult
from src.voice.providers import (
    ClientQuotaGuard,
    CloudSTTError,
    CloudSTTErrorKind,
    GroqSTTProvider,
    HybridTranscriptionEngine,
    build_transcription_engine,
    classify_utterance_language,
    describe_stt_setup,
    detect_script,
    encode_wav_16k_mono,
    normalize_language_code,
    redact_secrets,
)

FAKE_KEY = "gsk_TESTKEY1234567890abcdefghij"


# ----------------------------------------------------------------------- helpers
def make_audio(seconds: float = 2.0) -> ProcessedAudio:
    n = int(16000 * seconds)
    samples = (np.sin(np.linspace(0, 440 * 2 * np.pi * seconds, n)) * 0.3).astype(np.float32)
    return ProcessedAudio(
        samples=samples,
        sample_rate=16000,
        duration_seconds=seconds,
        original_sample_rate=16000,
        original_channels=1,
        peak_amplitude=0.3,
        rms_db=-20.0,
        was_resampled=False,
        was_clipped=False,
    )


def verbose_payload(text="Hello there", language="english", segments=None, duration=2.0) -> dict:
    if segments is None:
        segments = [
            {"start": 0.0, "end": 1.5, "text": f" {text}", "avg_logprob": -0.2,
             "no_speech_prob": 0.01, "compression_ratio": 1.1}
        ]
    return {"text": text, "language": language, "duration": duration, "segments": segments}


class FakeRaw:
    def __init__(self, payload: dict, headers: Optional[dict] = None) -> None:
        self._payload = payload
        self.headers = headers or {"x-ratelimit-limit-requests": "2000",
                                   "x-ratelimit-remaining-requests": "1999",
                                   "content-type": "application/json"}

    def parse(self) -> dict:
        return self._payload


class FakeGroqClient:
    """Scripted stand-in for the Groq SDK client (``audio.transcriptions.with_raw_response``)."""

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.calls: list[dict] = []
        outer = self

        class _Raw:
            def create(self, **kwargs: Any) -> Any:
                outer.calls.append(kwargs)
                item = outer.script.pop(0) if outer.script else outer.script_default()
                if callable(item):
                    item = item()
                if isinstance(item, BaseException):
                    raise item
                return item

        self.audio = SimpleNamespace(transcriptions=SimpleNamespace(with_raw_response=_Raw()))

    @staticmethod
    def script_default() -> FakeRaw:
        return FakeRaw(verbose_payload())


def _req() -> httpx.Request:
    return httpx.Request("POST", "https://api.groq.com/openai/v1/audio/transcriptions")


def status_error(cls, code: int, msg="err", headers: Optional[dict] = None):
    resp = httpx.Response(code, headers=headers or {}, request=_req())
    return cls(msg, response=resp, body=None)


def provider(client, **kw) -> GroqSTTProvider:
    kw.setdefault("consent_granted", True)
    p = GroqSTTProvider(client=client, api_key=FAKE_KEY, **kw)
    p._backoff_delay = lambda attempt, err: 0.0  # type: ignore[method-assign]
    return p


# ------------------------------------------------------------------ script helpers
@pytest.mark.parametrize(
    "text,expected",
    [
        ("Hello how are you", "latin"),
        ("आज मौसम कैसा है", "devanagari"),
        ("कल मेरा college में presentation है", "mixed"),
        ("1234 !!!", "none"),
        ("", "none"),
    ],
)
def test_detect_script(text, expected):
    assert detect_script(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Please set a reminder for three thirty", "english"),
        ("आज दिल्ली का मौसम कैसा रहेगा?", "hindi"),
        ("कल मेरा college में presentation है", "hinglish"),
        ("Kal mera college mein presentation hai, tum help karoge?", "hinglish"),
        ("Open Visual Studio Code and run the script", "english"),
        ("", "unknown"),
    ],
)
def test_classify_utterance_language(text, expected):
    assert classify_utterance_language(text) == expected


def test_normalize_language_code():
    assert normalize_language_code("hindi") == "hi"
    assert normalize_language_code("English") == "en"
    assert normalize_language_code("en") == "en"
    assert normalize_language_code("turkish") == "und"
    assert normalize_language_code(None) == "und"


# ----------------------------------------------------------------------- secrets
def test_redact_secrets_masks_literal_and_patterns(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)
    out = redact_secrets(f"failed with key {FAKE_KEY} and Bearer abcdefghijklmnop1234 and gsk_ABCDEFGHIJKLMNOP")
    assert FAKE_KEY not in out
    assert "abcdefghijklmnop1234" not in out
    assert "gsk_ABCDEFGHIJKLMNOP" not in out


def test_provider_repr_never_contains_key():
    p = GroqSTTProvider(consent_granted=True, api_key=FAKE_KEY)
    assert FAKE_KEY not in repr(p)
    assert FAKE_KEY not in str(p.__dict__.get("model"))


def test_error_messages_never_leak_key():
    leaky = status_error(groq.AuthenticationError, 401, f"Invalid API Key {FAKE_KEY}")
    p = provider(FakeGroqClient([leaky]))
    with pytest.raises(CloudSTTError) as ei:
        p.transcribe(make_audio())
    err = ei.value
    assert err.kind == CloudSTTErrorKind.AUTH
    assert FAKE_KEY not in str(err)
    assert FAKE_KEY not in err.technical_details
    assert FAKE_KEY not in err.user_message


# --------------------------------------------------------------------- WAV encode
def test_encode_wav_is_valid_16k_mono_pcm16():
    data = encode_wav_16k_mono(make_audio(1.0).samples)
    with wave.open(io.BytesIO(data), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000
        assert wf.getnframes() == 16000


def test_encode_wav_sanitises_nan_and_clips():
    arr = np.array([np.nan, np.inf, -np.inf, 5.0, -5.0], dtype=np.float32)
    data = encode_wav_16k_mono(arr)
    with wave.open(io.BytesIO(data), "rb") as wf:
        pcm = np.frombuffer(wf.readframes(5), dtype="<i2")
    assert pcm[0] == 0 and pcm[1] == 32767 and pcm[2] == -32767 and pcm[3] == 32767


# -------------------------------------------------------------------- Groq provider
def test_groq_success_maps_result_and_headers():
    client = FakeGroqClient([FakeRaw(verbose_payload("Meeting at 3.30", "english"))])
    p = provider(client)
    r = p.transcribe(make_audio(), "auto")
    assert r.text == "Meeting at 3.30"
    assert r.detected_language == "en"
    assert r.model_identifier == "groq-whisper-large-v3"
    assert r.compute_device == "cloud"
    assert r.diagnostics["script"] == "latin"
    assert r.diagnostics["provider"] == "groq"
    assert p.last_quota.get("x-ratelimit-limit-requests") == "2000"
    assert "content-type" not in p.last_quota.headers  # only rate-limit headers kept
    kw = client.calls[0]
    assert kw["model"] == "whisper-large-v3"
    assert kw["response_format"] == "verbose_json"
    assert kw["temperature"] == 0.0
    assert "language" not in kw  # auto: never force a language
    assert "task" not in kw and "translate" not in str(kw)  # transcription only


@pytest.mark.parametrize("mode,expected", [("en", "en"), ("hi", "hi")])
def test_groq_pinned_language_is_forwarded(mode, expected):
    client = FakeGroqClient([FakeRaw(verbose_payload())])
    provider(client).transcribe(make_audio(), mode)
    assert client.calls[0]["language"] == expected


def test_groq_unknown_language_mode_is_auto():
    client = FakeGroqClient([FakeRaw(verbose_payload())])
    provider(client).transcribe(make_audio(), "klingon")
    assert "language" not in client.calls[0]


def test_groq_hindi_devanagari_result():
    payload = verbose_payload("मुझे कल सुबह 6 बजे उठना है", "hindi")
    r = provider(FakeGroqClient([FakeRaw(payload)])).transcribe(make_audio(), "auto")
    assert r.detected_language == "hi"
    assert r.diagnostics["script"] == "devanagari"


def test_groq_hinglish_mixed_script_preserved():
    payload = verbose_payload("आज का weather कैसा है? umbrella ले जाना चाहिए", "hindi")
    r = provider(FakeGroqClient([FakeRaw(payload)])).transcribe(make_audio(), "auto")
    assert "weather" in r.text and "umbrella" in r.text
    assert r.diagnostics["script"] == "mixed"


def test_groq_unexpected_language_normalised_by_script():
    payload = verbose_payload("Arna", "turkish")
    r = provider(FakeGroqClient([FakeRaw(payload)])).transcribe(make_audio(), "auto")
    assert r.detected_language == "en"
    assert r.diagnostics["reported_language"] == "und"  # unknown name is never truncated/guessed


def test_groq_arabic_script_triggers_single_hindi_retry():
    urdu = verbose_payload("آج موسم کیسا ہے", "urdu")
    hindi = verbose_payload("आज मौसम कैसा है", "hindi")
    client = FakeGroqClient([FakeRaw(urdu), FakeRaw(hindi)])
    r = provider(client).transcribe(make_audio(), "auto")
    assert r.text == "आज मौसम कैसा है"
    assert len(client.calls) == 2
    assert client.calls[1]["language"] == "hi"
    assert r.diagnostics["retried_for_script"] is True


def test_groq_hallucination_filtered():
    payload = verbose_payload("1 1 1 1", "english", segments=[
        {"start": 0, "end": 1, "text": "1 1 1 1", "avg_logprob": -0.3, "no_speech_prob": 0.1, "compression_ratio": 1.0}])
    r = provider(FakeGroqClient([FakeRaw(payload)])).transcribe(make_audio(), "auto")
    assert r.text == ""


def test_groq_low_confidence_and_loop_segments_dropped():
    segs = [
        {"start": 0, "end": 1, "text": "real words", "avg_logprob": -0.2, "no_speech_prob": 0.01, "compression_ratio": 1.2},
        {"start": 1, "end": 2, "text": "ghost", "avg_logprob": -1.8, "no_speech_prob": 0.9, "compression_ratio": 1.0},
        {"start": 2, "end": 3, "text": "loop loop loop", "avg_logprob": -0.4, "no_speech_prob": 0.1, "compression_ratio": 3.5},
    ]
    r = provider(FakeGroqClient([FakeRaw(verbose_payload("x", segments=segs))])).transcribe(make_audio(), "auto")
    assert r.text == "real words"


def test_groq_requires_consent_and_sends_nothing():
    client = FakeGroqClient([FakeRaw(verbose_payload())])
    p = provider(client, consent_granted=False)
    with pytest.raises(CloudSTTError) as ei:
        p.transcribe(make_audio())
    assert ei.value.kind == CloudSTTErrorKind.CONSENT_REQUIRED
    assert client.calls == []


def test_groq_missing_key_not_configured(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    p = GroqSTTProvider(consent_granted=True)
    with pytest.raises(CloudSTTError) as ei:
        p.transcribe(make_audio())
    assert ei.value.kind == CloudSTTErrorKind.NOT_CONFIGURED


def test_groq_invalid_model_rejected():
    with pytest.raises(ValueError):
        GroqSTTProvider(consent_granted=True, model="whisper-large-v2")


def test_retry_on_timeout_then_success():
    client = FakeGroqClient([groq.APITimeoutError(request=_req()), FakeRaw(verbose_payload("ok"))])
    p = provider(client, max_retries=2)
    assert p.transcribe(make_audio()).text == "ok"
    assert len(client.calls) == 2 and p.last_attempts == 2


def test_retry_on_connection_error_and_5xx():
    client = FakeGroqClient([
        groq.APIConnectionError(request=_req()),
        status_error(groq.InternalServerError, 503),
        FakeRaw(verbose_payload("recovered")),
    ])
    p = provider(client, max_retries=2)
    assert p.transcribe(make_audio()).text == "recovered"
    assert len(client.calls) == 3


def test_retries_are_bounded():
    client = FakeGroqClient([groq.APITimeoutError(request=_req())] * 10)
    p = provider(client, max_retries=2)
    with pytest.raises(CloudSTTError) as ei:
        p.transcribe(make_audio())
    assert ei.value.kind == CloudSTTErrorKind.TIMEOUT
    assert len(client.calls) == 3  # 1 + 2 retries, never more


def test_zero_retries_means_single_call():
    client = FakeGroqClient([groq.APITimeoutError(request=_req())] * 3)
    p = provider(client, max_retries=0)
    with pytest.raises(CloudSTTError):
        p.transcribe(make_audio())
    assert len(client.calls) == 1


@pytest.mark.parametrize("cls,code,kind", [
    (groq.AuthenticationError, 401, CloudSTTErrorKind.AUTH),
    (groq.PermissionDeniedError, 403, CloudSTTErrorKind.AUTH),
    (groq.BadRequestError, 400, CloudSTTErrorKind.BAD_REQUEST),
])
def test_non_retryable_errors_do_not_retry(cls, code, kind):
    client = FakeGroqClient([status_error(cls, code)] * 3)
    p = provider(client, max_retries=3)
    with pytest.raises(CloudSTTError) as ei:
        p.transcribe(make_audio())
    assert ei.value.kind == kind and ei.value.status_code == code
    assert len(client.calls) == 1


def test_rate_limit_short_retry_after_is_retried():
    err = status_error(groq.RateLimitError, 429, "slow down", {"retry-after": "0.1"})
    client = FakeGroqClient([err, FakeRaw(verbose_payload("after wait"))])
    p = GroqSTTProvider(consent_granted=True, api_key=FAKE_KEY, client=client, max_retries=2)
    assert p.transcribe(make_audio()).text == "after wait"
    assert len(client.calls) == 2


def test_rate_limit_long_retry_after_is_quota_and_not_retried():
    err = status_error(groq.RateLimitError, 429, "limit", {"retry-after": "1800"})
    client = FakeGroqClient([err] * 3)
    p = provider(client, max_retries=3)
    with pytest.raises(CloudSTTError) as ei:
        p.transcribe(make_audio())
    assert ei.value.kind == CloudSTTErrorKind.QUOTA_EXHAUSTED
    assert ei.value.retry_after_seconds == 1800.0
    assert len(client.calls) == 1


def test_quota_wording_classified_as_quota():
    err = status_error(groq.RateLimitError, 429, "Rate limit reached: audio seconds per day (ASD) exceeded")
    with pytest.raises(CloudSTTError) as ei:
        provider(FakeGroqClient([err]), max_retries=0).transcribe(make_audio())
    assert ei.value.kind == CloudSTTErrorKind.QUOTA_EXHAUSTED


def test_cancel_before_request_sends_nothing():
    client = FakeGroqClient([FakeRaw(verbose_payload())])
    with pytest.raises(VoiceCancelledError):
        provider(client).transcribe(make_audio(), cancel_check=lambda: True)
    assert client.calls == []


def test_cancel_during_slow_request_returns_quickly():
    release = threading.Event()

    def slow() -> FakeRaw:
        release.wait(5.0)
        return FakeRaw(verbose_payload())

    client = FakeGroqClient([slow])
    flag = {"cancel": False}
    threading.Timer(0.15, lambda: flag.__setitem__("cancel", True)).start()
    t0 = time.perf_counter()
    try:
        with pytest.raises(VoiceCancelledError):
            provider(client, timeout_seconds=30.0).transcribe(make_audio(), cancel_check=lambda: flag["cancel"])
        assert time.perf_counter() - t0 < 2.0
    finally:
        release.set()


def test_cancel_during_backoff_wait():
    err = status_error(groq.RateLimitError, 429, "x", {"retry-after": "4"})
    client = FakeGroqClient([err, FakeRaw(verbose_payload())])
    flag = {"cancel": False}
    threading.Timer(0.2, lambda: flag.__setitem__("cancel", True)).start()
    p = GroqSTTProvider(consent_granted=True, api_key=FAKE_KEY, client=client, max_retries=2)
    t0 = time.perf_counter()
    with pytest.raises(VoiceCancelledError):
        p.transcribe(make_audio(), cancel_check=lambda: flag["cancel"])
    assert time.perf_counter() - t0 < 2.0
    assert len(client.calls) == 1


# ---------------------------------------------------------------------- quota guard
def test_client_quota_guard_rpm_and_min_billing():
    now = [0.0]
    g = ClientQuotaGuard(max_requests_per_minute=3, clock=lambda: now[0])
    for _ in range(3):
        assert g.allow(1.0)
        g.record(1.0)
    assert not g.allow(1.0)
    now[0] = 61.0
    assert g.allow(1.0)
    assert ClientQuotaGuard.billed_seconds(1.0) == 10.0  # Groq bills at least 10 s
    assert ClientQuotaGuard.billed_seconds(25.0) == 25.0


def test_client_quota_guard_hourly_and_daily_audio_caps():
    now = [0.0]
    g = ClientQuotaGuard(max_requests_per_minute=1000, max_audio_seconds_per_hour=30.0,
                         max_audio_seconds_per_day=45.0, clock=lambda: now[0])
    for _ in range(3):
        assert g.allow(5.0)
        g.record(5.0)          # 3 x 10 billed = 30
    assert not g.allow(5.0)    # hour cap
    now[0] = 3700.0
    assert g.allow(5.0)        # hour window slid, day total 30 + 10 = 40 <= 45
    g.record(5.0)
    assert not g.allow(5.0)    # day cap 50 > 45


# ----------------------------------------------------------------------- router
class FakeLocal:
    def __init__(self, text="local text") -> None:
        self.text = text
        self.calls = 0

    def get_model_status(self): return "status"
    def load_model(self): return "loaded"
    def unload_model(self): pass
    is_loaded = True

    def transcribe(self, audio, language_mode="auto", cancel_check=None) -> TranscriptionResult:
        self.calls += 1
        return TranscriptionResult(
            text=self.text, detected_language="en", language_probability=0.9,
            processing_duration_seconds=0.5, audio_duration_seconds=audio.duration_seconds,
            model_identifier="faster-whisper-small",
        )


class FakeCloud:
    name = "groq"

    def __init__(self, outcomes: list) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def transcribe(self, audio, language_mode="auto", cancel_check=None):
        self.calls += 1
        item = self.outcomes.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def cloud_result(text="cloud text") -> TranscriptionResult:
    return TranscriptionResult(
        text=text, detected_language="hi", language_probability=1.0,
        processing_duration_seconds=0.4, audio_duration_seconds=2.0,
        model_identifier="groq-whisper-large-v3", compute_device="cloud", compute_type="api",
    )


def hybrid(cloud, local=None, cfg=None, clock=None):
    cfg = cfg or VoiceConfig(stt_provider="groq", cloud_stt_enabled=True, cloud_consent_granted=True)
    return HybridTranscriptionEngine(cfg, local or FakeLocal(), cloud, clock=clock or time.monotonic)


def test_router_uses_cloud_when_healthy():
    cloud, local = FakeCloud([cloud_result()]), FakeLocal()
    r = hybrid(cloud, local).transcribe(make_audio())
    assert r.text == "cloud text" and local.calls == 0


def test_router_falls_back_to_local_on_cloud_failure():
    cloud = FakeCloud([CloudSTTError(CloudSTTErrorKind.CONNECTION, "down")])
    local = FakeLocal()
    eng = hybrid(cloud, local)
    r = eng.transcribe(make_audio())
    assert r.text == "local text"
    assert r.diagnostics["provider"] == "local"
    assert r.diagnostics["fallback_reason"] == "cloud_failed:connection"
    assert eng.last_provider == "local"


def test_router_cooldown_skips_cloud_then_recovers():
    now = [1000.0]
    cloud = FakeCloud([CloudSTTError(CloudSTTErrorKind.QUOTA_EXHAUSTED, "q"), cloud_result("back")])
    local = FakeLocal()
    eng = hybrid(cloud, local, clock=lambda: now[0])
    eng.transcribe(make_audio())                    # fails -> cooldown
    r2 = eng.transcribe(make_audio())               # cloud skipped
    assert cloud.calls == 1 and r2.diagnostics["fallback_reason"] == "cloud_paused:quota_exhausted"
    now[0] += 1000.0                                # cooldown (900 s) elapsed
    assert eng.transcribe(make_audio()).text == "back"
    assert cloud.calls == 2


def test_router_auth_failure_pauses_cloud_for_long():
    now = [0.0]
    cloud = FakeCloud([CloudSTTError(CloudSTTErrorKind.AUTH, "bad key")])
    eng = hybrid(cloud, clock=lambda: now[0])
    eng.transcribe(make_audio())
    now[0] += 1800.0
    eng.transcribe(make_audio())
    assert cloud.calls == 1


def test_router_fallback_disabled_raises_cloud_error():
    cfg = VoiceConfig(stt_provider="groq", cloud_stt_enabled=True, cloud_consent_granted=True, local_stt_fallback=False)
    cloud = FakeCloud([CloudSTTError(CloudSTTErrorKind.TIMEOUT, "slow")])
    local = FakeLocal()
    with pytest.raises(CloudSTTError):
        hybrid(cloud, local, cfg).transcribe(make_audio())
    assert local.calls == 0


def test_router_cancellation_is_not_swallowed():
    cloud = FakeCloud([VoiceCancelledError("c")])
    local = FakeLocal()
    with pytest.raises(VoiceCancelledError):
        hybrid(cloud, local).transcribe(make_audio())
    assert local.calls == 0


def test_router_validation_errors_propagate_without_network():
    cloud, local = FakeCloud([cloud_result()]), FakeLocal()
    eng = hybrid(cloud, local)
    with pytest.raises(RecordingTooShortError):
        eng.transcribe(make_audio(0.1))
    empty = make_audio()
    empty.samples = np.zeros((0,), dtype=np.float32)
    with pytest.raises(EmptyRecordingError):
        eng.transcribe(empty)
    assert cloud.calls == 0 and local.calls == 0


def test_router_client_throttle_routes_local_without_calling_cloud():
    cloud, local = FakeCloud([cloud_result()] * 5), FakeLocal()
    guard = ClientQuotaGuard(max_requests_per_minute=1)
    eng = HybridTranscriptionEngine(VoiceConfig(stt_provider="groq", cloud_stt_enabled=True,
                                                cloud_consent_granted=True), local, cloud, quota_guard=guard)
    assert eng.transcribe(make_audio()).text == "cloud text"
    r = eng.transcribe(make_audio())
    assert r.text == "local text" and cloud.calls == 1
    assert r.diagnostics["fallback_reason"] == "cloud_paused:client_throttled"


def test_router_without_cloud_provider_is_pure_local():
    local = FakeLocal()
    r = HybridTranscriptionEngine(VoiceConfig(), local, None).transcribe(make_audio())
    assert r.diagnostics["fallback_reason"] == "cloud_disabled" and local.calls == 1


def test_router_delegates_model_lifecycle_to_local():
    eng = hybrid(FakeCloud([]))
    assert eng.get_model_status() == "status" and eng.load_model() == "loaded" and eng.is_loaded


# ------------------------------------------------------------- config & factory
def test_voice_config_cloud_defaults_authorized():
    cfg = VoiceConfig()
    assert cfg.stt_provider == "groq"
    assert cfg.cloud_stt_enabled is True and cfg.cloud_consent_granted is True
    assert cfg.groq_stt_model == "whisper-large-v3"
    assert cfg.local_stt_fallback is True
    assert cfg.conversation_mode == "hands_free"


def test_voice_config_cloud_validation():
    with pytest.raises(ValueError):
        VoiceConfig(conversation_mode="telepathy")
    with pytest.raises(ValueError):
        VoiceConfig(stt_provider="openai")
    with pytest.raises(ValueError):
        VoiceConfig(groq_stt_model="whisper-1")
    with pytest.raises(ValueError):
        VoiceConfig(cloud_stt_timeout_seconds=0.1)
    with pytest.raises(ValueError):
        VoiceConfig(cloud_stt_max_retries=9)


def test_voice_config_env_consent_authorized(monkeypatch):
    for k in ("CLOUD_PROCESSING_CONSENT", "STT_PROVIDER", "CLOUD_STT_ENABLED", "VOICE_CONVERSATION_MODE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    cfg = VoiceConfig.load_from_env()
    assert cfg.stt_provider == "groq" and cfg.cloud_stt_enabled is True
    assert cfg.cloud_consent_granted is True
    assert cfg.conversation_mode == "hands_free"


def test_voice_config_env_explicit_disable_cloud(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("CLOUD_PROCESSING_CONSENT", "false")
    monkeypatch.setenv("STT_PROVIDER", "local")
    cfg = VoiceConfig.load_from_env()
    assert cfg.cloud_consent_granted is False
    assert cfg.stt_provider == "local"


def test_voice_config_env_overrides(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("CLOUD_PROCESSING_CONSENT", "true")
    monkeypatch.setenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")
    monkeypatch.setenv("CLOUD_STT_MAX_RETRIES", "99")
    monkeypatch.setenv("CLOUD_STT_TIMEOUT_SECONDS", "abc")
    monkeypatch.setenv("LOCAL_STT_FALLBACK", "false")
    cfg = VoiceConfig.load_from_env()
    assert cfg.cloud_consent_granted is True
    assert cfg.groq_stt_model == "whisper-large-v3-turbo"
    assert cfg.cloud_stt_max_retries == 5 and cfg.cloud_stt_timeout_seconds == 20.0
    assert cfg.local_stt_fallback is False


def test_describe_and_build_respect_consent_and_key(monkeypatch):
    local = FakeLocal()
    base = dict(stt_provider="groq", cloud_stt_enabled=True)
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)

    no_consent = VoiceConfig(**base, cloud_consent_granted=False)
    assert describe_stt_setup(no_consent) == ("local", "cloud consent not granted")
    assert build_transcription_engine(no_consent, local) is local

    disabled = VoiceConfig(stt_provider="groq", cloud_stt_enabled=False, cloud_consent_granted=True)
    assert describe_stt_setup(disabled)[0] == "local"
    assert build_transcription_engine(disabled, local) is local

    monkeypatch.delenv("GROQ_API_KEY")
    full = VoiceConfig(**base, cloud_consent_granted=True)
    assert describe_stt_setup(full) == ("local", "GROQ_API_KEY not set")
    assert build_transcription_engine(full, local) is local

    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)
    assert describe_stt_setup(full) == ("cloud+local", "")
    eng = build_transcription_engine(full, local)
    assert isinstance(eng, HybridTranscriptionEngine) and eng.local_engine is local
    assert FAKE_KEY not in repr(eng.cloud_provider)
