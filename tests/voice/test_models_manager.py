"""
Unit tests for WhisperModelManager (Phase C4).
Verifies model discovery, disk pre-flight checks, explicit downloading,
integrity verification, lazy loading, CUDA->CPU INT8 fallback, and unloading.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.voice.exceptions import SpeechModelError, VoiceErrorCategory
from src.voice.models import ModelAvailabilityState
from src.voice.models_manager import WhisperModelManager


def _create_fake_installed_model(m_dir: Path) -> None:
    """Helper to create dummy valid model.bin and config.json in model dir."""
    m_dir.mkdir(parents=True, exist_ok=True)
    bin_file = m_dir / "model.bin"
    cfg_file = m_dir / "config.json"
    bin_file.write_bytes(b"x" * 2048)
    cfg_file.write_bytes(b'{"model_type": "whisper"}')


def test_models_manager_init_and_get_model_dir(tmp_path: Path):
    mgr = WhisperModelManager(models_dir=tmp_path)
    assert mgr.models_dir == tmp_path
    assert mgr.get_model_dir("base") == tmp_path / "base"
    assert mgr.get_model_dir("TINY") == tmp_path / "tiny"

    with pytest.raises(ValueError, match="Unsupported model size"):
        mgr.get_model_dir("large-v3")


def test_is_model_installed_and_size(tmp_path: Path):
    mock_fw = MagicMock()
    mgr = WhisperModelManager(models_dir=tmp_path, fw_module=mock_fw)

    assert not mgr.is_model_installed("base")
    assert mgr.get_model_size_bytes("base") == 0

    base_dir = tmp_path / "base"
    _create_fake_installed_model(base_dir)

    assert mgr.is_model_installed("base")
    assert mgr.get_model_size_bytes("base") > 2000

    # Incomplete model (missing config.json)
    (base_dir / "config.json").unlink()
    assert not mgr.is_model_installed("base")


def test_get_model_status_transitions(tmp_path: Path):
    mock_fw = MagicMock()
    mgr = WhisperModelManager(models_dir=tmp_path, fw_module=mock_fw)

    # 1. Not installed
    st1 = mgr.get_model_status("tiny")
    assert st1.state == ModelAvailabilityState.NOT_INSTALLED
    assert not mgr.is_loaded

    # 2. Installed but not loaded
    _create_fake_installed_model(tmp_path / "tiny")
    st2 = mgr.get_model_status("tiny")
    assert st2.state == ModelAvailabilityState.INSTALLED_NOT_LOADED
    assert not mgr.is_loaded

    # 3. Loaded
    mock_fw.WhisperModel.return_value = MagicMock()
    mgr.load_model("tiny", compute_device="cpu", compute_type="int8")
    st3 = mgr.get_model_status("tiny")
    assert st3.state == ModelAvailabilityState.READY
    assert mgr.is_loaded
    assert mgr.active_model is not None


def test_download_model_insufficient_disk_space(tmp_path: Path):
    mock_fw = MagicMock()
    mgr = WhisperModelManager(models_dir=tmp_path, fw_module=mock_fw)

    # Mock disk_usage returning 100 MB free (< 1 GB)
    with patch("shutil.disk_usage", return_value=(1000 * 1024 * 1024, 900 * 1024 * 1024, 100 * 1024 * 1024)):
        with pytest.raises(SpeechModelError) as exc_info:
            mgr.download_model("tiny")

        assert exc_info.value.category == VoiceErrorCategory.MODEL_DOWNLOAD_FAILED
        assert "Insufficient disk space" in str(exc_info.value)


def test_download_model_already_installed_short_circuits(tmp_path: Path):
    mock_fw = MagicMock()
    mgr = WhisperModelManager(models_dir=tmp_path, fw_module=mock_fw)
    _create_fake_installed_model(tmp_path / "tiny")

    progress_reports = []
    def progress_cb(pct: float, msg: str):
        progress_reports.append((pct, msg))

    res_path = mgr.download_model("tiny", progress_callback=progress_cb)
    assert res_path == tmp_path / "tiny"
    assert len(progress_reports) == 1
    assert "already installed" in progress_reports[0][1]
    mock_fw.download_model.assert_not_called()


def test_download_model_cancelled(tmp_path: Path):
    mock_fw = MagicMock()
    mgr = WhisperModelManager(models_dir=tmp_path, fw_module=mock_fw)

    with pytest.raises(SpeechModelError) as exc_info:
        mgr.download_model("tiny", cancel_check=lambda: True)

    assert exc_info.value.category == VoiceErrorCategory.CANCELLED


def test_download_model_success(tmp_path: Path):
    mock_fw = MagicMock()
    mgr = WhisperModelManager(models_dir=tmp_path, fw_module=mock_fw)

    def fake_download(size_or_id, output_dir, local_files_only):
        _create_fake_installed_model(Path(output_dir))

    mock_fw.download_model.side_effect = fake_download

    progress_reports = []
    def progress_cb(pct: float, msg: str):
        progress_reports.append((pct, msg))

    res = mgr.download_model("base", progress_callback=progress_cb)
    assert res == tmp_path / "base"
    assert mgr.is_model_installed("base")
    assert len(progress_reports) >= 2


def test_download_model_integrity_failure_cleans_up(tmp_path: Path):
    mock_fw = MagicMock()
    mgr = WhisperModelManager(models_dir=tmp_path, fw_module=mock_fw)

    # Download completes but does NOT create required files
    mock_fw.download_model.return_value = None

    with pytest.raises(SpeechModelError) as exc_info:
        mgr.download_model("base")

    assert exc_info.value.category == VoiceErrorCategory.MODEL_LOAD_FAILED
    assert not (tmp_path / "base").is_dir()


def test_load_model_missing_raises(tmp_path: Path):
    mock_fw = MagicMock()
    mgr = WhisperModelManager(models_dir=tmp_path, fw_module=mock_fw)

    with pytest.raises(SpeechModelError) as exc_info:
        mgr.load_model("tiny")

    assert exc_info.value.category == VoiceErrorCategory.MODEL_MISSING


def test_load_model_cuda_fallback_to_cpu(tmp_path: Path):
    mock_fw = MagicMock()
    mgr = WhisperModelManager(models_dir=tmp_path, fw_module=mock_fw)
    _create_fake_installed_model(tmp_path / "tiny")

    # CUDA instantiation fails; CPU instantiation succeeds
    cpu_model = MagicMock()
    def fake_whisper_model(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            raise RuntimeError("CUDA driver out of memory or device unavailable")
        return cpu_model

    mock_fw.WhisperModel.side_effect = fake_whisper_model

    loaded = mgr.load_model("tiny", compute_device="cuda", compute_type="float16")
    assert loaded == cpu_model
    assert mgr.is_loaded
    st = mgr.get_model_status("tiny")
    assert st.compute_device == "cpu"
    assert st.compute_type == "int8"


def test_load_model_cached_return(tmp_path: Path):
    mock_fw = MagicMock()
    mgr = WhisperModelManager(models_dir=tmp_path, fw_module=mock_fw)
    _create_fake_installed_model(tmp_path / "tiny")

    first_model = MagicMock()
    mock_fw.WhisperModel.return_value = first_model

    m1 = mgr.load_model("tiny", compute_device="cpu", compute_type="int8")
    m2 = mgr.load_model("tiny", compute_device="cpu", compute_type="int8")

    assert m1 is m2
    assert mock_fw.WhisperModel.call_count == 1


def test_unload_model(tmp_path: Path):
    mock_fw = MagicMock()
    mgr = WhisperModelManager(models_dir=tmp_path, fw_module=mock_fw)
    _create_fake_installed_model(tmp_path / "tiny")

    mock_fw.WhisperModel.return_value = MagicMock()
    mgr.load_model("tiny", compute_device="cpu", compute_type="int8")
    assert mgr.is_loaded

    mgr.unload_model()
    assert not mgr.is_loaded
    assert mgr.active_model is None
