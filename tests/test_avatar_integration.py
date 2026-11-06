"""
Automated Phase B Desktop Avatar Integration Tests for Project AARNA.

Covers:
1. AvatarState specifications, allowlists, and Section C state semantics.
2. LocalAssetServer 127.0.0.1 loopback binding, dynamic port, and security allowlist / path-traversal blocking.
3. WebEngine failure fallback, missing model fallback, and model-loading timeout fallback.
4. Show/Hide Avatar toggle behavior and chat functionality when avatar is disabled.
5. AI lifecycle state transitions (THINKING during token streaming -> IDLE on finish; never SPEAKING on tokens;
   text input focus never triggers LISTENING; ERROR on failure).
6. Live subprocess WebGL readiness handshake, state updates, hide/pause, and clean application shutdown.
"""

from __future__ import annotations

import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from src.core.config import AppConfig
from src.ui.avatar import (
    ALLOWED_CAMERAS,
    ALLOWED_OUTFITS,
    AVATAR_STATE_SPECS,
    AarnaAvatarWidget,
    AvatarState,
    LocalAssetServer,
    get_state_spec,
    is_approved_relative_path,
)
from src.ui.main_window import AarnaMainWindow

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def qapp():
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def test_avatar_state_specifications_and_semantics():
    """Verify all 6 AvatarStates and their exact Section C behavioral rules."""
    expected_states = {
        AvatarState.GREETING,
        AvatarState.IDLE,
        AvatarState.LISTENING,
        AvatarState.THINKING,
        AvatarState.SPEAKING,
        AvatarState.ERROR,
    }
    assert set(AVATAR_STATE_SPECS.keys()) == expected_states

    # GREETING and ERROR auto-return to IDLE; others remain until explicitly transitioned
    assert get_state_spec("greeting").auto_return_state == AvatarState.IDLE
    assert get_state_spec("error").auto_return_state == AvatarState.IDLE
    assert get_state_spec("idle").auto_return_state is None
    assert get_state_spec("thinking").auto_return_state is None
    assert get_state_spec("listening").auto_return_state is None
    assert get_state_spec("speaking").auto_return_state is None

    # Only SPEAKING (reserved for TTS) enables viseme articulation
    for st, spec in AVATAR_STATE_SPECS.items():
        if st == AvatarState.SPEAKING:
            assert spec.enable_visemes is True
        else:
            assert spec.enable_visemes is False

    # Only LISTENING activates the scene's listening pulse
    for st, spec in AVATAR_STATE_SPECS.items():
        if st == AvatarState.LISTENING:
            assert spec.scene_assistant_state == "listening"
        else:
            assert spec.scene_assistant_state == "idle"

    with pytest.raises(ValueError):
        get_state_spec("invalid_state_name")


def test_asset_server_loopback_and_security_allowlist():
    """Verify LocalAssetServer binds to 127.0.0.1, serves approved assets, and blocks traversal/unapproved files."""
    assert is_approved_relative_path("embedded.html") is True
    assert is_approved_relative_path("index.html") is True
    assert is_approved_relative_path("app.js") is True
    assert is_approved_relative_path("vendor/three.module.min.js") is True
    assert is_approved_relative_path("models/aarna_teen_casual.glb") is True

    # Blocked paths
    assert is_approved_relative_path("../pyproject.toml") is False
    assert is_approved_relative_path("run_prototype.py") is False
    assert is_approved_relative_path("deliverables/aarna_v2_front_fullbody.png") is False
    assert is_approved_relative_path("captures/aarna_v2_front_fullbody.png") is False
    assert is_approved_relative_path(".env") is False

    server = LocalAssetServer()
    url = server.start()
    try:
        assert server.is_running is True
        assert server.host == "127.0.0.1"
        assert server.port > 0
        assert url == f"http://127.0.0.1:{server.port}/embedded.html"

        base = f"http://127.0.0.1:{server.port}"

        # 1. Approved embedded page & GLB HEAD request succeed
        with urllib.request.urlopen(url, timeout=5) as resp:
            assert resp.status == 200
            assert "no-store" in resp.headers.get("Cache-Control", "")
            assert resp.headers.get("X-Content-Type-Options") == "nosniff"
            html = resp.read().decode("utf-8")
            assert "aarna-phase-b-embedded-style" in html
            assert "avatar-canvas" in html

        req_glb = urllib.request.Request(f"{base}/models/aarna_teen_casual.glb", method="HEAD")
        with urllib.request.urlopen(req_glb, timeout=5) as resp_glb:
            assert resp_glb.status == 200
            assert "gltf-binary" in resp_glb.headers.get("Content-Type", "")

        # 2. Traversal & non-allowlisted files are strictly rejected with 403 Forbidden
        blocked_urls = [
            f"{base}/../pyproject.toml",
            f"{base}/%2e%2e/pyproject.toml",
            f"{base}/models/../../src/main.py",
            f"{base}/run_prototype.py",
            f"{base}/deliverables/aarna_v2_front_fullbody.png",
            f"{base}/captures/aarna_v2_front_fullbody.png",
        ]
        for bad_url in blocked_urls:
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(bad_url, timeout=3)
            assert exc_info.value.code == 403, f"Expected 403 for {bad_url}"
    finally:
        server.stop()
        assert server.is_running is False


def test_avatar_widget_webengine_failure_fallback_and_timeout(qapp, tmp_path):
    """Verify fallback activation on WebEngine failure, missing model file, and load timeout."""
    # 1. Simulated WebEngine failure
    w_fail = AarnaAvatarWidget(
        enable_webengine=True,
        force_webengine_failure=True,
    )
    assert w_fail.is_fallback_active is True
    assert "Simulated WebEngine" in w_fail.fallback_reason
    assert w_fail.stack.currentWidget() is w_fail.fallback_widget

    # State updates still work cleanly in fallback mode
    w_fail.start_thinking()
    assert w_fail.current_state == AvatarState.THINKING
    assert "THINKING" in w_fail.fallback_widget.state_label.text()
    w_fail.finish_thinking()
    assert w_fail.current_state == AvatarState.IDLE

    with pytest.raises(ValueError):
        w_fail.set_outfit("neon_green")
    with pytest.raises(ValueError):
        w_fail.set_camera("birdseye")

    w_fail.shutdown()

    # 2. Missing 3D model file in asset directory triggers fallback cleanly
    empty_dir = tmp_path / "empty_assets"
    empty_dir.mkdir()
    (empty_dir / "index.html").write_text("<html><body></body></html>", encoding="utf-8")

    w_missing = AarnaAvatarWidget(
        asset_dir=empty_dir,
        enable_webengine=True,
    )
    assert w_missing.is_fallback_active is True
    assert "aarna_teen_casual.glb" in w_missing.fallback_reason
    w_missing.shutdown()

    # 3. Load timeout triggers fallback cleanly when characterLoaded never becomes True
    slow_dir = tmp_path / "slow_assets"
    (slow_dir / "models").mkdir(parents=True)
    (slow_dir / "models" / "aarna_teen_casual.glb").write_bytes(b"glTF" + b"\x00" * 32)
    (slow_dir / "index.html").write_text("<html><head></head><body>No AARNA_PROTO</body></html>", encoding="utf-8")

    w_timeout = AarnaAvatarWidget(
        asset_dir=slow_dir,
        enable_webengine=False,
        load_timeout_ms=60,
    )
    # Manually simulate starting the timeout watchdog before readiness completes
    w_timeout._is_fallback_active = False
    w_timeout._is_ready = False
    w_timeout._timeout_timer.start(40)

    loop = QEventLoop()
    QTimer.singleShot(90, loop.quit)
    loop.exec()

    assert w_timeout.is_fallback_active is True
    assert "timed out" in w_timeout.fallback_reason
    w_timeout.shutdown()


def test_main_window_hide_show_and_chat_when_avatar_disabled(qapp):
    """Verify Show/Hide Avatar toggle, chat when avatar is disabled, and text focus does not trigger LISTENING."""
    cfg_disabled = AppConfig(enable_avatar=False, default_model="qwen2.5:3b")
    win = AarnaMainWindow(config=cfg_disabled)

    assert win.is_avatar_visible is False
    assert win.toggle_avatar_btn.text() == "Show Avatar"
    assert win.avatar_widget.is_paused is True

    # Text input focus must NEVER trigger LISTENING
    win.input_edit.setFocus()
    qapp.processEvents()
    assert win.avatar_widget.current_state == AvatarState.IDLE

    # Chat & TimeService work normally while avatar is disabled/hidden
    win.input_edit.setPlainText("What time is it?")
    win.transmit_message()
    history = win.conversation.get_history()
    assert len(history) == 2
    assert "The current local time is" in history[1].content
    assert win.avatar_widget.current_state == AvatarState.IDLE

    # Toggle avatar visibility ON and OFF
    win.toggle_avatar_visibility()
    assert win.is_avatar_visible is True
    assert win.toggle_avatar_btn.text() == "Hide Avatar"

    win.toggle_avatar_visibility()
    assert win.is_avatar_visible is False
    assert win.toggle_avatar_btn.text() == "Show Avatar"
    assert win.avatar_widget.is_paused is True

    win.close()
    qapp.processEvents()
    if win.status_worker and win.status_worker.isRunning():
        win.status_worker.wait(3000)


def test_main_window_ai_lifecycle_state_transitions(qapp, mocker):
    """
    Verify Section C state rules during AI request processing:
    - Avatar enters THINKING when request starts.
    - Incoming LLM tokens (_on_chunk_received) keep avatar in THINKING (do NOT trigger SPEAKING).
    - Finishing generation (_on_finished_response) returns avatar to IDLE.
    - Error (_on_error_occurred or offline send) transitions avatar to ERROR.
    - SPEAKING state is available for future TTS via start_speaking() / stop_speaking().
    """
    win = AarnaMainWindow(config=AppConfig(enable_avatar=True))
    win.is_connected = True

    # Prevent actual background thread network call so we can step through signals deterministically
    mocker.patch.object(win, "check_connection_status")
    mock_worker_start = mocker.patch("src.ui.main_window.ChatWorker.start")

    assert win.avatar_widget.current_state == AvatarState.IDLE

    # 1. User transmits prompt -> THINKING
    win.input_edit.setPlainText("Explain quantum tunneling briefly.")
    win.transmit_message()
    assert mock_worker_start.called
    assert win.avatar_widget.current_state == AvatarState.THINKING

    # 2. Streaming LLM tokens arrive -> MUST stay in THINKING (NOT SPEAKING)
    win._on_chunk_received("Quantum ")
    assert win.avatar_widget.current_state == AvatarState.THINKING
    win._on_chunk_received("tunneling is...")
    assert win.avatar_widget.current_state == AvatarState.THINKING
    assert AvatarState.SPEAKING not in win.avatar_widget.state_history

    # 3. Generation finishes -> returns to IDLE
    win._on_finished_response("Quantum tunneling is a quantum mechanical phenomenon.")
    win._on_worker_finished()
    assert win.avatar_widget.current_state == AvatarState.IDLE

    # 4. Future TTS hooks (SPEAKING) and Microphone hooks (LISTENING) work when explicitly invoked
    win.avatar_widget.start_listening()
    assert win.avatar_widget.current_state == AvatarState.LISTENING
    win.avatar_widget.stop_listening()
    assert win.avatar_widget.current_state == AvatarState.IDLE

    win.avatar_widget.start_speaking()
    assert win.avatar_widget.current_state == AvatarState.SPEAKING
    win.avatar_widget.stop_speaking()
    assert win.avatar_widget.current_state == AvatarState.IDLE

    # 5. Error transitions avatar to ERROR without disrupting chat
    win._on_error_occurred("Simulated timeout")
    assert win.avatar_widget.current_state == AvatarState.ERROR

    win.close()
    qapp.processEvents()


def test_live_integrated_main_window_webgl_handshake_and_shutdown():
    """
    Launch the full integrated AarnaMainWindow with live WebGL in a subprocess and verify:
    - JavaScript readiness handshake completes (is_ready == True)
    - Default Baby Pink frock is active (outfitKey == 'babypink')
    - Initial GREETING state triggers 'wave' gesture
    - State update to THINKING updates Three.js activeExpression to 'thinking'
    - Hiding avatar pauses WebGL renderer (rendererPaused == True)
    - Closing window shuts down LocalAssetServer and workers cleanly
    """
    script = f"""
import json, sys
from pathlib import Path
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

project_root = Path(r"{PROJECT_ROOT}")
sys.path.insert(0, str(project_root))

from src.core.config import AppConfig
from src.ui.avatar import AvatarState
from src.ui.main_window import AarnaMainWindow

app = QApplication(sys.argv)
cfg = AppConfig(enable_avatar=True, avatar_default_outfit="babypink")
win = AarnaMainWindow(config=cfg, enable_webengine=True)
win.show()

def wait_ms(ms: int):
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()

for _ in range(60):
    wait_ms(120)
    if win.avatar_widget.is_ready:
        break

assert win.avatar_widget.is_ready is True, f"Avatar failed to load: {{win.avatar_widget.fallback_reason}}"
assert win.avatar_widget.current_state == AvatarState.GREETING
assert win.avatar_widget.outfit_key == "babypink"

box = {{}}
win.avatar_widget.query_live_metrics(lambda m: box.update({{"m": m}}))
wait_ms(250)
m1 = box.get("m")
assert isinstance(m1, dict), "Failed to query live WebGL metrics"
assert m1["characterLoaded"] is True
assert m1["outfitKey"] == "babypink"
assert m1["activeGesture"] == "wave"

# Transition to THINKING and verify JS scene updated
win.avatar_widget.start_thinking()
wait_ms(250)
win.avatar_widget.query_live_metrics(lambda m: box.update({{"m2": m}}))
wait_ms(250)
m2 = box.get("m2")
assert m2["activeExpression"] == "thinking"

# Hide avatar and verify WebGL renderer pauses
win.set_avatar_visible(False)
wait_ms(200)
win.avatar_widget.query_live_metrics(lambda m: box.update({{"m3": m}}))
wait_ms(200)
m3 = box.get("m3")
assert m3["rendererPaused"] is True

# Clean shutdown
win.close()
assert win.avatar_widget.server is None
print("INTEGRATED_MAIN_WINDOW_WEBGL_OK")
"""
    env = dict(os.environ)
    env.pop("QT_QPA_PLATFORM", None)
    res = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=35,
    )
    assert res.returncode == 0, f"Subprocess failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
    assert "INTEGRATED_MAIN_WINDOW_WEBGL_OK" in res.stdout
