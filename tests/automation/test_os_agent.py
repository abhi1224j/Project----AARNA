"""
Unit and integration test suite for Phase 8-10 Windows Automation Skills & Smart OS Agent Hooks.
Verifies native OS tools, Hinglish intent parsing, resource monitoring, gaming/rendering detection,
volume auto-adjustment, and non-blocking async execution.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtCore import QCoreApplication

from src.automation.os_agent import (
    AARNA_AUTOMATION_TOOLS_SCHEMA,
    AutomationToolKit,
    HinglishIntentParser,
    OSAutomationAgent,
    SystemResourceMonitor,
)
from src.voice.controller import VoiceController
from src.voice.models import TranscriptionResult, TranscriptionSegment


@pytest.fixture(scope="module")
def qapp():
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])
    return app


# -----------------------------------------------------------------------------
# 1. Native Tool Execution Tests (os, subprocess, webbrowser, shutil)
# -----------------------------------------------------------------------------

def test_automation_toolkit_open_app() -> None:
    """Verify application launching maps known names to safe executables."""
    with patch("subprocess.Popen") as mock_popen:
        res = AutomationToolKit.open_app("notepad")
        assert res["status"] == "success"
        assert res["app"] == "notepad"
        mock_popen.assert_called_once()
        args, kwargs = mock_popen.call_args
        assert "notepad.exe" in args[0]

    with patch("webbrowser.open") as mock_wb:
        res = AutomationToolKit.open_app("chrome")
        assert res["status"] == "success"
        mock_wb.assert_called_once()


def test_automation_toolkit_search_web() -> None:
    """Verify Google and YouTube web search query construction and dispatch."""
    with patch("webbrowser.open") as mock_wb:
        # 1. Google search
        res_g = AutomationToolKit.search_web("Python async tutorial", platform="google")
        assert res_g["status"] == "success"
        assert "google.com" in res_g["url"]
        assert "Python+async+tutorial" in res_g["url"]

        # 2. YouTube search
        res_yt = AutomationToolKit.search_web("study music", platform="youtube")
        assert res_yt["status"] == "success"
        assert "youtube.com" in res_yt["url"]
        assert "study+music" in res_yt["url"]


def test_automation_toolkit_file_operations(tmp_path: Path) -> None:
    """Verify safe local file creation, reading, directory listing, and deletion."""
    test_file = tmp_path / "aarna_test.txt"

    # 1. Create file
    create_res = AutomationToolKit.create_file(str(test_file), content="Project AARNA OS Automation")
    assert create_res["status"] == "success"
    assert test_file.exists()

    # 2. Read file
    read_res = AutomationToolKit.read_file(str(test_file))
    assert read_res["status"] == "success"
    assert read_res["content"] == "Project AARNA OS Automation"

    # 3. List directory
    list_res = AutomationToolKit.list_directory(str(tmp_path))
    assert list_res["status"] == "success"
    names = [e["name"] for e in list_res["entries"]]
    assert "aarna_test.txt" in names

    # 4. Delete file
    del_res = AutomationToolKit.delete_file(str(test_file))
    assert del_res["status"] == "success"
    assert not test_file.exists()


# -----------------------------------------------------------------------------
# 2. Hinglish & Conversational Intent Parsing Tests
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "utterance,expected_tool,expected_platform,expected_query",
    [
        ("YouTube par study music chalao", "search_web", "youtube", "study music"),
        ("Play lofi beats on youtube", "search_web", "youtube", "lofi beats"),
        ("YouTube pe relax music search karo", "search_web", "youtube", "relax music"),
        ("Google par Python docs search karo", "search_web", "google", "Python docs"),
    ],
)
def test_hinglish_intent_youtube_and_web_search(
    utterance: str, expected_tool: str, expected_platform: str, expected_query: str
) -> None:
    """Verify mapping Hinglish music and web search intents to tool calls."""
    parsed = HinglishIntentParser.parse_intent(utterance)
    assert parsed is not None
    assert parsed["tool"] == expected_tool
    assert parsed["parameters"]["platform"] == expected_platform
    assert expected_query.lower() in parsed["parameters"]["query"].lower()


@pytest.mark.parametrize(
    "utterance,expected_app",
    [
        ("Mera Notepad kholo", "notepad"),
        ("Notepad open karo", "notepad"),
        ("Calculator open karo", "calculator"),
        ("Mera calc start karo", "calc"),
        ("Chrome browser kholo", "chrome"),
    ],
)
def test_hinglish_intent_app_launch(utterance: str, expected_app: str) -> None:
    """Verify mapping Hinglish app opening intents to open_app."""
    parsed = HinglishIntentParser.parse_intent(utterance)
    assert parsed is not None
    assert parsed["tool"] == "open_app"
    assert parsed["parameters"]["app_name"] == expected_app


def test_hinglish_intent_file_creation() -> None:
    """Verify mapping Hinglish file creation commands."""
    parsed = HinglishIntentParser.parse_intent("Ek naya text file banao")
    assert parsed is not None
    assert parsed["tool"] == "create_file"
    assert "file_path" in parsed["parameters"]


def test_hinglish_intent_extracts_llm_json_tool_call() -> None:
    """Verify parsing explicit tool call JSON generated by Gemini Flash."""
    llm_output = (
        "Bilkul! Main aapke liye Notepad open kar rahi hoon.\n"
        "```tool_call\n"
        '{"tool": "open_app", "parameters": {"app_name": "notepad"}}\n'
        "```"
    )
    parsed = HinglishIntentParser.parse_intent(llm_output)
    assert parsed is not None
    assert parsed["tool"] == "open_app"
    assert parsed["parameters"]["app_name"] == "notepad"


# -----------------------------------------------------------------------------
# 3. Dynamic Hardware Awareness & Disruption Mode Tests
# -----------------------------------------------------------------------------

def test_system_resource_monitor_metrics() -> None:
    """Verify metric collection returns valid numeric load values."""
    monitor = SystemResourceMonitor()
    metrics = monitor.get_metrics()
    assert "cpu_percent" in metrics
    assert "memory_percent" in metrics
    assert "is_fullscreen" in metrics
    assert "is_high_load" in metrics
    assert 0.0 <= metrics["cpu_percent"] <= 100.0
    assert 0.0 <= metrics["memory_percent"] <= 100.0


def test_disruption_mode_triggered_by_fullscreen(mocker) -> None:
    """Verify full-screen gaming/rendering triggers Quiet / Disrupted Mode."""
    monitor = SystemResourceMonitor()
    mocker.patch.object(monitor, "is_fullscreen_active", return_value=True)

    is_disrupted, reason = monitor.check_disruption_state()
    assert is_disrupted is True
    assert "Full-screen gaming or rendering" in reason


def test_disruption_mode_triggered_by_high_cpu(mocker) -> None:
    """Verify host CPU spikes >=85% trigger Quiet Mode."""
    monitor = SystemResourceMonitor()
    mocker.patch.object(monitor, "is_fullscreen_active", return_value=False)
    mocker.patch.object(monitor, "get_metrics", return_value={
        "cpu_percent": 92.5,
        "memory_percent": 60.0,
        "gpu_percent": None,
        "is_fullscreen": False,
        "is_high_load": True,
    })

    is_disrupted, reason = monitor.check_disruption_state()
    assert is_disrupted is True
    assert "High host CPU load" in reason


def test_os_agent_auto_adjusts_volume_and_ui_overlay(qapp, mocker) -> None:
    """Verify OSAutomationAgent softens playback volume and updates UI overlay when gaming/heavy load is detected."""
    mock_player = MagicMock()
    mock_ui = MagicMock()

    agent = OSAutomationAgent(audio_player=mock_player, ui_sink=mock_ui)

    # 1. Simulate transition into gaming / high load state
    mocker.patch.object(agent.monitor, "check_disruption_state", return_value=(True, "Full-screen gaming active"))
    mocker.patch.object(agent.monitor, "get_metrics", return_value={"cpu_percent": 50.0, "memory_percent": 50.0, "is_fullscreen": True, "is_high_load": True})

    agent._check_system_health()

    assert agent._is_quiet_mode is True
    mock_player.set_volume.assert_called_with(0.35)
    mock_ui.set_quiet_overlay_mode.assert_called_with(True, "Full-screen gaming active")

    # 2. Simulate return to normal workload
    mocker.patch.object(agent.monitor, "check_disruption_state", return_value=(False, "Normal"))
    agent._check_system_health()

    assert agent._is_quiet_mode is False
    mock_player.set_volume.assert_called_with(1.0)
    mock_ui.set_quiet_overlay_mode.assert_called_with(False, "Normal")


# -----------------------------------------------------------------------------
# 4. Asynchronous Non-Blocking Execution & Controller Linking
# -----------------------------------------------------------------------------

def test_async_tool_execution_non_blocking(tmp_path: Path) -> None:
    """Verify tool execution via execute_tool_async runs in thread pool without blocking event loop."""
    agent = OSAutomationAgent()
    target_file = tmp_path / "async_demo.txt"

    res = asyncio.run(agent.execute_tool_async(
        "create_file",
        {"file_path": str(target_file), "content": "Async non-blocking execution verified."},
    ))
    assert res["status"] == "success"
    assert target_file.exists()
    assert target_file.read_text(encoding="utf-8") == "Async non-blocking execution verified."


def test_voice_controller_integration_with_os_agent(qapp, mocker) -> None:
    """Verify VoiceController executes OS automation skills on user voice transcription."""
    mock_agent = MagicMock(spec=OSAutomationAgent)
    mock_agent.handle_user_utterance.return_value = {"status": "success", "tool": "open_app"}

    ctrl = VoiceController(os_agent=mock_agent)
    assert ctrl.os_agent is mock_agent

    dummy_result = TranscriptionResult(
        text="Mera Notepad kholo",
        detected_language="hi",
        language_probability=0.95,
        processing_duration_seconds=0.1,
        audio_duration_seconds=1.0,
        model_identifier="faster-whisper-tiny",
        compute_device="cpu",
        compute_type="int8",
        requested_language_mode="auto",
        segments=(TranscriptionSegment(0.0, 1.0, "Mera Notepad kholo", 0.0, 0.0),),
    )

    ctrl._on_transcription_finished(dummy_result)
    mock_agent.handle_user_utterance.assert_called_with("Mera Notepad kholo")

    ctrl.shutdown()
    mock_agent.stop_monitoring.assert_called_once()
