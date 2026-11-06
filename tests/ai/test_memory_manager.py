"""
Unit and integration tests for Phase 7 Persistent Memory Architecture in Project AARNA.
Validates SQLite3 memory.db schema, non-blocking asynchronous wrappers,
user profile persistence, conversational context snippet generation, and anchor prompting.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import pytest

from src.ai.memory_manager import MemoryManager, UserProfile, MemoryInteraction
from src.ai.prompts import AARNA_SYSTEM_PROMPT, inject_memory_context_into_prompt
from src.core.paths import get_database_path


def test_database_creation_and_default_profile_seeding(tmp_path: Path) -> None:
    """Verify memory.db creation and default user profile seeding."""
    db_file = tmp_path / "test_memory.db"
    manager = MemoryManager(db_path=db_file)

    assert db_file.exists()

    profile = manager.get_user_profile()
    assert profile.name == "Abhi"
    assert profile.preferred_language == "hinglish"
    assert "B.Tech" in profile.engineering_branch
    assert "AARNA" in profile.btech_context


def test_user_profile_update_and_custom_flags(tmp_path: Path) -> None:
    """Verify profile updating and custom flags JSON serialization."""
    db_file = tmp_path / "test_memory.db"
    manager = MemoryManager(db_path=db_file)

    updated = manager.update_user_profile(
        name="Abhishek",
        preferred_language="en",
        engineering_branch="AI & Data Science",
        btech_context="Autonomous Multi-Agent Audio Systems",
        custom_flags={"dark_mode": True, "voice_speed": 1.1},
    )

    assert updated.name == "Abhishek"
    assert updated.preferred_language == "en"
    assert updated.engineering_branch == "AI & Data Science"
    assert updated.custom_flags.get("dark_mode") is True
    assert updated.custom_flags.get("voice_speed") == 1.1

    # Verify reloading from disk
    reloaded = manager.get_user_profile()
    assert reloaded.name == "Abhishek"
    assert reloaded.custom_flags["voice_speed"] == 1.1


def test_conversation_history_logging_and_tag_retrieval(tmp_path: Path) -> None:
    """Verify interaction turn logging and recent history retrieval."""
    db_file = tmp_path / "test_memory.db"
    manager = MemoryManager(db_path=db_file)

    row_id1 = manager.log_interaction(
        user_query="Kal college me presentation hai BTech project ka.",
        assistant_response="All the best! Main aapke presentation slides revise karwa sakti hoon.",
        intent_summary="BTech project presentation preparation",
        context_tags=["btech", "college", "presentation"],
    )
    assert row_id1 > 0

    row_id2 = manager.log_interaction(
        user_query="Code ka real-time pipeline optimize kar do.",
        assistant_response="Sure, humne PyAudio in-memory buffer decode implement kar diya hai.",
        intent_summary="Real-time pipeline optimization",
        context_tags=["coding", "audio_pipeline"],
    )
    assert row_id2 > row_id1

    recent = manager.get_recent_interactions(limit=5)
    assert len(recent) == 2
    # Verify latest first (descending timestamp order)
    assert recent[0].intent_summary == "Real-time pipeline optimization"
    assert "audio_pipeline" in recent[0].personal_context_tags
    assert recent[1].intent_summary == "BTech project presentation preparation"


def test_conversational_context_snippet_generation(tmp_path: Path) -> None:
    """Verify generation of dynamic context snippet for LLM system prompt injection."""
    db_file = tmp_path / "test_memory.db"
    manager = MemoryManager(db_path=db_file)

    manager.log_interaction(
        user_query="Working on BTech project neural speech core today.",
        assistant_response="Great! Let's get that done.",
        intent_summary="Neural speech core engineering",
        context_tags=["btech_project", "speech"],
    )

    snippet = manager.get_conversational_context_snippet()

    assert "Abhi" in snippet
    assert "B.Tech" in snippet
    assert "Neural speech core engineering" in snippet
    assert "Memory Anchors" in snippet


@pytest.mark.anyio
async def test_async_memory_operations_non_blocking(tmp_path: Path) -> None:
    """Verify asynchronous non-blocking wrappers execute cleanly inside an event loop."""
    db_file = tmp_path / "async_memory.db"
    manager = MemoryManager(db_path=db_file)

    profile = await manager.get_user_profile_async()
    assert profile.name == "Abhi"

    updated = await manager.update_user_profile_async(name="Abhi V2")
    assert updated.name == "Abhi V2"

    log_id = await manager.log_interaction_async(
        user_query="Test async query",
        assistant_response="Test async response",
        intent_summary="Async non-blocking check",
    )
    assert log_id > 0

    recent = await manager.get_recent_interactions_async(limit=2)
    assert len(recent) == 1
    assert recent[0].intent_summary == "Async non-blocking check"

    snippet = await manager.get_conversational_context_snippet_async()
    assert "Abhi V2" in snippet


def test_conversational_anchor_prompt_injection() -> None:
    """Verify injection of memory snippet into AARNA system instructions."""
    memory_snippet = (
        "Persistent User Context & Memory Anchors:\n"
        "- User Name: Abhi\n"
        "- Academic / Project Context: BTech Final Year Project"
    )

    enhanced = inject_memory_context_into_prompt(AARNA_SYSTEM_PROMPT, memory_snippet)
    assert "You are AARNA" in enhanced
    assert "User Name: Abhi" in enhanced
    assert "Conversational Memory Anchoring" in enhanced


def test_database_path_resolution_and_env_override(monkeypatch) -> None:
    """Verify database path resolution using src.core.paths and env override."""
    default_path = get_database_path()
    assert default_path.is_absolute()
    assert default_path.name == "memory.db"

    # Test override
    override_val = "C:/custom/test/aarna_memory.db"
    monkeypatch.setenv("AARNA_DB_PATH", override_val)
    resolved_override = get_database_path()
    assert str(resolved_override).replace("\\", "/").endswith("custom/test/aarna_memory.db")
