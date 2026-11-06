"""
Tests for AARNA personality prompt and conversational behaviour regression.
Verifies friendly demeanor, multilingual guidelines, follow-up retention,
and absence of robotic tropes (no 'Hey', no repeated 'sir', no fake actions).
"""

import pytest
from src.ai.prompts import AARNA_SYSTEM_PROMPT, JARVIS_SYSTEM_PROMPT
from src.ai.conversation import ConversationManager


def test_personality_identity_and_nomenclature():
    """Verify core identity rules in AARNA system prompt."""
    assert "AARNA" in AARNA_SYSTEM_PROMPT
    assert '"Aarna!"' in AARNA_SYSTEM_PROMPT
    # Must strictly forbid requiring 'Hey'
    assert 'The prefix "Hey" is strictly NOT required.' in AARNA_SYSTEM_PROMPT


def test_personality_conversational_demeanor():
    """Verify that friendly, articulate, witty, and non-robotic guidelines are enforced."""
    prompt_lower = AARNA_SYSTEM_PROMPT.lower()
    # Friendly and approachable
    assert "friendly & approachable" in prompt_lower
    # Clever and articulate
    assert "articulate" in prompt_lower
    # No robotic 'sir' repetition
    assert 'not repeatedly address the user as "sir"' in prompt_lower
    # Occasionally witty
    assert "occasionally witty" in prompt_lower
    # Honest about limitations
    assert "confident yet honest" in prompt_lower
    # Concise by default
    assert "concise by default" in prompt_lower


def test_personality_multilingual_and_script_rules():
    """Verify multilingual support for English, Hindi, and Hinglish."""
    prompt_lower = AARNA_SYSTEM_PROMPT.lower()
    assert "english" in prompt_lower
    assert "hindi" in prompt_lower
    assert "hinglish" in prompt_lower
    assert "code-switching" in prompt_lower
    # Ensure script confusion warning is present
    assert "urdu" in prompt_lower  # warning against substituting Urdu script


def test_personality_operational_boundaries():
    """Verify that action hallucinations and local PC boundaries are established."""
    prompt = AARNA_SYSTEM_PROMPT
    assert "Windows workstation" in prompt or "Windows PC" in prompt
    assert "Never claim to have taken an action" in prompt


def test_backward_compatibility_prompt_alias():
    """Ensure JARVIS_SYSTEM_PROMPT remains an exact alias for existing code."""
    assert JARVIS_SYSTEM_PROMPT == AARNA_SYSTEM_PROMPT


def test_conversation_followup_tracking():
    """Verify multi-turn follow-up history is correctly maintained."""
    mgr = ConversationManager(system_prompt=AARNA_SYSTEM_PROMPT, max_history=10)
    mgr.add_user_message("What is quantum computing?")
    mgr.add_assistant_message("Quantum computing uses qubits that exploit superposition and entanglement.")
    mgr.add_user_message("How does superposition work?")
    mgr.add_assistant_message("Superposition allows a qubit to be in a linear combination of states |0> and |1>.")

    payload = mgr.get_messages_for_api(include_time_context=False)
    assert len(payload) == 5
    assert payload[0]["role"] == "system"
    assert payload[1] == {"role": "user", "content": "What is quantum computing?"}
    assert payload[2] == {
        "role": "assistant",
        "content": "Quantum computing uses qubits that exploit superposition and entanglement.",
    }
    assert payload[3] == {"role": "user", "content": "How does superposition work?"}
    assert payload[4] == {
        "role": "assistant",
        "content": "Superposition allows a qubit to be in a linear combination of states |0> and |1>.",
    }


def test_conversation_multilingual_turns():
    """Verify multi-turn conversation handles English, Hindi, and Hinglish turns smoothly."""
    mgr = ConversationManager(system_prompt=AARNA_SYSTEM_PROMPT)
    # English turn
    mgr.add_user_message("Hi Aarna, how are you today?")
    mgr.add_assistant_message("I'm doing well! Ready to help with your work today.")
    # Hindi turn
    mgr.add_user_message("क्या आज कोई महत्वपूर्ण अपडेट है?")
    mgr.add_assistant_message("आज सिस्टम पूरी तरह से सामान्य और सक्रिय है।")
    # Hinglish turn
    mgr.add_user_message("Aarna, meeting schedule check kar do.")
    mgr.add_assistant_message("Aapke calendar mein filhal koi naya meeting scheduled nahi hai.")

    payload = mgr.get_messages_for_api(include_time_context=False)
    assert len(payload) == 7
    assert payload[3]["content"] == "क्या आज कोई महत्वपूर्ण अपडेट है?"
    assert payload[5]["content"] == "Aarna, meeting schedule check kar do."


def test_personality_conversational_companion_dynamics():
    """Verify companion pacing, 1-2 sentence casual rule, and anti-chatbot mandates."""
    prompt = AARNA_SYSTEM_PROMPT
    prompt_lower = prompt.lower()

    # Pacing and short casual responses
    assert "1 to 2 short, spontaneous sentences" in prompt_lower
    assert "curious follow-up question" in prompt_lower
    assert "anti-chatbot" in prompt_lower or "anti-essay" in prompt_lower

    # Forbids robotic canned intros
    assert "as an ai language model" in prompt_lower
    assert "hello, i am aarna" in prompt_lower

    # Strict script matching
    assert "roman hinglish" in prompt_lower
    assert "devanagari hindi" in prompt_lower


def test_personality_dialogue_examples():
    """Verify standard natural conversational examples are documented in prompt."""
    prompt = AARNA_SYSTEM_PROMPT
    assert "Aaj college jaane ka mann nahi hai" in prompt
    assert "Arey, kya hua?" in prompt
    assert "Bas boring lagta hai" in prompt

