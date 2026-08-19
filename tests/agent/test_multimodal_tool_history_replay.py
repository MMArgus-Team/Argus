"""Replay contracts for sessions that completed the retired MM tool call.

The old tool name may remain in immutable conversation history even though it
is no longer exposed in a fresh schema.  Provider replay must preserve that
completed assistant/tool pair exactly: renaming only one side can break call-id
pairing, and dropping provider signatures can make the next request fail.
"""

from __future__ import annotations

import copy

from agent.agent_runtime_helpers import sanitize_api_messages


LEGACY_TOOL_NAME = "recall_multimodal_memory"
LEGACY_CALL_ID = "call_legacy_mm_1"


def test_legacy_completed_exchange_keeps_name_id_and_gemini_signature():
    """Sanitizer + Chat Completions adapter must not migrate stored history."""
    import agent.transports.chat_completions  # noqa: F401
    from agent.transports import get_transport

    history = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": LEGACY_CALL_ID,
                    "type": "function",
                    "function": {
                        "name": LEGACY_TOOL_NAME,
                        "arguments": '{"query":"what was on screen?"}',
                    },
                    "extra_content": {
                        "google": {"thought_signature": "gemini-sig-legacy"}
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": LEGACY_CALL_ID,
            "content": '{"found":true,"findings":"a red cup"}',
        },
    ]
    original = copy.deepcopy(history)

    sanitized = sanitize_api_messages(history)
    wire = get_transport("chat_completions").convert_messages(
        sanitized,
        model="google/gemini-3-pro-preview",
    )

    call = wire[0]["tool_calls"][0]
    assert call["id"] == LEGACY_CALL_ID
    assert call["function"]["name"] == LEGACY_TOOL_NAME
    assert call["function"]["arguments"] == '{"query":"what was on screen?"}'
    assert call["extra_content"] == {
        "google": {"thought_signature": "gemini-sig-legacy"}
    }
    assert wire[1]["tool_call_id"] == LEGACY_CALL_ID

    # Neither pre-call sanitization nor provider conversion may rewrite the
    # persisted transcript in place; another provider may replay it later.
    assert history == original


def test_legacy_completed_exchange_keeps_anthropic_tool_and_thinking_signature():
    """Anthropic ordered-block replay keeps the old completed call byte-stable."""
    from agent.anthropic_adapter import convert_messages_to_anthropic

    history = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": LEGACY_CALL_ID,
                    "type": "function",
                    "function": {
                        "name": LEGACY_TOOL_NAME,
                        "arguments": '{"query":"earlier sign"}',
                    },
                }
            ],
            "anthropic_content_blocks": [
                {
                    "type": "thinking",
                    "thinking": "I should inspect multimodal memory.",
                    "signature": "anthropic-sig-legacy",
                },
                {
                    "type": "tool_use",
                    "id": LEGACY_CALL_ID,
                    "name": LEGACY_TOOL_NAME,
                    "input": {"query": "earlier sign"},
                    # Output-only fields are allowed to be removed; the replay
                    # identity fields above must remain untouched.
                    "caller": None,
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": LEGACY_CALL_ID,
            "content": '{"found":true,"findings":"Lifan Center"}',
        },
    ]
    original = copy.deepcopy(history)

    sanitized = sanitize_api_messages(history)
    _system, wire = convert_messages_to_anthropic(sanitized)

    thinking, tool_use = wire[0]["content"]
    assert thinking == {
        "type": "thinking",
        "thinking": "I should inspect multimodal memory.",
        "signature": "anthropic-sig-legacy",
    }
    assert tool_use == {
        "type": "tool_use",
        "id": LEGACY_CALL_ID,
        "name": LEGACY_TOOL_NAME,
        "input": {"query": "earlier sign"},
    }
    assert wire[1]["content"][0]["tool_use_id"] == LEGACY_CALL_ID
    assert history == original
