"""Turn-boundary contracts for live-frame tool results.

``get_current_frame`` pixels are useful only inside the turn that requested
them.  They must remain available for that turn's next model iteration, but an
older turn's pixels must not be replayed as if they were current.  These tests
also protect user attachments, other multimodal tools, provider signatures,
and assistant/tool-result pairing.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.conversation_loop import (
    _current_turn_boundary_index,
    _project_message_for_current_turn,
)


OLD_IMAGE = "data:image/jpeg;base64,T0xE"
FRESH_IMAGE = "data:image/jpeg;base64,RlJFU0g="


def _image_part(url: str, *, part_type: str = "image_url") -> dict:
    if part_type == "image":
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": url.rsplit(",", 1)[-1],
            },
        }
    if part_type == "input_image":
        return {"type": "input_image", "image_url": url}
    return {"type": "image_url", "image_url": {"url": url}}


def _project(messages: list[dict], current_turn_user_idx: int) -> list[dict]:
    return [
        _project_message_for_current_turn(
            message,
            message_index=index,
            current_turn_user_idx=current_turn_user_idx,
        )
        for index, message in enumerate(messages)
    ]


def _contains_image_url(value: object, needle: str) -> bool:
    if isinstance(value, dict):
        return any(_contains_image_url(item, needle) for item in value.values())
    if isinstance(value, list):
        return any(_contains_image_url(item, needle) for item in value)
    return isinstance(value, str) and needle in value


def test_historical_get_current_frame_drops_pixels_but_keeps_text_and_pairing():
    message = {
        "role": "tool",
        "name": "get_current_frame",
        "tool_name": "get_current_frame",
        "tool_call_id": "call_old_frame",
        "content": [
            {"type": "text", "text": "Current view: three frames, ts=12.5s"},
            _image_part(OLD_IMAGE),
        ],
    }
    original = copy.deepcopy(message)

    projected = _project_message_for_current_turn(
        message,
        message_index=2,
        current_turn_user_idx=5,
    )

    assert projected["tool_call_id"] == "call_old_frame"
    assert projected["name"] == "get_current_frame"
    assert projected["tool_name"] == "get_current_frame"
    assert projected["content"] == [
        {"type": "text", "text": "Current view: three frames, ts=12.5s"}
    ]
    assert not _contains_image_url(projected, OLD_IMAGE)
    assert message == original


def test_same_turn_get_current_frame_keeps_fresh_pixels_for_next_iteration():
    message = {
        "role": "tool",
        "name": "get_current_frame",
        "tool_call_id": "call_fresh_frame",
        "content": [
            {"type": "text", "text": "Current view: fresh frame"},
            _image_part(FRESH_IMAGE),
        ],
    }

    projected = _project_message_for_current_turn(
        message,
        message_index=7,
        current_turn_user_idx=5,
    )

    assert _contains_image_url(projected, FRESH_IMAGE)
    assert projected["content"] == message["content"]


def test_user_image_attachment_is_never_treated_as_stale_tool_output():
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is this?"},
            _image_part(OLD_IMAGE),
        ],
    }

    projected = _project_message_for_current_turn(
        message,
        message_index=0,
        current_turn_user_idx=4,
    )

    assert _contains_image_url(projected, OLD_IMAGE)


def test_other_multimodal_tool_images_are_not_removed():
    message = {
        "role": "tool",
        "name": "show_memory_frame",
        "tool_name": "show_memory_frame",
        "tool_call_id": "call_memory_frame",
        "content": [
            {"type": "text", "text": "Historical key frame requested by user"},
            _image_part(OLD_IMAGE),
        ],
    }

    projected = _project_message_for_current_turn(
        message,
        message_index=2,
        current_turn_user_idx=5,
    )

    assert _contains_image_url(projected, OLD_IMAGE)


def test_all_supported_image_part_shapes_are_removed_from_old_current_frame():
    message = {
        "role": "tool",
        # Some restored/provider-neutral transcripts have only ``name``.
        "name": "get_current_frame",
        "tool_call_id": "call_shapes",
        "content": [
            {"type": "text", "text": "frame summary"},
            _image_part(OLD_IMAGE, part_type="image_url"),
            _image_part(OLD_IMAGE, part_type="input_image"),
            _image_part(OLD_IMAGE, part_type="image"),
        ],
    }

    projected = _project_message_for_current_turn(
        message,
        message_index=1,
        current_turn_user_idx=3,
    )

    assert projected["content"] == [{"type": "text", "text": "frame summary"}]


def test_image_only_old_result_keeps_nonempty_tool_result_placeholder():
    message = {
        "role": "tool",
        "tool_name": "get_current_frame",
        "tool_call_id": "call_image_only",
        "content": [_image_part(OLD_IMAGE)],
    }

    projected = _project_message_for_current_turn(
        message,
        message_index=1,
        current_turn_user_idx=2,
    )

    assert projected["tool_call_id"] == "call_image_only"
    assert projected["content"] == [{
        "type": "text",
        "text": "[previous current-frame image omitted from cross-turn replay]",
    }]


def test_boundary_recomputed_after_sequence_repair_keeps_fresh_frame():
    from agent.agent_runtime_helpers import repair_message_sequence_with_cursor

    messages = [
        # These malformed historical results are removed before request build,
        # shifting every later index left of build_turn_context's old index.
        *[
            {
                "role": "tool",
                "tool_call_id": f"orphan_{index}",
                "content": "orphan",
            }
            for index in range(5)
        ],
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "current question"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_fresh_after_repair",
                "type": "function",
                "function": {
                    "name": "get_current_frame",
                    "arguments": "{}",
                },
            }],
        },
        {
            "role": "tool",
            "name": "get_current_frame",
            "tool_call_id": "call_fresh_after_repair",
            "content": [
                {"type": "text", "text": "fresh after repair"},
                _image_part(FRESH_IMAGE),
            ],
        },
    ]
    stale_boundary = 7
    bare_agent = SimpleNamespace(_last_flushed_db_idx=0)

    assert repair_message_sequence_with_cursor(bare_agent, messages) == 5
    repaired_boundary = _current_turn_boundary_index(messages, stale_boundary)
    projected = _project(messages, repaired_boundary)

    assert repaired_boundary == 2
    assert _contains_image_url(projected, FRESH_IMAGE)


def test_internal_recovery_user_does_not_advance_current_turn_boundary():
    """Empty/tool recovery stays inside the original user's visual turn."""
    original_user = {"role": "user", "content": "What is visible now?"}
    messages = [
        {"role": "user", "content": "older question"},
        {"role": "assistant", "content": "older answer"},
        original_user,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_recovery_frame",
                "type": "function",
                "function": {
                    "name": "get_current_frame",
                    "arguments": "{}",
                },
            }],
        },
        {
            "role": "tool",
            "name": "get_current_frame",
            "tool_call_id": "call_recovery_frame",
            "content": [
                {"type": "text", "text": "fresh recovery frame"},
                _image_part(FRESH_IMAGE),
            ],
        },
        {
            "role": "assistant",
            "content": "(empty)",
            "_empty_recovery_synthetic": True,
        },
        {
            "role": "user",
            "content": "Please process the tool result and continue.",
            "_empty_recovery_synthetic": True,
            "_turn_internal_synthetic": True,
        },
    ]

    boundary = _current_turn_boundary_index(
        messages,
        fallback=2,
        boundary_message=original_user,
    )
    projected = _project(messages, boundary)

    assert boundary == 2
    assert _contains_image_url(projected, FRESH_IMAGE)


def test_boundary_fallback_ignores_internal_recovery_user_after_repair():
    """Even if repair replaces the original dict, marked nudges are not turns."""
    messages = [
        {"role": "user", "content": "older"},
        {"role": "assistant", "content": "older answer"},
        {"role": "user", "content": "current real user"},
        {
            "role": "assistant",
            "content": "(empty)",
            "_empty_recovery_synthetic": True,
        },
        {
            "role": "user",
            "content": "internal retry",
            "_turn_internal_synthetic": True,
        },
    ]

    missing_identity = {"role": "user", "content": "was replaced"}
    boundary = _current_turn_boundary_index(
        messages,
        fallback=2,
        boundary_message=missing_identity,
    )

    assert boundary == 2


def test_full_compression_with_active_todo_preserves_same_turn_frame_only():
    """Compression copies the turn tail; its TODO reinjection is not a turn.

    Exercise the real ``compress_context`` orchestration rather than manually
    constructing its output.  The compressor deliberately deep-copies the
    retained current-turn tail, which destroys the boundary user's object
    identity.  The active TODO is then appended by production code.  Boundary
    fallback must ignore that synthetic user: the current iteration keeps the
    fresh pixels, while a later real user turn strips them.
    """
    from run_agent import AIAgent

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("hermes_logging.setup_logging"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            model="openai/gpt-4o",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    current_user = {"role": "user", "content": "What is visible now?"}
    fresh_call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call_before_compression",
            "type": "function",
            "function": {
                "name": "get_current_frame",
                "arguments": '{"query":"inspect now"}',
            },
        }],
    }
    fresh_result = {
        "role": "tool",
        "name": "get_current_frame",
        "tool_call_id": "call_before_compression",
        "content": [
            {"type": "text", "text": "fresh frame before compression"},
            _image_part(FRESH_IMAGE),
        ],
    }
    messages = [
        {"role": "user", "content": "older question"},
        {"role": "assistant", "content": "older answer"},
        current_user,
        fresh_call,
        fresh_result,
    ]

    # Keep the current user/tool tail, but copy every dict exactly as the real
    # compressor does. The identity-based boundary marker is therefore gone.
    def _compress_copying_tail(
        source, current_tokens=None, focus_topic=None, force=False,
    ):
        del current_tokens, focus_topic, force
        return [
            {"role": "assistant", "content": "compressed older context"},
            *copy.deepcopy(source[-3:]),
        ]

    agent._session_db = None
    agent._memory_manager = None
    agent._compression_feasibility_checked = True
    agent.context_compressor.compress = _compress_copying_tail
    agent.context_compressor._last_compress_aborted = False
    agent.context_compressor._last_summary_error = None
    agent.context_compressor._last_aux_model_failure_model = None
    agent._todo_store.write([{
        "id": "inspect",
        "content": "finish inspecting the current view",
        "status": "in_progress",
    }])

    with patch.object(agent, "_build_system_prompt", return_value="stable prompt"):
        compressed, _ = agent._compress_context(
            messages,
            "stable prompt",
            approx_tokens=100_000,
        )

    assert compressed[-1]["role"] == "user"
    assert compressed[-1]["_turn_internal_synthetic"] is True
    assert "active task list" in compressed[-1]["content"]

    same_turn_boundary = _current_turn_boundary_index(
        compressed,
        fallback=2,
        boundary_message=current_user,
    )
    same_turn_projection = _project(compressed, same_turn_boundary)
    assert compressed[same_turn_boundary]["content"] == "What is visible now?"
    assert _contains_image_url(same_turn_projection, FRESH_IMAGE)

    next_user = {"role": "user", "content": "What about the next view?"}
    next_turn_messages = [*compressed, next_user]
    next_turn_boundary = _current_turn_boundary_index(
        next_turn_messages,
        fallback=len(next_turn_messages) - 1,
        boundary_message=next_user,
    )
    next_turn_projection = _project(next_turn_messages, next_turn_boundary)
    assert next_turn_boundary == len(next_turn_messages) - 1
    assert not _contains_image_url(next_turn_projection, FRESH_IMAGE)


def test_anthropic_replay_keeps_signed_tool_use_and_valid_tool_result_pair():
    from agent.anthropic_adapter import convert_messages_to_anthropic

    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_signed_frame",
                "type": "function",
                "function": {
                    "name": "get_current_frame",
                    "arguments": '{"query":"what is visible?"}',
                },
            }],
            "anthropic_content_blocks": [
                {
                    "type": "thinking",
                    "thinking": "I need the current view.",
                    "signature": "signed-thinking-block",
                },
                {
                    "type": "tool_use",
                    "id": "call_signed_frame",
                    "name": "get_current_frame",
                    "input": {"query": "what is visible?"},
                },
            ],
        },
        {
            "role": "tool",
            "name": "get_current_frame",
            "tool_name": "get_current_frame",
            "tool_call_id": "call_signed_frame",
            "content": [
                {"type": "text", "text": "Current view summary"},
                _image_part(OLD_IMAGE),
            ],
        },
        {"role": "user", "content": "A new turn starts here"},
    ]
    original = copy.deepcopy(messages)
    projected = _project(messages, current_turn_user_idx=2)

    _system, wire = convert_messages_to_anthropic(projected)

    assert wire[0]["content"][0] == {
        "type": "thinking",
        "thinking": "I need the current view.",
        "signature": "signed-thinking-block",
    }
    assert wire[0]["content"][1] == {
        "type": "tool_use",
        "id": "call_signed_frame",
        "name": "get_current_frame",
        "input": {"query": "what is visible?"},
    }
    tool_results = [
        block
        for message in wire
        for block in message.get("content", [])
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    assert len(tool_results) == 1
    assert tool_results[0]["tool_use_id"] == "call_signed_frame"
    assert not _contains_image_url(tool_results[0], OLD_IMAGE)
    assert messages == original


def test_messages_proxy_replay_keeps_pair_and_moves_no_old_image_to_user_blocks():
    from agent.chat_completion_helpers import _anthropic_messages_from_openai

    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_proxy_frame",
                "type": "function",
                "function": {
                    "name": "get_current_frame",
                    "arguments": '{"query":"read screen"}',
                },
            }],
        },
        {
            "role": "tool",
            "name": "get_current_frame",
            "tool_call_id": "call_proxy_frame",
            "content": [
                {"type": "text", "text": "screen text summary"},
                _image_part(OLD_IMAGE),
            ],
        },
        {"role": "user", "content": "new question"},
    ]

    _system, wire = _anthropic_messages_from_openai(
        _project(messages, current_turn_user_idx=2)
    )

    assert wire[0]["content"][0] == {
        "type": "tool_use",
        "id": "call_proxy_frame",
        "name": "get_current_frame",
        "input": {"query": "read screen"},
    }
    tool_results = [
        block
        for message in wire
        for block in message.get("content", [])
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    assert len(tool_results) == 1
    assert tool_results[0]["tool_use_id"] == "call_proxy_frame"
    assert "screen text summary" in str(tool_results[0]["content"])
    assert not _contains_image_url(wire, OLD_IMAGE)


def test_chat_completions_replay_preserves_gemini_thought_signature():
    import agent.transports.chat_completions  # noqa: F401
    from agent.transports import get_transport

    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_gemini_frame",
                "type": "function",
                "function": {
                    "name": "get_current_frame",
                    "arguments": '{"query":"inspect"}',
                },
                "extra_content": {
                    "google": {"thought_signature": "gemini-signed-thought"}
                },
            }],
        },
        {
            "role": "tool",
            "name": "get_current_frame",
            "tool_call_id": "call_gemini_frame",
            "content": [
                {"type": "text", "text": "old visual summary"},
                _image_part(OLD_IMAGE),
            ],
        },
        {"role": "user", "content": "new turn"},
    ]

    wire = get_transport("chat_completions").convert_messages(
        _project(messages, current_turn_user_idx=2),
        model="google/gemini-3-pro-preview",
    )

    assert wire[0]["tool_calls"][0]["extra_content"] == {
        "google": {"thought_signature": "gemini-signed-thought"}
    }
    assert wire[1]["tool_call_id"] == "call_gemini_frame"
    assert not _contains_image_url(wire[1], OLD_IMAGE)


def _tool_def(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _response(*, content: str = "done", tool_calls: list | None = None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason="tool_calls" if tool_calls else "stop")
    return SimpleNamespace(choices=[choice], model="openai/gpt-4o", usage=None)


def test_conversation_loop_strips_old_pixels_and_keeps_new_tool_pixels():
    """Exercise the actual two-iteration request construction, not just helper IO."""
    from run_agent import AIAgent

    old_tool_message = {
        "role": "tool",
        "name": "get_current_frame",
        "tool_name": "get_current_frame",
        "tool_call_id": "call_old_frame",
        "content": [
            {"type": "text", "text": "old frame summary"},
            _image_part(OLD_IMAGE),
        ],
    }
    history = [
        {"role": "user", "content": "What is visible?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_old_frame",
                "type": "function",
                "function": {
                    "name": "get_current_frame",
                    "arguments": '{"query":"what is visible?"}',
                },
            }],
        },
        old_tool_message,
        {"role": "assistant", "content": "I saw the old frame."},
    ]
    fresh_call = SimpleNamespace(
        id="call_fresh_frame",
        type="function",
        function=SimpleNamespace(
            name="get_current_frame",
            arguments='{"query":"what is visible now?"}',
        ),
    )
    fresh_result = {
        "_multimodal": True,
        "content": [
            {"type": "text", "text": "fresh frame summary"},
            _image_part(FRESH_IMAGE),
        ],
        "text_summary": "fresh frame summary",
    }

    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=[_tool_def("get_current_frame")],
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("hermes_logging.setup_logging"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            model="openai/gpt-4o",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = [
        _response(content="", tool_calls=[fresh_call]),
        _response(content="done"),
    ]
    agent._cached_system_prompt = "stable prompt"
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False

    with (
        patch("run_agent.handle_function_call", return_value=fresh_result),
        patch.object(agent, "_model_supports_vision", return_value=True),
        patch.object(
            agent,
            "_provider_supports_vision_tool_messages",
            return_value=True,
        ),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            "What is visible now?",
            conversation_history=history,
        )

    assert result["completed"] is True
    assert result["final_response"] == "done"
    calls = agent.client.chat.completions.create.call_args_list
    assert len(calls) == 2
    first_request = calls[0].kwargs["messages"]
    second_request = calls[1].kwargs["messages"]
    assert not _contains_image_url(first_request, OLD_IMAGE)
    assert not _contains_image_url(second_request, OLD_IMAGE)
    assert _contains_image_url(second_request, FRESH_IMAGE)
    assert _contains_image_url(old_tool_message, OLD_IMAGE)


def test_real_empty_recovery_request_keeps_same_turn_current_frame_pixels():
    """A synthetic recovery user cannot turn a fresh frame into old history."""
    from run_agent import AIAgent

    frame_call = SimpleNamespace(
        id="call_frame_before_empty",
        type="function",
        function=SimpleNamespace(
            name="get_current_frame",
            arguments='{"query":"inspect now"}',
        ),
    )
    fresh_result = {
        "_multimodal": True,
        "content": [
            {"type": "text", "text": "fresh frame before empty response"},
            _image_part(FRESH_IMAGE),
        ],
        "text_summary": "fresh frame before empty response",
    }

    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=[_tool_def("get_current_frame")],
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("hermes_logging.setup_logging"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            model="openai/gpt-4o",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = [
        _response(content="", tool_calls=[frame_call]),
        _response(content=""),
        _response(content="recovered answer"),
    ]
    agent._cached_system_prompt = "stable prompt"
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False

    with (
        patch("run_agent.handle_function_call", return_value=fresh_result),
        patch.object(agent, "_model_supports_vision", return_value=True),
        patch.object(
            agent,
            "_provider_supports_vision_tool_messages",
            return_value=True,
        ),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("What is visible now?")

    assert result["final_response"] == "recovered answer"
    requests = agent.client.chat.completions.create.call_args_list
    assert len(requests) == 3
    assert _contains_image_url(requests[1].kwargs["messages"], FRESH_IMAGE)
    assert _contains_image_url(requests[2].kwargs["messages"], FRESH_IMAGE)
    assert any(
        m.get("role") == "user"
        and "Please process the tool results" in str(m.get("content"))
        for m in requests[2].kwargs["messages"]
    )
