"""Shared wire-format contract for direct custom messages endpoints."""

from __future__ import annotations

from types import SimpleNamespace


OPENAI_WIRE_ENDPOINT = "https://doc.devops.example.com/u/project/v1/messages"
ANTHROPIC_WIRE_ENDPOINT = "https://gateway.example.com/anthropic/v1/messages"


def _request_kwargs():
    return {
        "model": "GPT-5.6 Luna",
        "messages": [
            {"role": "system", "content": "Be concise."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Inspect this frame"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,ZmFrZQ==",
                        },
                    },
                ],
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_frame",
                    "type": "function",
                    "function": {
                        "name": "get_current_frame",
                        "arguments": '{"query":"latest"}',
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_frame",
                "content": "frame captured",
            },
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_current_frame",
                "description": "Read a frame",
                "parameters": {"type": "object", "properties": {}},
            },
        }],
        "tool_choice": {
            "type": "function",
            "function": {"name": "get_current_frame"},
        },
    }


def _all_content_types(value):
    found = []
    if isinstance(value, dict):
        if isinstance(value.get("type"), str):
            found.append(value["type"])
        for child in value.values():
            found.extend(_all_content_types(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_all_content_types(child))
    return found


def test_hybrid_leaf_keeps_image_url_but_converts_historical_tool_pairs():
    from agent.chat_completion_helpers import _messages_proxy_payload
    from agent.multimodal.hermes_glue import _messages_payload

    main_payload = _messages_proxy_payload(
        _request_kwargs(), OPENAI_WIRE_ENDPOINT)
    multimodal_payload = _messages_payload(
        _request_kwargs(), base_url=OPENAI_WIRE_ENDPOINT)

    for payload in (main_payload, multimodal_payload):
        types = _all_content_types(payload["messages"])
        assert "image_url" in types
        assert "image" not in types
        assert "tool_use" in types
        assert "tool_result" in types
        assistant = next(
            message for message in payload["messages"]
            if message.get("role") == "assistant"
            and any(
                part.get("type") == "tool_use"
                for part in message.get("content") or []
                if isinstance(part, dict)
            )
        )
        tool_use = next(
            part for part in assistant["content"]
            if part.get("type") == "tool_use"
        )
        assert tool_use["id"] == "call_frame"
        assert tool_use["name"] == "get_current_frame"
        assert tool_use["input"] == {"query": "latest"}
        tool_result = next(
            part
            for message in payload["messages"]
            if message.get("role") == "user"
            for part in (
                message.get("content")
                if isinstance(message.get("content"), list)
                else []
            )
            if isinstance(part, dict) and part.get("type") == "tool_result"
        )
        assert tool_result["tool_use_id"] == "call_frame"
        assert tool_result["content"] == "frame captured"
        assert not any(message.get("role") == "tool"
                       for message in payload["messages"])
        assert not any(message.get("tool_calls")
                       for message in payload["messages"])
        assert payload["tools"] == [{
            "name": "get_current_frame",
            "description": "Read a frame",
            "input_schema": {"type": "object", "properties": {}},
        }]
        assert payload["tool_choice"] == {
            "type": "tool",
            "name": "get_current_frame",
        }


def test_explicit_anthropic_route_still_converts_content_and_tools():
    from agent.chat_completion_helpers import _messages_proxy_payload
    from agent.multimodal.hermes_glue import _messages_payload

    main_payload = _messages_proxy_payload(
        _request_kwargs(), ANTHROPIC_WIRE_ENDPOINT)
    multimodal_payload = _messages_payload(
        _request_kwargs(), base_url=ANTHROPIC_WIRE_ENDPOINT)

    for payload in (main_payload, multimodal_payload):
        types = _all_content_types(payload["messages"])
        assert "image" in types
        assert "image_url" not in types
        assert "tool_use" in types
        assert "tool_result" in types
        assert payload["tools"][0]["name"] == "get_current_frame"


def test_first_turn_continuous_watch_produces_live_watcher_tool_call(
    monkeypatch,
):
    """The Luna hybrid leaf can see and return set_live_watcher on turn one."""
    from agent.chat_completion_helpers import _call_messages_proxy
    from agent.prompt_builder import MM_LIVE_GUIDANCE
    from tools.live_watcher_tool import SET_LIVE_WATCHER_SCHEMA

    captured = {}

    class _Response:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {
                "id": "msg_watch_1",
                "type": "message",
                "role": "assistant",
                "model": "GPT-5.6 Luna",
                "stop_reason": "tool_use",
                "content": [{
                    "type": "tool_use",
                    "id": "call_watch_1",
                    "name": "set_live_watcher",
                    "input": {
                        "op": "create",
                        "task_instruction": "持续观看视频并在结束时总结内容",
                        "hook_main_agent": True,
                    },
                }],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }

    def _post(endpoint, *, headers, json, timeout):
        captured.update({
            "endpoint": endpoint,
            "headers": headers,
            "json": json,
            "timeout": timeout,
        })
        return _Response()

    monkeypatch.setattr("httpx.post", _post)
    agent = SimpleNamespace(
        base_url=OPENAI_WIRE_ENDPOINT,
        api_key="test-key",
        provider="custom",
        model="GPT-5.6 Luna",
    )
    response = _call_messages_proxy(agent, {
        "model": agent.model,
        "messages": [
            {"role": "system", "content": MM_LIVE_GUIDANCE},
            {
                "role": "user",
                "content": "给我持续看这个视频，视频结束告诉我讲了啥",
            },
        ],
        "tools": [{
            "type": "function",
            "function": SET_LIVE_WATCHER_SCHEMA,
        }],
        "tool_choice": "auto",
    })

    request = captured["json"]
    assert request["messages"] == [{
        "role": "user",
        "content": "给我持续看这个视频，视频结束告诉我讲了啥",
    }]
    assert "call set_live_watcher" in request["system"]
    assert request["tools"][0]["name"] == "set_live_watcher"
    assert request["tools"][0]["input_schema"] == (
        SET_LIVE_WATCHER_SCHEMA["parameters"])
    assert "function" not in request["tools"][0]
    assert request["tool_choice"] == {"type": "auto"}

    choice = response.choices[0]
    assert choice.finish_reason == "tool_calls"
    call = choice.message.tool_calls[0]
    assert call.function.name == "set_live_watcher"
    assert '"hook_main_agent": true' in call.function.arguments


def test_multimodal_send_log_reports_tool_count_and_wire():
    from agent.chat_completion_helpers import _mm_diag_before

    emitted = []
    agent = SimpleNamespace(
        _multimodal_session=True,
        _extra_body_additions=None,
        _mm_diag_emit=emitted.append,
        base_url=OPENAI_WIRE_ENDPOINT,
        api_mode="chat_completions",
        model="GPT-5.6 Luna",
    )

    _mm_diag_before(agent, {
        "model": "GPT-5.6 Luna",
        "messages": [{"role": "user", "content": "watch"}],
        "tools": _request_kwargs()["tools"],
    })

    assert emitted[0]["tool_count"] == 1
    assert emitted[0]["tool_wire"] == "anthropic"
