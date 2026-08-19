"""Offline wire-contract tests for ``MessagesMemoryClient``.

The memory writer always supplies OpenAI-shaped messages. A ``/v1/messages``
path alone does not determine its content schema: remote Luna and the local
127.0.0.1:8080 hybrid endpoint accept OpenAI content parts, while an explicit
``/anthropic`` route selects Anthropic blocks.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import AsyncMock, patch

import pytest

from agent.multimodal._workers import MessagesMemoryClient
from agent.multimodal.hermes_glue import (
    HermesClientFactory,
    MessagesChatCompletionsClient,
    build_submodule_client,
)


class _HTTPResponse:
    status_code = 200
    text = ""

    def __init__(self, body: Dict[str, Any]):
        self._body = body

    def json(self) -> Dict[str, Any]:
        return self._body


def _anthropic_ok(
    text: str = "ok", *, input_tokens: int = 3, output_tokens: int = 2,
) -> Dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


async def _call_with_mock_post(
    client: MessagesMemoryClient,
    messages: list[dict[str, Any]],
    response: Dict[str, Any],
) -> tuple[str | None, AsyncMock]:
    post = AsyncMock(return_value=_HTTPResponse(response))
    client._client.post = post
    try:
        result = await client.call_chat(
            messages, max_tokens=512, temperature=0.2,
        )
    finally:
        await client.aclose()
    return result, post


def test_remote_messages_lifts_system_and_preserves_openai_image_parts():
    client = MessagesMemoryClient(
        base_url="https://messages.example.test/proxy/v1/messages",
        api_key="test-key",
        model="GPT-5.6 Luna",
    )
    messages = [
        {"role": "system", "content": "You are the memory writer."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this frame."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,aW1hZ2UtYnl0ZXM="
                    },
                },
            ],
        },
    ]

    result, post = asyncio.run(
        _call_with_mock_post(client, messages, _anthropic_ok("converted"))
    )

    assert result == "converted"
    post.assert_awaited_once()
    endpoint = post.await_args.args[0]
    kwargs = post.await_args.kwargs
    payload = kwargs["json"]
    assert endpoint == "https://messages.example.test/proxy/v1/messages"
    assert kwargs["headers"]["anthropic-version"] == "2023-06-01"
    assert payload["system"] == "You are the memory writer."
    assert all(message["role"] != "system" for message in payload["messages"])
    assert payload["messages"] == [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this frame."},
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,aW1hZ2UtYnl0ZXM=",
                },
            },
        ],
    }]
    assert "image_url" in repr(payload["messages"])
    assert "'type': 'image'" not in repr(payload["messages"])


def test_local_8080_hybrid_preserves_openai_system_and_image_url():
    client = MessagesMemoryClient(
        base_url="http://127.0.0.1:8080/v1",
        api_key="test-key",
        model="GPT-5.6 Luna",
    )
    image_part = {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,bG9jYWwtaW1hZ2U="},
    }
    messages = [
        {"role": "system", "content": "local system"},
        {"role": "user", "content": [
            {"type": "text", "text": "local prompt"}, image_part,
        ]},
    ]

    result, post = asyncio.run(
        _call_with_mock_post(client, messages, _anthropic_ok("hybrid"))
    )

    assert result == "hybrid"
    payload = post.await_args.kwargs["json"]
    assert post.await_args.args[0] == "http://127.0.0.1:8080/v1/messages"
    assert "system" not in payload
    assert payload["messages"] == messages
    assert payload["messages"][1]["content"][1] == image_part
    assert payload["extra_body"]["enable_thinking"] is False


def test_remote_kimi_preserves_image_url_and_low_reasoning_effort():
    client = MessagesMemoryClient(
        base_url="https://messages.example.test/proxy/v1/messages",
        api_key="test-key",
        model="kimi/kimi-k3",
    )
    image_part = {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,a2ltaS1pbWFnZQ=="},
    }
    messages = [
        {"role": "system", "content": "You are the memory reviewer."},
        {"role": "user", "content": [
            {"type": "text", "text": "Review these frames."}, image_part,
        ]},
    ]

    result, post = asyncio.run(
        _call_with_mock_post(client, messages, _anthropic_ok("kimi"))
    )

    assert result == "kimi"
    payload = post.await_args.kwargs["json"]
    assert payload["system"] == "You are the memory reviewer."
    assert payload["reasoning_effort"] == "low"
    assert payload["messages"][0]["content"][1] == image_part
    assert "source" not in repr(payload["messages"])


def test_anthropic_response_joins_text_blocks_and_normalizes_usage():
    client = MessagesMemoryClient(
        base_url="https://messages.example.test/v1/messages",
        api_key="test-key",
        model="GPT-5.6 Luna",
    )
    usage_events: list[Dict[str, Any]] = []
    client.on_usage = usage_events.append
    response = _anthropic_ok(input_tokens=123, output_tokens=45)
    response["content"] = [
        {"type": "text", "text": '{"observation_text":"phone"'},
        {"type": "thinking", "thinking": "not user-visible"},
        {"type": "text", "text": ',"key_frame_indices":[0]}'},
    ]

    result, post = asyncio.run(_call_with_mock_post(client, [], response))

    assert post.await_count == 1
    assert result == '{"observation_text":"phone","key_frame_indices":[0]}'
    assert usage_events == [{
        "promptTokenCount": 123,
        "candidatesTokenCount": 45,
        "totalTokenCount": 168,
        "usage_kind": "",
    }]


@pytest.mark.parametrize("model", ["kimi-k2.6", "qwen3.7-plus"])
def test_explicit_messages_leaf_selects_messages_memory_client(model: str):
    cfg = SimpleNamespace(
        memory_provider="custom",
        memory_base_url="https://messages.example.test/v1/messages",
        memory_api_key="test-key",
        memory_model=model,
    )

    client = HermesClientFactory(cfg).memory_client(None)
    try:
        assert isinstance(client, MessagesMemoryClient)
        assert client.endpoint == "https://messages.example.test/v1/messages"
        assert client.model == model
    finally:
        asyncio.run(client.aclose())


@pytest.mark.parametrize("role", ["monitor", "watcher"])
@pytest.mark.parametrize("model", ["kimi-k2.6", "qwen3.7-plus"])
def test_explicit_messages_leaf_selects_messages_submodule_client(
    role: str, model: str,
):
    if role == "monitor":
        cfg = SimpleNamespace(
            monitor_provider="custom",
            monitor_base_url="https://messages.example.test/v1/messages",
            monitor_api_key="test-key",
            monitor_model=model,
        )
        method = "monitor_client"
    else:
        cfg = SimpleNamespace(
            watcher_provider="custom",
            watcher_base_url="https://messages.example.test/v1/messages",
            watcher_api_key="test-key",
            model=model,
        )
        method = "worker_client"

    client, resolved_model = getattr(HermesClientFactory(cfg), method)()
    try:
        assert isinstance(client, MessagesChatCompletionsClient)
        assert client.endpoint == "https://messages.example.test/v1/messages"
        assert resolved_model == model
    finally:
        asyncio.run(client.aclose())


@pytest.mark.parametrize("model", ["GPT-5.6 Luna", "kimi/kimi-k3"])
def test_memory_factory_preserves_model_messages_heuristic(model: str):
    cfg = SimpleNamespace(
        memory_provider="custom",
        memory_base_url="https://messages.example.test/v1",
        memory_api_key="test-key",
        memory_model=model,
    )

    client = HermesClientFactory(cfg).memory_client(None)
    try:
        assert isinstance(client, MessagesMemoryClient)
        assert client.endpoint == "https://messages.example.test/v1/messages"
        assert client.model == model
    finally:
        asyncio.run(client.aclose())


def test_recall_verify_factory_uses_dedicated_messages_endpoint():
    primary = SimpleNamespace(model="GPT-5.6 Luna")
    cfg = SimpleNamespace(
        memory_provider="custom",
        memory_api_key="primary-key",
        memory_model="GPT-5.6 Luna",
        recall_verify_provider="custom",
        recall_verify_base_url=(
            "https://doc.devops.beta.xiaohongshu.com/"
            "u/5b102a9d2690c243/v1/messages"
        ),
        recall_verify_api_key="verify-key",
        recall_verify_model="GPT-5.6 Luna",
    )

    client, model = HermesClientFactory(cfg).recall_verify_client(
        recall_client=primary,
        recall_model=primary.model,
    )
    try:
        assert client is not primary
        assert isinstance(client, MessagesMemoryClient)
        assert client.endpoint.endswith("/u/5b102a9d2690c243/v1/messages")
        assert model == "GPT-5.6 Luna"
    finally:
        asyncio.run(client.aclose())


def test_recall_verify_factory_falls_back_to_primary_without_override():
    primary = SimpleNamespace(model="GPT-5.6 Luna")
    cfg = SimpleNamespace(recall_verify_base_url="")

    client, model = HermesClientFactory(cfg).recall_verify_client(
        recall_client=primary,
        recall_model=primary.model,
    )

    assert client is primary
    assert model == primary.model


def test_generic_v1_root_does_not_select_messages_submodule_client():
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock()),
        ),
    )
    with (
        patch("openai.AsyncOpenAI", return_value=fake_client) as async_openai,
        patch(
            "agent.multimodal.hermes_glue._submodule_http_client",
            return_value=None,
        ),
    ):
        client, resolved_model = build_submodule_client(
            provider="custom",
            base_url="https://openai-compatible.example.test/v1",
            api_key="test-key",
            model="qwen3.7-plus",
            resolve_main=lambda: (_ for _ in ()).throw(
                AssertionError("explicit model must not resolve main")
            ),
            label="monitor",
        )

    assert client is fake_client
    assert not isinstance(client, MessagesChatCompletionsClient)
    assert resolved_model == "qwen3.7-plus"
    async_openai.assert_called_once_with(
        base_url="https://openai-compatible.example.test/v1",
        api_key="test-key",
    )


def test_messages_chat_client_forwards_caller_timeout_to_httpx_post():
    async def run_call():
        client = MessagesChatCompletionsClient(
            base_url="https://messages.example.test/v1/messages",
            api_key="test-key",
            model="qwen3.7-plus",
        )
        post = AsyncMock(return_value=_HTTPResponse(_anthropic_ok("bounded")))
        client._client.post = post
        try:
            response = await client.chat.completions.create(
                model="qwen3.7-plus",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=64,
                timeout=30.0,
            )
        finally:
            await client.aclose()
        return response, post

    response, post = asyncio.run(run_call())

    assert response.choices[0].message.content == "bounded"
    post.assert_awaited_once()
    assert post.await_args.kwargs["timeout"] == 30.0
