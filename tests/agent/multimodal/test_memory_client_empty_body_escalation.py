"""A reasoning route that eats its whole budget must not return silence.

Structured memory calls (frame verify, recall decide/distill) pass a tight
max_tokens because they only need a small JSON object. A reasoning endpoint
spends that budget on hidden reasoning first, so the call returns HTTP 200 with
finish_reason=length and an EMPTY body. The caller then reports "invalid JSON",
retries at the same cap, and fails identically: one production frame-verify burned
80s across two attempts and passed 0 of 8 frames. The client escalates the cap
once instead.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from agent.multimodal import _workers
from agent.multimodal._workers import OpenAIMemoryClient


def _response(text, finish_reason="stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=text),
            finish_reason=finish_reason,
        )],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=1,
                              total_tokens=11),
    )


def _client(responses):
    calls = []

    async def _create(**kwargs):
        calls.append(kwargs)
        return responses[min(len(calls) - 1, len(responses) - 1)]

    transport = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create)))
    return OpenAIMemoryClient(transport, "GPT-5.6 Sol"), calls


def _cap_of(kwargs):
    return kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")


def test_empty_length_capped_body_is_retried_with_a_bigger_budget():
    client, calls = _client([
        _response("", finish_reason="length"),
        _response('{"keep":["f1"]}'),
    ])

    text = asyncio.run(client.call_chat(
        [{"role": "user", "content": "verify"}],
        max_tokens=384, usage_kind="recall_verify_frames"))

    assert text == '{"keep":["f1"]}'
    assert len(calls) == 2, calls
    # The retry differs only in budget, and the escalation clears the floor that
    # lets a reasoning model emit its JSON after thinking.
    assert _cap_of(calls[0]) == 384
    assert _cap_of(calls[1]) >= 2048


def test_escalation_is_capped_so_a_large_caller_cannot_balloon():
    client, calls = _client([_response("", finish_reason="length")] * 2)

    asyncio.run(client.call_chat(
        [{"role": "user", "content": "write"}],
        max_tokens=5000, usage_kind="memory_writer"))

    assert len(calls) == 2
    assert _cap_of(calls[1]) <= _workers._CALL_CHAT_EMPTY_BODY_MAX_CAP


def test_a_portable_first_route_is_not_retried_at_the_same_effective_budget():
    """Escalation must be measured against the budget actually sent.

    Portable-first routes raise the cap internally to max(cap, 8192), so a
    1024-token verify already goes out at 8192. Escalating the *nominal* 1024
    lands back on 8192 — a byte-identical retry, i.e. exactly the wasted call
    this branch exists to remove.
    """
    client, calls = _client([_response("", finish_reason="length")] * 2)
    client.model = "gpt-5.6-luna"  # portable-first

    asyncio.run(client.call_chat(
        [{"role": "user", "content": "verify"}],
        max_tokens=1024, usage_kind="recall_verify_frames"))

    assert len(calls) == 2
    first, second = _cap_of(calls[0]), _cap_of(calls[1])
    assert first == 8192, first          # nominal 1024 was raised internally
    assert second > first, (first, second)


def test_truncated_but_non_empty_body_is_kept_without_a_second_call():
    """Partial output is the caller's to repair; only silence is unusable."""
    client, calls = _client([_response('{"keep":["f1"', finish_reason="length")])

    text = asyncio.run(client.call_chat(
        [{"role": "user", "content": "verify"}], max_tokens=384))

    assert text == '{"keep":["f1"'
    assert len(calls) == 1


def test_normal_completion_makes_exactly_one_call():
    client, calls = _client([_response("done")])

    assert asyncio.run(client.call_chat(
        [{"role": "user", "content": "x"}], max_tokens=384)) == "done"
    assert len(calls) == 1


def test_usage_is_reported_for_both_attempts():
    client, _calls = _client([
        _response("", finish_reason="length"),
        _response("ok"),
    ])
    seen = []
    client.on_usage = seen.append

    asyncio.run(client.call_chat(
        [{"role": "user", "content": "x"}], max_tokens=384,
        usage_kind="recall_verify_frames"))

    # Token accounting must not silently lose the wasted first attempt.
    assert len(seen) == 2
    assert {row["usage_kind"] for row in seen} == {"recall_verify_frames"}
