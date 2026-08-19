"""The zero-chunk stream guard must report what actually happened.

A stream that closes with no content and no ``finish_reason`` used to raise one
fixed sentence blaming a "possible upstream error or malformed SSE response".
Two unrelated failures land there:

- frames arrived but carried nothing and no terminator → genuinely malformed SSE
- nothing arrived at all, then a clean close → on a reverse-proxied endpoint
  this is normally the proxy's own response deadline firing while a reasoning
  model was still working toward its first token

Blaming "malformed SSE" for the second case is an actively misleading
diagnosis: it hides that the wait was a fixed deadline which every retry will
hit again. These tests pin that the message distinguishes the two and carries
the elapsed time and chunk count needed to tell them apart.

The guard had no test coverage before this file.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _make_stream_chunk(content=None, tool_calls=None, finish_reason=None):
    delta = SimpleNamespace(
        content=content, tool_calls=tool_calls,
        reasoning_content=None, reasoning=None,
    )
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model=None, usage=None)


def _make_agent():
    from run_agent import AIAgent
    agent = AIAgent(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    return agent


def _run(mock_create, stream_factory, monkeypatch):
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = (
        lambda *a, **kw: stream_factory()
    )
    mock_create.return_value = mock_client
    agent = _make_agent()
    agent._fire_stream_delta = lambda text: None
    monkeypatch.setenv("ARGUS_STREAM_RETRIES", "0")
    with pytest.raises(RuntimeError) as excinfo:
        agent._interruptible_streaming_api_call({})
    return str(excinfo.value)


class TestEmptyStreamDiagnosis:
    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_zero_chunks_blames_a_gateway_deadline_not_malformed_sse(
        self, _mock_close, mock_create, monkeypatch
    ):
        """Nothing arrived at all → name the proxy deadline, not bad SSE."""

        def _silent_stream():
            return iter(())

        msg = _run(mock_create, _silent_stream, monkeypatch)

        assert "without sending a single chunk" in msg
        # The actionable part: the reader must learn this is probably a
        # deadline and that retrying the same request will not help.
        assert "deadline" in msg
        assert "60s" in msg
        assert "malformed" in msg.lower(), (
            "the message should still say what it is NOT, so the old wrong "
            "diagnosis does not get re-derived"
        )
        assert "not a malformed" in msg

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_empty_chunks_still_reported_as_malformed_sse(
        self, _mock_close, mock_create, monkeypatch
    ):
        """Frames arrived but carried nothing → that really is malformed SSE."""

        def _empty_frames_stream():
            yield _make_stream_chunk()
            yield _make_stream_chunk()
            yield _make_stream_chunk()

        msg = _run(mock_create, _empty_frames_stream, monkeypatch)

        assert "3 chunk(s) arrived" in msg
        assert "malformed or truncated" in msg
        assert "without sending a single chunk" not in msg

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_elapsed_time_is_reported(self, _mock_close, mock_create, monkeypatch):
        """Elapsed is the signal that turns 'mystery' into 'fixed deadline'."""

        def _silent_stream():
            return iter(())

        msg = _run(mock_create, _silent_stream, monkeypatch)
        assert "after " in msg and "s " in msg

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_a_usable_stream_does_not_trip_the_guard(
        self, _mock_close, mock_create, monkeypatch
    ):
        """Sanity: real content must not be mistaken for an empty stream."""

        def _good_stream():
            yield _make_stream_chunk(content="hello")
            yield _make_stream_chunk(finish_reason="stop")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = (
            lambda *a, **kw: _good_stream()
        )
        mock_create.return_value = mock_client
        agent = _make_agent()
        agent._fire_stream_delta = lambda text: None
        monkeypatch.setenv("ARGUS_STREAM_RETRIES", "0")

        response = agent._interruptible_streaming_api_call({})
        assert response.choices[0].finish_reason == "stop"
        assert response.choices[0].message.content == "hello"
