"""Protocol contracts for graceful/abortive realtime ASR shutdown."""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import types

import pytest

from agent.multimodal.qwen_realtime import QwenRealtimeASR
from agent.multimodal.watcher_engine import WatcherAgent
from agent.multimodal import qwen_realtime, watcher_engine


class _FakeWebSocket:
    def __init__(self, finish_events=None):
        self.finish_events = list(finish_events or [])
        self.incoming: asyncio.Queue = asyncio.Queue()
        self.sent: list[dict] = []
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self.incoming.get()
        if item is None:
            raise StopAsyncIteration
        return json.dumps(item)

    async def send(self, raw):
        event = json.loads(raw)
        self.sent.append(event)
        if event.get("type") == "session.finish":
            for incoming in self.finish_events:
                await self.incoming.put(incoming)

    async def close(self):
        self.closed = True
        await self.incoming.put(None)


async def _attached_asr(ws, finals):
    async def on_final(text):
        finals.append(text)

    asr = QwenRealtimeASR("test-key", on_final=on_final)
    asr._ws = ws
    asr._connected = True
    asr._reader_task = asyncio.create_task(asr._reader())
    return asr


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("events", "expected_segments", "expected_joined"),
    [
        (
            [
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "\u4f60\u597d",
                },
                {"type": "session.finished", "transcript": "\u4f60\u597d\u4e16\u754c"},
            ],
            ["\u4f60\u597d"],
            "\u4f60\u597d\u4e16\u754c",
        ),
        (
            [
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "hello",
                },
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "world",
                },
                {"type": "session.finished", "transcript": "hello world"},
            ],
            ["hello", "world"],
            "hello world",
        ),
        (
            [
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "\u91cd\u590d",
                },
                {"type": "session.finished", "transcript": "\u91cd\u590d"},
            ],
            ["\u91cd\u590d"],
            "\u91cd\u590d",
        ),
        (
            [
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "event_id": "item-1",
                    "transcript": "\u597d",
                },
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "event_id": "item-2",
                    "transcript": "\u597d",
                },
                {"type": "session.finished", "transcript": "\u597d\u597d"},
            ],
            ["\u597d", "\u597d"],
            "\u597d\u597d",
        ),
    ],
)
async def test_graceful_close_waits_and_deduplicates_full_transcript(
    events, expected_segments, expected_joined,
):
    finals = []
    ws = _FakeWebSocket(events)
    asr = await _attached_asr(ws, finals)

    result = await asr.close(finish_timeout=0.5)

    assert ws.sent == [{"type": "session.finish"}]
    assert result == {
        "ok": True,
        "finish_sent": True,
        "completed": True,
        "session_finished": True,
        "timed_out": False,
        "aborted": False,
        "transcript": expected_joined,
    }
    assert finals == expected_segments


@pytest.mark.asyncio
async def test_duplicate_completed_event_id_is_idempotent():
    finals = []
    completed = {
        "type": "conversation.item.input_audio_transcription.completed",
        "event_id": "same-item",
        "transcript": "\u597d",
    }
    ws = _FakeWebSocket([
        completed,
        dict(completed),
        {"type": "session.finished", "transcript": "\u597d"},
    ])
    asr = await _attached_asr(ws, finals)

    result = await asr.close(finish_timeout=0.5)

    assert finals == ["\u597d"]
    assert result["transcript"] == "\u597d"


@pytest.mark.asyncio
async def test_finished_canonical_text_preserves_in_word_refinement():
    finals = []
    ws = _FakeWebSocket([
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "event_id": "cat-item",
            "transcript": "cat",
        },
        {"type": "session.finished", "transcript": "cats"},
    ])
    asr = await _attached_asr(ws, finals)

    result = await asr.close(finish_timeout=0.5)

    # A terminal full rewrite is exposed only through the canonical close
    # contract; replaying its suffix as another VAD segment is ambiguous.
    assert finals == ["cat"]
    assert result["transcript"] == "cats"


@pytest.mark.asyncio
async def test_finished_canonical_punctuation_refines_full_transcript():
    finals = []
    ws = _FakeWebSocket([
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "event_id": "first-item",
            "transcript": "\u4f60\u597d",
        },
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "event_id": "second-item",
            "transcript": "\u4e16\u754c",
        },
        {"type": "session.finished", "transcript": "\u4f60\u597d\uff0c\u4e16\u754c"},
    ])
    asr = await _attached_asr(ws, finals)

    result = await asr.close(finish_timeout=0.5)

    # Terminal punctuation is a canonical rewrite of the two completed items,
    # not a third segment containing the whole transcript again.
    assert finals == ["\u4f60\u597d", "\u4e16\u754c"]
    assert result["transcript"] == "\u4f60\u597d\uff0c\u4e16\u754c"


@pytest.mark.asyncio
async def test_finished_punctuated_full_transcript_is_canonical_only():
    finals = []
    ws = _FakeWebSocket([
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "event_id": "first-item",
            "transcript": "\u4f60\u597d",
        },
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "event_id": "second-item",
            "transcript": "\u4e16\u754c",
        },
        {"type": "session.finished", "transcript": "\u4f60\u597d\uff0c\u4e16\u754c\u518d\u89c1"},
    ])
    asr = await _attached_asr(ws, finals)

    result = await asr.close(finish_timeout=0.5)

    assert finals == ["\u4f60\u597d", "\u4e16\u754c"]
    assert result["transcript"] == "\u4f60\u597d\uff0c\u4e16\u754c\u518d\u89c1"


@pytest.mark.asyncio
async def test_finished_full_transcript_word_level_rewrite_is_authoritative():
    finals = []
    ws = _FakeWebSocket([
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "event_id": "first-item",
            "transcript": "turn on",
        },
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "event_id": "second-item",
            "transcript": "the light",
        },
        {
            "type": "session.finished",
            "transcript": "turn off the light",
        },
    ])
    asr = await _attached_asr(ws, finals)

    result = await asr.close(finish_timeout=0.5)

    assert finals == ["turn on", "the light"]
    assert result["transcript"] == "turn off the light"


@pytest.mark.asyncio
async def test_abortive_close_sends_no_finish_and_delivers_no_queued_final():
    finals = []
    ws = _FakeWebSocket()
    asr = await _attached_asr(ws, finals)
    await ws.incoming.put({
        "type": "conversation.item.input_audio_transcription.completed",
        "transcript": "must not escape cancellation",
    })

    result = await asr.close(graceful=False)

    assert ws.sent == []
    assert result["ok"] is True
    assert result["aborted"] is True
    assert finals == []


@pytest.mark.asyncio
async def test_graceful_close_timeout_is_bounded_and_idempotent():
    finals = []
    ws = _FakeWebSocket()
    asr = await _attached_asr(ws, finals)

    first = await asr.close(finish_timeout=0.01)
    second = await asr.close(finish_timeout=0.01)

    assert first == second
    assert first["timed_out"] is True
    assert first["session_finished"] is False
    assert ws.sent == [{"type": "session.finish"}]


@pytest.mark.asyncio
async def test_terminal_frame_without_completion_reports_incomplete_flush():
    finals = []
    ws = _FakeWebSocket([
        {
            "type": "conversation.item.input_audio_transcription.text",
            "text": "visible partial",
        },
        {"type": "session.finished", "transcript": ""},
    ])
    asr = await _attached_asr(ws, finals)

    result = await asr.close(finish_timeout=0.5)

    assert result["session_finished"] is True
    assert result["completed"] is False
    assert result["timed_out"] is True
    assert finals == []


@pytest.mark.asyncio
async def test_partial_after_prior_completed_segment_remains_incomplete():
    finals = []
    ws = _FakeWebSocket([
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "event_id": "completed-prefix",
            "transcript": "\u524d\u534a\u53e5",
        },
        {
            "type": "conversation.item.input_audio_transcription.text",
            "text": "\u672a\u5b8c\u6210\u5c3e\u97f3",
        },
        {"type": "session.finished", "transcript": ""},
    ])
    asr = await _attached_asr(ws, finals)

    result = await asr.close(finish_timeout=0.5)

    assert result["completed"] is True
    assert result["timed_out"] is True
    assert result["transcript"] == "\u524d\u534a\u53e5"
    assert finals == ["\u524d\u534a\u53e5"]


@pytest.mark.asyncio
async def test_upstream_error_wakes_finish_and_surfaces_bounded_reason():
    finals = []
    ws = _FakeWebSocket([
        {
            "type": "error",
            "error": {"message": "  upstream\nASR unavailable  " + "x" * 400},
        },
    ])
    asr = await _attached_asr(ws, finals)

    result = await asr.close(finish_timeout=0.5)

    assert result["ok"] is False
    assert result["reason"] == "upstream_error"
    assert result["error"].startswith("upstream ASR unavailable")
    assert "\n" not in result["error"]
    assert len(result["error"]) == 300
    assert result["timed_out"] is False
    assert result["session_finished"] is False
    assert finals == []


@pytest.mark.asyncio
async def test_late_reconnect_success_is_closed_after_stop_pops_owner():
    entered = asyncio.Event()
    release = asyncio.Event()

    class _ReconnectASR:
        def __init__(self):
            self.closed = []

        async def connect(self):
            entered.set()
            await release.wait()
            return True

        async def close(self, *, graceful=True):
            self.closed.append(graceful)
            return {"ok": True}

    asr = _ReconnectASR()
    watcher = WatcherAgent(object())
    watcher._asr = {"turn": asr}
    watcher._asr_reconnecting = {"turn": asr}

    reconnect = asyncio.create_task(watcher._asr_reconnect("turn"))
    await entered.wait()
    with watcher._asr_state_lock:
        watcher._asr.pop("turn")  # asr_stop linearization point
        watcher._asr_reconnecting.pop("turn", None)
    release.set()
    await reconnect

    assert asr.closed == [False]
    assert "turn" not in watcher._asr
    assert "turn" not in watcher._asr_reconnecting


@pytest.mark.asyncio
async def test_asr_audio_admission_is_nonblocking_and_drains_on_owner_loop():
    scheduled = []
    delivered = []
    drained = asyncio.Event()

    class _LoopProxy:
        def is_closed(self):
            return False

        def call_soon_threadsafe(self, callback, *args):
            scheduled.append((callback, args))

    class _ASR:
        is_connected = True

        async def append_audio(self, pcm):
            delivered.append(pcm)
            drained.set()
            return True

    watcher = WatcherAgent(object())
    watcher._loop = _LoopProxy()
    asr = _ASR()
    with watcher._asr_state_lock:
        watcher._asr["turn"] = asr

    assert watcher.asr_audio("turn", b"pcm") is True
    # Admission only reserved bounded queue state.  No coroutine or socket send
    # ran synchronously on the caller/gateway thread.
    assert delivered == []
    assert len(scheduled) == 1

    callback, args = scheduled.pop()
    callback(*args)
    await asyncio.wait_for(drained.wait(), timeout=1.0)
    assert delivered == [b"pcm"]


def test_concurrent_dead_asr_audio_schedules_one_reconnect_per_owner():
    scheduled = []
    scheduled_lock = threading.Lock()

    class _LoopProxy:
        def is_closed(self):
            return False

        def call_soon_threadsafe(self, callback, *args):
            with scheduled_lock:
                scheduled.append((callback, args))

    class _DeadASR:
        is_connected = False

    watcher = WatcherAgent(object())
    watcher._loop = _LoopProxy()
    asr = _DeadASR()
    with watcher._asr_state_lock:
        watcher._asr["turn"] = asr

    barrier = threading.Barrier(9)
    results = []

    def _feed():
        barrier.wait()
        results.append(watcher.asr_audio("turn", b"pcm"))

    threads = [threading.Thread(target=_feed) for _ in range(8)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=1.0)

    assert results == [False] * 8
    assert len(scheduled) == 1
    assert watcher._asr_reconnecting == {"turn": asr}


@pytest.mark.asyncio
async def test_close_while_socket_connects_cannot_publish_ghost_ws(monkeypatch):
    entered = asyncio.Event()
    release = asyncio.Event()
    ws = _FakeWebSocket()

    async def deferred_connect(_url, **_kwargs):
        entered.set()
        await release.wait()
        return ws

    monkeypatch.setitem(
        sys.modules,
        "websockets",
        types.SimpleNamespace(connect=deferred_connect),
    )
    asr = QwenRealtimeASR("test-key")

    connecting = asyncio.create_task(asr.connect())
    await entered.wait()
    close_result = await asr.close(graceful=False)
    release.set()

    assert await connecting is False
    assert close_result["aborted"] is True
    assert ws.closed is True
    assert asr._ws is None
    assert asr._reader_task is None
    assert asr.is_connected is False


@pytest.mark.asyncio
async def test_cancelled_connect_closes_published_candidate_and_reader(
    monkeypatch,
):
    ws = _FakeWebSocket()

    async def immediate_connect(_url, **_kwargs):
        return ws

    monkeypatch.setitem(
        sys.modules,
        "websockets",
        types.SimpleNamespace(connect=immediate_connect),
    )
    asr = QwenRealtimeASR("test-key")
    connecting = asyncio.create_task(asr.connect())
    for _ in range(50):
        if asr._reader_task is not None:
            break
        await asyncio.sleep(0)
    assert asr._reader_task is not None

    connecting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await connecting

    assert ws.closed is True
    assert asr._ws is None
    assert asr._reader_task is None
    assert asr.is_connected is False


def test_watcher_start_timeout_invalidates_late_success(monkeypatch):
    entered = threading.Event()
    cancellation_seen = threading.Event()
    closed = threading.Event()
    release = asyncio.Event()

    class _SlowASR:
        def __init__(self, *_args, **_kwargs):
            pass

        async def connect(self):
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                # Simulate a transport whose lower layer completes despite the
                # caller-side timeout.  The start-token CAS must still reject
                # and close this late candidate.
                cancellation_seen.set()
                await release.wait()
            return True

        async def close(self, **_kwargs):
            closed.set()
            return {"ok": True, "aborted": True}

    monkeypatch.setattr(qwen_realtime, "QwenRealtimeASR", _SlowASR)
    monkeypatch.setattr(watcher_engine, "_ASR_START_TIMEOUT_SEC", 0.05)

    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()
    watcher = object.__new__(WatcherAgent)
    watcher._loop = loop
    watcher.cfg = types.SimpleNamespace(
        dashscope_api_key="test-key",
        realtime_asr_enabled=True,
    )
    watcher._asr = {}
    watcher._asr_start_tokens = {}
    watcher._asr_state_lock = threading.RLock()
    try:
        assert watcher.asr_start("turn", lambda _t: None, lambda _t: None) is False
        assert entered.wait(timeout=1)
        assert cancellation_seen.wait(timeout=1)
        loop.call_soon_threadsafe(release.set)
        assert closed.wait(timeout=1)
        assert watcher._asr == {}
        assert watcher._asr_start_tokens == {}
    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=1)
        loop.close()


@pytest.mark.asyncio
async def test_reconnect_preserves_logical_turn_transcript(monkeypatch):
    finals = []

    async def on_final(text):
        finals.append(text)

    ws = _FakeWebSocket([
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "event_id": "post-reconnect",
            "transcript": "\u540e",
        },
        {"type": "session.finished", "transcript": "\u524d\u540e"},
    ])
    await ws.incoming.put({"type": "session.updated"})

    async def reconnect(_url, **_kwargs):
        return ws

    monkeypatch.setitem(
        sys.modules,
        "websockets",
        types.SimpleNamespace(connect=reconnect),
    )
    asr = QwenRealtimeASR("test-key", on_final=on_final)
    await asr._deliver_final("\u524d")
    asr._canonical_transcript = "\u524d"

    assert await asr.connect() is True
    result = await asr.close(finish_timeout=0.5)

    assert finals == ["\u524d", "\u540e"]
    assert result["transcript"] == "\u524d\u540e"
