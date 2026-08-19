"""Desktop push-to-talk ASR is one explicit, idempotent user turn."""

from __future__ import annotations

import base64
import threading
import types

import pytest

from tui_gateway import server


_REAL_THREAD = threading.Thread


class _ImmediateThread:
    def __init__(self, target=None, **_kwargs):
        self.target = target

    def start(self):
        if self.target is not None:
            self.target()


class _ObservedEvent(threading.Event):
    def __init__(self, waiter_entered):
        super().__init__()
        self.waiter_entered = waiter_entered

    def wait(self, timeout=None):
        self.waiter_entered.set()
        return super().wait(timeout)


class _FrameBuffer:
    def __init__(self, monitor_latest_ts=12.0, latest_ts=11.0):
        self.monitor_latest_ts = monitor_latest_ts
        self.latest_ts = latest_ts


class _AsrEngine:
    def __init__(self):
        self.audio = []
        self.audio_result = True
        self.audio_error = None
        self.starts = 0
        self.stops = []
        self.trailing_final = ""
        self.close_result = {
            "ok": True,
            "completed": True,
            "session_finished": True,
            "timed_out": False,
        }
        self.on_stop = None

    def asr_start(self, key, on_partial, on_final, on_speech_started):
        self.starts += 1
        self.key = key
        self.on_partial = on_partial
        self.on_final = on_final
        self.on_speech_started = on_speech_started
        return True

    def asr_audio(self, key, pcm):
        if self.audio_error is not None:
            raise self.audio_error
        self.audio.append((key, pcm))
        return self.audio_result

    def asr_stop(self, key, *, graceful=True):
        self.stops.append((key, graceful))
        if self.on_stop is not None:
            self.on_stop()
        if graceful and self.trailing_final:
            self.on_final(self.trailing_final)
        return dict(self.close_result)


def _ready_event():
    event = threading.Event()
    event.set()
    return event


def _session(engine, *, running=False, frame_buffer=None):
    return {
        "agent": types.SimpleNamespace(
            frame_buffer=frame_buffer or _FrameBuffer()),
        "agent_ready": _ready_event(),
        "attached_images": [],
        "history": [],
        "history_lock": threading.RLock(),
        "history_version": 0,
        "queued_prompt": None,
        "queued_prompts": [],
        "running": running,
        "session_key": "stored-manual-asr",
        "source": "multimodal",
        "_mm_live_watcher_agent": engine,
    }


@pytest.fixture
def manual_runtime(monkeypatch):
    sid = "live-manual-asr"
    engine = _AsrEngine()
    session = _session(engine)
    emitted = []
    submitted = []

    monkeypatch.setitem(server._sessions, sid, session)
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(
        server,
        "_emit",
        lambda event, actual_sid, payload=None: emitted.append(
            (event, actual_sid, payload or {})),
    )

    def fake_submit(
        _rid,
        actual_sid,
        target_session,
        text,
        **kwargs,
    ):
        submitted.append((actual_sid, text, dict(kwargs)))
        with target_session["history_lock"]:
            target_session["running"] = False

    monkeypatch.setattr(server, "_run_prompt_submit", fake_submit)
    return sid, session, engine, emitted, submitted


def _start(sid, turn_id="desktop-asr-1", mode="manual_turn"):
    return server._methods["multimodal.asr_start"](
        "rpc-start",
        {"session_id": sid, "turn_id": turn_id, "mode": mode},
    )["result"]


def _stop(sid, turn_id="desktop-asr-1", disposition="finish", **extra):
    return server._methods["multimodal.asr_stop"](
        "rpc-stop",
        {
            "session_id": sid,
            "turn_id": turn_id,
            "disposition": disposition,
            **extra,
        },
    )["result"]


def test_manual_vad_segments_preview_until_one_explicit_finish(manual_runtime):
    sid, _session, engine, emitted, submitted = manual_runtime
    assert _start(sid) == {
        "enabled": True,
        "turn_id": "desktop-asr-1",
        "mode": "manual_turn",
    }

    engine.on_partial("\u5148\u770b")
    engine.on_final("\u5148\u770b\u4e00\u4e0b")
    engine.on_final("\u7136\u540e\u7ee7\u7eed")

    assert submitted == []
    assert not [
        payload for event, _sid, payload in emitted
        if event == "multimodal.asr_final" and payload.get("text")
    ]
    buffers = [
        payload for event, _sid, payload in emitted
        if event == "multimodal.asr_buffer"
    ]
    assert buffers[-1] == {
        "segments": ["\u5148\u770b\u4e00\u4e0b", "\u7136\u540e\u7ee7\u7eed"],
        "turn_id": "desktop-asr-1",
    }
    engine.trailing_final = "\u6700\u540e\u7684\u5c3e词"

    result = _stop(sid)

    assert result["transcript"] == "\u5148\u770b\u4e00\u4e0b\u7136\u540e\u7ee7\u7eed\u6700\u540e\u7684\u5c3e词"
    assert result["submitted"] is True
    assert result["graceful"] is True
    assert len(submitted) == 1
    assert submitted[0][1] == result["transcript"]
    assert submitted[0][2]["anchor_ts"] == 12.0
    assert submitted[0][2]["anchor_frozen"] is True
    finals = [
        payload for event, _sid, payload in emitted
        if event == "multimodal.asr_final" and payload.get("text")
    ]
    assert len(finals) == 1
    assert finals[0]["turn_id"] == "desktop-asr-1"
    assert finals[0]["request_id"] == result["client_request_id"]

    again = _stop(sid)
    assert again == result
    assert len(submitted) == 1
    assert len([
        payload for event, _sid, payload in emitted
        if event == "multimodal.asr_final" and payload.get("text")
    ]) == 1


def test_manual_preserves_identical_vad_segments(manual_runtime):
    sid, _session, engine, _emitted, submitted = manual_runtime
    _start(sid)

    engine.on_final("\u597d")
    engine.on_final("\u597d")
    result = _stop(sid)

    assert result["transcript"] == "\u597d\u597d"
    assert result["submitted"] is True
    assert submitted[0][1] == "\u597d\u597d"


def test_manual_prefers_provider_canonical_finished_transcript(manual_runtime):
    sid, _session, engine, _emitted, submitted = manual_runtime
    engine.close_result["transcript"] = "cats"
    _start(sid)
    # These are the compatibility callbacks emitted while Qwen reconciles an
    # in-word session.finished refinement.  The final commit must use canonical
    # text from close_result rather than joining them as "cat s".
    engine.on_final("cat")
    engine.trailing_final = "s"

    result = _stop(sid)

    assert result["transcript"] == "cats"
    assert submitted[0][1] == "cats"


def test_audio_requires_exact_modern_turn_owner(manual_runtime):
    sid, _session, engine, _emitted, _submitted = manual_runtime
    _start(sid)
    pcm_b64 = base64.b64encode(b"pcm").decode("ascii")

    missing = server._methods["multimodal.asr_audio"](
        "rpc-a", {"session_id": sid, "pcm_b64": pcm_b64})["result"]
    stale = server._methods["multimodal.asr_audio"](
        "rpc-b",
        {"session_id": sid, "turn_id": "old", "pcm_b64": pcm_b64},
    )["result"]
    accepted = server._methods["multimodal.asr_audio"](
        "rpc-c",
        {"session_id": sid, "turn_id": "desktop-asr-1", "pcm_b64": pcm_b64},
    )["result"]

    assert missing["reason"] == "turn_id_required"
    assert stale["reason"] == "stale_turn"
    assert accepted == {"ok": True, "turn_id": "desktop-asr-1"}
    assert engine.audio == [(f"asr:{sid}", b"pcm")]


@pytest.mark.parametrize("failure_kind", ["false", "exception"])
def test_manual_audio_delivery_failure_aborts_whole_turn(
    manual_runtime, failure_kind,
):
    sid, _session, engine, emitted, submitted = manual_runtime
    _start(sid)
    engine.on_final("\u4e0d\u5b8c\u6574\u524d\u7f00\u4e0d\u80fd\u63d0\u4ea4")
    if failure_kind == "false":
        engine.audio_result = False
    else:
        engine.audio_error = RuntimeError("socket send failed")

    audio = server._methods["multimodal.asr_audio"](
        "rpc-failed-audio",
        {
            "session_id": sid,
            "turn_id": "desktop-asr-1",
            "pcm_b64": base64.b64encode(b"lost chunk").decode("ascii"),
        },
    )["result"]
    engine.trailing_final = "\u4e0a\u6e38\u5c3e\u97f3\u4e5f\u4e0d\u5f97\u8865\u63d0\u4ea4"
    result = _stop(sid)

    assert audio == {
        "ok": False,
        "reason": "audio_delivery_failed",
        "turn_id": "desktop-asr-1",
    }
    assert result["ok"] is False
    assert result["reason"] == "audio_delivery_failed"
    assert result["submitted"] is False
    assert result["transcript"] == ""
    assert engine.stops[-1] == (f"asr:{sid}", False)
    assert submitted == []
    assert not [
        payload for event, _actual_sid, payload in emitted
        if event == "multimodal.asr_final" and payload.get("text")
    ]


def test_cancel_remains_dominant_after_audio_delivery_failure(manual_runtime):
    sid, _session, engine, _emitted, submitted = manual_runtime
    _start(sid)
    engine.audio_result = False
    server._methods["multimodal.asr_audio"](
        "rpc-failed-audio",
        {
            "session_id": sid,
            "turn_id": "desktop-asr-1",
            "pcm_b64": base64.b64encode(b"lost chunk").decode("ascii"),
        },
    )

    result = _stop(sid, disposition="cancel")

    assert result["ok"] is True
    assert result["reason"] == "cancelled"
    assert result["submitted"] is False
    assert submitted == []


def test_manual_empty_and_cancel_never_submit(manual_runtime):
    sid, _session, engine, emitted, submitted = manual_runtime
    _start(sid)
    empty = _stop(sid)
    assert empty["submitted"] is False
    assert empty["reason"] == "empty"
    assert submitted == []

    assert _start(sid, turn_id="desktop-asr-2")["enabled"] is True
    engine.on_final("\u8fd9\u53e5\u8981\u53d6\u6d88")
    cancelled = _stop(
        sid, turn_id="desktop-asr-2", disposition="cancel")
    assert cancelled["submitted"] is False
    assert cancelled["transcript"] == ""
    assert cancelled["reason"] == "cancelled"
    assert engine.stops[-1] == (f"asr:{sid}", False)
    assert submitted == []
    assert not [
        payload for event, _sid, payload in emitted
        if event == "multimodal.asr_final" and payload.get("text")
    ]


def test_manual_upstream_error_without_transcript_is_visible_failure(
    manual_runtime,
):
    sid, _session, engine, _emitted, submitted = manual_runtime
    engine.close_result = {
        "ok": False,
        "completed": False,
        "session_finished": False,
        "timed_out": False,
        "reason": "upstream_error",
        "error": "ASR provider unavailable",
    }
    _start(sid)

    result = _stop(sid)

    assert result["ok"] is False
    assert result["reason"] == "upstream_error"
    assert result["error"] == "ASR provider unavailable"
    assert result["submitted"] is False
    assert result["transcript"] == ""
    assert submitted == []


def test_manual_finish_queues_once_while_main_agent_busy(manual_runtime):
    sid, session, engine, emitted, submitted = manual_runtime
    session["running"] = True
    _start(sid)
    engine.on_final("\u6392\u961f\u7684\u8bed\u97f3\u95ee\u9898")

    result = _stop(sid)

    assert result["submitted"] is True
    assert result["queued"] is True
    assert submitted == []
    assert len(session["queued_prompts"]) == 1
    queued = session["queued_prompts"][0]
    assert queued["text"] == "\u6392\u961f\u7684\u8bed\u97f3\u95ee\u9898"
    assert queued["voice_input"] is True
    assert queued["anchor_ts"] == 12.0
    assert queued["anchor_frozen"] is True
    assert len([
        payload for event, _sid, payload in emitted
        if event == "multimodal.asr_final" and payload.get("text")
    ]) == 1

    assert _stop(sid) == result
    assert len(session["queued_prompts"]) == 1


def test_continuous_remains_vad_driven_but_cancel_is_abortive(manual_runtime):
    sid, _session, engine, emitted, submitted = manual_runtime
    assert _start(sid, mode="continuous")["mode"] == "continuous"
    engine.on_final("\u5bf9\u8bdd\u6a21式\u7acb即\u63d0交")
    assert len(submitted) == 1

    # A trailing completion queued upstream at cancellation must not dispatch.
    engine.trailing_final = "\u53d6\u6d88\u540e的\u5e7d灵轮次"
    before = len(submitted)
    result = _stop(sid, disposition="cancel")
    assert result["reason"] == "cancelled"
    assert engine.stops[-1][1] is False
    assert len(submitted) == before
    assert not [
        p for e, _s, p in emitted
        if e == "multimodal.asr_final"
        and p.get("text") == "\u53d6\u6d88\u540e的\u5e7d灵轮次"
    ]


def test_explicit_manual_does_not_create_dialog_state(manual_runtime):
    sid, session, engine, _emitted, submitted = manual_runtime
    # Manual capture buffers VAD segments until the explicit stop.

    started = _start(sid, mode="manual_turn")
    engine.on_final("\u505c\u987f\u4e0d\u80fd\u81ea\u52a8\u53d1\u9001")

    assert started["mode"] == "manual_turn"
    assert "_mm_voice_dialog_on" not in session
    assert submitted == []
    assert _stop(sid)["submitted"] is True

    # Continuous remains a transport-level VAD mode without creating product
    # conversation-mode state.
    started_dialog = _start(
        sid, turn_id="desktop-asr-dialog", mode="continuous")
    assert started_dialog["mode"] == "continuous"
    assert "_mm_voice_dialog_on" not in session


def test_explicit_continuous_does_not_create_dialog_state(
    manual_runtime,
):
    sid, session, engine, _emitted, _submitted = manual_runtime

    started = _start(
        sid, turn_id="desktop-asr-dialog", mode="continuous")

    assert started == {
        "enabled": True,
        "turn_id": "desktop-asr-dialog",
        "mode": "continuous",
    }
    assert "_mm_voice_dialog_on" not in session

    # A repeated transport request is idempotent and opens no second session.
    repeated = _start(
        sid, turn_id="desktop-asr-dialog", mode="continuous")
    assert repeated["enabled"] is True
    assert repeated["idempotent"] is True
    assert "_mm_voice_dialog_on" not in session
    assert engine.starts == 1


def test_stale_transport_cannot_toggle_tts(manual_runtime):
    sid, session, _engine, _emitted, _submitted = manual_runtime
    transport_a = object()
    transport_b = object()
    session["transport"] = transport_b
    session["_mm_tts_on"] = True

    token_a = server.bind_transport(transport_a)
    try:
        stale_tts = server._methods["multimodal.tts_toggle"](
            "rpc-stale-tts", {"session_id": sid, "enabled": False},
        )["result"]
    finally:
        server.reset_transport(token_a)

    assert stale_tts == {
        "ok": False, "enabled": True, "reason": "stale_transport"}
    assert session["_mm_tts_on"] is True



def test_stop_before_start_tombstones_late_activation(manual_runtime):
    sid, _session, engine, _emitted, _submitted = manual_runtime

    stopped = _stop(sid, turn_id="late-turn", disposition="cancel")
    started = _start(sid, turn_id="late-turn")

    assert stopped["reason"] == "no_active_turn"
    assert stopped["submitted"] is False
    assert started == {
        "enabled": False,
        "reason": "retired_turn",
        "turn_id": "late-turn",
        "mode": "manual_turn",
    }
    assert engine.starts == 0


def test_stop_during_start_build_prevents_late_commit(
    manual_runtime, monkeypatch,
):
    sid, _session, engine, _emitted, _submitted = manual_runtime
    start_released_capture_lock = threading.Event()
    allow_start_to_commit = threading.Event()
    started_result = {}

    def pause_after_promotion(_tokens):
        start_released_capture_lock.set()
        assert allow_start_to_commit.wait(timeout=2)

    monkeypatch.setattr(server, "_clear_session_context", pause_after_promotion)

    def run_start():
        started_result.update(_start(sid, turn_id="racing-turn"))

    thread = _REAL_THREAD(target=run_start)
    thread.start()
    assert start_released_capture_lock.wait(timeout=2)

    stopped = _stop(sid, turn_id="racing-turn", disposition="cancel")
    allow_start_to_commit.set()
    thread.join(timeout=2)

    assert stopped["reason"] == "cancelled"
    assert stopped["submitted"] is False
    assert started_result["enabled"] is False
    assert started_result["reason"] == "retired_turn"
    assert engine.starts == 0


def test_cancel_is_responsive_while_runtime_promotion_is_blocked(
    manual_runtime, monkeypatch,
):
    sid, session, engine, _emitted, submitted = manual_runtime
    promotion_entered = threading.Event()
    release_promotion = threading.Event()
    start_result = {}
    session["_mm_live_watcher_agent"] = None

    def blocked_promotion(_sid, target_session):
        promotion_entered.set()
        assert release_promotion.wait(timeout=2)
        target_session["_mm_live_watcher_agent"] = engine
        return True

    monkeypatch.setattr(
        server, "_promote_session_to_multimodal", blocked_promotion)
    start_thread = _REAL_THREAD(
        target=lambda: start_result.update(
            _start(sid, turn_id="promotion-race")))
    start_thread.start()
    assert promotion_entered.wait(timeout=2)

    stopped = _stop(
        sid, turn_id="promotion-race", disposition="cancel")
    assert stopped["disposition"] == "cancel"
    assert stopped["submitted"] is False
    assert stopped["reason"] == "cancelled"
    assert submitted == []

    release_promotion.set()
    start_thread.join(timeout=2)
    assert start_result == {
        "enabled": False,
        "reason": "retired_turn",
        "turn_id": "promotion-race",
        "mode": "manual_turn",
    }
    assert engine.starts == 0
    assert "multimodal.asr_start" in server._LONG_HANDLERS
    assert "multimodal.asr_stop" in server._LONG_HANDLERS


def test_concurrent_duplicate_finish_linearizes_to_one_submit(manual_runtime):
    sid, _session, engine, _emitted, submitted = manual_runtime
    entered_close = threading.Event()
    allow_close = threading.Event()
    results = []
    _start(sid)
    engine.on_final("\u53ea能提交一次")

    def block_close():
        entered_close.set()
        assert allow_close.wait(timeout=2)

    engine.on_stop = block_close

    first = _REAL_THREAD(target=lambda: results.append(_stop(sid)))
    second = _REAL_THREAD(target=lambda: results.append(_stop(sid)))
    first.start()
    assert entered_close.wait(timeout=2)
    second.start()
    allow_close.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert len(results) == 2
    assert results[0] == results[1]
    assert len(engine.stops) == 1
    assert len(submitted) == 1


def test_cancel_overtakes_inflight_finish_before_commit(manual_runtime):
    sid, _session, engine, _emitted, submitted = manual_runtime
    entered_close = threading.Event()
    allow_close = threading.Event()
    cancel_waiting = threading.Event()
    results = []
    _start(sid)
    _session["_mm_asr_turn"]["stop_event"] = _ObservedEvent(cancel_waiting)
    engine.on_final("\u4e0d\u80fd\u5728\u8fb9\u754c\u540e\u63d0\u4ea4")
    engine.trailing_final = "\u8fdf\u5230\u7684\u5c3e\u97f3"

    def block_close():
        entered_close.set()
        assert allow_close.wait(timeout=2)

    engine.on_stop = block_close
    finish = _REAL_THREAD(target=lambda: results.append(_stop(sid)))
    cancel = _REAL_THREAD(
        target=lambda: results.append(_stop(sid, disposition="cancel")))
    finish.start()
    assert entered_close.wait(timeout=2)
    cancel.start()
    assert cancel_waiting.wait(timeout=2)
    assert _session["_mm_asr_turn"]["disposition"] == "cancel"
    allow_close.set()
    finish.join(timeout=2)
    cancel.join(timeout=2)

    assert len(results) == 2
    assert results[0] == results[1]
    assert results[0]["disposition"] == "cancel"
    assert results[0]["submitted"] is False
    assert results[0]["transcript"] == ""
    assert len(engine.stops) == 1
    assert submitted == []


def test_cancel_after_close_but_before_scheduler_claim_wins(
    manual_runtime, monkeypatch,
):
    sid, session, engine, emitted, submitted = manual_runtime
    preclaim_reached = threading.Event()
    allow_claim = threading.Event()
    cancel_waiting = threading.Event()
    results = []
    barrier_enabled = False

    def barrier_emit(event, actual_sid, payload=None):
        emitted.append((event, actual_sid, payload or {}))
        if (barrier_enabled and event == "multimodal.asr_partial"
                and not (payload or {}).get("text")):
            assert not session["_mm_asr_turn"].get("stop_committed", False)
            preclaim_reached.set()
            assert allow_claim.wait(timeout=2)

    monkeypatch.setattr(server, "_emit", barrier_emit)
    _start(sid)
    session["_mm_asr_turn"]["stop_event"] = _ObservedEvent(cancel_waiting)
    engine.on_final("\u8c03\u5ea6\u524d\u53d6\u6d88")
    barrier_enabled = True

    finish = _REAL_THREAD(target=lambda: results.append(_stop(sid)))
    cancel = _REAL_THREAD(
        target=lambda: results.append(_stop(sid, disposition="cancel")))
    finish.start()
    assert preclaim_reached.wait(timeout=2)
    cancel.start()
    assert cancel_waiting.wait(timeout=2)
    assert session["_mm_asr_turn"]["disposition"] == "cancel"
    allow_claim.set()
    finish.join(timeout=2)
    cancel.join(timeout=2)

    assert len(results) == 2
    assert results[0] == results[1]
    assert results[0]["disposition"] == "cancel"
    assert results[0]["submitted"] is False
    assert results[0]["transcript"] == ""
    assert submitted == []


def test_cancel_after_stop_commit_observes_same_finished_result(
    manual_runtime, monkeypatch,
):
    sid, session, engine, emitted, submitted = manual_runtime
    commit_reached = threading.Event()
    allow_dispatch = threading.Event()
    cancel_waiting = threading.Event()
    results = []
    barrier_enabled = False

    def barrier_emit(event, actual_sid, payload=None):
        emitted.append((event, actual_sid, payload or {}))
        if (barrier_enabled and event == "multimodal.asr_final"
                and (payload or {}).get("text")):
            assert session["_mm_asr_turn"]["stop_committed"] is True
            commit_reached.set()
            assert allow_dispatch.wait(timeout=2)

    monkeypatch.setattr(server, "_emit", barrier_emit)
    _start(sid)
    session["_mm_asr_turn"]["stop_event"] = _ObservedEvent(cancel_waiting)
    engine.on_final("\u63d0\u4ea4\u70b9\u5df2\u7ecf\u7ebf\u6027\u5316")
    barrier_enabled = True

    finish = _REAL_THREAD(target=lambda: results.append(_stop(sid)))
    cancel = _REAL_THREAD(
        target=lambda: results.append(_stop(sid, disposition="cancel")))
    finish.start()
    assert commit_reached.wait(timeout=2)
    assert server._abort_active_asr_turn(
        session,
        reason="transport_disconnected",
        reopenable=True,
    ) is None
    assert "desktop-asr-1" not in (
        session.get("_mm_asr_reopenable_turns") or {})
    cancel.start()
    assert cancel_waiting.wait(timeout=2)
    # The cancel follower cannot rewrite the leader's disposition after the
    # commit point; it waits and receives that exact terminal finish result.
    assert session["_mm_asr_turn"]["disposition"] == "finish"
    allow_dispatch.set()
    finish.join(timeout=2)
    cancel.join(timeout=2)

    assert len(results) == 2
    assert results[0] == results[1]
    assert results[0]["disposition"] == "finish"
    assert results[0]["submitted"] is True
    assert len(submitted) == 1


def test_detach_aborts_and_new_transport_reopens_same_continuous_turn(
    manual_runtime, monkeypatch,
):
    sid, session, engine, _emitted, _submitted = manual_runtime
    transport_a = object()
    transport_b = object()
    monkeypatch.setattr(server, "_schedule_ws_orphan_reap", lambda _sid: None)
    session["transport"] = transport_a
    token = server.bind_transport(transport_a)
    try:
        started = _start(sid, mode="continuous")
    finally:
        server.reset_transport(token)
    assert started["enabled"] is True

    assert server._close_sessions_for_transport(transport_a) == (0, 1)
    assert engine.stops[-1] == (f"asr:{sid}", False)
    assert session["_mm_asr_turn"]["state"] == "stopped"
    assert session["_mm_asr_turn"]["result"]["submitted"] is False

    # A reconnect/resume explicitly transfers the parked session to B; the
    # disconnect tombstone is the one retired-id class that B may reopen.
    session["transport"] = transport_b
    token = server.bind_transport(transport_b)
    try:
        resumed = _start(sid, mode="continuous")
    finally:
        server.reset_transport(token)
    assert resumed == {
        "enabled": True,
        "turn_id": "desktop-asr-1",
        "mode": "continuous",
    }
    assert session["_mm_asr_turn"]["owner_transport"] is transport_b
    assert engine.starts == 2

    token = server.bind_transport(transport_a)
    try:
        stale_audio = server._methods["multimodal.asr_audio"](
            "rpc-old-audio",
            {
                "session_id": sid,
                "turn_id": "desktop-asr-1",
                "pcm_b64": base64.b64encode(b"late").decode("ascii"),
            },
        )["result"]
        stale = _stop(sid, disposition="cancel")
    finally:
        server.reset_transport(token)
    assert stale_audio["ok"] is False
    assert stale_audio["reason"] == "stale_transport"
    assert stale["ok"] is False
    assert stale["reason"] == "stale_transport"


def test_detach_during_graceful_finish_forces_cancel_without_submit(
    manual_runtime, monkeypatch,
):
    sid, session, engine, emitted, submitted = manual_runtime
    transport_a = object()
    entered_finish = threading.Event()
    release_finish = threading.Event()
    stop_results = []
    monkeypatch.setattr(server, "_schedule_ws_orphan_reap", lambda _sid: None)
    session["transport"] = transport_a

    token = server.bind_transport(transport_a)
    try:
        assert _start(sid)["enabled"] is True
    finally:
        server.reset_transport(token)
    engine.on_final("断线前已识别")
    engine.trailing_final = "断线后不得逃逸"

    def block_graceful_finish():
        entered_finish.set()
        assert release_finish.wait(timeout=2)

    def finish_from_a():
        token_a = server.bind_transport(transport_a)
        try:
            stop_results.append(_stop(sid))
        finally:
            server.reset_transport(token_a)

    engine.on_stop = block_graceful_finish
    finish_thread = _REAL_THREAD(target=finish_from_a)
    finish_thread.start()
    assert entered_finish.wait(timeout=2)

    # The transport boundary flips the already-linearized graceful stop to an
    # effective cancellation.  Its trailing upstream final may still arrive,
    # but neither that callback nor the buffered prefix can become a user turn.
    assert server._close_sessions_for_transport(transport_a) == (0, 1)
    release_finish.set()
    finish_thread.join(timeout=2)

    assert len(stop_results) == 1
    assert stop_results[0]["disposition"] == "cancel"
    assert stop_results[0]["reason"] == "transport_disconnected"
    assert stop_results[0]["transcript"] == ""
    assert stop_results[0]["submitted"] is False
    assert submitted == []
    assert not [
        payload for event, _actual_sid, payload in emitted
        if event == "multimodal.asr_final" and payload.get("text")
    ]


def test_detach_parks_before_blocked_abort_and_never_overwrites_resume(
    manual_runtime, monkeypatch,
):
    sid, session, engine, _emitted, _submitted = manual_runtime
    transport_a = object()
    transport_b = object()
    entered_abort = threading.Event()
    release_abort = threading.Event()
    detach_results = []
    resumed_start = {}
    resumed_start_entered = threading.Event()
    resumed_start_done = threading.Event()
    monkeypatch.setattr(server, "_schedule_ws_orphan_reap", lambda _sid: None)
    session["transport"] = transport_a

    token = server.bind_transport(transport_a)
    try:
        assert _start(sid, mode="continuous")["enabled"] is True
    finally:
        server.reset_transport(token)

    def block_abort():
        entered_abort.set()
        assert release_abort.wait(timeout=2)

    engine.on_stop = block_abort
    detach_thread = _REAL_THREAD(
        target=lambda: detach_results.append(
            server._close_sessions_for_transport(transport_a)))
    detach_thread.start()
    assert entered_abort.wait(timeout=2)
    assert session["transport"] is server._detached_ws_transport

    # Model the session.resume fast path while abortive close is blocked.  It
    # can bind B immediately; B's asr_start then waits behind the capture lock
    # until A's exact turn has been retired and safely reopened.
    with server._session_resume_lock:
        server._live_session_payload(
            sid, session, touch=True, transport=transport_b)
    assert session["transport"] is transport_b

    def start_from_b():
        token_b = server.bind_transport(transport_b)
        try:
            resumed_start_entered.set()
            resumed_start.update(_start(sid, mode="continuous"))
        finally:
            server.reset_transport(token_b)
            resumed_start_done.set()

    start_thread = _REAL_THREAD(target=start_from_b)
    start_thread.start()
    assert resumed_start_entered.wait(timeout=2)
    assert not resumed_start_done.wait(timeout=0.05)

    release_abort.set()
    detach_thread.join(timeout=2)
    start_thread.join(timeout=2)

    assert detach_results == [(0, 1)]
    assert resumed_start == {
        "enabled": True,
        "turn_id": "desktop-asr-1",
        "mode": "continuous",
    }
    assert session["transport"] is transport_b
    assert session["_mm_asr_turn"]["owner_transport"] is transport_b
    assert engine.starts == 2


def test_matching_capture_anchor_is_used_and_mismatch_falls_back(manual_runtime):
    sid, session, engine, _emitted, submitted = manual_runtime
    session["_mm_capture_attempt_id"] = "capture-current"
    session["_mm_capture_active"] = True
    server._record_mm_capture_anchor_pair(
        session,
        capture_attempt_id="capture-current",
        client_ts=5.0,
        server_ts=10.25,
    )
    server._record_mm_capture_anchor_pair(
        session,
        capture_attempt_id="capture-current",
        client_ts=6.0,
        server_ts=11.25,
    )
    _start(sid)
    engine.on_final("\u770b\u770b我停止时的画面")
    matched = _stop(
        sid,
        capture_attempt_id="capture-current",
        anchor_ts=5.5,
    )
    # Raw 5.5 client-relative seconds never enters FrameBuffer's independent
    # server epoch; it maps to the latest accepted frame at/before the click.
    assert matched["anchor_ts"] == 10.25
    assert submitted[-1][2]["anchor_ts"] == 10.25

    _start(sid, turn_id="desktop-asr-2")
    engine.on_final("\u4e0d信任旧 capture")
    mismatched = _stop(
        sid,
        turn_id="desktop-asr-2",
        capture_attempt_id="capture-old",
        anchor_ts=1.0,
    )
    assert mismatched["anchor_ts"] == 12.0
    assert submitted[-1][2]["anchor_ts"] == 12.0

    _start(sid, turn_id="desktop-asr-before-first-frame")
    engine.on_final("\u9996\u5e27\u524d\u5c31\u505c\u6b62")
    before_first = _stop(
        sid,
        turn_id="desktop-asr-before-first-frame",
        capture_attempt_id="capture-current",
        anchor_ts=0.5,
    )
    assert before_first["anchor_ts"] is None
    assert submitted[-1][2]["anchor_ts"] is None
    assert submitted[-1][2]["anchor_frozen"] is True


def test_frozen_none_anchor_and_unconfirmed_partial_fail_closed(manual_runtime):
    sid, session, engine, emitted, submitted = manual_runtime
    frame_buffer = session["agent"].frame_buffer
    frame_buffer.monitor_latest_ts = None
    frame_buffer.latest_ts = None
    engine.close_result = {
        "ok": False,
        # A prior VAD segment completed, but the current live partial did not
        # flush before timeout.  Neither text is a safe complete command.
        "completed": True,
        "session_finished": False,
        "timed_out": True,
    }

    def frame_arrives_during_finish():
        frame_buffer.monitor_latest_ts = 99.0
        frame_buffer.latest_ts = 98.0

    engine.on_stop = frame_arrives_during_finish
    _start(sid)
    engine.on_final("\u5df2\u5b8c\u6210\u7684\u524d\u534a\u53e5")
    engine.on_partial("\u672a\u786e\u8ba4\u7684\u5c3e\u97f3")

    result = _stop(sid)

    assert result["ok"] is False
    assert result["submitted"] is False
    assert result["transcript"] == ""
    assert result["reason"] == "finish_timeout"
    assert result["graceful"] is False
    assert result["anchor_ts"] is None
    assert submitted == []
    assert session["queued_prompts"] == []
    assert not [
        payload for event, _actual_sid, payload in emitted
        if event == "multimodal.asr_final" and payload.get("text")
    ]
    assert _stop(sid) == result


def test_completed_segments_survive_terminal_ack_timeout(manual_runtime):
    sid, _session, engine, _emitted, submitted = manual_runtime
    engine.close_result = {
        "ok": False,
        "completed": True,
        "session_finished": False,
        "timed_out": True,
    }
    _start(sid)
    engine.on_final("\u8fd9\u53e5\u5df2\u7ecf\u5b8c\u6574\u786e\u8ba4")

    result = _stop(sid)

    assert result["ok"] is True
    assert result["submitted"] is True
    assert result["transcript"] == "\u8fd9\u53e5\u5df2\u7ecf\u5b8c\u6574\u786e\u8ba4"
    assert result["reason"] == "finish_timeout"
    assert [row[1] for row in submitted] == [
        "\u8fd9\u53e5\u5df2\u7ecf\u5b8c\u6574\u786e\u8ba4",
    ]
