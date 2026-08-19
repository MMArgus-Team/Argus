"""Monitor delivery must stay outside the main-agent canonical conversation."""

from __future__ import annotations

import contextlib
import json
import threading
import types
from unittest import mock

import pytest

from tui_gateway import server


class _ImmediateThread:
    def __init__(self, target=None, args=(), kwargs=None, **_unused):
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}

    def start(self):
        if self.target is not None:
            self.target(*self.args, **self.kwargs)

    def is_alive(self):
        return False


class _FakeMonitorEngine:
    latest = None

    def __init__(self, _frame_buffer, **kwargs):
        type(self).latest = self
        self.speak_cb = kwargs["speak_cb"]
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def is_healthy(self):
        return self.started and not self.stopped


def _session(*, running: bool = False):
    history = [
        {"role": "user", "content": "ordinary question"},
        {"role": "assistant", "content": "ordinary answer"},
    ]
    return {
        "agent": types.SimpleNamespace(mm_monitors={}),
        "attached_images": [],
        "cols": 80,
        "history": list(history),
        "history_lock": threading.Lock(),
        "history_version": 7,
        "queued_prompt": None,
        "queued_prompts": [],
        "running": running,
        "session_key": "durable-monitor-session",
        "source": "multimodal",
        "transport": None,
    }, history


def _build_engine(monkeypatch, session):
    from agent.multimodal import monitor_engine

    _FakeMonitorEngine.latest = None
    monkeypatch.setattr(monitor_engine, "MonitorEngine", _FakeMonitorEngine)
    engine = server._maybe_start_monitor_engine(
        "live-monitor", session, frame_buffer=object())
    assert engine is _FakeMonitorEngine.latest
    assert engine is not None
    return engine


def _build_query_callbacks(monkeypatch, session):
    watcher = mock.Mock()
    watcher.start.return_value = True
    watcher_cls = mock.Mock(return_value=watcher)
    monkeypatch.setattr(server, "_MM_ACTIVE_WATCHERS", {})
    monkeypatch.setattr(
        "agent.multimodal.hermes_glue.flatten_mm_config",
        lambda *_args, **_kwargs: {"enabled": True},
    )
    monkeypatch.setattr(
        "agent.multimodal.watcher_engine.WatcherAgent", watcher_cls)
    assert server._maybe_start_live_watcher_agent(
        "live-monitor", object(), object(), session) is watcher
    return watcher_cls.call_args.kwargs


def _complete_query_tool(
    *,
    tool_call_id: str,
    task_id: str,
    parent_id: str,
    handoff_mode: str = "deferred_reply",
) -> None:
    server._on_tool_complete(
        "live-monitor",
        tool_call_id,
        "query_multimodal",
        {"query": f"question {task_id}"},
        json.dumps({
            "control": "handoff",
            "reply_owner": "query_worker",
            "handoff_mode": handoff_mode,
            "status": "running" if handoff_mode == "deferred_reply" else "busy",
            "task_id": task_id,
            "parent_user_message_id": parent_id,
            "original_user_text": f"question {task_id}",
        }),
    )


def test_plain_monitor_alert_is_only_emitted_and_persisted_to_sidechannel(
    monkeypatch,
):
    session, original_history = _session()
    emitted = []
    inserted = []
    evidence = {
        "input_count": 3,
        "shown_count": 1,
        "frames": [{"ts": 1.5, "source_type": "screen", "thumb_b64": "dGh1bWI="}],
    }

    class _SideDb:
        def insert_mm_monitor_alert(
            self, session_id, monitor_id, text, label=None, evidence=None,
        ):
            inserted.append((session_id, monitor_id, text, label, evidence))

    @contextlib.contextmanager
    def _side_db(_session):
        yield _SideDb()

    monkeypatch.setattr(server, "_emit", lambda event, sid, payload=None: emitted.append(
        (event, sid, payload or {})))
    monkeypatch.setattr(server, "_session_db", _side_db)
    engine = _build_engine(monkeypatch, session)

    delivered = engine.speak_cb(
        "mon-plain",
        {
            "brief": "watch for a phone",
            "hook_main_agent": False,
            "_delivery_evidence": evidence,
        },
        "A phone appeared.",
    )

    assert delivered is True
    assert [event for event, _sid, _payload in emitted] == [
        "message.start", "message.delta", "message.complete",
    ]
    assert all(
        payload["source"] == "monitor" and payload["monitor_id"] == "mon-plain"
        for _event, _sid, payload in emitted
    )
    assert len(inserted) == 1
    assert inserted[0][:3] == (
        "durable-monitor-session", "mon-plain", "A phone appeared.",
    )
    assert inserted[0][3].startswith("watch for")
    assert inserted[0][4] == evidence
    assert emitted[-1][2]["evidence"] == evidence
    assert session["history"] == original_history
    assert session["history_version"] == 7


def test_immediate_monitor_hook_is_ephemeral_and_has_no_synthetic_user_echo(
    monkeypatch,
):
    session, original_history = _session()
    session["_mm_tts_on"] = True
    pending_images = [{"path": "/tmp/user-staged.png", "mime_type": "image/png"}]
    session["attached_images"] = list(pending_images)
    emitted = []
    captured = {}

    class _Agent:
        compression_enabled = True
        mm_monitors = {}

        def clear_interrupt(self):
            pass

        def run_conversation(
            self, prompt, conversation_history=None, stream_callback=None,
        ):
            captured["prompt"] = prompt
            captured["ephemeral"] = self._ephemeral_internal_turn
            captured["compression_enabled_during_run"] = self.compression_enabled
            captured["history_is_detached"] = (
                conversation_history[0] is not session["history"][0]
            )
            conversation_history[0]["content"] = "mutated private working copy"
            self._last_flushed_db_idx = 999
            self._session_messages = [
                {"role": "user", "content": "private monitor hook instruction"}
            ]
            if stream_callback:
                stream_callback("hook result")
            return {
                "final_response": "hook result",
                "messages": list(conversation_history or []) + [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "hook result"},
                ],
            }

    session["agent"] = _Agent()
    session["agent"]._session_messages = [
        {"role": "user", "content": "ordinary question"},
        {"role": "assistant", "content": "ordinary answer"},
    ]
    session["agent"]._last_flushed_db_idx = 2
    original_session_messages = session["agent"]._session_messages
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "_emit", lambda event, sid, payload=None: emitted.append(
        (event, sid, payload or {})))
    monkeypatch.setattr(server, "render_message", lambda *_args: "")
    monkeypatch.setattr(server, "_get_usage", lambda _agent: {})
    monkeypatch.setattr(
        server,
        "_get_voice_agent",
        lambda _session: (_ for _ in ()).throw(
            AssertionError("an internal monitor hook must not enter VoiceAgent")
        ),
    )
    monkeypatch.setattr(
        server,
        "_voice_tts_enabled",
        lambda: (_ for _ in ()).throw(
            AssertionError("an internal monitor hook must not trigger global TTS")
        ),
    )
    monkeypatch.setattr(
        "agent.title_generator.maybe_auto_title",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("an internal monitor hook must not auto-title the chat")
        ),
    )
    monkeypatch.setattr(
        server, "_reserve_mm_query_turn_projection", lambda *_args, **_kwargs: [])
    engine = _build_engine(monkeypatch, session)

    delivered = engine.speak_cb(
        "mon-hook",
        {
            "brief": "watch for a phone",
            "hook_main_agent": True,
            "hook_instruction": "Tell the user what happened.",
        },
        "A phone appeared.",
    )

    assert delivered is True
    assert captured["ephemeral"] is True
    assert captured["compression_enabled_during_run"] is False
    assert captured["history_is_detached"] is True
    assert session["agent"].compression_enabled is True
    assert session["agent"]._session_messages is original_session_messages
    assert session["agent"]._last_flushed_db_idx == 2
    assert captured["prompt"] == (
        "Tell the user what happened.\n"
        "monitor / watcher result for reference: A phone appeared."
    )
    assert not any(event == "message.user_echo" for event, _sid, _p in emitted)
    assert len([event for event, _sid, _p in emitted if event == "message.start"]) == 1
    assert any(event == "message.complete" for event, _sid, _p in emitted)
    assert session["history"] == original_history
    assert session["history_version"] == 7
    assert session["attached_images"] == pending_images
    assert session["_monitor_hook_running"] is False


def test_busy_monitor_hook_keeps_origin_and_ephemeral_guard_through_fifo(
    monkeypatch,
):
    session, original_history = _session(running=True)
    dispatched = []
    engine = _build_engine(monkeypatch, session)

    delivered = engine.speak_cb(
        "mon-queued",
        {
            "brief": "watch for a phone",
            "hook_main_agent": True,
            "hook_instruction": "Tell the user what happened.",
        },
        "A phone appeared.",
    )

    assert delivered is True
    assert len(session["queued_prompts"]) == 1
    assert session["queued_prompts"][0]["origin"] == "monitor_hook"
    assert session["queued_prompts"][0]["user_originated"] is False

    def _fake_submit(
        _rid,
        _sid,
        active_session,
        text,
        *,
        user_originated=False,
        client_request_id="",
        internal_origin="",
    ):
        dispatched.append({
            "text": text,
            "user_originated": user_originated,
            "client_request_id": client_request_id,
            "internal_origin": internal_origin,
            "guard": active_session.get("_monitor_hook_running"),
        })
        with active_session["history_lock"]:
            active_session["running"] = False
            active_session["_monitor_hook_running"] = False

    monkeypatch.setattr(server, "_run_prompt_submit", _fake_submit)
    session["running"] = False

    assert server._drain_queued_prompt("rpc", "live-monitor", session) is True
    assert dispatched == [{
        "text": (
            "Tell the user what happened.\n"
            "monitor / watcher result for reference: A phone appeared."
        ),
        "user_originated": False,
        "client_request_id": "",
        "internal_origin": "monitor_hook",
        "guard": True,
    }]
    assert session["history"] == original_history
    assert session["history_version"] == 7


def test_internal_query_worker_answer_stays_visible_but_never_projects(
    monkeypatch,
):
    """A fast QueryWorker completion may beat tool.complete; both stay private."""
    session, original_history = _session()
    session["agent"] = types.SimpleNamespace(
        _active_parent_user_message_id="internal-turn",
        _ephemeral_internal_turn=True,
        mm_monitors={},
        mm_watchers={},
        session_id="durable-monitor-session",
    )
    emitted = []
    notices = []
    watcher = mock.Mock()
    watcher.start.return_value = True

    monkeypatch.setattr(server, "_MM_ACTIVE_WATCHERS", {})
    monkeypatch.setitem(server._sessions, "live-monitor", session)
    monkeypatch.setattr(
        server,
        "_emit",
        lambda event, sid, payload=None: emitted.append(
            (event, sid, payload or {})),
    )
    monkeypatch.setattr(
        server,
        "_append_mm_context",
        lambda *args, **kwargs: notices.append((args, kwargs)),
    )
    server._on_tool_start(
        "live-monitor",
        "tool-internal",
        "query_multimodal",
        {"query": "private visual follow-up"},
    )
    marker = session["_mm_internal_request_origins"]["internal-turn"]
    assert marker["origin"] == "internal_hook"
    assert marker["calls"]["tool-internal"]["tool_complete"] is False
    assert marker["calls"]["tool-internal"]["query_complete"] is False
    with (
        mock.patch(
            "agent.multimodal.hermes_glue.flatten_mm_config",
            return_value={"enabled": True},
        ),
        mock.patch(
            "agent.multimodal.watcher_engine.WatcherAgent",
            return_value=watcher,
        ) as watcher_cls,
    ):
        assert server._maybe_start_live_watcher_agent(
            "live-monitor", object(), object(), session) is watcher
        callbacks = watcher_cls.call_args.kwargs

        # Deliberately complete before tool.complete to cover the real race.
        callbacks["on_query_complete"](
            "qry-internal",
            "internal-turn",
            "private visual follow-up",
            "worker-visible answer",
            "complete",
        )
        server._on_tool_complete(
            "live-monitor",
            "tool-internal",
            "query_multimodal",
            {"query": "private visual follow-up"},
            json.dumps({
                "control": "handoff",
                "reply_owner": "query_worker",
                "handoff_mode": "deferred_reply",
                "status": "running",
                "task_id": "qry-internal",
                "parent_user_message_id": "internal-turn",
                "original_user_text": "private visual follow-up",
            }),
        )

    assert any(
        event == "message.complete"
        and payload.get("request_id") == "internal-turn"
        and payload.get("text") == "worker-visible answer"
        for event, _sid, payload in emitted
    )
    assert session.get("_mm_query_results", []) == []
    assert session["_mm_worker_tasks"]["qry-internal"]["status"] == "complete"
    assert session["_mm_worker_tasks"]["qry-internal"]["result"] == (
        "worker-visible answer"
    )
    assert session.get("_mm_internal_request_origins", {}) == {}
    assert notices == []
    assert session["history"] == original_history
    assert server._reserve_mm_query_turn_projection(
        session, "next-real-turn") == []


@pytest.mark.parametrize(
    "phase_order",
    [
        (
            ("worker", "a"),
            ("tool", "a"),
            ("worker", "b"),
            ("tool", "b"),
        ),
        (
            ("tool", "a"),
            ("worker", "a"),
            ("tool", "b"),
            ("worker", "b"),
        ),
    ],
    ids=("worker-before-tool", "tool-before-worker"),
)
def test_two_internal_query_workers_share_parent_without_context_leak(
    monkeypatch,
    phase_order,
):
    """Settling one private worker must not expose its sibling as user Q/A."""
    session, original_history = _session()
    session["agent"] = types.SimpleNamespace(
        _active_parent_user_message_id="internal-multi",
        _ephemeral_internal_turn=True,
        mm_monitors={},
        mm_watchers={},
        session_id="durable-monitor-session",
    )
    emitted = []
    notices = []
    monkeypatch.setitem(server._sessions, "live-monitor", session)
    monkeypatch.setattr(
        server,
        "_emit",
        lambda event, sid, payload=None: emitted.append(
            (event, sid, payload or {})),
    )
    monkeypatch.setattr(
        server,
        "_append_mm_context",
        lambda *args, **kwargs: notices.append((args, kwargs)),
    )
    callbacks = _build_query_callbacks(monkeypatch, session)

    for suffix in ("a", "b"):
        server._on_tool_start(
            "live-monitor",
            f"tool-{suffix}",
            "query_multimodal",
            {"query": f"question qry-{suffix}"},
        )

    marker = session["_mm_internal_request_origins"]["internal-multi"]
    assert set(marker["calls"]) == {"tool-a", "tool-b"}

    for phase, suffix in phase_order:
        task_id = f"qry-{suffix}"
        if phase == "worker":
            callbacks["on_query_complete"](
                task_id,
                "internal-multi",
                f"question {task_id}",
                f"private answer {suffix}",
                "complete",
            )
        else:
            _complete_query_tool(
                tool_call_id=f"tool-{suffix}",
                task_id=task_id,
                parent_id="internal-multi",
            )

    assert session.get("_mm_internal_request_origins", {}) == {}
    assert session.get("_mm_query_results", []) == []
    assert notices == []
    assert session["history"] == original_history
    assert {
        task_id: row["status"]
        for task_id, row in session["_mm_worker_tasks"].items()
    } == {"qry-a": "complete", "qry-b": "complete"}
    assert {
        payload.get("task_id")
        for event, _sid, payload in emitted
        if event == "message.complete"
    } == {"qry-a", "qry-b"}


def test_internal_query_receipt_retires_only_its_sibling_call(
    monkeypatch,
):
    """A non-deferred receipt cannot unhide another worker on the same turn."""
    session, original_history = _session()
    session["agent"] = types.SimpleNamespace(
        _active_parent_user_message_id="internal-mixed",
        _ephemeral_internal_turn=True,
        mm_monitors={},
        mm_watchers={},
        session_id="durable-monitor-session",
    )
    notices = []
    monkeypatch.setitem(server._sessions, "live-monitor", session)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server,
        "_append_mm_context",
        lambda *args, **kwargs: notices.append((args, kwargs)),
    )
    callbacks = _build_query_callbacks(monkeypatch, session)

    server._on_tool_start(
        "live-monitor", "tool-receipt", "query_multimodal", {"query": "busy"})
    server._on_tool_start(
        "live-monitor", "tool-deferred", "query_multimodal", {"query": "watch"})

    _complete_query_tool(
        tool_call_id="tool-receipt",
        task_id="qry-receipt",
        parent_id="internal-mixed",
        handoff_mode="receipt",
    )
    marker = session["_mm_internal_request_origins"]["internal-mixed"]
    assert set(marker["calls"]) == {"tool-deferred"}

    _complete_query_tool(
        tool_call_id="tool-deferred",
        task_id="qry-deferred",
        parent_id="internal-mixed",
    )
    callbacks["on_query_complete"](
        "qry-deferred",
        "internal-mixed",
        "watch",
        "private deferred answer",
        "complete",
    )

    assert session.get("_mm_internal_request_origins", {}) == {}
    assert session.get("_mm_query_results", []) == []
    assert notices == []
    assert session["history"] == original_history
    assert session["_mm_worker_tasks"]["qry-deferred"]["status"] == "complete"


def test_internal_query_receipt_retires_marker_without_waiting_for_worker(
    monkeypatch,
):
    """A queue-full receipt has no worker completion and must not leak a marker."""
    session, _history = _session()
    session["agent"] = types.SimpleNamespace(
        _active_parent_user_message_id="internal-receipt",
        _ephemeral_internal_turn=True,
        mm_monitors={},
    )
    monkeypatch.setitem(server._sessions, "live-monitor", session)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)

    server._on_tool_start(
        "live-monitor", "tool-receipt", "query_multimodal", {"query": "q"})
    server._on_tool_complete(
        "live-monitor",
        "tool-receipt",
        "query_multimodal",
        {"query": "q"},
        json.dumps({
            "control": "handoff",
            "reply_owner": "query_worker",
            "handoff_mode": "receipt",
            "status": "busy",
            "task_id": "qry-receipt",
            "parent_user_message_id": "internal-receipt",
            "original_user_text": "q",
        }),
    )

    assert session.get("_mm_internal_request_origins", {}) == {}
