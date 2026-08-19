"""Request correlation contracts for multimodal foreground turns."""

from __future__ import annotations

import sys
import threading
import types

from tui_gateway import server


class _ImmediateThread:
    def __init__(self, target=None, daemon=None, name=None):
        self.target = target

    def start(self):
        if self.target is not None:
            self.target()

    def is_alive(self):
        return False


class _DeferredThread(_ImmediateThread):
    def start(self):
        return None


def _ready_event(*, set_now: bool = True) -> threading.Event:
    event = threading.Event()
    if set_now:
        event.set()
    return event


def _session(**overrides):
    session = {
        "agent": types.SimpleNamespace(),
        "agent_ready": _ready_event(),
        "attached_images": [],
        "cols": 80,
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "queued_prompt": None,
        "queued_prompts": [],
        "running": False,
        "session_key": "stored-session",
        "source": "multimodal",
    }
    session.update(overrides)
    return session


def test_request_id_preserves_supplied_value_and_generates_unique_fallbacks():
    assert server._ensure_client_request_id("  caller-id  ") == "caller-id"
    assert server._ensure_client_request_id("x" * 140) == "x" * 128

    first = server._ensure_client_request_id()
    second = server._ensure_client_request_id("")
    assert first.startswith("turn_")
    assert second.startswith("turn_")
    assert first != second


def test_run_prompt_submit_tags_every_internal_turn_once(monkeypatch):
    emitted = []
    monkeypatch.setattr(server.threading, "Thread", _DeferredThread)
    monkeypatch.setattr(
        server,
        "_reserve_mm_query_turn_projection",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        server,
        "_emit",
        lambda event, sid, payload=None: emitted.append(
            (event, sid, payload or {})),
    )
    session = _session(running=True)

    server._run_prompt_submit("rpc-1", "live", session, "first")
    first_events = list(emitted)
    emitted.clear()
    server._run_prompt_submit("rpc-2", "live", session, "second")
    second_events = list(emitted)

    first_ids = {
        payload["request_id"]
        for event, _sid, payload in first_events
        if event in {"message.user_echo", "message.start"}
    }
    second_ids = {
        payload["request_id"]
        for event, _sid, payload in second_events
        if event in {"message.user_echo", "message.start"}
    }
    assert len(first_ids) == 1
    assert len(second_ids) == 1
    assert first_ids != second_ids


def test_generated_request_id_enables_mm_deferred_persistence(monkeypatch):
    captured = {}

    class Agent:
        def run_conversation(
            self, _prompt, conversation_history=None, stream_callback=None,
        ):
            captured["defer"] = self._defer_current_turn_persistence
            captured["request_id"] = self._active_parent_user_message_id
            return {"final_response": "", "messages": []}

    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "render_message", lambda *_args: "")
    monkeypatch.setattr(server, "_get_usage", lambda _agent: {})
    monkeypatch.setattr(
        server,
        "_reserve_mm_query_turn_projection",
        lambda *_args, **_kwargs: [],
    )
    session = _session(agent=Agent(), running=True)

    server._run_prompt_submit(
        "rpc-mm", "live-mm", session, "inspect", user_originated=True)

    assert captured["defer"] is True
    assert captured["request_id"].startswith("turn_")


def test_internal_watcher_hook_has_no_user_echo_and_does_not_replace_history(
    monkeypatch,
):
    emitted = []
    captured = {}

    class Agent:
        def run_conversation(
            self, prompt, conversation_history=None, stream_callback=None,
        ):
            captured["prompt"] = prompt
            captured["ephemeral"] = self._ephemeral_internal_turn
            if stream_callback:
                stream_callback("最终总结")
            return {
                "final_response": "最终总结",
                # Deliberately include the internal prompt to prove the gateway
                # refuses to install this working transcript as canonical history.
                "messages": list(conversation_history or []) + [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "最终总结"},
                ],
            }

    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(
        server,
        "_emit",
        lambda event, sid, payload=None: emitted.append(
            (event, sid, payload or {})),
    )
    monkeypatch.setattr(server, "render_message", lambda *_args: "")
    monkeypatch.setattr(server, "_get_usage", lambda _agent: {})
    monkeypatch.setattr(
        server,
        "_reserve_mm_query_turn_projection",
        lambda *_args, **_kwargs: [],
    )
    original_history = [
        {"role": "user", "content": "开始看视频"},
        {"role": "assistant", "content": "已启动"},
    ]
    session = _session(
        agent=Agent(),
        running=True,
        history=list(original_history),
        history_version=3,
    )

    server._run_prompt_submit(
        "hook-rpc",
        "live-hook",
        session,
        "internal completion report",
        internal_origin="watcher_hook",
    )

    assert captured == {
        "prompt": "internal completion report",
        "ephemeral": True,
    }
    assert not any(event == "message.user_echo" for event, _sid, _p in emitted)
    assert any(event == "message.complete" for event, _sid, _p in emitted)
    assert session["history"] == original_history
    assert session["history_version"] == 3


def test_prompt_submit_stamps_raw_frame_anchor_for_query_worker(monkeypatch):
    captured = {}

    class Agent:
        frame_buffer = types.SimpleNamespace(
            monitor_latest_ts=8.5,
            latest_ts=7.5,
        )

        def run_conversation(
            self, _prompt, conversation_history=None, stream_callback=None,
        ):
            captured["ran"] = True
            return {"final_response": "", "messages": []}

    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "render_message", lambda *_args: "")
    monkeypatch.setattr(server, "_get_usage", lambda _agent: {})
    monkeypatch.setattr(
        server,
        "_reserve_mm_query_turn_projection",
        lambda *_args, **_kwargs: [],
    )
    session = _session(agent=Agent(), running=True)

    server._run_prompt_submit(
        "rpc-mm-anchor", "live-mm-anchor", session, "inspect",
        user_originated=True,
    )

    assert captured["ran"] is True
    assert session["_mm_send_anchor_ts"] == 8.5


def test_prompt_submit_returns_and_passes_generated_request_id(monkeypatch):
    captured = []
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda *_: None)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda *_: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda *_: None)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)

    def fake_run(
        _rid,
        _sid,
        session,
        _text,
        *,
        user_originated=False,
        client_request_id="",
    ):
        captured.append((client_request_id, user_originated))
        with session["history_lock"]:
            session["running"] = False

    monkeypatch.setattr(server, "_run_prompt_submit", fake_run)
    session = _session()
    monkeypatch.setitem(server._sessions, "live-submit", session)

    first = server._methods["prompt.submit"](
        "rpc-1", {"session_id": "live-submit", "text": "one"})
    second = server._methods["prompt.submit"](
        "rpc-2", {"session_id": "live-submit", "text": "two"})

    first_id = first["result"]["client_request_id"]
    second_id = second["result"]["client_request_id"]
    assert first_id and second_id and first_id != second_id
    assert captured == [(first_id, True), (second_id, True)]


class _AsrEngine:
    def asr_start(self, _key, _on_partial, on_final, _on_speech_started):
        self.on_final = on_final
        return True


def _start_asr(monkeypatch, session):
    emitted = []
    submitted = []
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(
        server,
        "_emit",
        lambda event, sid, payload=None: emitted.append(
            (event, sid, payload or {})),
    )

    def fake_run(
        _rid,
        _sid,
        target_session,
        text,
        *,
        user_originated=False,
        client_request_id="",
    ):
        submitted.append((text, user_originated, client_request_id))
        with target_session["history_lock"]:
            target_session["running"] = False

    monkeypatch.setattr(server, "_run_prompt_submit", fake_run)
    monkeypatch.setitem(server._sessions, "live-asr", session)
    response = server._methods["multimodal.asr_start"](
        "rpc-asr", {"session_id": "live-asr"})
    assert response["result"]["enabled"] is True
    return emitted, submitted


def test_legacy_asr_final_and_submit_share_one_request_id(monkeypatch):
    engine = _AsrEngine()
    session = _session(_mm_live_watcher_agent=engine)
    emitted, submitted = _start_asr(monkeypatch, session)

    engine.on_final("打开日志")

    finals = [
        payload for event, _sid, payload in emitted
        if event == "multimodal.asr_final" and payload.get("text")
    ]
    assert len(finals) == 1
    assert submitted == [("打开日志", True, finals[0]["request_id"])]


def test_busy_legacy_asr_queues_same_request_id(monkeypatch):
    engine = _AsrEngine()
    session = _session(_mm_live_watcher_agent=engine, running=True)
    emitted, submitted = _start_asr(monkeypatch, session)

    engine.on_final("继续检查")

    finals = [
        payload for event, _sid, payload in emitted
        if event == "multimodal.asr_final" and payload.get("text")
    ]
    assert len(finals) == 1
    assert submitted == []
    assert session["queued_prompts"][0]["client_request_id"] == (
        finals[0]["request_id"])


def test_voice_eou_emits_one_user_bubble_with_submit_request_id(monkeypatch):
    engine = _AsrEngine()
    session = _session(_mm_live_watcher_agent=engine)

    class Voice:
        def is_interactive(self):
            return False

        def submit_user(self, text):
            session["_mm_voice_turn_cb"](text)

    session["_mm_voice_agent"] = Voice()
    emitted, submitted = _start_asr(monkeypatch, session)

    engine.on_final("分析当前页面")

    finals = [
        payload for event, _sid, payload in emitted
        if event == "multimodal.asr_final" and payload.get("text")
    ]
    assert len(finals) == 1
    assert submitted == [(
        "分析当前页面", True, finals[0]["request_id"])]


def test_voice_main_route_emits_asr_final_and_reuses_id(monkeypatch):
    import agent as agent_package

    captured = {}
    fake_multimodal = types.ModuleType("agent.multimodal")
    fake_multimodal.__path__ = []
    fake_voice_module = types.ModuleType("agent.multimodal.voice_agent")

    class VoiceAgent:
        def __init__(self, **kwargs):
            captured["submit"] = kwargs["submit_main_agent_cb"]

        def start(self):
            return None

    fake_voice_module.VoiceAgent = VoiceAgent
    fake_multimodal.voice_agent = fake_voice_module
    monkeypatch.setattr(
        agent_package, "multimodal", fake_multimodal, raising=False)
    monkeypatch.setitem(sys.modules, "agent.multimodal", fake_multimodal)
    monkeypatch.setitem(
        sys.modules, "agent.multimodal.voice_agent", fake_voice_module)
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)

    emitted = []
    submitted = []
    monkeypatch.setattr(
        server,
        "_emit",
        lambda event, sid, payload=None: emitted.append(
            (event, sid, payload or {})),
    )

    def fake_run(
        _rid,
        _sid,
        target_session,
        text,
        *,
        user_originated=False,
        client_request_id="",
    ):
        submitted.append((text, user_originated, client_request_id))
        with target_session["history_lock"]:
            target_session["running"] = False

    monkeypatch.setattr(server, "_run_prompt_submit", fake_run)
    session = _session(
        _mm_live_watcher_agent=object(),
        _mm_live_sid="live-voice",
    )
    server._get_voice_agent(session)

    captured["submit"]("看看编译错误", 7)

    finals = [
        payload for event, _sid, payload in emitted
        if event == "multimodal.asr_final" and payload.get("text")
    ]
    assert len(finals) == 1
    assert submitted == [(
        "看看编译错误", True, finals[0]["request_id"])]


def test_list_registries_ready_requires_event_and_agent(monkeypatch):
    not_ready = _session(agent=None, agent_ready=_ready_event(set_now=False))
    monkeypatch.setitem(server._sessions, "not-ready", not_ready)
    response = server._methods["multimodal.list_registries"](
        "rpc-1", {"session_id": "not-ready"})
    assert response["result"] == {
        "ready": False,
        "monitors": [],
        "watchers": [],
    }

    ready = _session(agent=types.SimpleNamespace(), agent_ready=_ready_event())
    monkeypatch.setitem(server._sessions, "ready-empty", ready)
    response = server._methods["multimodal.list_registries"](
        "rpc-2", {"session_id": "ready-empty"})
    assert response["result"]["ready"] is True
    assert response["result"]["monitors"] == []
    assert response["result"]["watchers"] == []
