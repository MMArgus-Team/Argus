import copy
import json
import threading
import uuid
from types import SimpleNamespace
from unittest import mock

import pytest

from tui_gateway import server


class _Buffer:
    def __init__(self):
        self.source_types = []
        self.frames = []

    @property
    def size(self):
        return len(self.frames)

    def set_source_type(self, value):
        self.source_types.append(value)

    def push_live(self, jpeg_b64, **metadata):
        self.frames.append((jpeg_b64, metadata))
        return {"stored": True, "size": self.size}


class _Watcher:
    healthy = True
    is_ready = True

    def __init__(self):
        self.started = 0
        self.stopped = 0

    def mark_source_started(self):
        self.started += 1

    def mark_source_stopped(self):
        self.stopped += 1

    def asr_start(self, key, *_callbacks):
        self.asr_key = key
        return True


class _HealthyMemory:
    healthy = True
    is_ready = True


class _HealthyMonitor:
    def __init__(self):
        self.removed = []

    def is_healthy(self):
        return True

    def remove_monitor(self, monitor_id):
        self.removed.append(monitor_id)


def _ordinary_multiturn_history():
    return [
        {"role": "user", "content": "First ordinary question"},
        {
            "role": "assistant",
            "content": "First ordinary answer",
            "reasoning": {"summary": ["kept in the cached prefix"]},
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Second ordinary question"},
                {"type": "text", "text": "with structured content"},
            ],
        },
        {"role": "assistant", "content": "Second ordinary answer"},
    ]


def _history_contract(history):
    serialized = json.dumps(
        history, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return len(history), copy.deepcopy(history), serialized


def _assert_history_contract(history, before):
    depth, deep_snapshot, serialized = before
    assert len(history) == depth
    assert history == deep_snapshot
    assert (
        json.dumps(history, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        == serialized
    )


@pytest.fixture
def live_session(monkeypatch):
    sid = "desktop-mm"
    session = {
        "agent": SimpleNamespace(frame_buffer=_Buffer()),
        "session_key": "stored-desktop-mm",
        "source": "tui",
    }
    monkeypatch.setitem(server._sessions, sid, session)
    return sid, session


def test_source_start_promotes_an_existing_desktop_chat(monkeypatch, live_session):
    sid, session = live_session
    watcher = _Watcher()
    calls = []

    def promote(actual_sid, actual_session):
        calls.append((actual_sid, actual_session))
        actual_session["source"] = "multimodal"
        actual_session["_mm_live_watcher_agent"] = watcher
        return True

    monkeypatch.setattr(server, "_promote_session_to_multimodal", promote)

    response = server._methods["multimodal.source_stopped"](
        "rpc-1",
        {"session_id": sid, "started": True, "source_type": "camera"},
    )

    assert response["result"]["ok"] is True
    assert calls == [(sid, session)]
    assert session["agent"].frame_buffer.source_types == ["camera"]
    assert watcher.started == 1


def test_voice_only_start_promotes_plain_desktop_without_video_lifecycle(
    monkeypatch, live_session
):
    sid, session = live_session
    session["agent_ready"] = threading.Event()
    session["agent_ready"].set()
    watcher = _Watcher()
    calls = []

    def promote(actual_sid, actual_session):
        calls.append((actual_sid, actual_session))
        actual_session["source"] = "multimodal"
        actual_session["_mm_live_watcher_agent"] = watcher
        return True

    monkeypatch.setattr(server, "_start_agent_build", lambda *_args: None)
    monkeypatch.setattr(server, "_promote_session_to_multimodal", promote)

    response = server._methods["multimodal.asr_start"](
        "rpc-voice", {"session_id": sid}
    )

    assert response["result"] == {"enabled": True}
    assert calls == [(sid, session)]
    assert watcher.asr_key == f"asr:{sid}"
    assert session["_mm_asr_on"] is True
    assert session.get("_mm_capture_active") is not True
    assert watcher.started == 0


def test_voice_only_start_does_not_repromote_an_already_ready_session(
    monkeypatch, live_session
):
    sid, session = live_session
    session["agent_ready"] = threading.Event()
    session["agent_ready"].set()
    watcher = _Watcher()
    session["_mm_live_watcher_agent"] = watcher
    promote = mock.Mock(return_value=True)

    monkeypatch.setattr(server, "_start_agent_build", lambda *_args: None)
    monkeypatch.setattr(server, "_promote_session_to_multimodal", promote)

    response = server._methods["multimodal.asr_start"](
        "rpc-ready-voice", {"session_id": sid}
    )

    assert response["result"] == {"enabled": True}
    promote.assert_not_called()
    assert watcher.asr_key == f"asr:{sid}"


def test_voice_only_start_rechecks_transport_after_delayed_promotion(
    monkeypatch, live_session
):
    sid, session = live_session
    session["agent_ready"] = threading.Event()
    session["agent_ready"].set()
    transport_a = object()
    transport_b = object()
    session["transport"] = transport_a
    watcher = _Watcher()
    entered_promotion = threading.Event()
    finish_promotion = threading.Event()
    response = {}

    def promote(_sid, actual_session):
        entered_promotion.set()
        assert finish_promotion.wait(timeout=2)
        actual_session["_mm_live_watcher_agent"] = watcher
        return True

    def start_from_a():
        token = server.bind_transport(transport_a)
        try:
            response.update(server._methods["multimodal.asr_start"](
                "rpc-stale-voice", {"session_id": sid}
            ))
        finally:
            server.reset_transport(token)

    monkeypatch.setattr(server, "_start_agent_build", lambda *_args: None)
    monkeypatch.setattr(server, "_promote_session_to_multimodal", promote)
    thread = threading.Thread(target=start_from_a)
    thread.start()
    assert entered_promotion.wait(timeout=2)

    # A newer connection takes ownership while A is still promoting.
    session["transport"] = transport_b
    finish_promotion.set()
    thread.join(timeout=2)

    assert response["result"] == {
        "enabled": False,
        "reason": "stale_transport",
    }
    assert not hasattr(watcher, "asr_key")
    assert session.get("_mm_asr_on") is not True


def test_mid_session_promotion_preserves_cached_multiturn_history_and_restores_registry(
    monkeypatch, live_session
):
    sid, session = live_session
    session["history"] = _ordinary_multiturn_history()
    session["history_lock"] = threading.RLock()
    before = _history_contract(session["history"])
    agent = session["agent"]
    agent.mm_monitors = {}
    watcher = _Watcher()
    memory = _HealthyMemory()
    pushed_registries = []

    def start_memory(*_args, **kwargs):
        kwargs["session"]["_mm_memory_backend"] = memory
        return memory

    def start_watcher(_sid, _frame_buffer, _backend, actual_session):
        actual_session["_mm_live_watcher_agent"] = watcher

    def start_monitor(_sid, actual_session, _frame_buffer):
        actual_session["_mm_monitor_engine"] = _HealthyMonitor()

    def restore_registry(detached_history, actual_agent, session_id=""):
        assert detached_history is not session["history"]
        assert session_id == session["session_key"]
        # The reconciler historically rewrote stale tool receipts in place.
        # Promotion must isolate that repair from the provider-cached prefix.
        detached_history[0]["content"] = "reconciled only on detached history"
        actual_agent.mm_monitors["restored-monitor"] = {
            "enabled": False,
            "status": "interrupted",
        }
        return 1

    monkeypatch.setattr(server, "_maybe_start_memory_backend", start_memory)
    monkeypatch.setattr(server, "_maybe_start_live_watcher_agent", start_watcher)
    monkeypatch.setattr(server, "_maybe_start_monitor_engine", start_monitor)
    monkeypatch.setattr(server, "_reconcile_stale_mm_jobs", restore_registry)
    monkeypatch.setattr(
        server,
        "_push_mm_registries",
        lambda actual_sid, actual_agent: pushed_registries.append(
            (actual_sid, copy.deepcopy(actual_agent.mm_monitors))
        ),
    )

    response = server._methods["multimodal.source_stopped"](
        "rpc-promote-history",
        {
            "session_id": sid,
            "started": True,
            "source_type": "camera",
            "capture_generation": 3,
        },
    )

    assert response["result"]["ok"] is True
    _assert_history_contract(session["history"], before)
    assert agent.mm_monitors == {
        "restored-monitor": {"enabled": False, "status": "interrupted"}
    }
    assert pushed_registries == [(sid, copy.deepcopy(agent.mm_monitors))]
    assert watcher.started == 1
    assert session["source"] == "multimodal"


def test_memory_failure_never_builds_standalone_watcher_and_retry_completes_runtime(
    monkeypatch, live_session
):
    sid, session = live_session
    session["session_key"] = f"memory-retry-{uuid.uuid4().hex}"
    memory = _HealthyMemory()
    memory_results = iter([None, memory])
    watcher_instances = []

    class ResidentWatcher(_Watcher):
        def __init__(self, frame_buffer, memory_backend=None, **_kwargs):
            super().__init__()
            self.frame_buffer = frame_buffer
            self._memory_backend = memory_backend
            self.is_stopped = False
            self._stopped_callbacks = []
            watcher_instances.append(self)

        def add_stopped_callback(self, callback):
            self._stopped_callbacks.append(callback)

        def start(self, timeout=None):
            return True

        def stop(self, timeout=None):
            self.is_stopped = True
            for callback in list(self._stopped_callbacks):
                callback(self)
            return True

    def start_memory(*_args, **_kwargs):
        backend = next(memory_results)
        session["_mm_memory_backend"] = backend
        return backend

    def start_monitor(_sid, actual_session, _frame_buffer):
        monitor = actual_session.get("_mm_monitor_engine") or _HealthyMonitor()
        actual_session["_mm_monitor_engine"] = monitor
        return monitor

    monkeypatch.setattr(server, "_maybe_start_memory_backend", start_memory)
    monkeypatch.setattr(server, "_maybe_start_monitor_engine", start_monitor)
    monkeypatch.setattr(server, "_reconcile_stale_mm_jobs", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(server, "_push_mm_registries", lambda *_args: None)

    watcher_key = server._mm_memory_backend_registry_key(
        session["session_key"], sid
    )
    try:
        with (
            mock.patch(
                "agent.multimodal.hermes_glue.flatten_mm_config",
                return_value={"enabled": True},
            ),
            mock.patch(
                "agent.multimodal.watcher_engine.WatcherAgent",
                ResidentWatcher,
            ),
        ):
            first = server._methods["multimodal.source_stopped"](
                "rpc-memory-failed",
                {"session_id": sid, "started": True, "source_type": "camera"},
            )

            assert first["error"]["code"] == 5027
            assert watcher_instances == []
            assert session.get("_mm_live_watcher_agent") is None
            assert session["_mm_capture_active"] is False

            second = server._methods["multimodal.source_stopped"](
                "rpc-memory-retry",
                {"session_id": sid, "started": True, "source_type": "camera"},
            )

        assert second["result"]["ok"] is True
        assert len(watcher_instances) == 1
        assert watcher_instances[0]._memory_backend is memory
        assert session["_mm_memory_backend"] is memory
        assert session["_mm_live_watcher_agent"] is watcher_instances[0]
        assert isinstance(session["_mm_monitor_engine"], _HealthyMonitor)
        assert server._multimodal_runtime_ready(session) is True
        assert watcher_instances[0].started == 1
    finally:
        for watcher in watcher_instances:
            watcher.stop()
        with server._MM_ACTIVE_MEMORY_BACKENDS_LOCK:
            if server._MM_ACTIVE_WATCHERS.get(watcher_key) in watcher_instances:
                server._MM_ACTIVE_WATCHERS.pop(watcher_key, None)


def test_source_start_is_a_long_handler():
    """Promotion may wait for the ordinary chat's agent build to finish."""
    assert "multimodal.source_stopped" in server._LONG_HANDLERS


def test_ordinary_session_create_does_not_eagerly_start_multimodal(
    monkeypatch, tmp_path
):
    sid = "ordinary-live"
    monkeypatch.setattr(server.uuid, "uuid4", lambda: SimpleNamespace(hex=sid))
    monkeypatch.setattr(server, "_new_session_key", lambda: "ordinary-stored")
    monkeypatch.setattr(server, "_completion_cwd", lambda _params: str(tmp_path))
    monkeypatch.setattr(server, "_profile_home", lambda _profile: None)
    monkeypatch.setattr(server, "_claim_active_session_slot", lambda *_a, **_k: (None, None))
    monkeypatch.setattr(server, "_register_session_cwd", lambda *_args: None)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *_args: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(
        server,
        "_promote_session_to_multimodal",
        lambda *_args: pytest.fail("ordinary session.create must not promote"),
    )
    # Record the initially-absent key so monkeypatch removes the handler's live
    # session record at teardown.
    monkeypatch.setitem(server._sessions, sid[:8], {})

    response = server._methods["session.create"]("rpc-create", {})

    assert response["result"]["session_id"] == sid[:8]
    session = server._sessions[sid[:8]]
    assert session["source"] == "tui"
    assert session["agent"] is None
    assert "_mm_live_watcher_agent" not in session
    assert "_mm_monitor_engine" not in session


def test_source_start_waits_for_agent_ready_before_promotion(
    monkeypatch, live_session
):
    sid, session = live_session
    wait_entered = threading.Event()
    release_wait = threading.Event()
    promotion_called = threading.Event()
    response = []

    def wait_agent(actual_session, rid, timeout=30.0):
        assert actual_session is session
        assert rid == "rpc-waits"
        assert timeout >= 120.0
        wait_entered.set()
        assert release_wait.wait(timeout=2.0)
        return None

    def promote(actual_sid, actual_session):
        assert actual_sid == sid
        assert actual_session is session
        promotion_called.set()
        actual_session["source"] = "multimodal"
        return True

    monkeypatch.setattr(server, "_start_agent_build", lambda *_args: None)
    monkeypatch.setattr(server, "_wait_agent", wait_agent)
    monkeypatch.setattr(server, "_promote_session_to_multimodal", promote)

    worker = threading.Thread(
        target=lambda: response.append(
            server._methods["multimodal.source_stopped"](
                "rpc-waits",
                {"session_id": sid, "started": True, "source_type": "screen"},
            )
        )
    )
    worker.start()

    assert wait_entered.wait(timeout=1.0)
    assert not promotion_called.is_set(), (
        "source_started promoted a partially built ordinary chat before "
        "agent_ready"
    )

    release_wait.set()
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert promotion_called.is_set()
    assert response[0]["result"]["ok"] is True


def test_concurrent_stop_waits_for_delayed_promotion_then_interrupts(
    monkeypatch, live_session
):
    sid, session = live_session
    session["history"] = _ordinary_multiturn_history()
    before = _history_contract(session["history"])
    promotion_entered = threading.Event()
    release_promotion = threading.Event()
    promotion_finished = threading.Event()
    start_responses = []
    stop_responses = []
    interrupted = []
    watcher = _Watcher()

    def promote(actual_sid, actual_session):
        assert actual_sid == sid
        assert actual_session is session
        promotion_entered.set()
        assert release_promotion.wait(timeout=2.0)
        actual_session["source"] = "multimodal"
        actual_session["_mm_live_watcher_agent"] = watcher
        actual_session["_mm_monitor_engine"] = _HealthyMonitor()
        promotion_finished.set()
        return True

    def interrupt(actual_sid, actual_session):
        assert promotion_finished.is_set()
        assert actual_sid == session["session_key"]
        assert actual_session is session
        assert actual_session["_mm_capture_active"] is False
        interrupted.append(actual_sid)
        return 1

    monkeypatch.setattr(server, "_start_agent_build", lambda *_args: None)
    monkeypatch.setattr(server, "_wait_agent", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_promote_session_to_multimodal", promote)
    monkeypatch.setattr(server, "_interrupt_running_mm_jobs", interrupt)

    start = threading.Thread(
        target=lambda: start_responses.append(
            server._methods["multimodal.source_stopped"](
                "rpc-delayed-start",
                {
                    "session_id": sid,
                    "started": True,
                    "source_type": "screen",
                    "capture_client_id": "renderer-lock",
                    "capture_generation": 5,
                },
            )
        )
    )
    start.start()
    assert promotion_entered.wait(timeout=1.0)

    stop = threading.Thread(
        target=lambda: stop_responses.append(
            server._methods["multimodal.source_stopped"](
                "rpc-concurrent-stop",
                {
                    "session_id": sid,
                    "started": False,
                    "capture_client_id": "renderer-lock",
                    "capture_generation": 5,
                },
            )
        )
    )
    stop.start()
    stop.join(timeout=0.05)

    assert stop.is_alive(), "stop bypassed the in-flight activation transaction"
    assert interrupted == []

    release_promotion.set()
    start.join(timeout=2.0)
    stop.join(timeout=2.0)

    assert not start.is_alive()
    assert not stop.is_alive()
    assert start_responses[0]["result"]["ok"] is True
    assert stop_responses[0]["result"]["ok"] is True
    assert interrupted == [session["session_key"]]
    assert session["_mm_capture_active"] is False
    assert session["agent"].frame_buffer.source_types == ["screen", ""]
    assert watcher.started == 1
    _assert_history_contract(session["history"], before)


def test_source_start_does_not_repromote_a_ready_multimodal_session(
    monkeypatch, live_session
):
    sid, session = live_session
    session["source"] = "multimodal"
    watcher = _Watcher()
    session["_mm_live_watcher_agent"] = watcher
    monkeypatch.setattr(server, "_multimodal_runtime_ready", lambda actual: actual is session)
    promote = monkeypatch.setattr(
        server,
        "_promote_session_to_multimodal",
        lambda *_args: pytest.fail("already-multimodal session was promoted again"),
    )

    response = server._methods["multimodal.source_stopped"](
        "rpc-2",
        {"session_id": sid, "started": True, "source_type": "screen"},
    )

    assert response["result"]["ok"] is True
    assert session["agent"].frame_buffer.source_types == ["screen"]
    assert watcher.started == 1
    assert promote is None


def test_source_start_reports_promotion_failure(monkeypatch, live_session):
    sid, _session = live_session
    monkeypatch.setattr(
        server, "_promote_session_to_multimodal", lambda *_args: False
    )

    response = server._methods["multimodal.source_stopped"](
        "rpc-3",
        {"session_id": sid, "started": True, "source_type": "camera"},
    )

    assert response["error"]["code"] == 5027
    assert "initialize the multimodal runtime" in response["error"]["message"]


def test_source_stop_does_not_promote_an_ordinary_chat(monkeypatch, live_session):
    sid, session = live_session
    monkeypatch.setattr(
        server,
        "_promote_session_to_multimodal",
        lambda *_args: pytest.fail("source stop must not promote"),
    )

    response = server._methods["multimodal.source_stopped"](
        "rpc-4", {"session_id": sid, "started": False}
    )

    assert response["result"]["ok"] is True
    assert session["source"] == "tui"


def test_source_stop_preserves_cached_multiturn_history_while_interrupting_jobs(
    monkeypatch, live_session
):
    sid, session = live_session
    session["source"] = "multimodal"
    session["history"] = _ordinary_multiturn_history()
    session["history_lock"] = threading.RLock()
    before = _history_contract(session["history"])
    watcher = _Watcher()
    monitor_engine = _HealthyMonitor()
    agent = session["agent"]
    agent.mm_monitor_active = True
    agent.mm_monitors = {
        "running-monitor": {"enabled": True, "status": "running"},
    }
    agent.mm_watchers = {
        "queued-watcher": {"status": "queued"},
    }
    session["_mm_live_watcher_agent"] = watcher
    session["_mm_monitor_engine"] = monitor_engine
    pushed = []

    def reconcile(detached_history, actual_agent):
        assert detached_history is not session["history"]
        detached_history[-1]["content"] = "detached reconciliation only"
        actual_agent.mm_watchers["restored-watcher"] = {
            "status": "interrupted",
            "_interrupted": True,
        }
        return 1

    monkeypatch.setattr(server, "_reconcile_stale_mm_jobs", reconcile)
    import tools.live_watcher_tool as live_watcher_tool
    import tools.monitor_tool as monitor_tool

    monkeypatch.setattr(
        monitor_tool,
        "_push_monitors_event",
        lambda actual_sid, actual_agent: pushed.append(
            ("monitor", actual_sid, copy.deepcopy(actual_agent.mm_monitors))
        ),
    )
    monkeypatch.setattr(
        live_watcher_tool,
        "_push_watchers_event",
        lambda actual_sid, actual_agent: pushed.append(
            ("watcher", actual_sid, copy.deepcopy(actual_agent.mm_watchers))
        ),
    )

    response = server._methods["multimodal.source_stopped"](
        "rpc-stop-history", {"session_id": sid, "started": False}
    )

    assert response["result"]["ok"] is True
    _assert_history_contract(session["history"], before)
    assert monitor_engine.removed == ["running-monitor"]
    assert agent.mm_monitors["running-monitor"] == {
        "enabled": False,
        "status": "interrupted",
        "_interrupted": True,
    }
    assert agent.mm_watchers["queued-watcher"] == {
        "status": "interrupted",
        "_interrupted": True,
    }
    assert agent.mm_watchers["restored-watcher"]["status"] == "interrupted"
    assert agent.mm_monitor_active is False
    assert watcher.stopped == 1
    assert session["agent"].frame_buffer.source_types == [""]
    assert [kind for kind, *_rest in pushed] == ["monitor", "watcher"]


def test_stale_stop_generation_cannot_tear_down_a_newer_source(
    monkeypatch, live_session
):
    sid, session = live_session
    session["source"] = "multimodal"
    watcher = _Watcher()
    session["_mm_live_watcher_agent"] = watcher
    interrupted = []
    monkeypatch.setattr(server, "_multimodal_runtime_ready", lambda _session: True)
    monkeypatch.setattr(
        server,
        "_interrupt_running_mm_jobs",
        lambda *args: interrupted.append(args),
    )

    started = server._methods["multimodal.source_stopped"](
        "rpc-gen-start",
        {
            "session_id": sid,
            "started": True,
            "source_type": "camera",
            "capture_generation": 8,
        },
    )
    stale_stop = server._methods["multimodal.source_stopped"](
        "rpc-gen-stale-stop",
        {"session_id": sid, "started": False, "capture_generation": 7},
    )

    assert started["result"]["capture_generation"] == 8
    assert stale_stop["result"] == {"ok": True, "stale": True}
    assert session["_mm_capture_active"] is True
    assert session["agent"].frame_buffer.source_types == ["camera"]
    assert watcher.started == 1
    assert interrupted == []


def test_newer_source_start_interrupts_old_jobs_before_late_old_stop(
    monkeypatch, live_session
):
    sid, session = live_session
    session["source"] = "multimodal"
    watcher = _Watcher()
    session["_mm_live_watcher_agent"] = watcher
    old_generation = 30
    new_generation = old_generation + 2
    session.update(
        {
            "_mm_capture_active": True,
            "_mm_capture_client_id": "renderer-replace",
            "_mm_capture_client_started_at_ms": 50_000,
            "_mm_capture_generation": old_generation,
        }
    )
    agent = session["agent"]
    agent.mm_monitors = {
        "old-monitor": {"enabled": True, "status": "running"}
    }
    interrupted = []

    def interrupt(actual_sid, actual_session):
        interrupted.append((actual_sid, actual_session["_mm_capture_generation"]))
        actual_session["agent"].mm_monitors["old-monitor"] = {
            "enabled": False,
            "status": "interrupted",
        }
        return 1

    monkeypatch.setattr(server, "_multimodal_runtime_ready", lambda _session: True)
    monkeypatch.setattr(server, "_interrupt_running_mm_jobs", interrupt)

    replacement = server._methods["multimodal.source_stopped"](
        "rpc-replacement-start",
        {
            "session_id": sid,
            "started": True,
            "source_type": "screen",
            "capture_client_id": "renderer-replace",
            "capture_client_started_at_ms": 50_000,
            "capture_generation": new_generation,
        },
    )
    stale_old_stop = server._methods["multimodal.source_stopped"](
        "rpc-late-old-stop",
        {
            "session_id": sid,
            "started": False,
            "capture_client_id": "renderer-replace",
            "capture_client_started_at_ms": 50_000,
            "capture_generation": old_generation,
        },
    )

    assert replacement["result"]["capture_generation"] == new_generation
    assert interrupted == [(session["session_key"], new_generation)]
    assert agent.mm_monitors["old-monitor"] == {
        "enabled": False,
        "status": "interrupted",
    }
    assert stale_old_stop["result"] == {"ok": True, "stale": True}
    assert session["_mm_capture_active"] is True
    assert session["_mm_capture_generation"] == new_generation
    assert session["agent"].frame_buffer.source_types == ["screen"]
    assert watcher.started == 1


def test_duplicate_start_cannot_resurrect_a_stopped_capture_generation(
    monkeypatch, live_session
):
    sid, session = live_session
    session["source"] = "multimodal"
    watcher = _Watcher()
    session["_mm_live_watcher_agent"] = watcher
    monkeypatch.setattr(server, "_multimodal_runtime_ready", lambda _session: True)
    monkeypatch.setattr(server, "_interrupt_running_mm_jobs", lambda *_args: 0)

    params = {
        "session_id": sid,
        "started": True,
        "source_type": "screen",
        "capture_generation": 11,
    }
    server._methods["multimodal.source_stopped"]("rpc-gen-start", params)
    server._methods["multimodal.source_stopped"](
        "rpc-gen-stop",
        {"session_id": sid, "started": False, "capture_generation": 11},
    )
    late_start = server._methods["multimodal.source_stopped"](
        "rpc-gen-late-start", params
    )

    assert late_start["result"] == {"ok": True, "stale": True}
    assert session["_mm_capture_active"] is False
    assert session["agent"].frame_buffer.source_types == ["screen", ""]
    assert watcher.started == 1


def test_new_attempt_can_reuse_stopped_client_generation_but_old_attempt_stays_stale(
    monkeypatch, live_session
):
    sid, session = live_session
    session["source"] = "multimodal"
    watcher = _Watcher()
    session["_mm_live_watcher_agent"] = watcher
    monkeypatch.setattr(server, "_multimodal_runtime_ready", lambda _session: True)
    monkeypatch.setattr(server, "_interrupt_running_mm_jobs", lambda *_args: 0)
    owner = {
        "session_id": sid,
        "capture_client_id": "renderer-attempt-restart",
        "capture_client_started_at_ms": 15_000,
        "capture_generation": 11,
    }
    start_a_params = {
        **owner,
        "started": True,
        "source_type": "camera",
        "capture_attempt_id": "attempt-A",
    }

    start_a = server._methods["multimodal.source_stopped"](
        "rpc-attempt-a-start", start_a_params
    )
    stop_a = server._methods["multimodal.source_stopped"](
        "rpc-attempt-a-stop",
        {**owner, "started": False, "capture_attempt_id": "attempt-A"},
    )
    start_b = server._methods["multimodal.source_stopped"](
        "rpc-attempt-b-start",
        {
            **owner,
            "started": True,
            "source_type": "camera",
            "capture_attempt_id": "attempt-B",
        },
    )
    retry_a = server._methods["multimodal.source_stopped"](
        "rpc-attempt-a-retry", start_a_params
    )
    frame_b = server._methods["multimodal.frame"](
        "rpc-attempt-b-frame",
        {
            **owner,
            "jpeg_b64": "A" * 128,
            "source_type": "camera",
            "capture_attempt_id": "attempt-B",
        },
    )

    assert start_a["result"].get("stale") is not True
    assert stop_a["result"].get("stale") is not True
    assert start_b["result"]["capture_generation"] == owner["capture_generation"]
    assert start_b["result"].get("stale") is not True
    assert retry_a["result"] == {"ok": True, "stale": True}
    assert frame_b["result"]["buffered"] is True
    assert session["_mm_capture_active"] is True
    assert session["_mm_capture_client_id"] == owner["capture_client_id"]
    assert session["_mm_capture_generation"] == owner["capture_generation"]
    assert session["_mm_capture_attempt_id"] == "attempt-B"
    assert watcher.started == 2


def test_duplicate_active_start_is_idempotent_for_current_watcher_generation(
    monkeypatch, live_session
):
    sid, session = live_session
    session["source"] = "multimodal"
    buffer = session["agent"].frame_buffer
    buffer.frames.append(("old-source-frame", {}))
    interrupted = []

    class EpochWatcher(_Watcher):
        def __init__(self):
            super().__init__()
            self.epoch = 0
            self.clears = 0

        def mark_source_started(self):
            super().mark_source_started()
            self.epoch += 1
            self.clears += 1
            buffer.frames.clear()

    watcher = EpochWatcher()
    session["_mm_live_watcher_agent"] = watcher
    monkeypatch.setattr(server, "_multimodal_runtime_ready", lambda _session: True)
    monkeypatch.setattr(
        server,
        "_interrupt_running_mm_jobs",
        lambda *args: interrupted.append(args),
    )
    params = {
        "session_id": sid,
        "started": True,
        "source_type": "camera",
        "capture_client_id": "renderer-idempotent",
        "capture_client_started_at_ms": 10_000,
        "capture_generation": 17,
    }

    first = server._methods["multimodal.source_stopped"](
        "rpc-idempotent-first", params
    )
    buffer.frames.append(("current-source-frame", {}))
    duplicate = server._methods["multimodal.source_stopped"](
        "rpc-idempotent-duplicate", params
    )

    assert first["result"]["capture_generation"] == 17
    assert duplicate["result"]["capture_generation"] == 17
    assert watcher.started == 1
    assert watcher.epoch == 1
    assert watcher.clears == 1
    assert buffer.frames == [("current-source-frame", {})]
    assert session["_mm_capture_active"] is True
    assert interrupted == []


def test_new_renderer_client_can_take_over_generation_zero_and_stales_old_owner(
    monkeypatch, live_session
):
    sid, session = live_session
    session["source"] = "multimodal"
    session["_mm_live_watcher_agent"] = _Watcher()
    interrupted = []
    monkeypatch.setattr(server, "_multimodal_runtime_ready", lambda _session: True)
    monkeypatch.setattr(
        server,
        "_interrupt_running_mm_jobs",
        lambda *args: interrupted.append(args),
    )

    transport_a = object()
    transport_b = object()
    session["transport"] = transport_a

    def invoke(method, rid, params, transport):
        token = server.bind_transport(transport)
        try:
            return server._methods[method](rid, params)
        finally:
            server.reset_transport(token)

    start_a = invoke(
        "multimodal.source_stopped",
        "rpc-client-a-start",
        {
            "session_id": sid,
            "started": True,
            "source_type": "camera",
            "capture_client_id": "renderer-A",
            "capture_generation": 9,
        },
        transport_a,
    )
    # session.resume under the reloaded renderer transfers the live session's
    # transport before its new capture client announces generation zero.
    session["transport"] = transport_b
    start_b = invoke(
        "multimodal.source_stopped",
        "rpc-client-b-start",
        {
            "session_id": sid,
            "started": True,
            "source_type": "screen",
            "capture_client_id": "renderer-B",
            "capture_generation": 0,
        },
        transport_b,
    )

    jpeg = "A" * 128
    frame_b = invoke(
        "multimodal.frame",
        "rpc-client-b-frame",
        {
            "session_id": sid,
            "jpeg_b64": jpeg,
            "source_type": "screen",
            "capture_client_id": "renderer-B",
            "capture_generation": 0,
        },
        transport_b,
    )
    stale_frame_a = invoke(
        "multimodal.frame",
        "rpc-client-a-late-frame",
        {
            "session_id": sid,
            "jpeg_b64": jpeg,
            "source_type": "camera",
            "capture_client_id": "renderer-A",
            "capture_generation": 9,
        },
        transport_a,
    )
    stale_stop_a = invoke(
        "multimodal.source_stopped",
        "rpc-client-a-late-stop",
        {
            "session_id": sid,
            "started": False,
            "capture_client_id": "renderer-A",
            "capture_generation": 9,
        },
        transport_a,
    )

    assert start_a["result"]["capture_generation"] == 9
    assert start_b["result"]["capture_generation"] == 0
    assert frame_b["result"]["buffered"] is True
    assert stale_frame_a["result"] == {
        "buffered": False,
        "reason": "stale_capture",
    }
    assert stale_stop_a["result"] == {"ok": True, "stale": True}
    assert session["_mm_capture_client_id"] == "renderer-B"
    assert session["_mm_capture_generation"] == 0
    assert session["_mm_capture_active"] is True
    assert len(session["agent"].frame_buffer.frames) == 1
    assert session["agent"].frame_buffer.source_types == ["camera", "screen"]
    assert len(interrupted) == 1
    assert interrupted[0][0] == session["session_key"]
    assert interrupted[0][1] is session


def test_new_transport_retry_survives_stale_inflight_start_tombstone(
    monkeypatch, live_session
):
    sid, session = live_session
    session["source"] = "multimodal"
    watcher = _Watcher()
    session["_mm_live_watcher_agent"] = watcher
    wait_entered = threading.Event()
    release_wait = threading.Event()
    responses = {}
    transport_a = object()
    transport_b = object()
    session["transport"] = transport_a

    def wait_agent(_session, rid, **_kwargs):
        if rid == "rpc-transport-a":
            wait_entered.set()
            assert release_wait.wait(timeout=2.0)
        return None

    def invoke(rid, transport):
        token = server.bind_transport(transport)
        try:
            responses[rid] = server._methods["multimodal.source_stopped"](
                rid,
                {
                    "session_id": sid,
                    "started": True,
                    "source_type": "screen",
                    "capture_client_id": "renderer-retry",
                    "capture_client_started_at_ms": 20_000,
                    "capture_generation": 4,
                },
            )
        finally:
            server.reset_transport(token)

    monkeypatch.setattr(server, "_start_agent_build", lambda *_args: None)
    monkeypatch.setattr(server, "_wait_agent", wait_agent)
    monkeypatch.setattr(server, "_multimodal_runtime_ready", lambda _session: True)

    request_a = threading.Thread(
        target=invoke, args=("rpc-transport-a", transport_a)
    )
    request_a.start()
    assert wait_entered.wait(timeout=1.0)

    session["transport"] = transport_b
    request_b = threading.Thread(
        target=invoke, args=("rpc-transport-b", transport_b)
    )
    request_b.start()
    request_b.join(timeout=0.05)
    assert request_b.is_alive(), "replacement retry bypassed capture ordering"

    release_wait.set()
    request_a.join(timeout=2.0)
    request_b.join(timeout=2.0)

    assert not request_a.is_alive()
    assert not request_b.is_alive()
    assert responses["rpc-transport-a"]["result"] == {"ok": True, "stale": True}
    assert responses["rpc-transport-b"]["result"]["capture_generation"] == 4
    assert responses["rpc-transport-b"]["result"].get("stale") is not True
    assert session["_mm_capture_active"] is True
    assert session["_mm_capture_client_id"] == "renderer-retry"
    assert session["_mm_capture_generation"] == 4
    assert session["agent"].frame_buffer.source_types == ["screen"]
    assert watcher.started == 1


def test_new_capture_attempt_owns_same_generation_after_reconnect(
    monkeypatch, live_session
):
    sid, session = live_session
    session["source"] = "multimodal"
    watcher = _Watcher()
    session["_mm_live_watcher_agent"] = watcher
    wait_entered = threading.Event()
    release_wait = threading.Event()
    responses = {}
    interrupted = []
    transport_a = object()
    transport_b = object()
    session["transport"] = transport_a
    common = {
        "session_id": sid,
        "capture_client_id": "renderer-same-owner",
        "capture_client_started_at_ms": 30_000,
        "capture_generation": 6,
    }

    def wait_agent(_session, rid, **_kwargs):
        if rid == "rpc-attempt-a-start":
            wait_entered.set()
            assert release_wait.wait(timeout=2.0)
        return None

    def invoke(method, rid, params, transport):
        token = server.bind_transport(transport)
        try:
            return server._methods[method](rid, params)
        finally:
            server.reset_transport(token)

    def start_attempt(rid, attempt_id, transport):
        responses[rid] = invoke(
            "multimodal.source_stopped",
            rid,
            {
                **common,
                "started": True,
                "source_type": "camera",
                "capture_attempt_id": attempt_id,
            },
            transport,
        )

    monkeypatch.setattr(server, "_start_agent_build", lambda *_args: None)
    monkeypatch.setattr(server, "_wait_agent", wait_agent)
    monkeypatch.setattr(server, "_multimodal_runtime_ready", lambda _session: True)
    monkeypatch.setattr(
        server,
        "_interrupt_running_mm_jobs",
        lambda *args: interrupted.append(args),
    )

    request_a = threading.Thread(
        target=start_attempt,
        args=("rpc-attempt-a-start", "attempt-A", transport_a),
    )
    request_a.start()
    assert wait_entered.wait(timeout=1.0)

    session["transport"] = transport_b
    request_b = threading.Thread(
        target=start_attempt,
        args=("rpc-attempt-b-start", "attempt-B", transport_b),
    )
    request_b.start()
    request_b.join(timeout=0.05)
    assert request_b.is_alive(), "attempt B bypassed the capture transaction"

    release_wait.set()
    request_a.join(timeout=2.0)
    request_b.join(timeout=2.0)

    jpeg = "A" * 128
    frame_b = invoke(
        "multimodal.frame",
        "rpc-attempt-b-frame",
        {
            **common,
            "jpeg_b64": jpeg,
            "source_type": "camera",
            "capture_attempt_id": "attempt-B",
        },
        transport_b,
    )
    stale_frame_a = invoke(
        "multimodal.frame",
        "rpc-attempt-a-frame",
        {
            **common,
            "jpeg_b64": jpeg,
            "source_type": "camera",
            "capture_attempt_id": "attempt-A",
        },
        transport_a,
    )
    stale_rollback_a = invoke(
        "multimodal.source_stopped",
        "rpc-attempt-a-rollback",
        {
            **common,
            "started": False,
            "capture_attempt_id": "attempt-A",
        },
        transport_a,
    )
    next_frame_b = invoke(
        "multimodal.frame",
        "rpc-attempt-b-periodic-frame",
        {
            **common,
            "jpeg_b64": jpeg,
            "source_type": "camera",
            "capture_attempt_id": "attempt-B",
        },
        transport_b,
    )

    assert responses["rpc-attempt-a-start"]["result"] == {
        "ok": True,
        "stale": True,
    }
    assert responses["rpc-attempt-b-start"]["result"]["capture_generation"] == 6
    assert responses["rpc-attempt-b-start"]["result"].get("stale") is not True
    assert frame_b["result"]["buffered"] is True
    assert stale_frame_a["result"] == {
        "buffered": False,
        "reason": "stale_capture",
    }
    assert stale_rollback_a["result"] == {"ok": True, "stale": True}
    assert next_frame_b["result"]["buffered"] is True
    assert session["_mm_capture_active"] is True
    assert session["_mm_capture_attempt_id"] == "attempt-B"
    assert session["_mm_capture_generation"] == 6
    assert len(session["agent"].frame_buffer.frames) == 2
    assert interrupted == []


def test_legacy_capture_without_attempt_id_remains_compatible(
    monkeypatch, live_session
):
    sid, session = live_session
    session["source"] = "multimodal"
    watcher = _Watcher()
    session["_mm_live_watcher_agent"] = watcher
    interrupted = []
    monkeypatch.setattr(server, "_multimodal_runtime_ready", lambda _session: True)
    monkeypatch.setattr(
        server,
        "_interrupt_running_mm_jobs",
        lambda *args: interrupted.append(args),
    )
    owner = {
        "session_id": sid,
        "capture_client_id": "legacy-renderer",
        "capture_generation": 12,
    }

    started = server._methods["multimodal.source_stopped"](
        "rpc-legacy-start",
        {**owner, "started": True, "source_type": "screen"},
    )
    frame = server._methods["multimodal.frame"](
        "rpc-legacy-frame",
        {**owner, "jpeg_b64": "A" * 128, "source_type": "screen"},
    )
    stopped = server._methods["multimodal.source_stopped"](
        "rpc-legacy-stop",
        {**owner, "started": False},
    )
    retry = server._methods["multimodal.source_stopped"](
        "rpc-legacy-retry",
        {**owner, "started": True, "source_type": "screen"},
    )

    assert started["result"]["capture_generation"] == 12
    assert frame["result"]["buffered"] is True
    assert stopped["result"]["ok"] is True
    assert stopped["result"].get("stale") is not True
    assert retry["result"] == {"ok": True, "stale": True}
    assert session["_mm_capture_active"] is False
    assert session["agent"].frame_buffer.source_types == ["screen", ""]
    assert len(session["agent"].frame_buffer.frames) == 1
    assert watcher.started == 1
    assert len(interrupted) == 1
