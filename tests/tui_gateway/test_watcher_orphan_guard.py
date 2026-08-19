"""Gateway ownership regressions for the resident multimodal Watcher."""

from __future__ import annotations

import threading
import time
import uuid
from types import SimpleNamespace
from unittest import mock


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


def _clear_guards(server) -> None:
    with server._MM_ACTIVE_MEMORY_BACKENDS_LOCK:
        server._MM_ACTIVE_WATCHERS.clear()


def test_finalize_during_start_sees_and_stops_published_watcher():
    import tui_gateway.server as server
    from agent.multimodal.watcher_engine import WatcherAgent

    _clear_guards(server)
    entered = threading.Event()
    instances = []
    result = []
    session = {
        "session_key": f"watcher-close-start-{uuid.uuid4().hex}",
        "history": [],
    }
    frame_buffer = object()
    memory_backend = object()

    def watcher_factory(*args, **kwargs):
        engine = WatcherAgent(*args, **kwargs)
        instances.append(engine)

        def _blocked_build() -> bool:
            entered.set()
            engine._stop.wait(2.0)
            return False

        engine._build = _blocked_build
        return engine

    try:
        with (
            mock.patch.object(server, "_load_cfg", return_value={}),
            mock.patch.object(server, "_get_db", return_value=None),
            mock.patch.object(server, "_notify_session_boundary"),
            mock.patch(
                "agent.multimodal.hermes_glue.flatten_mm_config",
                return_value={"enabled": True, "memory_enabled": False},
            ),
            mock.patch(
                "agent.multimodal.watcher_engine.WatcherAgent",
                side_effect=watcher_factory,
            ),
            mock.patch.object(server, "_MM_WATCHER_STARTUP_TIMEOUT_SEC", 2.0),
            mock.patch.object(server, "_MM_WATCHER_STOP_TIMEOUT_SEC", 0.5),
            mock.patch.object(server, "_MM_WATCHER_DEPENDENCY_JOIN_SEC", 0.1),
        ):
            starter = threading.Thread(
                target=lambda: result.append(
                    server._maybe_start_live_watcher_agent(
                        "live-close", frame_buffer, memory_backend, session,
                    )
                )
            )
            starter.start()
            assert entered.wait(1.0)

            # Publication happens before start waits, so finalize cannot miss
            # the engine even though its provider build is still blocked.
            assert session["_mm_live_watcher_agent"] is instances[0]
            server._finalize_session(session, end_reason="test_close")
            starter.join(2.0)

        assert result == [None]
        assert session["_finalized"] is True
        assert len(instances) == 1
        assert instances[0].wait_stopped(1.0) is True
        assert instances[0]._thread is not None
        assert not instances[0]._thread.is_alive()
        key = server._mm_memory_backend_registry_key(
            session["session_key"], "live-close")
        assert _wait_until(
            lambda: server._MM_ACTIVE_WATCHERS.get(key) is not instances[0])
    finally:
        for engine in instances:
            engine.stop(timeout=1.0)
        _clear_guards(server)


def test_timed_out_watcher_stays_guarded_until_stopped_then_rebuilds():
    import tui_gateway.server as server
    from agent.multimodal.watcher_engine import WatcherAgent

    _clear_guards(server)
    entered = threading.Event()
    release = threading.Event()
    instances = []
    durable_id = f"watcher-timeout-{uuid.uuid4().hex}"
    frame_buffer = object()
    memory_backend = object()

    def watcher_factory(*args, **kwargs):
        engine = WatcherAgent(*args, **kwargs)
        instances.append(engine)
        if len(instances) == 1:
            def _blocked_build() -> bool:
                entered.set()
                # Simulate a provider constructor that cannot observe stop until
                # it returns. The registry must guard this whole interval.
                release.wait(2.0)
                return False

            engine._build = _blocked_build
        else:
            def _minimal_build() -> bool:
                engine.cfg = SimpleNamespace(
                    query_worker_max_concurrency=1,
                    query_worker_max_pending=0,
                )
                return True

            engine._build = _minimal_build
        return engine

    first_session = {"session_key": durable_id, "history": []}
    duplicate_session = {"session_key": durable_id, "history": []}
    rebuilt_session = {"session_key": durable_id, "history": []}
    try:
        with (
            mock.patch.object(server, "_load_cfg", return_value={}),
            mock.patch.object(server, "_get_db", return_value=None),
            mock.patch.object(server, "_notify_session_boundary"),
            mock.patch(
                "agent.multimodal.hermes_glue.flatten_mm_config",
                return_value={"enabled": True, "memory_enabled": False},
            ),
            mock.patch(
                "agent.multimodal.watcher_engine.WatcherAgent",
                side_effect=watcher_factory,
            ),
            mock.patch(
                "agent.multimodal.watch_file.reconcile_stale",
                return_value=0,
            ),
            mock.patch.object(server, "_MM_WATCHER_STARTUP_TIMEOUT_SEC", 0.03),
            mock.patch.object(server, "_MM_WATCHER_STOP_TIMEOUT_SEC", 0.03),
            mock.patch.object(server, "_MM_WATCHER_DEPENDENCY_JOIN_SEC", 0.03),
        ):
            assert server._maybe_start_live_watcher_agent(
                "live-1", frame_buffer, memory_backend, first_session,
            ) is None
            assert entered.is_set()
            first = instances[0]
            assert first_session["_mm_live_watcher_agent"] is first
            assert first._thread is not None and first._thread.is_alive()
            assert len(instances) == 1

            # A reopened session with callbacks bound to a different sid is not
            # allowed to reuse or overlap the timed-out owner.
            assert server._maybe_start_live_watcher_agent(
                "live-2", frame_buffer, memory_backend, duplicate_session,
            ) is None
            assert len(instances) == 1
            assert sum(
                int(item._thread is not None and item._thread.is_alive())
                for item in instances
            ) == 1

            server._finalize_session(first_session, end_reason="test_close")
            assert first._thread is not None and first._thread.is_alive()
            assert len(instances) == 1

            release.set()
            assert first.wait_stopped(1.0) is True

            ready = server._maybe_start_live_watcher_agent(
                "live-3", frame_buffer, memory_backend, rebuilt_session,
            )
            assert ready is instances[1]
            assert rebuilt_session["_mm_live_watcher_agent"] is ready
            assert len(instances) == 2
            assert sum(
                int(item._thread is not None and item._thread.is_alive())
                for item in instances
            ) == 1
    finally:
        release.set()
        for engine in instances:
            engine.stop(timeout=1.0)
        _clear_guards(server)


def test_finalize_defers_memory_stop_until_watcher_really_stops():
    import tui_gateway.server as server

    callbacks = []

    class _SlowWatcher:
        is_stopped = False

        def stop(self, timeout=0.0):
            return False

        def add_stopped_callback(self, callback):
            callbacks.append(callback)

    class _Memory:
        def __init__(self):
            self.stop_calls = 0

        def stop(self, timeout=0.0):
            self.stop_calls += 1
            return True

    watcher = _SlowWatcher()
    memory = _Memory()
    session = {
        "session_key": f"watcher-deferred-{uuid.uuid4().hex}",
        "history": [],
        "_mm_live_watcher_agent": watcher,
        "_mm_memory_backend": memory,
    }

    with (
        mock.patch.object(server, "_get_db", return_value=None),
        mock.patch.object(server, "_notify_session_boundary"),
    ):
        server._finalize_session(session, end_reason="test_close")

    assert memory.stop_calls == 0
    assert len(callbacks) == 1
    watcher.is_stopped = True
    callbacks[0](watcher)
    assert memory.stop_calls == 1
