"""Regression coverage for multimodal MemoryBackend lifecycle ownership."""

from __future__ import annotations

import asyncio
import threading
import uuid
from unittest import mock


def test_distinct_sessions_keep_independent_backends_for_same_profile():
    import tui_gateway.server as server

    class TrackingBackend:
        def __init__(self, frame_buffer, **kwargs):
            self.frame_buffer = frame_buffer
            self.session_id = kwargs["session_id"]
            self.is_ready = False
            self.is_stopped = False
            self.startup_error = None
            self.state = "new"
            self.stop_calls = 0
            self._stopped_callbacks = []

        def add_stopped_callback(self, callback):
            self._stopped_callbacks.append(callback)

        def start(self, timeout=None):
            self.is_ready = True
            self.state = "ready"
            return True

        def stop(self, timeout=None):
            self.stop_calls += 1
            self.is_ready = False
            self.is_stopped = True
            self.state = "stopped"
            for callback in list(self._stopped_callbacks):
                callback(self)
            return True

    durable_a = f"independent-a-{uuid.uuid4().hex}"
    durable_b = f"independent-b-{uuid.uuid4().hex}"
    buffer_a = object()
    buffer_b = object()
    session_a = {}
    session_b = {}
    instances = []

    def backend_factory(*args, **kwargs):
        backend = TrackingBackend(*args, **kwargs)
        instances.append(backend)
        return backend

    key_a = server._mm_memory_backend_registry_key(durable_a, "live-a")
    key_b = server._mm_memory_backend_registry_key(durable_b, "live-b")
    try:
        with (
            mock.patch.object(server, "_load_cfg", return_value={}),
            mock.patch(
                "agent.multimodal.hermes_glue.build_config",
                return_value=object(),
            ),
            mock.patch(
                "agent.multimodal.memory_backend.MemoryBackend",
                side_effect=backend_factory,
            ),
        ):
            backend_a = server._maybe_start_memory_backend(
                "live-a", durable_a, buffer_a, session=session_a,
            )
            backend_b = server._maybe_start_memory_backend(
                "live-b", durable_b, buffer_b, session=session_b,
            )

        assert backend_a is instances[0]
        assert backend_b is instances[1]
        assert backend_a is not backend_b
        assert session_a["_mm_memory_backend"] is backend_a
        assert session_b["_mm_memory_backend"] is backend_b
        assert backend_a.stop_calls == 0
        assert backend_a.is_ready is True
        with server._MM_ACTIVE_MEMORY_BACKENDS_LOCK:
            assert server._MM_ACTIVE_MEMORY_BACKENDS.get(key_a) is backend_a
            assert server._MM_ACTIVE_MEMORY_BACKENDS.get(key_b) is backend_b
    finally:
        for backend in instances:
            if not backend.is_stopped:
                backend.stop(timeout=1.0)
        with server._MM_ACTIVE_MEMORY_BACKENDS_LOCK:
            if server._MM_ACTIVE_MEMORY_BACKENDS.get(key_a) in instances:
                server._MM_ACTIVE_MEMORY_BACKENDS.pop(key_a, None)
            if server._MM_ACTIVE_MEMORY_BACKENDS.get(key_b) in instances:
                server._MM_ACTIVE_MEMORY_BACKENDS.pop(key_b, None)


def test_blocked_startup_stays_guarded_until_thread_stops_then_rebuilds():
    import tui_gateway.server as server
    from agent.multimodal.memory_backend import MemoryBackend

    entered = threading.Event()
    release = threading.Event()
    instances = []
    durable_id = f"orphan-guard-{uuid.uuid4().hex}"
    frame_buffer = object()

    def backend_factory(*args, **kwargs):
        backend = MemoryBackend(*args, **kwargs)
        instances.append(backend)

        if len(instances) == 1:
            def _blocked_build() -> bool:
                entered.set()
                release.wait(2.0)
                if backend._build_stop_requested("test blocking build"):
                    return False
                return True

            backend._build = _blocked_build
        else:
            backend._build = lambda: True

        async def _minimal_main() -> None:
            if not backend._mark_ready():
                return
            while not backend._stop.is_set():
                await asyncio.sleep(0.01)

        backend._main = _minimal_main
        return backend

    first_session = {}
    duplicate_session = {}
    rebuilt_session = {}
    try:
        with (
            mock.patch.object(server, "_load_cfg", return_value={}),
            mock.patch(
                "agent.multimodal.hermes_glue.flatten_mm_config",
                return_value={"memory_enabled": True},
            ),
            mock.patch(
                "agent.multimodal.hermes_glue.build_config",
                return_value=object(),
            ),
            mock.patch(
                "agent.multimodal.memory_backend.MemoryBackend",
                side_effect=backend_factory,
            ),
            mock.patch.object(server, "_MM_MEMORY_STARTUP_TIMEOUT_SEC", 0.03),
            mock.patch.object(server, "_MM_MEMORY_STOP_TIMEOUT_SEC", 0.03),
        ):
            # Startup and the follow-up stop both time out while _build is
            # blocked. The backend must remain reachable from the session and
            # guarded by durable identity.
            assert server._maybe_start_memory_backend(
                "live-1", durable_id, frame_buffer, session=first_session,
            ) is None
            assert entered.is_set()
            first = instances[0]
            assert first_session["_mm_memory_backend"] is first
            assert first._thread is not None and first._thread.is_alive()
            assert len(instances) == 1

            # A concurrent/reopened session cannot construct another resource
            # bundle while the old build thread is still alive.
            assert server._maybe_start_memory_backend(
                "live-2", durable_id, frame_buffer, session=duplicate_session,
            ) is None
            assert duplicate_session["_mm_memory_backend"] is first
            assert len(instances) == 1
            assert sum(
                int(item._thread is not None and item._thread.is_alive())
                for item in instances
            ) == 1

            # Closing the first session still finds the timed-out backend. A
            # bounded close is allowed to time out, but must not drop the guard.
            server._finalize_session(first_session, end_reason="test_close")
            assert first_session["_finalized"] is True
            assert first._thread is not None and first._thread.is_alive()
            assert len(instances) == 1

            # Once the blocked stage returns, _build observes stop, skips the
            # remainder, and the stopped callback removes the registry entry.
            release.set()
            assert first.wait_stopped(1.0) is True

            ready = server._maybe_start_memory_backend(
                "live-3", durable_id, frame_buffer, session=rebuilt_session,
            )
            assert ready is instances[1]
            assert rebuilt_session["_mm_memory_backend"] is ready
            assert len(instances) == 2
            assert sum(
                int(item._thread is not None and item._thread.is_alive())
                for item in instances
            ) == 1
    finally:
        release.set()
        for backend in instances:
            backend.stop(timeout=1.0)
        key = server._mm_memory_backend_registry_key(durable_id, "live-1")
        with server._MM_ACTIVE_MEMORY_BACKENDS_LOCK:
            if server._MM_ACTIVE_MEMORY_BACKENDS.get(key) in instances:
                server._MM_ACTIVE_MEMORY_BACKENDS.pop(key, None)
