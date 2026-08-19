"""Focused QueryWorker admission and Watcher lifecycle regressions."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest import mock

from agent.multimodal.watcher_engine import WatcherAgent


class _FrameBuffer:
    latest_ts = 1.0

    @staticmethod
    def latest(_count):
        return []


class _BlockingResponder:
    def __init__(self):
        self.release = threading.Event()
        self._lock = threading.Lock()
        self.started = 0
        self.running = 0
        self.max_running = 0

    async def _spawn_delegation(
        self, *, task_instruction, sink, on_event, **_kwargs,
    ):
        async def _drive():
            with self._lock:
                self.started += 1
                self.running += 1
                self.max_running = max(self.max_running, self.running)
            try:
                while not self.release.is_set():
                    await asyncio.sleep(0.005)
                answer = f"{task_instruction}-answer"
                await sink(answer)
                await on_event({
                    "type": "answer_ready",
                    "answer_full": answer,
                })
            finally:
                with self._lock:
                    self.running -= 1

        return asyncio.create_task(_drive())


def _wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


def _engine(*, concurrency=2, pending=8, responder=None, on_complete=None):
    worker = responder or _BlockingResponder()
    engine = WatcherAgent(
        _FrameBuffer(), on_query_complete=on_complete)

    def _build():
        engine.cfg = SimpleNamespace(
            cont_recent_frames=3,
            query_worker_max_concurrency=concurrency,
            query_worker_max_pending=pending,
        )
        engine.responder = worker
        return True

    engine._build = _build
    return engine, worker


def test_same_parent_is_atomically_deduplicated_across_submit_threads():
    engine, responder = _engine()
    assert engine.start(timeout=2.0)
    barrier = threading.Barrier(8)
    results = []
    results_lock = threading.Lock()

    def _submit(index):
        barrier.wait()
        result = engine.submit_query_async(
            "same question",
            task_id=f"qry_{index}",
            parent_user_message_id="turn_same",
        )
        with results_lock:
            results.append(result)

    threads = [threading.Thread(target=_submit, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert len(results) == 8
    assert len(set(results)) == 1
    assert results[0]
    with engine._query_lock:
        assert len(engine._active_queries) == 1
        assert engine._query_by_parent == {"turn_same": results[0]}

    responder.release.set()
    assert _wait_until(lambda: not engine._active_queries)
    assert engine.stop(timeout=2.0)


def test_query_execution_and_pending_queue_are_bounded_then_cleaned():
    completed = []
    completed_lock = threading.Lock()

    def _complete(*row):
        with completed_lock:
            completed.append(row)

    engine, responder = _engine(
        concurrency=2, pending=8, on_complete=_complete)
    assert engine.start(timeout=2.0)

    # Two execution slots + eight waiting slots are admitted.  The eleventh
    # distinct parent is rejected instead of creating another asyncio task.
    accepted = [
        engine.submit_query_async(
            f"Q{i}", task_id=f"qry_{i}",
            parent_user_message_id=f"turn_{i}",
        )
        for i in range(10)
    ]
    assert all(accepted)
    assert engine.submit_query_async(
        "overflow", task_id="qry_overflow",
        parent_user_message_id="turn_overflow",
    ) == ""

    assert _wait_until(lambda: responder.started == 2)
    assert responder.max_running == 2
    with engine._query_lock:
        assert len(engine._query_running) == 2
        assert len(engine._query_pending) == 8

    responder.release.set()
    assert _wait_until(lambda: len(completed) == 10)
    assert responder.max_running <= 2
    assert _wait_until(lambda: not engine._active_queries)
    with engine._query_lock:
        assert not engine._query_by_parent
        assert not engine._query_pending
        assert not engine._query_running
    assert engine.stop(timeout=2.0)


def test_stop_rejects_new_queries_cancels_active_work_and_joins():
    completions = []
    emissions = []
    engine, responder = _engine(
        concurrency=1, pending=2,
        on_complete=lambda *row: completions.append(row),
    )
    engine._emit_cb = lambda event, payload: emissions.append((event, payload))
    assert engine.start(timeout=2.0)
    assert engine.submit_query_async(
        "running", task_id="qry_run", parent_user_message_id="turn_run")
    assert engine.submit_query_async(
        "queued", task_id="qry_wait", parent_user_message_id="turn_wait")
    assert _wait_until(lambda: responder.started == 1)

    assert engine.stop(timeout=2.0)
    assert engine._thread is not None
    assert not engine._thread.is_alive()
    assert engine.submit_query_async(
        "late", task_id="qry_late", parent_user_message_id="turn_late") == ""
    with engine._query_lock:
        assert not engine._active_queries
        assert not engine._query_by_parent
        assert not engine._query_pending
        assert not engine._query_running
    assert completions == []
    assert not any(event == "message.complete" for event, _ in emissions)


def test_start_reports_build_failure_and_query_runtime_stays_unavailable():
    engine = WatcherAgent(_FrameBuffer())
    engine._build = lambda: False

    assert not engine.start(timeout=1.0)
    assert not engine._healthy
    assert engine.submit_query_async(
        "question", task_id="qry_1", parent_user_message_id="turn_1") == ""
    assert engine.stop(timeout=1.0)


def test_watcher_stop_closes_only_owned_submodule_clients():
    class _Client:
        def __init__(self, owned):
            self.close_calls = 0
            if owned:
                self._hermes_submodule_owned = True

        async def close(self):
            self.close_calls += 1

    worker_client = _Client(owned=True)
    recall_client = _Client(owned=True)
    shared_client = _Client(owned=False)
    engine = WatcherAgent(_FrameBuffer())

    def _build():
        engine.cfg = SimpleNamespace(
            query_worker_max_concurrency=1,
            query_worker_max_pending=0,
        )
        engine.responder = _BlockingResponder()
        engine.client = worker_client
        engine.recall_agent = SimpleNamespace(client=recall_client)
        # A shared client is deliberately not reachable through either owned
        # role; keeping it here documents that ownership, not closeability,
        # controls teardown.
        engine._shared_test_client = shared_client
        return True

    engine._build = _build
    assert engine.start(timeout=2.0)
    assert engine.stop(timeout=2.0)
    assert worker_client.close_calls == 1
    assert recall_client.close_calls == 1
    assert shared_client.close_calls == 0


def test_stop_cannot_join_watcher_thread_before_start(monkeypatch):
    """Thread publication/start is atomic with stop's lifecycle transition."""
    from agent.multimodal import watcher_engine as watcher_engine_module

    real_thread = threading.Thread
    start_entered = threading.Event()
    allow_start = threading.Event()

    class _SlowStartThread(real_thread):
        def start(self):
            start_entered.set()
            assert allow_start.wait(1.0)
            return super().start()

    monkeypatch.setattr(
        watcher_engine_module.threading, "Thread", _SlowStartThread)
    engine, _responder = _engine(concurrency=1, pending=0)
    results = {}
    errors = []

    def _start():
        try:
            results["start"] = engine.start(timeout=1.0)
        except Exception as exc:  # pragma: no cover - assertion below
            errors.append(exc)

    def _stop():
        try:
            results["stop"] = engine.stop(timeout=1.0)
        except Exception as exc:  # pragma: no cover - assertion below
            errors.append(exc)

    starter = real_thread(target=_start)
    starter.start()
    assert start_entered.wait(1.0)
    stopper = real_thread(target=_stop)
    stopper.start()
    time.sleep(0.03)
    assert stopper.is_alive()

    allow_start.set()
    starter.join(2.0)
    stopper.join(2.0)

    assert errors == []
    assert results.get("stop") is True
    assert engine.wait_stopped(1.0) is True
    assert engine.state == engine.STATE_STOPPED


def test_watcher_thread_start_failure_is_terminal_and_join_safe(monkeypatch):
    from agent.multimodal import watcher_engine as watcher_engine_module

    class _FailingThread:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            raise RuntimeError("thread start failed")

    monkeypatch.setattr(
        watcher_engine_module.threading, "Thread", _FailingThread)
    engine = WatcherAgent(_FrameBuffer())
    stopped = []
    engine.add_stopped_callback(stopped.append)

    assert engine.start(timeout=1.0) is False
    assert engine.is_failed is True
    assert engine.is_stopped is True
    assert engine._thread is None
    assert stopped == [engine]
    assert engine.stop(timeout=0.0) is True


def test_build_failure_closes_pending_owned_clients_once():
    """Failure before RecallAgent publication still closes both transports."""
    class _Client:
        _hermes_submodule_owned = True

        def __init__(self):
            self.close_calls = 0

        async def close(self):
            self.close_calls += 1

    worker_client = _Client()
    recall_client = _Client()
    engine = WatcherAgent(_FrameBuffer())

    def _fail_after_clients_created():
        engine.client = worker_client
        engine._recall_client_pending = recall_client
        engine._startup_error = RuntimeError("RecallAgent constructor failed")
        return False

    engine._build = _fail_after_clients_created

    assert engine.start(timeout=1.0) is False
    assert engine.wait_stopped(1.0) is True
    assert engine.is_failed is True
    assert worker_client.close_calls == 1
    assert recall_client.close_calls == 1
    assert engine._recall_client_pending is None

    # Defensive/repeated close paths remain identity-idempotent.
    assert engine.stop(timeout=0.0) is True
    assert engine.stop(timeout=0.0) is True
    asyncio.run(engine._close_owned_llm_clients())
    assert worker_client.close_calls == 1
    assert recall_client.close_calls == 1


def test_post_build_startup_error_still_runs_full_teardown():
    """Semaphore/config initialization exceptions cannot bypass finally."""
    class _Client:
        _hermes_submodule_owned = True

        def __init__(self):
            self.close_calls = 0

        async def close(self):
            self.close_calls += 1

    client = _Client()
    engine = WatcherAgent(_FrameBuffer())

    def _build():
        engine.cfg = SimpleNamespace(
            query_worker_max_concurrency="not-an-int",
            query_worker_max_pending=0,
        )
        engine.client = client
        return True

    engine._build = _build
    assert engine.start(timeout=1.0) is False
    assert engine.wait_stopped(1.0) is True
    assert engine.is_failed is True
    assert client.close_calls == 1


def test_real_build_recall_constructor_failure_closes_factory_clients():
    """The raw Recall client is retained before RecallAgent can raise."""
    from agent.multimodal._config import Config

    class _Client:
        _hermes_submodule_owned = True

        def __init__(self):
            self.close_calls = 0

        async def close(self):
            self.close_calls += 1

    cfg = Config()
    cfg.model = "watcher-model"
    cfg._worker_model_explicit = True
    cfg.query_worker_max_concurrency = 1
    cfg.query_worker_max_pending = 0
    worker_client = _Client()
    recall_client = _Client()
    factory = mock.Mock()
    factory.worker_client.return_value = (worker_client, "watcher-model")
    factory.recall_client.return_value = (recall_client, "recall-model")
    engine = WatcherAgent(_FrameBuffer())
    mem = SimpleNamespace(db_path="watcher-build-failure.sqlite")

    with (
        mock.patch(
            "agent.multimodal.hermes_glue.build_config", return_value=cfg),
        mock.patch(
            "agent.multimodal.hermes_glue.HermesClientFactory",
            return_value=factory,
        ),
        mock.patch("agent.multimodal._memory.MemoryStore", return_value=mem),
        mock.patch("agent.multimodal._memory.SearchFactStore", return_value=object()),
        mock.patch("agent.multimodal._memory.ConversationLog", return_value=object()),
        mock.patch("agent.multimodal._memory.FrameStore", return_value=object()),
        mock.patch("agent.multimodal._memory.ScreenTextStore", return_value=object()),
        mock.patch("agent.multimodal._memory.ScreenTableStore", return_value=object()),
        mock.patch("agent.multimodal._memory.TaskStateStore", return_value=object()),
        mock.patch("agent.multimodal._workers.ToolBox", return_value=object()),
        mock.patch(
            "agent.multimodal._workers.RecallAgent",
            side_effect=RuntimeError("recall constructor exploded"),
        ),
        mock.patch("agent.multimodal._workers.WatcherWorker") as watcher_worker,
    ):
        assert engine.start(timeout=1.0) is False

    assert engine.wait_stopped(1.0) is True
    assert engine.is_failed is True
    assert worker_client.close_calls == 1
    assert recall_client.close_calls == 1
    assert engine._recall_client_pending is None
    watcher_worker.assert_not_called()
