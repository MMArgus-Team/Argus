"""Cross-loop ownership tests for WatcherAgent's backend Recall facade."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.multimodal._config import Config
from agent.multimodal.watcher_engine import BackendRecallProxy, WatcherAgent


class _LoopThread:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.ready.set()
        self.loop.run_forever()
        self.loop.close()

    def __enter__(self):
        self.thread.start()
        assert self.ready.wait(2.0)
        return self

    def __exit__(self, *_exc) -> None:
        if not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=2.0)
        assert not self.thread.is_alive()


def _backend(loop, recall_agent, **overrides):
    values = {
        "_loop": loop,
        "_stop": threading.Event(),
        "is_ready": True,
        "recall_agent": recall_agent,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_proxy_runs_recall_on_backend_loop_and_progress_on_watcher_loop():
    target_observed = {}
    progress_observed = {}
    expected_result = object()

    class _Recall:
        async def run(self, **kwargs):
            target_observed["loop"] = asyncio.get_running_loop()
            target_observed["thread"] = threading.get_ident()
            await kwargs["on_progress"]({"phase": "tool_obs", "round": 1})
            return expected_result

    with _LoopThread() as backend_runtime:
        proxy = BackendRecallProxy(
            _backend(backend_runtime.loop, _Recall()))

        async def _watcher_side():
            watcher_loop = asyncio.get_running_loop()
            watcher_thread = threading.get_ident()

            async def _progress(event):
                progress_observed["loop"] = asyncio.get_running_loop()
                progress_observed["thread"] = threading.get_ident()
                progress_observed["event"] = event

            result = await proxy.run(
                initial_calls=[], brief="b", user_text="u", ask_ts=1.0,
                on_progress=_progress)
            return watcher_loop, watcher_thread, result

        watcher_loop, watcher_thread, result = asyncio.run(_watcher_side())

    assert result is expected_result
    assert target_observed["loop"] is backend_runtime.loop
    assert target_observed["thread"] == backend_runtime.thread.ident
    assert progress_observed["loop"] is watcher_loop
    assert progress_observed["thread"] == watcher_thread
    assert progress_observed["event"]["phase"] == "tool_obs"


def test_proxy_cancellation_cancels_backend_recall_coroutine():
    started = threading.Event()
    cancelled = threading.Event()

    class _Recall:
        async def run(self, **_kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

    with _LoopThread() as backend_runtime:
        proxy = BackendRecallProxy(
            _backend(backend_runtime.loop, _Recall()))

        async def _watcher_side():
            task = asyncio.create_task(proxy.run(
                initial_calls=[], brief="b", user_text="u", ask_ts=1.0))
            assert await asyncio.to_thread(started.wait, 2.0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert await asyncio.to_thread(cancelled.wait, 2.0)

        asyncio.run(_watcher_side())


def _build_engine(memory_backend):
    engine = WatcherAgent.__new__(WatcherAgent)
    engine._hermes_cfg = {}
    engine.frame_buffer = object()
    engine._memory_backend = memory_backend
    return engine


def _ready_resource_backend():
    cfg = Config()
    cfg.model = "watcher-model"
    cfg._worker_model_explicit = True
    return _backend(
        object(),
        object(),
        cfg=cfg,
        mem=SimpleNamespace(db_path="shared.sqlite"),
        search_fact_store=object(),
        store=None,
        conversation=object(),
        frame_store=object(),
        screen_text_store=object(),
        screen_table_store=object(),
        task_state_store=object(),
    )


def test_build_reuses_one_complete_backend_bundle_and_installs_proxy():
    backend = _ready_resource_backend()
    engine = _build_engine(backend)
    factory = MagicMock()
    factory.worker_client.return_value = (object(), "watcher-model")

    with (
        patch("agent.multimodal.hermes_glue.build_config") as build_config,
        patch(
            "agent.multimodal.hermes_glue.HermesClientFactory",
            return_value=factory,
        ),
        patch("agent.multimodal._memory.MemoryStore") as memory_store,
        patch("agent.multimodal._workers.ToolBox", return_value=MagicMock()),
        patch("agent.multimodal._workers.RecallAgent") as recall_cls,
        patch("agent.multimodal._workers.WatcherWorker", return_value=MagicMock()),
    ):
        assert engine._build() is True

    build_config.assert_not_called()
    memory_store.assert_not_called()
    recall_cls.assert_not_called()
    factory.recall_client.assert_not_called()
    assert engine.cfg is not backend.cfg
    assert vars(engine.cfg) == vars(backend.cfg)
    assert engine.mem is backend.mem
    assert engine.search_fact_store is backend.search_fact_store
    assert engine.store is backend.search_fact_store
    assert engine.screen_table_store is backend.screen_table_store
    assert isinstance(engine.recall_agent, BackendRecallProxy)
    assert engine.recall_agent._memory_backend is backend


def test_incomplete_backend_fails_instead_of_creating_local_fallback():
    backend = _ready_resource_backend()
    backend.screen_table_store = None
    engine = _build_engine(backend)

    with (
        patch("agent.multimodal.hermes_glue.build_config") as build_config,
        patch("agent.multimodal._memory.MemoryStore") as memory_store,
        patch("agent.multimodal._workers.RecallAgent") as recall_cls,
    ):
        assert engine._build() is False

    build_config.assert_not_called()
    memory_store.assert_not_called()
    recall_cls.assert_not_called()


def test_unready_backend_fails_instead_of_creating_local_fallback():
    backend = _ready_resource_backend()
    backend.is_ready = False
    engine = _build_engine(backend)

    with (
        patch("agent.multimodal.hermes_glue.build_config") as build_config,
        patch("agent.multimodal._memory.MemoryStore") as memory_store,
        patch("agent.multimodal._workers.RecallAgent") as recall_cls,
    ):
        assert engine._build() is False

    build_config.assert_not_called()
    memory_store.assert_not_called()
    recall_cls.assert_not_called()
