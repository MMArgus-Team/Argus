"""Focused contracts for the resident multimodal memory backend lifecycle."""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from types import SimpleNamespace


class _MemStub:
    db_path = "/tmp/lifecycle-test.sqlite"

    def cleanup(self) -> None:
        return None

    def tokens_txt_path(self) -> str:
        return ""


def _install_minimal_build(backend) -> None:
    """Avoid provider/store construction while exercising the real loop."""

    def _build() -> bool:
        backend.cfg = SimpleNamespace()
        backend.mem = _MemStub()
        backend.recall_agent = SimpleNamespace(recall_limiter=None)
        return True

    backend._build = _build


def test_start_with_timeout_publishes_ready_only_after_runtime_init():
    from agent.multimodal.memory_backend import MemoryBackend

    backend = MemoryBackend(object(), session_id="session-a")
    _install_minimal_build(backend)

    assert backend.start(offline=True, timeout=1.0) is True
    assert backend.state == backend.STATE_READY
    assert backend.is_ready is True
    assert backend.healthy is True
    assert backend._recall_limiter is not None
    assert backend.recall_agent.recall_limiter is backend._recall_limiter

    assert backend.stop(timeout=1.0) is True
    assert backend.state == backend.STATE_STOPPED
    assert backend.is_stopped is True
    # Stop is deliberately idempotent.
    assert backend.stop(timeout=0.0) is True


def test_startup_failure_is_observable_and_never_reports_ready():
    from agent.multimodal.memory_backend import MemoryBackend

    backend = MemoryBackend(object(), session_id="session-b")
    failure = RuntimeError("provider resolution failed")

    def _fail_build() -> bool:
        backend._startup_error = failure
        return False

    backend._build = _fail_build

    assert backend.start(timeout=1.0) is False
    assert backend.state == backend.STATE_FAILED
    assert backend.is_failed is True
    assert backend.is_ready is False
    assert backend.startup_error is failure
    assert backend.stop(timeout=1.0) is True
    assert backend.is_stopped is True


def test_recall_failure_preserves_structured_error_trace():
    """The real backend bridge must not collapse a Recall error into a miss."""
    from agent.multimodal.memory_backend import MemoryBackend

    loop = asyncio.new_event_loop()
    loop_ready = threading.Event()

    def _run_loop():
        asyncio.set_event_loop(loop)
        loop_ready.set()
        loop.run_forever()

    thread = threading.Thread(target=_run_loop, daemon=True)
    thread.start()
    assert loop_ready.wait(1.0)

    class _FailingRecall:
        async def run(self, *, on_progress, **_kwargs):
            await on_progress({
                "phase": "error",
                "stage": "decision",
                "round": 0,
                "model": "gpt-5.6-luna",
                "error": "HTTP 400 Unknown parameter: top_k",
                "elapsed_sec": 4.2,
            })
            raise RuntimeError("Recall decision failed")

    emitted = []
    backend = MemoryBackend(
        SimpleNamespace(latest_ts=12.5, current_source_type="screen"),
        emit_cb=lambda event, payload: emitted.append((event, payload)),
    )
    backend._loop = loop
    backend._state = backend.STATE_READY
    backend.recall_agent = _FailingRecall()

    try:
        result = backend.recall("店主找谁探店", timeout=2.0)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2.0)
        loop.close()

    assert result["ok"] is False
    assert "Recall decision failed" in result["error"]
    assert result["rounds"] == 1
    assert result["elapsed_sec"] == 4.2
    assert result["recall_trace"] == [{
        "phase": "error",
        "round": 0,
        "stage": "decision",
        "error": "HTTP 400 Unknown parameter: top_k",
        "elapsed_sec": 4.2,
    }]
    assert emitted[-1][0] == "multimodal.trajectory"
    assert emitted[-1][1]["phase"] == "error"
    assert emitted[-1][1]["worker"] == "RecallWorker"


def test_recall_fast_table_trace_preserves_tool_call_and_result():
    from agent.multimodal.memory_backend import MemoryBackend

    loop = asyncio.new_event_loop()
    loop_ready = threading.Event()

    def _run_loop():
        asyncio.set_event_loop(loop)
        loop_ready.set()
        loop.run_forever()

    thread = threading.Thread(target=_run_loop, daemon=True)
    thread.start()
    assert loop_ready.wait(1.0)

    fast_event = {
        "phase": "fast_table",
        "tool_name": "search_screen_text",
        "args": {"query": "表 2 Argus Score", "limit": 8},
        "query": "表 2 Argus Score",
        "obs_len": 80,
        "obs_summary": "[00:42-00:45] Argus | 91.2",
        "findings_len": 120,
        "findings_preview": "已命中表 2：Argus | 91.2",
        "elapsed_sec": 0.12,
        "frame_ids": ["f_1234567890"],
        "evidence_segments": [{
            "kind": "screen",
            "t_start": 42.0,
            "t_end": 45.0,
            "frame_ids": ["f_1234567890"],
        }],
    }
    tool_event = {
        "phase": "tool_obs",
        "round": 0,
        "parallel_elapsed_sec": 0.2,
        "observations": [{
            "name": "search_entity",
            "args": {"query": "Argus", "top_k": 3},
            "obs_len": 64,
            "elapsed_sec": 0.08,
            "obs_summary": "entity Argus first=00:42 last=00:45",
            "frame_ids": ["f_1234567890"],
            "evidence_segments": [{
                "kind": "memory",
                "t_start": 42.0,
                "t_end": 45.0,
                "frame_ids": ["f_1234567890"],
            }],
        }],
    }

    class _FastRecall:
        async def run(self, *, on_progress, **_kwargs):
            await on_progress(tool_event)
            await on_progress(fast_event)
            return SimpleNamespace(
                findings="已命中表 2：Argus | 91.2",
                clues=["Argus | 91.2"],
                frame_ids=["f_1234567890"],
                rounds=0,
                elapsed_sec=0.12,
            )

    backend = MemoryBackend(
        SimpleNamespace(latest_ts=60.0, current_source_type="screen"),
    )
    backend._loop = loop
    backend._state = backend.STATE_READY
    backend.recall_agent = _FastRecall()

    try:
        result = backend.recall("表 2 里 Argus 的 Score", timeout=2.0)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2.0)
        loop.close()

    assert result["ok"] is True
    assert result["recall_trace"] == [{
        "phase": "tool_obs",
        "round": 0,
        "tools": [{
            "name": "search_entity",
            "args": {"query": "Argus", "top_k": 3},
            "obs_len": 64,
            "elapsed_sec": 0.08,
            "obs_summary": "entity Argus first=00:42 last=00:45",
            "frame_ids": ["f_1234567890"],
            "evidence_segments": [{
                "kind": "memory",
                "t_start": 42.0,
                "t_end": 45.0,
                "frame_ids": ["f_1234567890"],
            }],
        }],
        "parallel_elapsed_sec": 0.2,
    }, {
        "phase": "fast_table",
        "tool_name": "search_screen_text",
        "args": {"query": "表 2 Argus Score", "limit": 8},
        "query": "表 2 Argus Score",
        "obs_len": 80,
        "obs_summary": "[00:42-00:45] Argus | 91.2",
        "findings_len": 120,
        "findings_preview": "已命中表 2：Argus | 91.2",
        "elapsed_sec": 0.12,
        "frame_ids": ["f_1234567890"],
        "evidence_segments": [{
            "kind": "screen",
            "t_start": 42.0,
            "t_end": 45.0,
            "frame_ids": ["f_1234567890"],
        }],
    }]


def test_stop_cannot_join_memory_thread_before_start(monkeypatch):
    from agent.multimodal import memory_backend as memory_backend_module
    from agent.multimodal.memory_backend import MemoryBackend

    real_thread = threading.Thread
    start_entered = threading.Event()
    allow_start = threading.Event()

    class _SlowStartThread(real_thread):
        def start(self):
            start_entered.set()
            assert allow_start.wait(1.0)
            return super().start()

    monkeypatch.setattr(
        memory_backend_module.threading, "Thread", _SlowStartThread)
    backend = MemoryBackend(object(), session_id="start-stop-race")
    _install_minimal_build(backend)
    results = {}
    errors = []

    def _start():
        try:
            results["start"] = backend.start(offline=True, timeout=1.0)
        except Exception as exc:  # pragma: no cover - assertion below
            errors.append(exc)

    def _stop():
        try:
            results["stop"] = backend.stop(timeout=1.0)
        except Exception as exc:  # pragma: no cover - assertion below
            errors.append(exc)

    starter = real_thread(target=_start)
    starter.start()
    assert start_entered.wait(1.0)
    stopper = real_thread(target=_stop)
    stopper.start()
    time.sleep(0.03)
    # stop() is waiting on the lifecycle lock, not joining a never-started
    # Thread object.
    assert stopper.is_alive()
    allow_start.set()
    starter.join(2.0)
    stopper.join(2.0)

    assert errors == []
    assert results.get("stop") is True
    assert backend.wait_stopped(1.0) is True


def test_memory_thread_start_failure_is_terminal_and_join_safe(monkeypatch):
    from agent.multimodal import memory_backend as memory_backend_module
    from agent.multimodal.memory_backend import MemoryBackend

    class _FailingThread:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            raise RuntimeError("thread start failed")

    monkeypatch.setattr(
        memory_backend_module.threading, "Thread", _FailingThread)
    backend = MemoryBackend(object(), session_id="start-failure")
    stopped = []
    backend.add_stopped_callback(stopped.append)

    assert backend.start(timeout=1.0) is False
    assert backend.is_failed is True
    assert backend.is_stopped is True
    assert backend._thread is None
    assert stopped == [backend]
    assert backend.stop(timeout=0.0) is True


def test_submodule_client_marks_only_dedicated_endpoint_owned(monkeypatch):
    from agent.multimodal import hermes_glue

    shared = SimpleNamespace()
    resolved = hermes_glue.build_submodule_client(
        provider="", base_url="", api_key="", model="",
        resolve_main=lambda: (shared, "shared-model"), label="recall",
    )
    assert resolved == (shared, "shared-model")
    assert not hasattr(shared, "_hermes_submodule_owned")

    class _FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(
        sys.modules, "openai", SimpleNamespace(AsyncOpenAI=_FakeAsyncOpenAI))
    monkeypatch.setattr(
        hermes_glue, "_submodule_http_client", lambda *_args: None)
    monkeypatch.setattr(
        hermes_glue, "wrap_kimi_client", lambda client, **_kwargs: client)

    dedicated, model = hermes_glue.build_submodule_client(
        provider="custom", base_url="https://recall.invalid/v1",
        api_key="test", model="recall-model",
        resolve_main=lambda: (shared, "shared-model"), label="recall",
    )
    assert model == "recall-model"
    assert dedicated._hermes_submodule_owned is True


def test_owned_llm_cleanup_is_identity_deduped_and_idempotent():
    from agent.multimodal.memory_backend import MemoryBackend

    class _RawClient:
        def __init__(self, *, owned=False):
            self.close_calls = 0
            if owned:
                self._hermes_submodule_owned = True

        async def close(self):
            self.close_calls += 1

    class _MemoryAdapter:
        def __init__(self, raw):
            self.client = raw
            self.close_calls = 0

        async def aclose(self):
            self.close_calls += 1

    backend = MemoryBackend.__new__(MemoryBackend)
    backend._closed_llm_client_ids = set()
    recall = _RawClient(owned=True)
    memory = _MemoryAdapter(_RawClient())
    reviewer = _MemoryAdapter(_RawClient())
    backend.recall_agent = SimpleNamespace(client=recall)
    backend.memory_client = memory
    backend.reviewer_client = reviewer

    asyncio.run(backend._close_owned_llm_clients())
    asyncio.run(backend._close_owned_llm_clients())

    assert recall.close_calls == 1
    assert memory.close_calls == 1
    assert reviewer.close_calls == 1


def test_startup_build_failure_closes_created_clients_once():
    from agent.multimodal.memory_backend import MemoryBackend

    class _Adapter:
        def __init__(self):
            self.close_calls = 0

        async def aclose(self):
            self.close_calls += 1

    class _RecallClient:
        _hermes_submodule_owned = True

        def __init__(self):
            self.close_calls = 0

        async def close(self):
            self.close_calls += 1

    backend = MemoryBackend(object(), session_id="session-build-failure")
    shared_adapter = _Adapter()
    recall = _RecallClient()

    def _fail_after_client_build() -> bool:
        backend.mem = _MemStub()
        backend.memory_client = shared_adapter
        # Writer and Reviewer intentionally reuse one adapter.
        backend.reviewer_client = shared_adapter
        backend.recall_agent = SimpleNamespace(client=recall)
        backend._startup_error = RuntimeError("later build step failed")
        return False

    backend._build = _fail_after_client_build

    assert backend.start(timeout=1.0) is False
    assert backend._stopped.wait(1.0) is True
    assert shared_adapter.close_calls == 1
    assert recall.close_calls == 1


def test_default_database_identity_is_unique_even_in_same_second(
    tmp_path, monkeypatch,
):
    import hermes_constants
    from agent.multimodal import hermes_glue

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(
        hermes_glue.time, "strftime", lambda *_args, **_kwargs: "20260807_120000")

    a = hermes_glue.build_config(
        {"multimodal": {}}, session_id="durable-session")
    b = hermes_glue.build_config(
        {"multimodal": {}}, session_id="durable-session")

    assert a.mem_db_path != b.mem_db_path
    assert a.mem_db_path.endswith(".sqlite")
    assert b.mem_db_path.endswith(".sqlite")
    assert "durable-session" not in a.mem_db_path
    assert "20260807_120000_" in a.mem_db_path

    # A caller may intentionally precompute one runtime identity. Reusing it
    # is stable for that backend, while a different durable session still gets
    # a distinct privacy-preserving namespace.
    same_runtime_a = hermes_glue.build_config(
        {"multimodal": {}}, session_id="session-a", runtime_id="run-fixed")
    same_runtime_a_again = hermes_glue.build_config(
        {"multimodal": {}}, session_id="session-a", runtime_id="run-fixed")
    same_runtime_b = hermes_glue.build_config(
        {"multimodal": {}}, session_id="session-b", runtime_id="run-fixed")
    assert same_runtime_a.mem_db_path == same_runtime_a_again.mem_db_path
    assert same_runtime_a.mem_db_path != same_runtime_b.mem_db_path


def test_explicit_database_path_still_wins(tmp_path, monkeypatch):
    import hermes_constants
    from agent.multimodal.hermes_glue import build_config

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    explicit = tmp_path / "operator-selected.sqlite"
    cfg = build_config({
        "multimodal": {"mem_db_path": str(explicit)},
    }, session_id="ignored-for-explicit-path")

    assert cfg.mem_db_path == str(explicit)


def test_writer_recall_channel_lock_only_for_the_same_endpoint():
    from agent.multimodal.memory_backend import _clients_share_llm_channel

    cfg = SimpleNamespace(
        memory_base_url="https://memory.example/v1",
        recall_base_url="https://recall.example/v1",
        memory_provider="custom",
    )
    memory = SimpleNamespace(
        client=SimpleNamespace(base_url="https://memory.example/v1/"))
    recall = SimpleNamespace(base_url="https://recall.example/v1")
    assert _clients_share_llm_channel(memory, recall, cfg) is False

    recall.base_url = "https://memory.example/v1"
    assert _clients_share_llm_channel(memory, recall, cfg) is True

    # With no dedicated role endpoints both clients follow the main Hermes
    # endpoint even when a provider adapter does not expose ``base_url``.
    hidden_cfg = SimpleNamespace(
        memory_base_url="", recall_base_url="", memory_provider="custom")
    assert _clients_share_llm_channel(object(), object(), hidden_cfg) is True
