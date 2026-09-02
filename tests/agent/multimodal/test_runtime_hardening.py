from __future__ import annotations

import asyncio
import base64
import io
import struct
import threading
import time
from types import SimpleNamespace

from PIL import Image

from agent.multimodal._memory import Frame, FrameBuffer, SearchFact, SearchFactStore
from agent.multimodal.memory_backend import MemoryBackend, _pcm16_signal_metrics
from agent.multimodal.watcher_engine import BackendRecallProxy, WatcherAgent
from tui_gateway.server import _audio_container_signature, _resolve_env_audio_window


def _jpeg_b64(color=(80, 120, 160)) -> str:
    image = Image.new("RGB", (32, 24), color)
    for x in range(image.width):
        shade = max(0, 255 - x * 6)
        for y in range(image.height):
            image.putpixel((x, y), (shade, color[1], color[2]))
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=90)
    return base64.b64encode(out.getvalue()).decode("ascii")


def _frame_cfg():
    return SimpleNamespace(
        buffer_seconds=120,
        buffer_capture_fps=2,
        framebuffer_dhash_threshold_init=6,
        framebuffer_dhash_threshold_min=2,
        framebuffer_dhash_threshold_max=20,
    )


def test_monitor_raw_buffer_keeps_frames_dropped_by_long_term_dedup():
    buf = FrameBuffer(_frame_cfg())
    encoded = _jpeg_b64()

    buf.push(Frame(ts=1.0, jpeg_b64=encoded, source_type="screen"))
    buf.push(Frame(ts=1.5, jpeg_b64=encoded, source_type="screen"))

    assert buf.size == 1
    assert buf.monitor_size == 2
    assert [frame.ts for frame in buf.monitor_all_after(0.0)] == [1.0, 1.5]


def test_search_fact_store_preserves_provenance_bounds_and_ttl():
    cfg = SimpleNamespace(
        search_facts_max=2,
        search_fact_ttl_sec=60.0,
        search_fact_value_max_chars=4000,
    )
    store = SearchFactStore(cfg)

    def fact(query: str, value: str, fetched_at: float) -> SearchFact:
        return SearchFact(
            key=SearchFactStore.normalize_query(query),
            query=query,
            value=value,
            source_tool="text_search",
            source_urls=("https://example.test/source",),
            fetched_at=fetched_at,
            expires_at=200.0,
            confidence=0.8,
        )

    snapshot = store.upsert_many([
        fact("Alpha?", "A", 101.0),
        fact("Beta", "B", 102.0),
        fact("Gamma", "C", 103.0),
    ], now=100.0)

    assert list(snapshot.facts) == ["beta", "gamma"]
    assert store.get_by_query("  GAMMA？ ", now=150.0).value == "C"
    assert snapshot.facts["gamma"].source_urls == (
        "https://example.test/source",)
    assert store.snapshot(now=201.0).facts == {}


class _LoopThread:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.ready.set()
        self.loop.run_forever()
        self.loop.close()

    def __enter__(self):
        self.thread.start()
        assert self.ready.wait(2.0)
        return self

    def __exit__(self, *_exc):
        if not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=2.0)
        assert not self.thread.is_alive()


def test_backend_recall_proxy_runs_recall_and_progress_on_owning_loops():
    observed = {}

    class Recall:
        async def run(self, **kwargs):
            observed["recall_loop"] = asyncio.get_running_loop()
            await kwargs["on_progress"]({"phase": "tool_obs"})
            return "done"

    with _LoopThread() as backend_runtime:
        backend = SimpleNamespace(
            _loop=backend_runtime.loop,
            _stop=threading.Event(),
            is_ready=True,
            recall_agent=Recall(),
        )
        proxy = BackendRecallProxy(backend)

        async def caller():
            caller_loop = asyncio.get_running_loop()

            async def progress(event):
                observed["progress_loop"] = asyncio.get_running_loop()
                observed["event"] = event

            result = await proxy.run(
                brief="query", user_text="query", ask_ts=1.0,
                initial_calls=[], on_progress=progress)
            return caller_loop, result

        caller_loop, result = asyncio.run(caller())

    assert result == "done"
    assert observed["recall_loop"] is backend_runtime.loop
    assert observed["progress_loop"] is caller_loop
    assert observed["event"] == {"phase": "tool_obs"}


class _MemStub:
    db_path = "/tmp/runtime-hardening.sqlite"

    def cleanup(self):
        return None

    def tokens_txt_path(self):
        return ""


def test_memory_backend_publishes_ready_and_stops_cleanly():
    backend = MemoryBackend(object())

    def build():
        backend.cfg = SimpleNamespace()
        backend.mem = _MemStub()
        backend.recall_agent = SimpleNamespace(llm_channel_lock=None)
        return True

    backend._build = build
    assert backend.start(offline=True, timeout=1.0) is True
    assert backend.state == backend.STATE_READY
    assert backend.healthy is True
    assert backend.stop(timeout=2.0) is True
    assert backend.state == backend.STATE_STOPPED
    assert backend.is_stopped is True


def test_query_admission_limits_running_and_pending_work():
    release = threading.Event()
    started = threading.Event()

    def recall(*_args, **_kwargs):
        started.set()
        assert release.wait(2.0)
        return {"ok": True}

    # Use the production object shape: __init__ owns set-based query state and
    # the semaphore is created on the Watcher event loop.  The previous test
    # injected integer counters + threading.BoundedSemaphore and therefore
    # validated an object that can never exist in production.
    with _LoopThread() as watcher_runtime:
        engine = WatcherAgent(object())
        engine._loop = watcher_runtime.loop
        engine._thread = watcher_runtime.thread
        engine._query_max_concurrency = 1
        engine._query_max_pending = 1

        async def _make_semaphore():
            return asyncio.Semaphore(1)

        engine._query_semaphore = asyncio.run_coroutine_threadsafe(
            _make_semaphore(), watcher_runtime.loop).result(timeout=1.0)
        engine._recall_memory_unbounded = recall
        assert isinstance(engine._query_running, set)
        assert isinstance(engine._query_pending, set)
        assert isinstance(engine._query_semaphore, asyncio.Semaphore)

        results = []
        first = threading.Thread(
            target=lambda: results.append(
                engine.recall_memory("first", timeout=2.0)))
        second = threading.Thread(
            target=lambda: results.append(
                engine.recall_memory("second", timeout=2.0)))
        first.start()
        assert started.wait(1.0)
        second.start()

        deadline = time.time() + 1.0
        while time.time() < deadline:
            with engine._query_lock:
                if len(engine._query_pending) == 1:
                    break
            time.sleep(0.01)
        rejected = engine.recall_memory("third", timeout=0.1)
        assert rejected["queue_full"] is True

        release.set()
        first.join(2.0)
        second.join(2.0)
        assert not first.is_alive() and not second.is_alive()
        assert len(results) == 2 and all(item["ok"] for item in results)
        with engine._query_lock:
            assert engine._query_running == set()
            assert engine._query_pending == set()


def test_audio_diagnostics_detect_headers_silence_and_time_mapping():
    assert _audio_container_signature(b"\x1a\x45\xdf\xa3payload", "audio/webm") == (
        "webm_ebml", True)
    assert _audio_container_signature(b"\x1f\x43\xb6\x75payload", "audio/webm") == (
        "webm_cluster", False)

    silence = struct.pack("<" + "h" * 16, *([0] * 16))
    signal = _pcm16_signal_metrics(silence, sample_rate=8)
    assert signal["rms"] == 0.0
    assert signal["duration_sec"] == 2.0

    start, end, duration = _resolve_env_audio_window(
        server_end_ts=100.0,
        client_start_ts=3.0,
        client_end_ts=8.0,
        client_duration_sec=5.0,
        default_duration_sec=5.0,
    )
    assert (start, end, duration) == (95.0, 100.0, 5.0)
