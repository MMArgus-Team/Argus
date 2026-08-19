from __future__ import annotations

import asyncio
import json
import threading
import time
from types import SimpleNamespace

from agent.multimodal._memory import (
    ContextStore,
    SearchFact,
    SearchFactSnapshot,
    SearchFactStore,
    SharedContext,
)
from agent.multimodal._workers import (
    MEMORY_WRITER_SYSTEM,
    MemoryReviewer,
    MemoryWriter,
    ReactStep,
    WatcherWorker,
)


def _cfg(**overrides):
    values = {
        "search_facts_max": 8,
        "search_fact_ttl_sec": 60.0,
        "search_fact_value_max_chars": 4000,
        "react_max_rounds": 3,
        "react_search_tasks_max": 5,
        "cont_recent_frames": 12,
        "cont_now_frames": 4,
        "search_recent_frames": 8,
        "cont_recall_frames_max": 6,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _fact(query: str, value: str, *, fetched_at: float = 100.0,
          expires_at: float = 200.0) -> SearchFact:
    return SearchFact(
        key=SearchFactStore.normalize_query(query),
        query=query,
        value=value,
        source_tool="text_search:anysearch",
        source_urls=("https://example.test/source",),
        fetched_at=fetched_at,
        expires_at=expires_at,
        confidence=0.75,
    )


def test_search_fact_store_is_structured_atomic_bounded_and_expiring():
    store = SearchFactStore(_cfg(search_facts_max=2))
    notifications = []
    store.add_listener(notifications.append)

    snap = store.upsert_many([
        _fact("Alpha?", "A"),
        _fact("Beta", "B"),
        _fact("Gamma", "C"),
    ], now=100.0)

    # One batch is one version change, and the oldest entry is evicted.
    assert snap.version == 1
    assert list(snap.facts) == ["beta", "gamma"]
    assert snap.facts["gamma"].source_urls == (
        "https://example.test/source",)
    assert snap.to_dict()["facts"]["gamma"]["confidence"] == 0.75
    assert snap.display_values() == {"Beta": "B", "Gamma": "C"}
    assert [item.version for item in notifications] == [1]

    # Query normalization supports cache reuse across case/trailing punctuation.
    assert store.get_by_query("  GAMMA？ ", now=150.0) == snap.facts["gamma"]

    # Expired evidence disappears from both reads and snapshots.
    expired = store.snapshot(now=201.0)
    assert expired.facts == {}
    assert expired.version == 2
    assert [item.version for item in notifications] == [1, 2]


def test_search_fact_store_is_thread_safe_for_parallel_upserts_and_reads():
    store = SearchFactStore(_cfg(search_facts_max=16))
    errors = []

    def worker(worker_id: int) -> None:
        try:
            for idx in range(40):
                query = f"worker {worker_id} item {idx}"
                store.upsert_many([
                    _fact(query, f"value-{idx}", expires_at=10_000.0)
                ], now=100.0)
                store.snapshot(now=100.0)
        except Exception as exc:  # pragma: no cover - assertion reports details
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(idx,)) for idx in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(store.snapshot(now=100.0).facts) <= 16


def test_older_search_result_cannot_overwrite_newer_evidence():
    store = SearchFactStore(_cfg())
    newer = _fact("Current Price?", "new", fetched_at=120.0,
                  expires_at=300.0)
    older = _fact("current price", "old", fetched_at=110.0,
                  expires_at=300.0)

    first = store.upsert_many([newer], now=100.0)
    second = store.upsert_many([older], now=100.0)

    assert second.version == first.version
    assert store.get_by_query("CURRENT PRICE!", now=150.0).value == "new"


class _Conversation:
    def __init__(self):
        self.appended = []

    async def append(self, role, content, rel_ts=None):
        self.appended.append((role, content, rel_ts))

    def recent_n(self, n):
        return []


class _ToolBox:
    def __init__(self):
        self.calls = 0

    async def call(self, name, args, *, anchor=None, crop_progress_cb=None):
        self.calls += 1
        return (
            f"[text_search query={args['query']!r}]\n"
            "哈利法塔高 828 米。 URL: https://example.test/burj-khalifa"
        )


def _watcher(store: SearchFactStore, toolbox: _ToolBox) -> WatcherWorker:
    worker = WatcherWorker.__new__(WatcherWorker)
    worker.cfg = _cfg()
    worker.client = object()
    worker.buf = SimpleNamespace(
        size=0, latest_ts=100.0, latest=lambda n: [], latest_one=lambda: None)
    worker.mem = None
    worker.store = store
    worker.conversation = _Conversation()
    worker.frame_store = None
    worker.recorder = None
    worker.toolbox = toolbox
    worker.recall_agent = None
    worker.inflight = {}
    worker._front_lock = asyncio.Lock()
    return worker


async def _run_delegation(worker: WatcherWorker, steps, *, fallback=""):
    step_iter = iter(steps)
    events = []

    async def react_step(**kwargs):
        return next(step_iter)

    async def answer(**kwargs):
        return fallback, 0.0, len(fallback)

    async def sink(text):
        return None

    async def on_event(event):
        events.append(event)

    worker.react_step = react_step
    worker.answer = answer
    task = await worker._spawn_delegation(
        task_instruction="查哈利法塔高度",
        prelude="",
        sink=sink,
        on_event=on_event,
        ask_frames_override=[],
    )
    await task
    return events


def test_search_candidate_commits_after_answer_then_hits_normalized_cache():
    store = SearchFactStore(_cfg())
    toolbox = _ToolBox()
    worker = _watcher(store, toolbox)

    first_steps = [
        ReactStep(tool_calls=[{
            "name": "text_search",
            "args": {"query": "Burj Khalifa 最高楼？"},
            "anchor": "current",
        }]),
        ReactStep(answer="哈利法塔高 828 米。"),
    ]
    asyncio.run(_run_delegation(worker, first_steps))

    fact = store.get_by_query("burj khalifa 最高楼")
    assert fact is not None
    assert fact.value == "哈利法塔高 828 米。 URL: https://example.test/burj-khalifa"
    assert fact.source_urls == ("https://example.test/burj-khalifa",)
    assert fact.expires_at > fact.fetched_at
    assert toolbox.calls == 1
    prompt_block = worker._search_fact_prompt_block()
    assert "Burj Khalifa 最高楼？" in prompt_block
    assert "https://example.test/burj-khalifa" in prompt_block

    # Same normalized query is served from the session cache; no second search.
    second_steps = [
        ReactStep(tool_calls=[{
            "name": "text_search",
            "args": {"query": "burj khalifa 最高楼"},
            "anchor": "current",
        }]),
        ReactStep(answer="仍然是 828 米。"),
    ]
    events = asyncio.run(_run_delegation(worker, second_steps))
    assert toolbox.calls == 1
    assert any(event.get("type") == "search_done"
               and event.get("cache_hit") is True for event in events)


def test_search_candidate_is_discarded_when_answer_fails():
    store = SearchFactStore(_cfg())
    toolbox = _ToolBox()
    worker = _watcher(store, toolbox)

    steps = [
        ReactStep(tool_calls=[{
            "name": "text_search",
            "args": {"query": "Burj Khalifa"},
            "anchor": "current",
        }]),
        ReactStep(answer=""),
    ]
    asyncio.run(_run_delegation(worker, steps, fallback=""))

    assert toolbox.calls == 1
    assert store.snapshot().facts == {}


def test_memory_writer_has_no_search_fact_contract_or_store_reference():
    assert "facts_update" not in MEMORY_WRITER_SYSTEM
    store = SearchFactStore(_cfg())
    mem_stub = type("MemStub", (), {"get_meta": lambda self, k, d="0": d})()
    writer = MemoryWriter(
        _cfg(), store, mem_stub, object(), object(), object(), object())
    assert not hasattr(writer, "store")
    reviewer = MemoryReviewer(
        _cfg(), store, object(), object(), object(), object(), object())
    assert not hasattr(reviewer, "store")

    # Old imports resolve to the new structured implementation during migration.
    assert ContextStore is SearchFactStore
    assert SharedContext is SearchFactSnapshot


def test_memory_backend_projects_search_facts_as_json_safe_ui_values():
    from agent.multimodal.memory_backend import MemoryBackend

    emitted = []
    backend = MemoryBackend.__new__(MemoryBackend)
    backend._emit_cb = lambda event, payload: emitted.append((event, payload))
    backend._last_ctx_sig = None
    backend.conversation = SimpleNamespace(
        latest_obs=lambda _n: [], latest_audio_obs=lambda _n: [])
    store = SearchFactStore(_cfg())
    backend.search_fact_store = store
    store.add_listener(lambda _snapshot: backend._push_ctx())

    now = time.time()
    store.upsert_many([SearchFact(
        key="price", query="Example price?", value="$123",
        source_tool="text_search:anysearch",
        source_urls=("https://example.test/price",),
        fetched_at=now, expires_at=now + 60, confidence=0.75,
    )], now=now)

    assert len(emitted) == 1
    event, payload = emitted[0]
    assert event == "multimodal.ctx"
    assert payload["facts"] == {"Example price?": "$123"}
    # Regression: SearchFact dataclasses must never leak into the websocket
    # payload, where json.dumps would raise TypeError.
    json.dumps(payload)


def test_memory_backend_serializes_cross_thread_context_pushes():
    from agent.multimodal.memory_backend import MemoryBackend

    backend = MemoryBackend.__new__(MemoryBackend)
    backend._ctx_push_lock = threading.RLock()
    entered = threading.Event()
    release = threading.Event()
    calls = []
    active = 0
    max_active = 0
    counter_lock = threading.Lock()

    def _push_unlocked():
        nonlocal active, max_active
        with counter_lock:
            active += 1
            max_active = max(max_active, active)
            calls.append("start")
            first = len(calls) == 1
        if first:
            entered.set()
            assert release.wait(1.0)
        with counter_lock:
            calls.append("end")
            active -= 1

    backend._push_ctx_unlocked = _push_unlocked
    first = threading.Thread(target=backend._push_ctx)
    second = threading.Thread(target=backend._push_ctx)
    first.start()
    assert entered.wait(1.0)
    second.start()
    time.sleep(0.03)
    assert calls == ["start"]
    release.set()
    first.join(1.0)
    second.join(1.0)

    assert not first.is_alive() and not second.is_alive()
    assert calls == ["start", "end", "start", "end"]
    assert max_active == 1


def test_search_fact_listener_marshals_ui_push_to_backend_loop():
    from agent.multimodal.memory_backend import MemoryBackend

    queued = []
    pushed = []

    class _Loop:
        @staticmethod
        def is_closed():
            return False

        @staticmethod
        def is_running():
            return True

        @staticmethod
        def call_soon_threadsafe(callback):
            queued.append(callback)

    backend = MemoryBackend.__new__(MemoryBackend)
    backend._loop = _Loop()
    backend._push_ctx = lambda: pushed.append(True)

    backend._schedule_ctx_push()

    assert pushed == []
    assert len(queued) == 1
    queued[0]()
    assert pushed == [True]
