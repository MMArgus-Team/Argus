"""WatcherAgent — resident engine for background live-watcher delegations.

Sibling of :class:`MemoryBackend`. While MemoryBackend keeps writing/reviewing
multimodal memory whenever the video stream is on, WatcherAgent runs the
continuous deep-analysis loop the main agent delegates via set_live_watcher,
plus one-shot QueryWorker jobs that own simple Recall/Search answers.

A live watcher has ONE mode: it walks the video from the head, round after
round as frames accumulate, and keeps going until the video source stops, the
worker conclusively observes the task's finite completion condition, or the
user stops it. There is no simple/complex query classification. Each round
runs WatcherWorker._spawn_delegation (multi-round ReAct on a frame batch), logs
to analyse/watch_<rid>.md, streams progress to the DeepPanel via the gateway's
``_emit`` channel (``multimodal.bg`` events), streams each round's report via
``on_round_report``. At successful completion the full accumulated log is
consolidated once and delivered through ``on_delegation_complete``; an optional
main-agent hook therefore runs once, never once per segment.

``submit_query_async()`` runs the existing WatcherWorker ReAct planner once,
letting RecallAgent and the stateless Search ToolBox fan out concurrently and
reply directly to the original user-message slot. ``recall_memory()`` remains a
synchronous compatibility entry for non-gateway callers and tests.

The engine shares MemoryStore / FrameBuffer / ConversationLog / SearchFactStore /
FrameStore with :class:`MemoryBackend` — these are **injected**, not rebuilt,
so RecallAgent actually sees what MemoryWriter wrote.

Lifecycle (driven by the gateway):
    engine = WatcherAgent(frame_buffer, memory_backend, hermes_cfg, emit_cb, sid, ...)
    engine.start()
    ...
    engine.stop()
"""

from __future__ import annotations

import asyncio
import base64
import copy
import contextvars
import inspect
import json
import logging
import math
import re
import threading
import time
import uuid
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any, Callable, Dict, Optional

log = logging.getLogger("hermes.multimodal.watcher_engine")

# ★ #1 p95 埋点: 每收满这么多 chunk-gap 样本打印一次分位数。
_TTS_LAT_REPORT_EVERY = 200

# ``tui_gateway.server._trajectory_safe`` keeps complete JPEG thumbnails only
# up to this size.  Reject larger/invalid fallbacks here rather than sending a
# huge value that the gateway would replace with an unusable truncation marker.
_QUERY_DEBUG_JPEG_B64_MAX_CHARS = 500_000

# Debug OCR is deliberately much smaller than the model-facing evidence.  At
# most three records * 1,800 serialized characters plus JSON list punctuation
# fits below the strict aggregate ceiling.
_QUERY_OCR_DEBUG_MAX_RECORDS = 3
_QUERY_OCR_DEBUG_RECORD_MAX_CHARS = 1_800
_QUERY_OCR_DEBUG_TOTAL_MAX_CHARS = 5_500

# Synchronous gateway callers must not wait forever for DashScope's connect +
# session.updated handshake.  Kept as a constant so lifecycle tests can use a
# short deterministic bound.
_ASR_START_TIMEOUT_SEC = 12.0


def sanitize_query_ocr_debug_evidence(evidence: Any) -> list[dict]:
    """Return a small, secret-redacted OCR projection for UI trajectories.

    This surface is intentionally independent from the richer model prompt
    evidence.  It contains no JPEG/base64 payloads, OCR boxes, frame ids, or
    provider objects.  Both each serialized record and the serialized list have
    hard character ceilings, so dense desktop OCR cannot bloat the WebSocket or
    retained debug trajectory.
    """
    from agent.redact import redact_sensitive_text

    records: list[dict] = []
    for item in list(evidence or [])[:_QUERY_OCR_DEBUG_MAX_RECORDS]:
        if not isinstance(item, dict):
            continue
        try:
            ts = float(item.get("frame_ts"))
            frame_ts: Optional[float] = ts if math.isfinite(ts) else None
        except (TypeError, ValueError):
            frame_ts = None

        def _safe_text(value: Any, limit: int) -> str:
            char_limit = max(0, int(limit))
            # OCR providers can return whole dense pages.  The UI retains only
            # this prefix, so bound secret scanning too; the small look-ahead
            # lets a credential starting near the retained edge be recognized
            # as a complete pattern before final truncation.
            candidate = str(value or "")[:char_limit + 512]
            redacted = redact_sensitive_text(candidate, force=True)
            return redacted[:char_limit]

        record = {
            "frame_ts": frame_ts,
            "source_type": _safe_text(item.get("source_type"), 80),
            "evidence_source": _safe_text(item.get("evidence_source"), 120),
            "app": _safe_text(item.get("app"), 160),
            "window_title": _safe_text(item.get("window_title"), 240),
            "raw_text": _safe_text(item.get("raw_text"), 1_600),
        }

        # Metadata contributes to the per-record bound.  Shrink only raw_text,
        # preserving timestamp/source fields needed to correlate with frames.
        serialized = json.dumps(
            record, ensure_ascii=False, separators=(",", ":"))
        if len(serialized) > _QUERY_OCR_DEBUG_RECORD_MAX_CHARS:
            overflow = len(serialized) - _QUERY_OCR_DEBUG_RECORD_MAX_CHARS
            raw_text = record["raw_text"]
            record["raw_text"] = raw_text[:max(0, len(raw_text) - overflow)]
            serialized = json.dumps(
                record, ensure_ascii=False, separators=(",", ":"))
        # Defensive metadata-only fallback should field limits ever grow.
        if len(serialized) > _QUERY_OCR_DEBUG_RECORD_MAX_CHARS:
            record["app"] = ""
            record["window_title"] = ""
            record["raw_text"] = ""
        records.append(record)

    # The per-record invariant makes this normally unreachable.  Keep a hard
    # aggregate assertion in behavior rather than relying on that arithmetic if
    # record shape changes later.
    while records and len(json.dumps(
            records, ensure_ascii=False, separators=(",", ":"),
    )) > _QUERY_OCR_DEBUG_TOTAL_MAX_CHARS:
        record = records[-1]
        raw_text = str(record.get("raw_text") or "")
        if raw_text:
            record["raw_text"] = raw_text[:max(0, len(raw_text) - 256)]
        else:
            records.pop()
    return records


def _percentile(samples, q: float) -> float:
    """样本的第 q 百分位 (q∈[0,1]), 最近邻法。空 → 0.0。"""
    if not samples:
        return 0.0
    xs = sorted(samples)
    if len(xs) == 1:
        return float(xs[0])
    idx = int(round(q * (len(xs) - 1)))
    idx = max(0, min(len(xs) - 1, idx))
    return float(xs[idx])


async def _invoke_progress_callback(
    callback: Callable[[Dict[str, Any]], Any], event: Dict[str, Any],
) -> None:
    """Invoke a recall progress callback on the loop that owns it."""
    result = callback(event)
    if inspect.isawaitable(result):
        await result


class BackendRecallProxy:
    """Loop-safe async facade for a MemoryBackend-owned RecallAgent.

    ``RecallAgent`` owns an ``AsyncOpenAI`` client and asyncio primitives built
    on the MemoryBackend thread.  WatcherWorker runs on a different loop, so it
    must never await that agent directly.  This facade preserves the agent's
    async ``run(...)`` interface while marshalling the real coroutine to the
    backend loop.  Progress events are marshalled in the opposite direction so
    Watcher callbacks continue to execute on the Watcher loop.
    """

    def __init__(self, memory_backend: Any):
        self._memory_backend = memory_backend

    async def run(self, **kwargs: Any) -> Any:
        caller_loop = asyncio.get_running_loop()
        backend = self._memory_backend
        if not _backend_is_ready(backend):
            raise RuntimeError("memory backend recall is not ready")
        backend_loop = getattr(backend, "_loop", None)
        recall_agent = getattr(backend, "recall_agent", None)
        if recall_agent is None or backend_loop is None:
            raise RuntimeError("memory backend recall is not ready")
        if backend_loop.is_closed() or not backend_loop.is_running():
            raise RuntimeError("memory backend loop is not running")

        stop_event = getattr(backend, "_stop", None)
        if stop_event is not None:
            try:
                if stop_event.is_set():
                    raise RuntimeError("memory backend is stopping")
            except AttributeError:
                pass

        progress_cb = kwargs.get("on_progress")
        if progress_cb is not None and backend_loop is not caller_loop:
            async def _relay_progress(event: Dict[str, Any]) -> None:
                # This function is awaited by RecallAgent on the backend loop.
                # Schedule the user callback back onto the Watcher loop instead
                # of executing a Watcher-owned coroutine on the backend thread.
                event_copy = dict(event or {})
                callback_coro = _invoke_progress_callback(
                    progress_cb, event_copy)
                try:
                    callback_future = asyncio.run_coroutine_threadsafe(
                        callback_coro, caller_loop)
                except Exception:
                    callback_coro.close()
                    raise
                try:
                    await asyncio.wrap_future(callback_future)
                except asyncio.CancelledError:
                    callback_future.cancel()
                    raise

            kwargs["on_progress"] = _relay_progress

        target_coro = recall_agent.run(**kwargs)
        # A same-loop call is useful for focused embedding/tests and avoids the
        # needless thread-safe round trip.  Production Watcher and backend loops
        # are distinct.
        if backend_loop is caller_loop:
            return await target_coro
        try:
            backend_future = asyncio.run_coroutine_threadsafe(
                target_coro, backend_loop)
        except Exception:
            target_coro.close()
            raise
        try:
            return await asyncio.wrap_future(backend_future)
        except asyncio.CancelledError:
            # asyncio.wrap_future normally chains cancellation, but make the
            # ownership transfer explicit so backend LLM work cannot outlive a
            # cancelled QueryWorker/session.
            backend_future.cancel()
            raise


class BackendQueryOCRProxy:
    """Loop-safe facade for MemoryBackend's long-lived OCR worker.

    The OCR client (local RapidOCR — the only OCR backend; remote VLM OCR was
    removed) is constructed on the MemoryBackend loop.  QueryWorker runs on the
    Watcher loop, so the collection coroutine must be marshalled back to its
    owner just like Recall.
    """

    def __init__(self, memory_backend: Any):
        self._memory_backend = memory_backend

    async def collect_query_evidence(
        self, frames: list, *, ask_ts: float,
    ) -> list[dict]:
        caller_loop = asyncio.get_running_loop()
        backend = self._memory_backend
        if not _backend_is_ready(backend):
            return []
        backend_loop = getattr(backend, "_loop", None)
        worker = getattr(backend, "screen_ocr_worker", None)
        if worker is None or backend_loop is None:
            return []
        if backend_loop.is_closed() or not backend_loop.is_running():
            return []

        stop_event = getattr(backend, "_stop", None)
        if stop_event is not None:
            try:
                if stop_event.is_set():
                    return []
            except AttributeError:
                pass

        target_coro = worker.collect_query_evidence(
            list(frames or []), ask_ts=float(ask_ts))
        if backend_loop is caller_loop:
            return list(await target_coro or [])
        try:
            backend_future = asyncio.run_coroutine_threadsafe(
                target_coro, backend_loop)
        except Exception:
            target_coro.close()
            raise
        try:
            return list(await asyncio.wrap_future(backend_future) or [])
        except asyncio.CancelledError:
            backend_future.cancel()
            raise


def _backend_is_ready(memory_backend: Any) -> bool:
    """Best-effort compatibility check for MemoryBackend lifecycle versions."""
    if memory_backend is None:
        return False

    # New lifecycle API.  Accept either properties or zero-argument methods so
    # this remains compatible while the backend lifecycle lands independently.
    marker = getattr(memory_backend, "is_ready", None)
    if marker is not None:
        try:
            ready = bool(marker() if callable(marker) else marker)
        except Exception:
            return False
        if not ready:
            return False
        healthy = getattr(memory_backend, "healthy", None)
        if healthy is not None:
            try:
                return bool(healthy() if callable(healthy) else healthy)
            except Exception:
                return False
        return True

    state = getattr(memory_backend, "state", None)
    if state is not None:
        state_value = getattr(state, "value", state)
        return str(state_value).strip().lower() == "ready"

    # Compatibility with an older backend that has no explicit lifecycle: an
    # explicitly-unhealthy or not-yet-signalled backend is never considered
    # usable.  Final full-bundle validation in _build handles the remaining
    # legacy case.
    healthy = getattr(memory_backend, "_healthy", None)
    if healthy is False:
        return False
    ready_event = getattr(memory_backend, "_ready", None)
    if ready_event is not None:
        try:
            if not ready_event.is_set():
                return False
        except AttributeError:
            return False
    return True


def _scene_label_from(thought: str, max_len: int = 16) -> str:
    """Cheaply extract a short "scene label" from this segment's model thought,
    for the DeepPanel segment-card title row.

    Pure heuristic, no LLM call:
      1) prefer content inside 《…》/〈…〉 title brackets (a film/work name best
         captures the scene);
      2) else take the fragment before the first sentence break
         (。！？.!?；;、\\n);
      3) strip common lead-ins ("这是" / "画面显示" / "从…中", etc.) and
         surrounding punctuation/whitespace;
      4) truncate to max_len chars (counted per character for both CJK and
         Latin), appending an ellipsis when over-length.
    Returns "" when nothing meaningful is found (the title row then shows only
    the segment number + timestamp).
    """
    t = (thought or "").strip()
    if not t:
        return ""
    # 跳过 _workers 在 thought 为空时合成的占位句 (它们不是真实场景描述)。
    from agent.multimodal._sentinels import SYNTH_THOUGHTS
    if t in SYNTH_THOUGHTS:
        return ""
    # 1) 书名号内容
    m = re.search(r"[《〈]([^》〉]{1,20})[》〉]", t)
    if m:
        label = m.group(1).strip()
    else:
        # 2) 首句
        seg = re.split(r"[。！？.!?；;、\n]", t, maxsplit=1)[0].strip()
        # 3) 去起手词
        seg = re.sub(r"^(从|这是|这个|画面(上|中)?显示了?|画面(上|中)?是|可以(识别|看)出?)[:：,，\s]*",
                     "", seg).strip(" :：,，.。-—")
        label = seg
    if not label:
        return ""
    if len(label) > max_len:
        label = label[:max_len].rstrip() + "…"
    return label


class WatcherAgent:
    """Owns the WatcherWorker (single-step ReAct + the ReAct loop, merged into
    one object) plus the RecallAgent.

    All worker LLM calls run on this engine's own daemon-thread asyncio loop,
    so the main agent's synchronous turn thread never blocks on async I/O.
    Thread-safe submit APIs marshal in via ``run_coroutine_threadsafe``.
    """

    STATE_NEW = "new"
    STATE_STARTING = "starting"
    STATE_READY = "ready"
    STATE_FAILED = "failed"
    STATE_STOPPING = "stopping"
    STATE_STOPPED = "stopped"

    def __init__(
        self,
        frame_buffer: Any,
        memory_backend: Optional[Any] = None,
        hermes_cfg: Optional[dict] = None,
        emit_cb: Optional[Callable[[str, dict], None]] = None,
        sid: Optional[str] = None,
        on_delegation_complete: Optional[
            Callable[[str, str, str, str], None]
        ] = None,
        on_delegation_start: Optional[Callable[[str, str], None]] = None,
        on_round_report: Optional[Callable[[str, int, str], None]] = None,
        research_registry_cb: Optional[Callable[[str], Optional[dict]]] = None,
        on_query_complete: Optional[Callable[
            [str, str, str, str, str], None
        ]] = None,
    ):
        # Shared with MemoryBackend (injected refs — not rebuilt).
        self.frame_buffer = frame_buffer
        self._memory_backend = memory_backend
        self._hermes_cfg = hermes_cfg
        # emit_cb(event, payload) — thread-safe push to the dashboard
        # (gateway binds it to _emit(event, sid, payload)).
        self._emit_cb = emit_cb
        self._sid = sid
        # on_delegation_complete(request_id, task_instruction, final_summary,
        # stop_reason) — called exactly once after a successful or interrupted
        # run is finalized. The gateway decides whether the optional main-agent
        # completion hook should fire from stop_reason. Thread-safe.
        self._on_delegation_complete = on_delegation_complete
        # on_delegation_start(request_id, task_instruction) — fired right after a background
        # delegation is submitted, so the gateway can write a "running" placeholder
        # assistant message (by rid). Thread-safe (called from submit thread).
        self._on_delegation_start = on_delegation_start
        # on_round_report(request_id, round_idx, report_text) — fired after EACH
        # productive analysis round (non-empty answer), so the gateway can append
        # it to the watcher panel / debug sidecar. It never re-enters the main
        # agent. Fired from the engine loop thread; must be thread-safe.
        self._on_round_report = on_round_report
        # on_query_complete(task_id, parent_user_message_id, original_user_query,
        # text, status). The gateway uses it to persist the completed Q/A ledger
        # and complete the preallocated answer slot.
        self._on_query_complete = on_query_complete
        # research_registry_cb(request_id) -> the tool-layer research entry
        # (agent.mm_watchers[rid]) or None. _run_delegation reads it at each
        # batch boundary so op=update (new task_instruction next round) and op=delete
        # (finish current round, then end + suppress the completion hook) take
        # effect without restarting the delegation. Thread-safe: the callback just
        # reads a dict the tool thread mutates (last-writer-wins on plain fields).
        self._research_registry_cb = research_registry_cb

        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._state_lock = threading.RLock()
        self._state = self.STATE_NEW
        # ``_ready`` is retained as the legacy "startup resolved" event: old
        # callers/tests expect it to be set for both success and failure.  The
        # explicit lifecycle API uses ``_startup_done`` + the state value.
        self._ready = threading.Event()
        self._startup_done = threading.Event()
        self._stopped = threading.Event()
        self._stopped_callbacks: list[Callable[[Any], None]] = []
        self._stopped_callbacks_fired = False
        self._startup_error: Optional[BaseException] = None
        self._runtime_error: Optional[BaseException] = None
        self._closed_llm_client_ids: set[int] = set()
        self._stop = threading.Event()
        self._healthy = False  # True only after _build() succeeds
        # In-flight delegation tasks keyed by request_id (for cancel on stop /
        # inspection). Router tasks are never mutated once started; this only
        # tracks them so stop() can cancel and nothing leaks.
        self._active: dict = {}
        # QueryWorker submissions enter from gateway/tool threads while their
        # completion callbacks run on the Watcher loop thread.  Keep every map
        # mutation behind one lock; relying on CPython's individual dict ops
        # was not enough for parent-id dedupe + capacity checks to be atomic.
        self._query_lock = threading.RLock()
        self._active_queries: dict = {}
        self._query_by_parent: dict[str, str] = {}
        self._query_pending: set[str] = set()
        self._query_running: set[str] = set()
        self._query_semaphore: Optional[asyncio.Semaphore] = None
        self._query_max_concurrency = 2
        self._query_max_pending = 8

        # Built lazily on the engine thread.
        self.cfg = None
        self.client = None
        self.model: str = ""
        self.mem = None
        self.store = None
        self.search_fact_store = None
        self.conversation = None
        self.frame_store = None
        self.screen_text_store = None
        self.screen_table_store = None
        self.task_state_store = None
        self.query_ocr_worker = None
        self._owned_query_ocr_client = None
        self.toolbox = None
        self.live_worker = None     # ReAct 单步+循环合一 (WatcherWorker)
        self.router_worker = None   # alias of live_worker (backward compat)
        self.recall_agent = None
        # A dedicated Recall transport exists before RecallAgent.__init__ may
        # succeed.  Keep it reachable so a constructor failure can still close
        # the pool on this engine's owner loop.
        self._recall_client_pending = None
        self.recall_worker = None   # alias of recall_agent (backward compat)
        self.responder = None       # alias of live_worker (holds _spawn_delegation)
        # Streaming realtime ASR sessions (DashScope) keyed by client key.
        # Managed on THIS engine's loop; see asr_* methods below.
        self._asr: dict = {}
        # Pending start ownership tokens linearize a synchronous timeout/stop
        # against the loop-owned async connect.  A late connect may publish
        # only while its exact token is still current.
        self._asr_start_tokens: dict[str, object] = {}
        self._asr_state_lock = threading.RLock()
        # ★ 在飞重连去抖: 上游 ASR WS 死掉时 asr_audio 会触发 _asr_reconnect, 同一 key
        #   同时只允许一个重连协程 (音频热路径每 200ms 来一片, 不去抖会疯狂并发 connect)。
        self._asr_reconnecting: set = set()
        # Serial TTS queue: per-turn segments (interim text blocks + final
        # answer) are enqueued and played back-to-back by a single consumer so
        # a later segment never cuts off an earlier one. All segments of one
        # turn share a response_id (the frontend appends same-rid PCM to its
        # playback timeline instead of stopping the current source). Built
        # lazily on first enqueue. See enqueue_tts / _tts_consumer.
        self._tts_queue: Optional[asyncio.Queue] = None
        self._tts_consumer_task: Optional[asyncio.Task] = None
        # ★ #1 p95 埋点: TTS chunk 延迟采样。收两类样本:
        #   first_chunk_ms — 从开始合成一段到第一个 PCM chunk 发出 (首字延迟);
        #   gap_ms         — 相邻 chunk 发出的间隔 (流是否卡顿, p95 慢帧看这个)。
        #   每累计 _TTS_LAT_REPORT_EVERY 个样本打印一次 p50/p95, 常驻可观测、零外部依赖。
        from collections import deque as _deque
        self._tts_first_ms: "Any" = _deque(maxlen=512)
        self._tts_gap_ms: "Any" = _deque(maxlen=2048)
        self._tts_lat_since_report = 0
        # Deep-analysis stop signals: a continuous (analysis/research) delegation
        # checks its per-rid Event each round; the gateway RPC multimodal.
        # stop_analysis sets it so the user / main agent can end the run early.
        # All managed on THIS engine loop.
        self._stop_events: dict = {}      # rid -> asyncio.Event
        self._stop_reasons: dict[str, str] = {}
        # A persisted cursor is valid only inside this exact engine runtime.
        # After process restart FrameBuffer timestamps begin on a new timeline.
        self._runtime_id = uuid.uuid4().hex
        # ★ Video source (screen share / camera / video call) closed signal.
        # The frontend sends multimodal.source_stopped when the user stops
        # capture; a continuous deep-analysis loop keeps waiting for new frames
        # UNTIL this is set (or the user stops the run), instead of guessing
        # "stopped" from a frame-idle heuristic (which false-stopped on lulls).
        # Session-scoped (one video source per session). Set on the engine loop.
        self._source_stopped: bool = False
        # Monotonic source lifecycle epoch. A stop followed immediately by a
        # start must not clear the terminal condition of a delegation that
        # belonged to the previous stream. Each run snapshots this value; any
        # later source transition permanently ends that run even when the
        # session-wide live flag becomes True again for the new stream.
        self._source_epoch: int = 0

    # ------------------------------------------------------------------ #
    @property
    def state(self) -> str:
        with self._state_lock:
            return self._state

    @property
    def is_ready(self) -> bool:
        return self.state == self.STATE_READY

    @property
    def is_failed(self) -> bool:
        return self.state == self.STATE_FAILED

    @property
    def is_stopped(self) -> bool:
        return self._stopped.is_set()

    @property
    def healthy(self) -> bool:
        thread = self._thread
        loop = self._loop
        return bool(
            self.is_ready
            and self._healthy
            and thread is not None
            and thread.is_alive()
            and loop is not None
            and not loop.is_closed()
            and loop.is_running()
        )

    @property
    def startup_error(self) -> Optional[BaseException]:
        with self._state_lock:
            return self._startup_error

    @property
    def runtime_error(self) -> Optional[BaseException]:
        with self._state_lock:
            return self._runtime_error

    def wait_ready(self, timeout: Optional[float] = None) -> bool:
        """Wait until startup reaches READY or a terminal non-ready state."""
        wait_s = None if timeout is None else max(0.0, float(timeout))
        self._startup_done.wait(timeout=wait_s)
        return self.healthy

    def wait_stopped(self, timeout: Optional[float] = None) -> bool:
        """Wait until the owner thread has drained tasks and closed its loop."""
        wait_s = None if timeout is None else max(0.0, float(timeout))
        return self._stopped.wait(timeout=wait_s)

    def add_stopped_callback(self, callback: Callable[[Any], None]) -> None:
        """Invoke ``callback(self)`` once teardown has released all resources."""
        call_now = False
        with self._state_lock:
            if self._stopped.is_set():
                call_now = True
            else:
                self._stopped_callbacks.append(callback)
        if call_now:
            try:
                callback(self)
            except Exception:
                log.debug("[watcher] stopped callback failed", exc_info=True)

    def _publish_startup_done(self) -> None:
        self._startup_done.set()
        self._ready.set()

    def _publish_stopped(self) -> None:
        callbacks: list[Callable[[Any], None]] = []
        with self._state_lock:
            self._healthy = False
            self._publish_startup_done()
            self._stopped.set()
            if not self._stopped_callbacks_fired:
                self._stopped_callbacks_fired = True
                callbacks = list(self._stopped_callbacks)
                self._stopped_callbacks.clear()
        for callback in callbacks:
            try:
                callback(self)
            except Exception:
                log.debug("[watcher] stopped callback failed", exc_info=True)

    def _mark_startup_failed(self, exc: BaseException) -> None:
        with self._state_lock:
            if self._startup_error is None:
                self._startup_error = exc
            self._state = self.STATE_FAILED
            self._healthy = False
        self._publish_startup_done()

    def _mark_ready(self) -> bool:
        with self._state_lock:
            if self._stop.is_set() or self._state == self.STATE_STOPPING:
                self._state = self.STATE_STOPPING
                self._healthy = False
                self._publish_startup_done()
                return False
            self._state = self.STATE_READY
            self._healthy = True
        self._publish_startup_done()
        return True

    def _build_stop_requested(self, after_stage: str) -> bool:
        stop = getattr(self, "_stop", None)
        if stop is None or not stop.is_set():
            return False
        log.info(
            "[watcher] startup cancelled after %s; skipping remaining build",
            after_stage,
        )
        return True

    def start(self, timeout: float = 10.0) -> bool:
        """Start the session-owned Watcher runtime and wait for full readiness.

        Returning merely because the thread exists recreates the original
        half-built-worker race.  Success therefore means the build completed,
        the event loop is running, and the runtime is still healthy.
        """
        start_error: Optional[BaseException] = None
        with self._state_lock:
            if self._state == self.STATE_READY:
                return self.healthy
            if self._state in {
                self.STATE_FAILED, self.STATE_STOPPING, self.STATE_STOPPED,
            }:
                return False
            if self._state == self.STATE_NEW:
                self._state = self.STATE_STARTING
                # New threads do not inherit ContextVar state.  Preserve the
                # session/profile home selected by the gateway so Watcher
                # config, watch files, and side-channel state never fall back
                # to the process launch profile.
                runtime_context = contextvars.copy_context()
                self._thread = threading.Thread(
                    target=runtime_context.run,
                    args=(self._run,),
                    name="mm-router-engine",
                    daemon=True,
                )
                # Assignment + Thread.start are atomic with respect to stop().
                # Otherwise stop could join a never-started Thread object.
                try:
                    self._thread.start()
                except BaseException as exc:
                    # Failed Thread objects are not joinable.  Remove the handle
                    # before a concurrent stop can observe the FAILED state.
                    self._thread = None
                    self._mark_startup_failed(exc)
                    start_error = exc
        # Callbacks may acquire the gateway lifecycle lock. Publish outside the
        # state lock so the global order never becomes state -> lifecycle.
        if start_error is not None:
            self._publish_stopped()
            return False
        return self.wait_ready(timeout)

    def stop(self, timeout: float = 5.0) -> bool:
        """Cancel owned work, stop the loop, and join the worker thread.

        The join is bounded so gateway teardown cannot hang forever.  The
        return value tells the owner whether the Watcher thread actually
        exited within that bound.
        """
        stopped_before_launch = False
        teardown_running_loop = False
        with self._state_lock:
            if self._state == self.STATE_STOPPED:
                return True
            if self._state == self.STATE_NEW:
                self._state = self.STATE_STOPPED
                stopped_before_launch = True
            elif self._state != self.STATE_FAILED:
                teardown_running_loop = (
                    self._state == self.STATE_READY and self._healthy)
                self._state = self.STATE_STOPPING
            self._healthy = False
        with self._query_lock:
            self._stop.set()
        self._publish_startup_done()
        if stopped_before_launch:
            self._publish_stopped()
            return True
        loop = self._loop
        thread = self._thread
        if (teardown_running_loop
                and loop is not None
                and not loop.is_closed()
                and loop.is_running()):
            # Cancel in-flight delegations AND close any live ASR WS sessions
            # so their reader tasks + DashScope WebSockets don't leak past
            # session teardown. Without this, a client that disconnects
            # without calling multimodal.asr_stop would leave the WS pinned.
            async def _teardown():
                for t in list(self._active.values()):
                    try:
                        t.cancel()
                    except Exception:
                        pass
                with self._query_lock:
                    query_futures = list(self._active_queries.values())
                    self._active_queries.clear()
                    self._query_by_parent.clear()
                    self._query_pending.clear()
                    self._query_running.clear()
                for t in query_futures:
                    try:
                        t.cancel()
                    except Exception:
                        pass
                # Cancel the serial TTS consumer so a half-drained queue doesn't
                # keep the loop alive past teardown.
                if self._tts_consumer_task is not None:
                    try:
                        self._tts_consumer_task.cancel()
                    except Exception:
                        pass
                # ★ C11: Collect the async ASR close() coroutines and await them
                # BEFORE stopping the loop. Scheduling them with ensure_future
                # and then calling loop.stop() in the same callback would let
                # run_forever return before close() ever ran → WS/reader leak.
                closes = []
                async def _abort_asr(asr):
                    try:
                        return await asr.close(graceful=False)
                    except TypeError:
                        return await asr.close()
                for _asr in list(self._asr.values()):
                    try:
                        closes.append(_abort_asr(_asr))
                    except Exception:
                        pass
                self._asr.clear()
                # Wake any continuous deep-analysis loop blocked polling for
                # frames so it unwinds instead of hanging past teardown.
                for _ev in list(self._stop_events.values()):
                    try:
                        _ev.set()
                    except Exception:
                        pass
                self._stop_events.clear()
                if closes:
                    try:
                        await asyncio.gather(*closes, return_exceptions=True)
                    except Exception:
                        pass
                # Give the just-cancelled active/query/TTS tasks scheduling
                # turns to process CancelledError before run_forever exits.
                await asyncio.sleep(0)
                await asyncio.sleep(0)
            try:
                async def _teardown_and_stop():
                    try:
                        await _teardown()
                    finally:
                        loop.stop()

                def _schedule_teardown() -> None:
                    coro = _teardown_and_stop()
                    try:
                        loop.create_task(coro)
                    except Exception:
                        coro.close()
                        loop.stop()

                loop.call_soon_threadsafe(_schedule_teardown)
            except Exception:
                pass
        else:
            with self._query_lock:
                query_futures = list(self._active_queries.values())
                self._active_queries.clear()
                self._query_by_parent.clear()
                self._query_pending.clear()
                self._query_running.clear()
            for future in query_futures:
                try:
                    future.cancel()
                except Exception:
                    pass

        # Joining our own thread would deadlock; the scheduled teardown above
        # will still stop it immediately after the current callback returns.
        if (thread is not None
                and thread is not threading.current_thread()
                and thread.ident is not None):
            thread.join(timeout=max(0.0, float(timeout)))
        stopped = bool(thread is None or not thread.is_alive())
        if not stopped:
            log.warning(
                "[watcher] engine stop timed out after %.1fs (state=%s)",
                max(0.0, float(timeout)), self.state,
            )
        return stopped

    async def _close_owned_llm_clients(self) -> None:
        """Close only dedicated Watcher/standalone-Recall transports once."""
        closed_ids = getattr(self, "_closed_llm_client_ids", None)
        if closed_ids is None:
            closed_ids = set()
            self._closed_llm_client_ids = closed_ids
        recall_agent = getattr(self, "recall_agent", None)
        recall_client = (
            getattr(recall_agent, "client", None)
            or getattr(self, "_recall_client_pending", None)
        )
        for label, client in (
            ("worker", getattr(self, "client", None)),
            ("recall", recall_client),
        ):
            if (client is None
                    or not bool(getattr(
                        client, "_hermes_submodule_owned", False))
                    or id(client) in closed_ids):
                continue
            closed_ids.add(id(client))
            close = getattr(client, "close", None)
            if close is None:
                continue
            try:
                result = close()
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                log.debug("[watcher] %s client close failed: %s", label, exc)
        # Standalone mode constructs its own OCR transport.  A MemoryBackend-
        # shared worker is never stored here and remains owned by that backend.
        owned_ocr = getattr(self, "_owned_query_ocr_client", None)
        ocr_transport = getattr(owned_ocr, "client", None)
        if ocr_transport is not None and id(ocr_transport) not in closed_ids:
            closed_ids.add(id(ocr_transport))
            close = getattr(ocr_transport, "close", None)
            if close is not None:
                try:
                    result = close()
                    if inspect.isawaitable(result):
                        await result
                except Exception as exc:
                    log.debug("[watcher] query OCR client close failed: %s", exc)
        self._owned_query_ocr_client = None
        self._recall_client_pending = None

    # ------------------------------------------------------------------ #
    def _build(self) -> bool:
        """Build the WatcherWorker (+ RecallAgent) and shared stores on the
        engine thread. Returns True on success, False if construction failed."""
        try:
            if self._build_stop_requested("thread launch"):
                return False
            from .hermes_glue import build_config, HermesClientFactory
            from . import _memory as memory_types
            from ._workers import (
                WatcherWorker, RecallAgent, ScreenOCRWorker, ToolBox)

            mb = self._memory_backend
            owns_recall = mb is None
            if mb is not None:
                if not _backend_is_ready(mb):
                    detail = str(getattr(mb, "startup_error", "") or "").strip()
                    raise RuntimeError(
                        "memory backend was provided but is not ready"
                        + (f": {detail}" if detail else ""))

                # One all-or-nothing resource bundle.  Never fill an absent
                # backend field with a new local store: that was the startup race
                # which produced two MemoryStore/Embedding/RecallAgent stacks.
                fact_store = getattr(mb, "search_fact_store", None)
                if fact_store is None:
                    fact_store = getattr(mb, "store", None)
                bundle = {
                    "cfg": getattr(mb, "cfg", None),
                    "mem": getattr(mb, "mem", None),
                    "search_fact_store": fact_store,
                    "conversation": getattr(mb, "conversation", None),
                    "frame_store": getattr(mb, "frame_store", None),
                    "screen_text_store": getattr(mb, "screen_text_store", None),
                    "screen_table_store": getattr(mb, "screen_table_store", None),
                    "task_state_store": getattr(mb, "task_state_store", None),
                    "recall_agent": getattr(mb, "recall_agent", None),
                    "loop": getattr(mb, "_loop", None),
                }
                missing = [
                    name for name, value in bundle.items() if value is None
                ]
                if missing:
                    raise RuntimeError(
                        "ready memory backend has an incomplete resource bundle: "
                        + ", ".join(missing))

                # Derive from the exact config snapshot built by MemoryBackend;
                # calling build_config() again could select a different
                # timestamp-derived DB path even after the ready race was fixed.
                # Watcher may resolve/override its worker-facing model fields.
                # Keep the backend's config immutable after READY: MemoryWriter
                # and its RecallAgent continue using that authoritative object,
                # while stores and Recall itself remain shared by identity.
                cfg = copy.copy(bundle["cfg"])
                self.cfg = cfg
                self.mem = bundle["mem"]
                self.search_fact_store = bundle["search_fact_store"]
                self.store = self.search_fact_store  # compatibility name
                self.conversation = bundle["conversation"]
                self.frame_store = bundle["frame_store"]
                self.screen_text_store = bundle["screen_text_store"]
                self.screen_table_store = bundle["screen_table_store"]
                self.task_state_store = bundle["task_state_store"]
                self.recall_agent = BackendRecallProxy(mb)
                if getattr(mb, "screen_ocr_worker", None) is not None:
                    self.query_ocr_worker = BackendQueryOCRProxy(mb)
            else:
                # Standalone Watcher mode owns one complete local bundle.  This
                # is deliberately selected only when no MemoryBackend exists;
                # an unready/failed backend above is never silently replaced.
                cfg = build_config(self._hermes_cfg)
                self.cfg = cfg
                memory_store_cls = memory_types.MemoryStore
                fact_store_cls = getattr(
                    memory_types, "SearchFactStore", None)
                if fact_store_cls is None:
                    fact_store_cls = getattr(memory_types, "ContextStore")
                self.mem = memory_store_cls(cfg)
                self.search_fact_store = fact_store_cls(cfg)
                self.store = self.search_fact_store  # compatibility name
                self.conversation = memory_types.ConversationLog(
                    max_chars=cfg.conv_max_chars,
                    min_turns=cfg.conv_min_turns,
                    max_bg_obs=cfg.conv_max_bg_obs)
                self.frame_store = memory_types.FrameStore(cfg)
                self.screen_text_store = memory_types.ScreenTextStore(
                    cfg, db_path=self.mem.db_path)
                self.screen_table_store = memory_types.ScreenTableStore(
                    cfg, db_path=self.mem.db_path)
                self.task_state_store = memory_types.TaskStateStore(
                    cfg, db_path=self.mem.db_path)
                # Standalone Watcher has no MemoryBackend-owned OCR worker.
                # Build exactly one session-long worker and reuse it for every
                # QueryWorker request; construction failure is a soft OCR
                # degradation and must not make the Watcher unavailable.
                try:
                    self.query_ocr_worker = ScreenOCRWorker(
                        cfg, self.frame_buffer, self.frame_store,
                        self.screen_text_store, self.screen_table_store,
                        self._stop,
                    )
                    self._owned_query_ocr_client = (
                        self.query_ocr_worker.ocr_client)
                except RuntimeError as exc:
                    log.warning("[query-ocr] standalone OCR disabled: %s", exc)
                    self.query_ocr_worker = None
                    self._owned_query_ocr_client = None
            if self._build_stop_requested("resource bundle"):
                return False

            # Deep-analysis worker LLM client. worker_client() honors an optional
            # dedicated endpoint (model.watcher.{base_url,api_key,model} — e.g.
            # a Kimi endpoint independent of the main agent); when unset it falls
            # back to the main resolved model (original behavior).
            client_factory = HermesClientFactory(cfg)
            self.client, self.model = client_factory.worker_client()
            if self._build_stop_requested("worker LLM client"):
                return False
            # ★ Sync the RESOLVED model back into cfg so the workers actually use
            # it. Workers call the LLM with cfg.model / (cfg.worker_fallback_model
            # or cfg.model); build_config left cfg.model at its dataclass default
            # ("qwen3.5"), which does NOT exist on the resolved endpoint → every
            # WatcherWorker/Search/decide call 404'd ("Model not exist") and the
            # delegation produced an empty answer (blank deep-research window).
            # SKIP this follow when the user pinned multimodal.worker_model
            # (build_config already set cfg.model to it) — explicit override wins.
            if self.model and not getattr(cfg, "_worker_model_explicit", False):
                try:
                    cfg.model = self.model
                    if not (getattr(cfg, "worker_fallback_model", "") or "").strip():
                        cfg.worker_fallback_model = self.model
                except Exception:
                    pass

            self.toolbox = ToolBox(
                cfg, self.frame_buffer, frame_store=self.frame_store)
            if self._build_stop_requested("ToolBox"):
                return False

            if owns_recall:
                # Standalone fallback still uses the dedicated recall role and
                # includes the structured ScreenTableStore.  It is built on the
                # Watcher loop, so direct awaits remain loop-safe.
                recall_client, recall_model = client_factory.recall_client()
                self._recall_client_pending = recall_client
                if self._build_stop_requested("recall LLM client"):
                    return False
                self.recall_agent = RecallAgent(
                    cfg, self.mem, recall_client, self.conversation,
                    buf=self.frame_buffer, frame_store=self.frame_store,
                    screen_text_store=self.screen_text_store,
                    screen_table_store=self.screen_table_store,
                    task_state_store=self.task_state_store,
                    model=recall_model)
                self._recall_client_pending = None
                if self._build_stop_requested("RecallAgent"):
                    return False
            self.recall_worker = self.recall_agent  # backward-compat alias

            # ★ 合并后: WatcherWorker 同时是 ReAct 单步(react_step/answer)与
            #   ReAct 循环(_spawn_delegation)。search 由无状态 ToolBox 直接执行,
            #   recall 走 RecallAgent 子agent。一个对象即可, 不再有独立 Runner。
            inflight: dict = {}
            self.live_worker = WatcherWorker(
                cfg, self.client, self.frame_buffer, self.mem, self.store,
                self.conversation, frame_store=self.frame_store,
                search_fact_store=self.search_fact_store,
                toolbox=self.toolbox, recall_agent=self.recall_agent,
                inflight=inflight)
            if self._build_stop_requested("WatcherWorker"):
                return False
            # backward-compat aliases (旧代码/日志引用这些属性名)。
            self.router_worker = self.live_worker
            self.responder = self.live_worker
            return True
        except Exception as exc:
            self._startup_error = exc
            log.warning("[watcher] build failed: %s", exc, exc_info=True)
            return False

    def _run(self) -> None:
        loop: Optional[asyncio.AbstractEventLoop] = None
        try:
            loop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)
            if not self._build():
                if self._stop.is_set():
                    with self._state_lock:
                        if self._state != self.STATE_FAILED:
                            self._state = self.STATE_STOPPING
                    self._publish_startup_done()
                else:
                    self._mark_startup_failed(
                        self._startup_error
                        or RuntimeError("multimodal Watcher build failed"))
                return

            # Everything after loop creation remains inside this try so an
            # invalid config/reconcile/semaphore failure receives exactly the
            # same task + client + loop teardown as a normal stop.
            self._query_max_concurrency = max(1, int(getattr(
                self.cfg, "query_worker_max_concurrency", 2) or 2))
            self._query_max_pending = max(0, int(getattr(
                self.cfg, "query_worker_max_pending", 8) or 0))
            self._query_semaphore = asyncio.Semaphore(
                self._query_max_concurrency)

            # 启动校准: 异常退出可能留下 stale running/stopping。
            try:
                from . import watch_file as _df
                _fixed = _df.reconcile_stale(
                    active_ids=list(self._stop_events.keys()))
                if _fixed:
                    log.info(
                        "[watcher] startup reconcile: %d watcher 文件被校准",
                        _fixed,
                    )
            except Exception as _rec_exc:
                log.debug("[watcher] reconcile_stale failed: %s", _rec_exc)

            # Mark ready from inside the running event loop.  Setting _ready
            # before run_forever allowed the gateway to submit into a loop that
            # had been built but had not started dispatching callbacks yet.
            def _mark_loop_ready() -> None:
                if not self._mark_ready():
                    loop.stop()
                    return
                log.info("[watcher] engine ready (sid=%s model=%s)",
                         self._sid, self.model)

            loop.call_soon(_mark_loop_ready)
            # Park the loop — work is submitted via run_coroutine_threadsafe.
            loop.run_forever()
        except BaseException as exc:
            if not self._startup_done.is_set():
                self._mark_startup_failed(exc)
            elif not isinstance(exc, (KeyboardInterrupt, SystemExit)):
                with self._state_lock:
                    self._runtime_error = exc
                    if not self._stop.is_set():
                        self._state = self.STATE_FAILED
                log.warning("[watcher] runtime failed: %s", exc, exc_info=True)
            else:
                raise
        finally:
            self._healthy = False
            if loop is not None and not loop.is_closed():
                # Drain anything that survived an abrupt loop.stop, then close
                # dedicated transports on the exact loop that constructed them.
                async def _drain_owned_tasks() -> None:
                    current = asyncio.current_task()
                    pending = [
                        task for task in asyncio.all_tasks(loop)
                        if task is not current and not task.done()
                    ]
                    for task in pending:
                        task.cancel()
                    if pending:
                        await asyncio.gather(*pending, return_exceptions=True)
                    await self._close_owned_llm_clients()

                try:
                    loop.run_until_complete(_drain_owned_tasks())
                except Exception as exc:
                    log.debug("[watcher] async teardown failed: %s", exc)
                try:
                    loop.close()
                except Exception:
                    pass
            try:
                asyncio.set_event_loop(None)
            except Exception:
                pass
            self._loop = None
            with self._state_lock:
                if self._state != self.STATE_FAILED:
                    self._state = self.STATE_STOPPED
            self._publish_stopped()

    # ------------------------------------------------------------------ #
    # Public sync APIs (called from main-agent tool handlers)
    # ------------------------------------------------------------------ #
    def submit_tts(self, text: str, response_id: str = "") -> bool:
        """Non-blocking: stream TTS for ``text`` to the dashboard as a
        fire-and-forget task on the engine loop.

        Each PCM chunk is forwarded as ``multimodal.tts``
        ``{response_id, pcm_b64, sample_rate, is_final}`` via ``emit_cb``;
        a terminal ``is_final: True`` empty chunk closes the response. When
        ``response_id`` is empty a fresh ``tts_<hex>`` id is generated.
        Returns True on submit, False when the engine/cfg isn't ready or text
        is empty.
        """
        text = (text or "").strip()
        # Guard on cfg too: a half-built engine (_build failed) has cfg=None,
        # which would crash TTSClient(self.cfg) inside the fire-and-forget task.
        if not text or self._loop is None or self.cfg is None:
            return False
        try:
            fut = asyncio.run_coroutine_threadsafe(
                self._run_tts(text, response_id), self._loop)

            def _done(f):
                try:
                    exc = f.exception()
                except Exception:
                    exc = None
                if exc is not None and not isinstance(exc, asyncio.CancelledError):
                    log.warning("[watcher] TTS task crashed: %s", exc)
            fut.add_done_callback(_done)
            return True
        except Exception as exc:
            log.warning("[watcher] submit_tts failed: %s", exc)
            return False

    async def _run_tts(self, text: str, response_id: str,
                       send_final: bool = True) -> None:
        emit = self._emit_cb
        if emit is None:
            return
        import base64
        import secrets as _secrets
        # Prefer the DashScope realtime TTS when configured (streaming PCM24k);
        # else fall back to the legacy internal TTSClient. Both yield (pcm, sr).
        cfg = self.cfg
        ds_key = (getattr(cfg, "dashscope_api_key", "") or "").strip()
        use_realtime = bool(ds_key) and getattr(cfg, "realtime_tts_enabled", True)
        if use_realtime:
            from .qwen_realtime import QwenRealtimeTTS
            client = QwenRealtimeTTS(
                ds_key,
                model=getattr(cfg, "realtime_tts_model", "qwen3-tts-flash-realtime"),
                voice=getattr(cfg, "realtime_tts_voice", "Cherry"),
                sample_rate=int(getattr(cfg, "realtime_tts_sample_rate", 24000)),
                speech_rate=float(getattr(cfg, "realtime_tts_speech_rate", 1.3)))
            _stream = client.synthesize(text)
        else:
            try:
                from ._dual_agent import TTSClient
            except Exception as exc:
                log.warning("[watcher] TTS unavailable: %s", exc)
                return
            _stream = TTSClient(self.cfg).stream(text)
        # Use a fresh token, not id(text): id() of a short-lived string can be
        # reused after GC → two sequential TTS calls could collide on rid.
        rid = response_id or ("tts_" + _secrets.token_hex(4))
        sent = 0
        # ★ #1 p95 埋点: 计时锚点。_t_synth_start=开始合成; _t_prev_chunk=上个 chunk 发出。
        _t_synth_start = time.monotonic()
        _t_prev_chunk = 0.0
        try:
            async for pcm, sr in _stream:
                _t_now = time.monotonic()
                try:
                    emit("multimodal.tts", {
                        "response_id": rid,
                        "pcm_b64": base64.b64encode(pcm).decode("ascii"),
                        "sample_rate": int(sr),
                        "is_final": False,
                    })
                except Exception:
                    pass
                # 采样: 首 chunk 记首字延迟, 其余记与上个 chunk 的间隔。
                if sent == 0:
                    self._tts_first_ms.append((_t_now - _t_synth_start) * 1000.0)
                else:
                    self._tts_gap_ms.append((_t_now - _t_prev_chunk) * 1000.0)
                    self._tts_lat_since_report += 1
                    if self._tts_lat_since_report >= _TTS_LAT_REPORT_EVERY:
                        self._report_tts_latency()
                        self._tts_lat_since_report = 0
                _t_prev_chunk = _t_now
                sent += 1
        except Exception as exc:
            log.warning("[watcher] TTS stream error: %s", exc)
        finally:
            # Terminal is_final closes the response on the frontend. Skip it for
            # mid-turn segments in the serial queue (send_final=False) so the
            # frontend keeps appending later segments' PCM to the same timeline
            # instead of treating each segment as a finished response.
            if send_final:
                try:
                    emit("multimodal.tts", {
                        "response_id": rid, "pcm_b64": "",
                        "sample_rate": 24000, "is_final": True,
                    })
                except Exception:
                    pass
            log.info("[watcher] TTS done %d chunks (%d chars)", sent, len(text))

    def _report_tts_latency(self) -> None:
        """★ #1 p95 埋点: 打印 TTS 首字延迟 + chunk 间隔的 p50/p95/max。
        grep '[TTS_LAT]' 看趋势; gap p95 明显 > p50 = 存在偶发慢帧 (文章说的 p95 元凶)。"""
        try:
            first = list(self._tts_first_ms)
            gap = list(self._tts_gap_ms)
            log.info(
                "[TTS_LAT] first_chunk_ms p50=%.0f p95=%.0f max=%.0f (n=%d) | "
                "chunk_gap_ms p50=%.0f p95=%.0f max=%.0f (n=%d)",
                _percentile(first, 0.5), _percentile(first, 0.95),
                max(first) if first else 0.0, len(first),
                _percentile(gap, 0.5), _percentile(gap, 0.95),
                max(gap) if gap else 0.0, len(gap),
            )
        except Exception as exc:
            log.debug("[TTS_LAT] report failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Serial per-turn TTS queue. enqueue_tts() appends a (text, response_id)
    # segment; a single consumer plays them one after another (await _run_tts),
    # so a later segment never cuts off an earlier one. finish_tts() enqueues a
    # sentinel that emits the single terminal is_final for the turn's rid.
    # ------------------------------------------------------------------ #
    def enqueue_tts(self, text: str, response_id: str) -> bool:
        """Thread-safe: queue one TTS segment for serial playback.

        All segments of a turn should pass the SAME ``response_id`` so the
        frontend appends their PCM to one playback timeline. Returns True on
        enqueue, False when the engine/cfg isn't ready or text is empty.
        """
        text = (text or "").strip()
        if not text or self._loop is None or self.cfg is None:
            return False
        try:
            self._loop.call_soon_threadsafe(
                self._tts_enqueue_on_loop, ("seg", text, response_id))
            return True
        except Exception as exc:
            log.warning("[watcher] enqueue_tts failed: %s", exc)
            return False

    def finish_tts(self, response_id: str) -> bool:
        """Thread-safe: mark the end of a turn's TTS segments. The consumer
        emits the single terminal is_final for ``response_id`` once all queued
        segments before it have played."""
        if self._loop is None:
            return False
        try:
            self._loop.call_soon_threadsafe(
                self._tts_enqueue_on_loop, ("end", "", response_id))
            return True
        except Exception as exc:
            log.warning("[watcher] finish_tts failed: %s", exc)
            return False

    def _tts_enqueue_on_loop(self, item: tuple) -> None:
        """Runs on the engine loop: lazily build the queue + consumer, enqueue."""
        if self._tts_queue is None:
            self._tts_queue = asyncio.Queue()
        if self._tts_consumer_task is None or self._tts_consumer_task.done():
            self._tts_consumer_task = asyncio.ensure_future(
                self._tts_consumer(), loop=self._loop)
        self._tts_queue.put_nowait(item)

    async def _tts_consumer(self) -> None:
        """Single consumer: play queued segments serially."""
        emit = self._emit_cb
        q = self._tts_queue
        if q is None:
            return
        while True:
            try:
                kind, text, rid = await q.get()
            except asyncio.CancelledError:
                raise
            try:
                if kind == "seg":
                    # No terminal is_final between segments — keep the timeline
                    # open so the next segment appends rather than restarts.
                    await self._run_tts(text, rid, send_final=False)
                elif kind == "end":
                    # Close the turn's response so the frontend clears the
                    # "playing" badge once the queued audio drains.
                    if emit is not None:
                        try:
                            emit("multimodal.tts", {
                                "response_id": rid, "pcm_b64": "",
                                "sample_rate": 24000, "is_final": True,
                            })
                        except Exception:
                            pass
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("[watcher] TTS queue segment failed: %s", exc)

    def interrupt_tts(self) -> bool:
        """Thread-safe: 立即打断当前 TTS 播放 + 清空 TTS 播队.

        用户开口时 (VoiceAgent v2) 会调这个: 取消当前 _run_tts 的 async for 迭代,
        drain 队列里所有等待的 segments, emit is_final 收尾。下次 enqueue_tts 时
        _tts_enqueue_on_loop 会 lazily 重建 consumer, 干净重启。
        """
        if self._loop is None:
            return False
        try:
            self._loop.call_soon_threadsafe(self._interrupt_tts_on_loop)
            return True
        except Exception as exc:
            log.warning("[watcher] interrupt_tts failed: %s", exc)
            return False

    def _interrupt_tts_on_loop(self) -> None:
        """Runs on engine loop: cancel current consumer + drain queue."""
        task = self._tts_consumer_task
        if task is not None and not task.done():
            try:
                task.cancel()
            except Exception:
                pass
        self._tts_consumer_task = None
        # Drain 队列 (丢弃所有等待中的 seg/end sentinel)
        q = self._tts_queue
        if q is not None:
            drained = 0
            while not q.empty():
                try:
                    q.get_nowait()
                    drained += 1
                except Exception:
                    break
            if drained:
                log.info("[watcher] interrupt_tts drained %d queued segments", drained)
        # 通知前端把当前 rid 关掉 (清 "playing" 状态)
        # 不知道当前 rid 是什么, 发一个 empty is_final 的通用信号
        emit = self._emit_cb
        if emit is not None:
            try:
                emit("multimodal.tts", {
                    "response_id": "__interrupt__", "pcm_b64": "",
                    "sample_rate": 24000, "is_final": True,
                })
            except Exception:
                pass


    # ------------------------------------------------------------------ #
    # Streaming realtime ASR (DashScope) — thread-safe entrypoints. Each
    # session opens a QwenRealtimeASR on THIS engine's loop; partial/final
    # transcripts are surfaced via the two callbacks passed by the gateway.
    # ------------------------------------------------------------------ #
    def asr_start(self, key: str,
                  on_partial: Callable[[str], None],
                  on_final: Callable[[str], None],
                  on_speech_started: Optional[Callable[[], None]] = None) -> bool:
        """Open a streaming DashScope ASR session, blocking until the WS is
        connected or the attempt fails.

        ``on_partial(text)`` / ``on_final(text)`` are plain sync callables the
        gateway supplies; they're invoked from the engine loop. An existing
        session under the same ``key`` is closed first. Returns True when the WS
        connected, False if the engine/cfg isn't ready, DashScope/realtime ASR
        is unconfigured, or the connect timed out/failed.

        ``on_speech_started`` (optional): sync callable fired the moment DashScope
        VAD detects user speech start. Used to barge-in: interrupt the currently
        playing TTS the instant the user opens their mouth, instead of waiting
        for ASR final + intent classification (which adds 2-3s of "why isn't it
        stopping" latency). Gateway wires this to voice._interrupt_current_playback.
        """
        if self._loop is None or self.cfg is None:
            return False
        api_key = (getattr(self.cfg, "dashscope_api_key", "") or "").strip()
        if not api_key or not getattr(self.cfg, "realtime_asr_enabled", True):
            return False
        state_lock = getattr(self, "_asr_state_lock", None)
        if state_lock is None:
            state_lock = threading.RLock()
            self._asr_state_lock = state_lock
        if not hasattr(self, "_asr_start_tokens"):
            self._asr_start_tokens = {}
        start_token = object()
        with state_lock:
            self._asr_start_tokens[key] = start_token
        fut = asyncio.run_coroutine_threadsafe(
            self._asr_start(
                key,
                on_partial,
                on_final,
                on_speech_started,
                start_token=start_token,
            ),
            self._loop,
        )
        try:
            return bool(fut.result(timeout=_ASR_START_TIMEOUT_SEC))
        except FuturesTimeout:
            # Invalidate before cancellation: even a connect implementation
            # that delays/suppresses CancelledError will fail the publish CAS.
            with state_lock:
                if self._asr_start_tokens.get(key) is start_token:
                    self._asr_start_tokens.pop(key, None)
            fut.cancel()
            log.warning("[watcher] asr_start timed out key=%s", key)
            return False
        except Exception as exc:
            with state_lock:
                if self._asr_start_tokens.get(key) is start_token:
                    self._asr_start_tokens.pop(key, None)
            log.warning("[watcher] asr_start failed: %s", exc)
            return False

    async def _asr_start(self, key, on_partial, on_final,
                         on_speech_started=None, *, start_token=None) -> bool:
        from .qwen_realtime import QwenRealtimeASR
        state_lock = getattr(self, "_asr_state_lock", None)
        if state_lock is None:
            state_lock = threading.RLock()
            self._asr_state_lock = state_lock
        if not hasattr(self, "_asr_start_tokens"):
            self._asr_start_tokens = {}
        if start_token is None:
            start_token = object()
            with state_lock:
                self._asr_start_tokens[key] = start_token
        with state_lock:
            if self._asr_start_tokens.get(key) is not start_token:
                return False
        # Replace any stale session for this key.
        with state_lock:
            old = self._asr.pop(key, None)
        if old is not None:
            try:
                try:
                    await old.close(graceful=False)
                except TypeError:
                    await old.close()
            except Exception:
                pass
        cfg = self.cfg

        async def _p(t):
            try:
                on_partial(t)
            except Exception:
                pass

        async def _f(t):
            try:
                on_final(t)
            except Exception:
                pass

        async def _ss():
            if on_speech_started is None:
                return
            try:
                on_speech_started()
            except Exception:
                pass

        asr = QwenRealtimeASR(
            (cfg.dashscope_api_key or "").strip(),
            model=getattr(cfg, "realtime_asr_model", "qwen3-asr-flash-realtime"),
            language=getattr(cfg, "realtime_asr_language", "zh"),
            sample_rate=int(getattr(cfg, "realtime_asr_sample_rate", 16000)),
            vad_threshold=float(getattr(cfg, "realtime_asr_vad_threshold", 0.5)),
            vad_silence_ms=int(getattr(cfg, "realtime_asr_vad_silence_ms", 1200)),
            on_partial=_p, on_final=_f,
            on_speech_started=(_ss if on_speech_started is not None else None))
        try:
            ok = await asr.connect()
        except asyncio.CancelledError:
            # run_coroutine_threadsafe.cancel() propagates here on a gateway
            # timeout.  Abort the candidate before re-raising so its reader and
            # socket cannot survive outside the ownership map.
            try:
                try:
                    await asr.close(graceful=False)
                except TypeError:
                    await asr.close()
            except Exception:
                pass
            with state_lock:
                if self._asr_start_tokens.get(key) is start_token:
                    self._asr_start_tokens.pop(key, None)
            raise
        except Exception:
            ok = False

        with state_lock:
            owns_start = self._asr_start_tokens.get(key) is start_token
            if owns_start:
                self._asr_start_tokens.pop(key, None)
            if ok and owns_start:
                self._asr[key] = asr
                return True

        # Timeout, stop, or a replacement start invalidated this candidate
        # while connect awaited the network/session-ready acknowledgement.
        try:
            try:
                await asr.close(graceful=False)
            except TypeError:
                await asr.close()
        except Exception:
            pass
        return False

    def asr_audio(self, key: str, pcm: bytes) -> bool:
        """Thread-safe: feed a PCM16 chunk into an open ASR session.

        ★ 自愈: 上游 DashScope ASR WS 可能在长时会话中静默死掉 (网络抖动/服务端 idle
          超时/其它帧错误连带), 死后 append_audio 会静默丢弃音频 → "说什么没反应"。
          这里在喂音频前检测连接: 死了就触发一次去抖重连 (本片音频丢弃, 下一片到时若
          已重连就正常识别), 让会话自愈, 不再永久失灵。"""
        loop = self._loop
        asr = self._asr.get(key)
        if loop is None or asr is None or not pcm:
            return False
        # 上游死了 → 触发去抖重连 (同 key 只跑一个), 本片丢弃。
        if not getattr(asr, "is_connected", True):
            if key not in self._asr_reconnecting:
                self._asr_reconnecting.add(key)
                try:
                    asyncio.run_coroutine_threadsafe(self._asr_reconnect(key), loop)
                except Exception as exc:
                    self._asr_reconnecting.discard(key)
                    log.debug("[watcher] asr reconnect schedule failed: %s", exc)
            return False
        try:
            future = asyncio.run_coroutine_threadsafe(
                asr.append_audio(pcm), loop)
            return bool(future.result(timeout=1.0))
        except FuturesTimeout:
            log.warning("[watcher] asr_audio delivery timed out key=%s", key)
            return False
        except Exception as exc:
            log.debug("[watcher] asr_audio failed: %s", exc)
            return False

    async def _asr_reconnect(self, key: str) -> None:
        """重连一个死掉的上游 ASR 会话 (复用同对象 + 同回调, 重建 WS)。去抖: asr_audio
        保证同 key 同时只有一个本协程在跑。"""
        try:
            asr = self._asr.get(key)
            if asr is None:
                return   # 已被 asr_stop 移除 → 放弃重连
            ok = False
            try:
                ok = await asr.connect()
            except Exception as exc:
                log.warning("[watcher] asr reconnect error key=%s: %s", key, exc)
            if ok:
                # CAS ownership check: asr_stop/replacement may have popped this
                # exact object while connect() was awaiting the network.  A
                # late successful connect must not leave an untracked ghost WS
                # whose callbacks survive the stopped renderer turn.
                if self._asr.get(key) is not asr:
                    try:
                        try:
                            await asr.close(graceful=False)
                        except TypeError:
                            await asr.close()
                    except Exception:
                        pass
                    log.info(
                        "[watcher] discarded stale ASR reconnect key=%s", key)
                    return
                log.info("[watcher] asr reconnected key=%s", key)
            else:
                log.warning("[watcher] asr reconnect failed key=%s (will retry on next audio)", key)
        finally:
            self._asr_reconnecting.discard(key)

    def asr_stop(
        self,
        key: str,
        *,
        finish_timeout: float = 5.0,
        graceful: bool = True,
    ) -> dict:
        """Thread-safe: gracefully finish + close an ASR session.

        ``QwenRealtimeASR.close`` waits for the server's final transcription
        callbacks and ``session.finished``.  Block this synchronous gateway
        entrypoint only for that bounded grace period so the caller can safely
        merge the callback-owned segments before committing one manual turn.
        The ASR is removed from the live key map *before* waiting, preventing
        late PCM from entering a finishing turn.
        """
        loop = self._loop
        state_lock = getattr(self, "_asr_state_lock", None)
        if state_lock is None:
            state_lock = threading.RLock()
            self._asr_state_lock = state_lock
        with state_lock:
            if hasattr(self, "_asr_start_tokens"):
                self._asr_start_tokens.pop(key, None)
            asr = self._asr.pop(key, None)
        self._asr_reconnecting.discard(key)   # 清在飞重连标记 (会话已停)
        if loop is None or asr is None:
            return {
                "ok": True,
                "reason": "not_active",
                "completed": False,
                "session_finished": False,
                "timed_out": False,
            }
        try:
            async def _close_asr():
                try:
                    return await asr.close(
                        finish_timeout=finish_timeout,
                        graceful=graceful,
                    )
                except TypeError:
                    return await asr.close()

            fut = asyncio.run_coroutine_threadsafe(
                _close_asr(), loop)
            return dict(fut.result(timeout=max(0.1, finish_timeout + 1.0)))
        except FuturesTimeout:
            log.warning(
                "[watcher] asr_stop timed out key=%s after %.1fs",
                key, finish_timeout + 1.0,
            )
            return {
                "ok": False,
                "reason": "finish_timeout",
                "completed": False,
                "session_finished": False,
                "timed_out": True,
            }
        except Exception as exc:
            log.debug("[watcher] asr_stop failed: %s", exc)
            return {
                "ok": False,
                "reason": "finish_failed",
                "completed": False,
                "session_finished": False,
                "timed_out": False,
            }

    def submit_complex_async(self, task_instruction: str,
                             request_id: str = "") -> str:
        """Non-blocking: start a background live-watcher delegation.

        The live watcher has ONE mode: it walks the video from the head, round
        after round as frames accumulate, and keeps going until the video source
        stops (or the user stops it / it auto-wraps on prolonged no-new-content).
        There is no query-type classification.

        task_instruction: the user's observation/research instruction for this
        watcher (what to watch for + what to produce). Constant for the whole run
        (unless op=update changes it).

        request_id: when the caller (set_live_watcher) already created the
        analyse file with a rid, pass it so the file and the delegation share one
        id. Empty → a fresh id is generated.

        Idempotent: if a delegation for ``request_id`` is still live, returns the
        existing rid without starting a duplicate. Fires ``on_delegation_start``
        so the gateway can write a "running" placeholder message.

        Returns the request_id (``req_<8hex>``), or ``""`` when the engine isn't
        ready or submission failed.
        """
        if self._loop is None or self.responder is None:
            return ""
        import secrets
        rid = (request_id or "").strip() or ("req_" + secrets.token_hex(4))
        # BUG 7: refuse to start a SECOND delegation for a rid whose run is still
        # live (e.g. op=enable on an already-running watcher). Two runs share one
        # _stop_events[rid] + registry entry → duplicate reports/hooks. Return the
        # existing rid (idempotent) instead of spawning a duplicate.
        _existing = self._active.get(rid)
        if _existing is not None and not _existing.done():
            log.info("[watcher] delegation %s already live; not starting a duplicate", rid)
            return rid
        try:
            fut = asyncio.run_coroutine_threadsafe(
                self._run_delegation(rid, task_instruction),
                self._loop)
            self._active[rid] = fut

            # Surface exceptions that would otherwise vanish into the Future,
            # and drop the tracking entry when done.
            def _done(f, _rid=rid):
                self._active.pop(_rid, None)
                try:
                    exc = f.exception()
                except Exception:
                    exc = None
                if exc is not None and not isinstance(exc, asyncio.CancelledError):
                    log.warning("[watcher] delegation %s crashed: %s", _rid, exc)
            fut.add_done_callback(_done)
            # Fire the start hook so the gateway can write a "running" placeholder
            # assistant message keyed by this rid (backfilled on completion).
            if self._on_delegation_start is not None:
                try:
                    self._on_delegation_start(rid, task_instruction)
                except Exception as _exc:
                    log.warning("[watcher] on_delegation_start cb failed (%s): %s",
                                rid, _exc)
            return rid
        except Exception as exc:
            log.warning("[watcher] submit_complex_async failed: %s", exc)
            return ""

    def submit_query_async(
        self,
        task_instruction: str,
        *,
        task_id: str,
        parent_user_message_id: str,
        original_user_query: Optional[str] = None,
        ask_ts: Optional[float] = None,
    ) -> str:
        """Non-blocking one-shot Recall/Search answer owned by QueryWorker.

        This is intentionally a thin runtime wrapper around the already mature
        :meth:`WatcherWorker._spawn_delegation` ReAct loop.  QueryWorker receives
        the ask-time recent frames, lets the VLM Router decide whether direct
        current-frame answering is enough, and only fans out RecallAgent/Search
        when the question needs history or outside facts.  The terminal natural
        language answer streams straight to the original UI answer slot.  The
        main AIAgent is not in this return path.

        Returns the accepted task id once scheduled, the already-live task id
        when ``parent_user_message_id`` is submitted again, or ``""`` when the
        runtime/queue is unavailable.  Execution and waiting counts are bounded
        per session.
        """
        query = (task_instruction or "").strip()
        user_query = (original_user_query or query).strip()
        parent_id = (parent_user_message_id or "").strip()
        qid = (task_id or "").strip()
        if not query or not qid or not parent_id:
            return ""
        loop = self._loop
        if (
            self._stop.is_set()
            or not self._healthy
            or not self._ready.is_set()
            or loop is None
            or loop.is_closed()
            or not loop.is_running()
            or self.responder is None
            or self._query_semaphore is None
        ):
            return ""

        # Parent-message identity, not caller-generated random ids, defines one
        # logical query.  Reserve the parent + pending slot atomically before
        # scheduling so simultaneous tool calls cannot both pass admission.
        with self._query_lock:
            if self._stop.is_set() or not self._healthy:
                return ""
            existing_qid = self._query_by_parent.get(parent_id)
            if existing_qid:
                existing = self._active_queries.get(existing_qid)
                if (
                    existing_qid in self._query_pending
                    or existing_qid in self._query_running
                    or (existing is not None and not existing.done())
                ):
                    return existing_qid
                # A completed Future whose callback has not yet cleaned the
                # reverse index must not permanently block this parent.
                self._query_by_parent.pop(parent_id, None)
                self._active_queries.pop(existing_qid, None)
                self._query_pending.discard(existing_qid)
                self._query_running.discard(existing_qid)

            qid_future = self._active_queries.get(qid)
            if (
                qid in self._query_pending
                or qid in self._query_running
                or (qid_future is not None and not qid_future.done())
            ):
                # Same random id attached to another parent is ambiguous; only
                # the parent-id dedupe path above may reuse a task id.
                return ""

            max_admitted = (
                self._query_max_concurrency + self._query_max_pending)
            if len(self._query_by_parent) >= max_admitted:
                log.warning(
                    "[query-worker] queue full (running=%d pending=%d cap=%d+%d)",
                    len(self._query_running), len(self._query_pending),
                    self._query_max_concurrency, self._query_max_pending)
                return ""
            self._query_by_parent[parent_id] = qid
            self._query_pending.add(qid)

        emit = self._emit_cb

        def _emit_trajectory(phase: str, **payload) -> None:
            if emit is None:
                return
            try:
                emit("multimodal.trajectory", {
                    "worker": str(payload.pop("worker", "QueryWorker")),
                    "phase": phase,
                    "task_id": qid,
                    "parent_user_message_id": parent_id,
                    "query": user_query,
                    **({"worker_instruction": query}
                       if query != user_query else {}),
                    **payload,
                })
            except Exception:
                pass

        def _bounded_jpeg_b64(value: Any) -> str:
            """Return one complete bounded JPEG data payload, or ``""``.

            ``FrameStore.thumbnail_b64`` deliberately falls back to the input
            image when Pillow cannot decode/resize it.  That is useful for
            model paths, but unsafe for a debug WebSocket event: a full camera
            frame can be several megabytes and gateway-side string truncation
            would no longer be a renderable image.  Validate before emitting
            and omit an unusable preview instead of shipping partial base64.
            """
            raw_value = str(value or "").strip()
            if raw_value.startswith("data:"):
                comma = raw_value.find(",")
                raw_value = raw_value[comma + 1:] if comma >= 0 else ""
            if (
                not raw_value
                or len(raw_value) > _QUERY_DEBUG_JPEG_B64_MAX_CHARS
            ):
                return ""
            try:
                decoded = base64.b64decode(raw_value, validate=True)
            except Exception:
                return ""
            # The browser renders these specifically as image/jpeg.  Checking
            # both markers is cheap and prevents valid-base64 garbage from
            # becoming a broken debug image.
            if not (decoded.startswith(b"\xff\xd8")
                    and decoded.endswith(b"\xff\xd9")):
                return ""
            return raw_value

        def _build_ask_frame_previews(ask_frames: list) -> list[dict]:
            """Thumbnail the exact QueryWorker snapshot for debug display.

            The frames are intentionally not assigned fake ``frame_id`` values:
            ask-time FrameBuffer entries are not necessarily persisted in the
            FrameStore.  Preview construction is called only when an event sink
            exists, keeping non-UI/headless QueryWorker runs free of JPEG work.
            """
            if emit is None:
                return []
            previews: list[dict] = []
            frame_store = getattr(self, "frame_store", None)
            try:
                max_side = int(getattr(
                    self.cfg, "ui_event_thumb_max_side", 480) or 480)
            except (TypeError, ValueError):
                max_side = 480
            try:
                quality = int(getattr(
                    self.cfg, "ui_event_thumb_jpeg_quality", 70) or 70)
            except (TypeError, ValueError):
                quality = 70
            for frame in list(ask_frames or [])[:3]:
                original = str(getattr(frame, "jpeg_b64", "") or "")
                candidate = ""
                if frame_store is not None:
                    try:
                        candidate = frame_store.thumbnail_b64(
                            original,
                            max_side=max_side,
                            quality=quality,
                        )
                    except Exception as exc:
                        log.debug(
                            "[query-worker] ask-frame thumbnail failed: %s",
                            exc,
                        )
                # A resize failure may return/raise with the original image.
                # Use it only when it is already a complete, bounded JPEG.
                safe_b64 = (
                    _bounded_jpeg_b64(candidate)
                    or _bounded_jpeg_b64(original)
                )
                if not safe_b64:
                    continue
                previews.append({
                    "ts": float(getattr(frame, "ts", 0.0) or 0.0),
                    "source_type": str(
                        getattr(frame, "source_type", "") or ""),
                    "jpeg_b64": safe_b64,
                })
            return previews

        # Freeze an explicit ask-time snapshot synchronously, before the query
        # waits for a worker slot.  An empty list is meaningful: no frame was
        # available at/before ask_ts, and later frames must never be substituted.
        # When ask_ts is absent we intentionally keep the existing live behavior
        # and sample latest frames when the worker actually starts.
        frozen_ask_frames: Optional[list] = None
        if ask_ts is not None:
            try:
                frame_cap = min(3, max(1, int(
                    getattr(self.cfg, "cont_recent_frames", 3) or 3)))
                raw_all_le = getattr(self.frame_buffer, "raw_all_le", None)
                if callable(raw_all_le):
                    # One-shot visual QA intentionally bypasses the long-term
                    # dHash-sparse buffer.  Freeze the exact server-received
                    # captures at/before ask_ts; preview generation below uses
                    # this same object list, while the model receives their
                    # original JPEG payloads through ask_frames_override.
                    frozen_ask_frames = list(
                        raw_all_le(float(ask_ts), frame_cap) or [])
                else:
                    # Backward-compatible contract for custom/test buffers.
                    frozen_ask_frames = list(
                        self.frame_buffer.all_le(float(ask_ts)) or [])[-frame_cap:]
            except Exception:
                frozen_ask_frames = []

        async def _collect_query_ocr(
            ask_frames: list, effective_ask_ts: float,
        ) -> tuple[list[dict], str, str]:
            worker = getattr(self, "query_ocr_worker", None)
            if not ask_frames:
                return [], "skipped", "no_frozen_frames"
            if worker is None:
                return [], "skipped", "ocr_unavailable"
            try:
                records = list(await worker.collect_query_evidence(
                    list(ask_frames), ask_ts=float(effective_ask_ts)) or [])
                return records, ("available" if records else "empty"), ""
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # OCR is supplemental evidence.  A broken/missing OCR backend
                # must never prevent the exact frozen images from reaching the
                # QueryWorker VLM.
                log.warning("[query-ocr] evidence collection failed: %s", exc)
                return [], "error", "collection_failed"

        async def _execute_query() -> None:
            streamed_parts: list[str] = []
            authoritative_answer = {"text": ""}
            started_at = time.time()

            if frozen_ask_frames is not None:
                # Copy the frozen container so downstream code cannot mutate the
                # submission snapshot.  In particular, preserve explicit [].
                ask_frames = list(frozen_ask_frames)
            else:
                try:
                    frame_cap = min(3, max(1, int(
                        getattr(self.cfg, "cont_recent_frames", 3) or 3)))
                    ask_frames = list(
                        self.frame_buffer.latest(frame_cap) or [])
                except Exception:
                    ask_frames = []
            try:
                effective_ask_ts = float(
                    ask_ts
                    if ask_ts is not None
                    else (getattr(self.frame_buffer, "latest_ts", None) or 0.0))
            except Exception:
                effective_ask_ts = 0.0

            async def _sink(token: str) -> None:
                if not token:
                    return
                streamed_parts.append(token)
                if emit is not None:
                    try:
                        emit("message.delta", {
                            "source": "query_worker",
                            "request_id": parent_id,
                            "task_id": qid,
                            "text": token,
                        })
                    except Exception:
                        pass

            async def _on_event(event: Dict[str, Any]) -> None:
                ev = dict(event or {})
                if ev.get("type") == "answer_ready":
                    full = ev.get("answer_full")
                    if isinstance(full, str) and full.strip():
                        authoritative_answer["text"] = full.strip()
                typ = str(ev.get("type") or "progress")
                worker = "QueryWorker"
                if ("recall" in typ or ev.get("channel") == "recall"):
                    worker = "RecallWorker"
                elif "search" in typ or ev.get("channel") == "search":
                    worker = "SearchWorker"
                elif typ.startswith("router"):
                    worker = "QueryRouter"
                # Recall thumbnails are promoted to the normalized trajectory
                # surface consumed by the inspector. Keep them out of the nested
                # event copy to avoid storing the same base64 payload twice.
                trace_event = dict(ev)
                trace_frames = trace_event.pop("frames", None)
                # RecallAgent may emit the same thumbnails on fast-path/done and
                # the outer orchestration emits the verified set again in
                # recall_done.  For QueryWorker's live trajectory, serialize
                # images only once on recall_done; inner steps still carry
                # frame_ids/counts and remain fully auditable without doubling
                # large base64 payloads on the WebSocket.
                if (trace_event.get("channel") == "recall"
                        and typ == "bg_progress"):
                    trace_frames = None
                extra = {"frames": trace_frames} if isinstance(trace_frames, list) else {}
                _emit_trajectory(
                    typ, worker=worker, event=trace_event, **extra)

            # Attach thumbnails from the exact list passed below to
            # ``ask_frames_override``.  Never re-read FrameBuffer here: doing so
            # could show a newer frame than the QueryWorker actually inspected.
            try:
                ask_frame_previews = (
                    _build_ask_frame_previews(ask_frames)
                    if emit is not None else []
                )
            except Exception as exc:
                # Debug rendering must never prevent the actual QueryWorker
                # request from running or completing its parent answer slot.
                log.debug(
                    "[query-worker] ask-frame preview build failed: %s", exc)
                ask_frame_previews = []
            _emit_trajectory(
                "started", n_frames=len(ask_frames), ask_ts=effective_ask_ts,
                status="running", frames=ask_frame_previews)

            # The started event and exact frozen-frame previews must be visible
            # immediately.  OCR is supplemental work and may consume its full
            # deadline, so report it only after the task has visibly taken
            # ownership of the user's answer slot.
            ocr_started = time.time()
            try:
                total_ocr_timeout = max(0.1, float(
                    getattr(self.cfg, "ocr_timeout_sec", 8.0) or 8.0))
                (query_ocr_evidence, ocr_evidence_state,
                 ocr_evidence_reason) = await asyncio.wait_for(
                    _collect_query_ocr(ask_frames, effective_ask_ts),
                    timeout=total_ocr_timeout + 0.25,
                )
            except asyncio.TimeoutError:
                log.warning("[query-ocr] outer deadline exceeded")
                query_ocr_evidence = []
                ocr_evidence_state = "error"
                ocr_evidence_reason = "deadline_exceeded"
            debug_ocr_evidence = sanitize_query_ocr_debug_evidence(
                query_ocr_evidence)
            _emit_trajectory(
                "ocr_evidence",
                status=("error" if ocr_evidence_state == "error"
                        else "complete"),
                evidence_state=ocr_evidence_state,
                reason=ocr_evidence_reason,
                record_count=len(query_ocr_evidence),
                evidence=debug_ocr_evidence,
                evidence_sources=sorted({
                    str(item.get("evidence_source") or "")
                    for item in query_ocr_evidence
                    if isinstance(item, dict)
                    and str(item.get("evidence_source") or "")
                }),
                elapsed_sec=round(time.time() - ocr_started, 3),
            )
            status = "complete"
            answer = ""
            try:
                driving = await self.responder._spawn_delegation(
                    task_instruction=query,
                    prelude="",
                    sink=_sink,
                    on_event=_on_event,
                    # A no-anchor/no-frame live query stays unanchored so the
                    # inner worker may retain its legacy latest-frame fallback.
                    # Explicit ask_ts (including an empty snapshot) is strict.
                    ask_ts=(
                        effective_ask_ts
                        if ask_ts is not None or ask_frames
                        else None
                    ),
                    # ask_ts establishes snapshot semantics.  Preserve an empty
                    # snapshot as [] instead of collapsing it to "not provided".
                    # Without ask_ts, None retains the legacy live fallback.
                    ask_frames_override=(
                        ask_frames
                        if ask_ts is not None
                        else (ask_frames or None)
                    ),
                    force_initial_recall=False,
                    query_worker_mode=True,
                    query_ocr_evidence=query_ocr_evidence,
                    router_enable_thinking=False,
                )
                await driving
                answer = (
                    authoritative_answer["text"].strip()
                    or "".join(streamed_parts).strip()
                )
                if not answer:
                    answer = "QueryWorker 没有返回可靠结果。"
                    status = "error"
            except asyncio.CancelledError:
                status = "cancelled"
                answer = "该查询任务已取消。"
                raise
            except Exception as exc:
                status = "error"
                answer = f"QueryWorker 执行失败：{exc}"
                log.warning("[query-worker] task %s failed: %s", qid, exc,
                            exc_info=True)
            finally:
                elapsed = round(time.time() - started_at, 3)
                _emit_trajectory(
                    status, status=status, elapsed_sec=elapsed,
                    answer_len=len(answer), answer_preview=answer[:1000])
                # Session teardown sets _stop before cancelling query Futures.
                # Such cancellation must not complete an answer slot after the
                # session is already closing.  A non-teardown cancellation still
                # reports normally so an explicit task cancel remains visible.
                with self._query_lock:
                    suppress_delivery = self._stop.is_set()
                if not suppress_delivery:
                    delivered = False
                    cb = self._on_query_complete
                    if cb is not None:
                        try:
                            cb(qid, parent_id, user_query, answer, status)
                            delivered = True
                        except Exception as exc:
                            log.warning(
                                "[query-worker] completion callback failed (%s): %s",
                                qid, exc, exc_info=True)
                    if not delivered and emit is not None:
                        try:
                            emit("message.complete", {
                                "source": "query_worker",
                                "request_id": parent_id,
                                "task_id": qid,
                                "text": answer,
                                "status": status,
                            })
                        except Exception:
                            pass

        async def _run_query() -> None:
            semaphore = self._query_semaphore
            if semaphore is None:
                raise RuntimeError("QueryWorker semaphore is not initialized")
            acquired = False
            try:
                await semaphore.acquire()
                acquired = True
                with self._query_lock:
                    self._query_pending.discard(qid)
                    self._query_running.add(qid)
                await _execute_query()
            finally:
                with self._query_lock:
                    self._query_running.discard(qid)
                if acquired:
                    semaphore.release()

        try:
            with self._query_lock:
                # stop() flips health while holding the same lock.  Recheck at
                # the actual scheduling boundary because constructing the
                # callbacks above happens after the initial reservation.
                if (
                    self._stop.is_set()
                    or not self._healthy
                    or self._loop is not loop
                    or loop.is_closed()
                    or not loop.is_running()
                ):
                    if self._query_by_parent.get(parent_id) == qid:
                        self._query_by_parent.pop(parent_id, None)
                    self._query_pending.discard(qid)
                    return ""
                future = asyncio.run_coroutine_threadsafe(_run_query(), loop)
                self._active_queries[qid] = future

            def _done(fut, _qid=qid, _parent_id=parent_id):
                with self._query_lock:
                    if self._active_queries.get(_qid) is fut:
                        self._active_queries.pop(_qid, None)
                    if self._query_by_parent.get(_parent_id) == _qid:
                        self._query_by_parent.pop(_parent_id, None)
                    self._query_pending.discard(_qid)
                    self._query_running.discard(_qid)
                try:
                    exc = fut.exception()
                except Exception:
                    exc = None
                if exc is not None and not isinstance(
                    exc, (asyncio.CancelledError, FuturesTimeout)
                ):
                    log.warning("[query-worker] task %s crashed: %s", _qid, exc)

            future.add_done_callback(_done)
            return qid
        except Exception as exc:
            with self._query_lock:
                if self._query_by_parent.get(parent_id) == qid:
                    self._query_by_parent.pop(parent_id, None)
                self._active_queries.pop(qid, None)
                self._query_pending.discard(qid)
                self._query_running.discard(qid)
            log.warning("[query-worker] submit failed: %s", exc, exc_info=True)
            return ""

    def query_visual_evidence(
        self,
        task_instruction: str,
        *,
        original_user_query: str = "",
        ask_ts: Optional[float] = None,
        timeout: float = 45.0,
    ) -> Dict[str, Any]:
        """Synchronously collect ask-time visual grounding for the Main Agent.

        Unlike :meth:`submit_query_async`, this path does not own or complete a
        user message. It runs QueryWorker as a perception-only VLM with no
        Recall/Search tools, then returns its bounded observation as an ordinary
        tool result so the Main Agent can continue with PDF/file/terminal/
        browser/skill orchestration.
        """
        instruction = (task_instruction or "").strip()
        if not instruction:
            return {"ok": False, "error": "visual evidence query is required"}

        loop = self._loop
        if (
            self._stop.is_set()
            or not self._healthy
            or not self._ready.is_set()
            or loop is None
            or loop.is_closed()
            or not loop.is_running()
            or self.responder is None
            or self._query_semaphore is None
        ):
            return {"ok": False, "error": "QueryWorker is not ready"}
        if self._thread is threading.current_thread():
            return {
                "ok": False,
                "error": "visual evidence collection cannot block its own event loop",
            }

        started = time.monotonic()
        timeout_sec = max(1.0, float(timeout or 45.0))
        try:
            frame_cap = min(3, max(1, int(
                getattr(self.cfg, "cont_recent_frames", 3) or 3)))
            if ask_ts is not None:
                raw_all_le = getattr(self.frame_buffer, "raw_all_le", None)
                if callable(raw_all_le):
                    frozen_frames = list(
                        raw_all_le(float(ask_ts), frame_cap) or [])
                else:
                    frozen_frames = list(
                        self.frame_buffer.all_le(float(ask_ts)) or [])[-frame_cap:]
            else:
                frozen_frames = list(
                    self.frame_buffer.latest(frame_cap) or [])
        except Exception:
            frozen_frames = []

        try:
            effective_ask_ts = float(
                ask_ts
                if ask_ts is not None
                else (getattr(self.frame_buffer, "latest_ts", None) or 0.0)
            )
        except (TypeError, ValueError):
            effective_ask_ts = 0.0

        if not frozen_frames:
            return {
                "ok": True,
                "evidence": (
                    "Observed: no ask-time frame was available.\n"
                    "Relevant text/identifiers: none.\n"
                    "Grounding: no visual claim can be established.\n"
                    "Uncertainty: high.\n"
                    "Recommended next capability: ask the user to share the "
                    "screen or expose the target artifact explicitly."
                ),
                "ask_ts": effective_ask_ts,
                "n_frames": 0,
                "t_start": None,
                "t_end": None,
                "limitations": [
                    "No frame was captured at or before the question timestamp.",
                    "No Search, Recall, file, browser, terminal, or skill was run.",
                ],
                "elapsed_sec": round(time.monotonic() - started, 3),
            }

        user_query = (original_user_query or instruction).strip()
        worker_instruction = instruction
        if user_query and user_query != instruction:
            worker_instruction = (
                "Original user request (context only; do not answer it end-to-end):\n"
                f"{user_query}\n\n"
                "Main Agent visual-grounding brief:\n"
                f"{instruction}"
            )

        evidence_id = "evi_" + uuid.uuid4().hex[:8]
        with self._query_lock:
            admitted = len(self._query_pending) + len(self._query_running)
            capacity = self._query_max_concurrency + self._query_max_pending
            if admitted >= capacity:
                return {
                    "ok": False,
                    "error": "visual evidence query queue is full",
                    "queue_full": True,
                }
            self._query_pending.add(evidence_id)

        async def _run_evidence() -> Dict[str, Any]:
            semaphore = self._query_semaphore
            if semaphore is None:
                raise RuntimeError("QueryWorker semaphore is not initialized")
            acquired = False
            streamed_parts: list[str] = []
            authoritative_answer = {"text": ""}
            try:
                remaining = max(
                    0.1, timeout_sec - (time.monotonic() - started))
                await asyncio.wait_for(semaphore.acquire(), timeout=remaining)
                acquired = True
                with self._query_lock:
                    self._query_pending.discard(evidence_id)
                    self._query_running.add(evidence_id)

                query_ocr_evidence: list[dict] = []
                ocr_worker = getattr(self, "query_ocr_worker", None)
                if ocr_worker is not None:
                    try:
                        ocr_timeout = max(0.1, min(
                            remaining,
                            float(getattr(
                                self.cfg, "ocr_timeout_sec", 8.0) or 8.0)
                            + 0.25,
                        ))
                        query_ocr_evidence = list(await asyncio.wait_for(
                            ocr_worker.collect_query_evidence(
                                list(frozen_frames),
                                ask_ts=effective_ask_ts,
                            ),
                            timeout=ocr_timeout,
                        ) or [])
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        log.warning(
                            "[query-evidence] supplemental OCR failed: %s", exc)

                async def _sink(token: str) -> None:
                    if token:
                        streamed_parts.append(token)

                async def _on_event(event: Dict[str, Any]) -> None:
                    ev = dict(event or {})
                    if ev.get("type") == "answer_ready":
                        full = ev.get("answer_full")
                        if isinstance(full, str) and full.strip():
                            authoritative_answer["text"] = full.strip()

                driving = await self.responder._spawn_delegation(
                    task_instruction=worker_instruction,
                    prelude="",
                    sink=_sink,
                    on_event=_on_event,
                    ask_ts=effective_ask_ts,
                    ask_frames_override=list(frozen_frames),
                    force_initial_recall=False,
                    query_worker_mode=True,
                    evidence_only_mode=True,
                    query_ocr_evidence=query_ocr_evidence,
                    router_enable_thinking=False,
                )
                await driving
                evidence = (
                    authoritative_answer["text"].strip()
                    or "".join(streamed_parts).strip()
                )
                if not evidence:
                    return {
                        "ok": False,
                        "error": "QueryWorker returned no visual evidence",
                    }
                return {
                    "ok": True,
                    "evidence": evidence,
                    "ask_ts": effective_ask_ts,
                    "n_frames": len(frozen_frames),
                    "t_start": float(frozen_frames[0].ts),
                    "t_end": float(frozen_frames[-1].ts),
                    "limitations": [
                        "Evidence is limited to the frozen ask-time frames and supplemental OCR.",
                        "No Search, Recall, file, browser, terminal, or skill was run.",
                        "The Main Agent must verify downstream artifact contents with the appropriate tool.",
                    ],
                    "elapsed_sec": round(time.monotonic() - started, 3),
                }
            finally:
                with self._query_lock:
                    self._query_pending.discard(evidence_id)
                    self._query_running.discard(evidence_id)
                if acquired:
                    semaphore.release()

        future = None
        try:
            future = asyncio.run_coroutine_threadsafe(_run_evidence(), loop)
            return future.result(timeout=timeout_sec)
        except FuturesTimeout:
            if future is not None:
                future.cancel()
            with self._query_lock:
                self._query_pending.discard(evidence_id)
                self._query_running.discard(evidence_id)
            return {
                "ok": False,
                "error": "visual evidence collection timed out",
                "timed_out": True,
            }
        except Exception as exc:
            if future is not None:
                future.cancel()
            with self._query_lock:
                self._query_pending.discard(evidence_id)
                self._query_running.discard(evidence_id)
            log.warning(
                "[query-evidence] collection failed: %s", exc, exc_info=True)
            return {"ok": False, "error": f"visual evidence collection failed: {exc}"}

    # ------------------------------------------------------------------ #
    def recall_memory(self, brief: str, user_text: str = "",
                      timeout: Optional[float] = None) -> Dict[str, Any]:
        """Run one bounded QueryWorker recall job.

        At most ``query_worker_max_concurrency`` jobs execute concurrently and
        only ``query_worker_max_pending`` additional callers may wait. This
        protects the backend loop without changing the main-agent tool contract.
        """
        started = time.monotonic()
        with self._query_lock:
            admitted = self._query_running + self._query_pending
            cap = self._query_max_concurrency + self._query_max_pending
            if admitted >= cap:
                log.warning(
                    "[query-worker] queue full running=%d pending=%d cap=%d+%d",
                    self._query_running, self._query_pending,
                    self._query_max_concurrency, self._query_max_pending)
                return {
                    "ok": False,
                    "error": "memory query queue is full",
                    "queue_full": True,
                }
            self._query_pending += 1

        wait_timeout = None if timeout is None else max(0.0, float(timeout))
        acquired = self._query_semaphore.acquire(timeout=wait_timeout)
        with self._query_lock:
            self._query_pending = max(0, self._query_pending - 1)
            if acquired:
                self._query_running += 1
        if not acquired:
            return {
                "ok": False,
                "error": "memory query timed out while waiting for a worker",
                "timed_out": True,
            }

        try:
            remaining = timeout
            if timeout is not None:
                remaining = max(1.0, float(timeout) - (time.monotonic() - started))
            return self._recall_memory_unbounded(
                brief, user_text=user_text, timeout=remaining)
        finally:
            with self._query_lock:
                self._query_running = max(0, self._query_running - 1)
            self._query_semaphore.release()

    def _recall_memory_unbounded(self, brief: str, user_text: str = "",
                                 timeout: Optional[float] = None) -> Dict[str, Any]:
        """BLOCKING: run the RecallAgent's full ReAct memory-recall loop and
        return its distilled findings. This is the main agent's synchronous entry
        into the mature RecallAgent (vs. set_live_watcher, which re-watches
        the raw video in the background).

        RecallAgent.run() is a multi-round LLM loop over the memory graph with
        per-round distillation and optional frame verification. When a
        MemoryBackend with its own recall_agent + loop is present, the call is
        delegated to backend.recall() (unified entry, backend loop); otherwise
        it marshals onto THIS engine's loop via run_coroutine_threadsafe. Either
        way it blocks (with optional timeout) so the main agent can answer THIS
        turn.

        Returns a dict:
          {ok, found, findings, clues, frame_ids, rounds, elapsed_sec}
        ``ok=False`` + ``error`` on engine-not-ready / timeout / crash so the tool
        can report cleanly (``timed_out=True`` on timeout). Never raises.
        """
        brief = (brief or "").strip()
        # ★ 重构: 记忆召回子 agent (RecallAgent) 归 MemoryBackend 管理并有独立模型。
        #   有 backend 就走 backend 统一入口(marshal 到 backend loop); 否则用本引擎的
        #   本地兜底 recall_agent(engine 单跑场景)。主 Agent 工具与 WatcherWorker 同源。
        mb = self._memory_backend
        if mb is not None and getattr(mb, "recall_agent", None) is not None \
                and getattr(mb, "_loop", None) is not None:
            return mb.recall(brief, user_text=user_text, timeout=timeout)
        if self._loop is None or self.recall_agent is None:
            return {"ok": False, "error": "engine not ready (recall unavailable)"}
        if not brief:
            return {"ok": False, "error": "brief is required"}

        # Anchor reads at the current stream time (RecallWorker enforces the D3
        # anti-dirty-read guard against this ask_ts). None → 0.0 lets the worker
        # fall back internally.
        try:
            # FrameBuffer.latest_ts is a @property, not a method.
            ask_ts = self.frame_buffer.latest_ts if self.frame_buffer else None
        except Exception:
            ask_ts = None
        ask_ts = float(ask_ts) if ask_ts is not None else 0.0

        try:
            fut = asyncio.run_coroutine_threadsafe(
                self.recall_agent.run(
                    initial_calls=[], brief=brief,
                    user_text=(user_text or brief), ask_ts=ask_ts),
                self._loop)
        except Exception as exc:
            log.warning("[watcher] recall_memory submit failed: %s", exc)
            return {"ok": False, "error": f"submit failed: {exc}"}

        # timeout=None → 不设等待墙 (调用方未显式给)。等待墙是编排层的选择,
        # 引擎不预设魔法默认。单次 LLM 调用超时统一走主 Agent 的
        # providers.<id>.request_timeout_seconds (见 hermes_glue._submodule_http_client)。
        _wall = None if timeout is None else max(1.0, float(timeout))
        try:
            res = fut.result(timeout=_wall)
        except FuturesTimeout:
            # Don't leave the coroutine running unbounded on the engine loop.
            try:
                fut.cancel()
            except Exception:
                pass
            log.warning("[watcher] recall_memory timed out after %.1fs", _wall)
            return {"ok": False, "error": f"recall timed out after {_wall:.0f}s",
                    "timed_out": True}
        except Exception as exc:
            log.warning("[watcher] recall_memory crashed: %s", exc)
            return {"ok": False, "error": f"recall failed: {exc}"}

        findings = (getattr(res, "findings", "") or "").strip()
        clues = list(getattr(res, "clues", None) or [])
        frame_ids = list(getattr(res, "frame_ids", None) or [])
        # RecallWorker returns a sentinel string when nothing was found.
        from agent.multimodal._sentinels import RECALL_NO_CLUES
        found = bool(findings) and findings != RECALL_NO_CLUES
        return {
            "ok": True,
            "found": found,
            "findings": findings,
            "clues": clues,
            "frame_ids": frame_ids,
            "rounds": int(getattr(res, "rounds", 0) or 0),
            "elapsed_sec": round(float(getattr(res, "elapsed_sec", 0.0) or 0.0), 2),
        }

    # ------------------------------------------------------------------ #
    async def _run_delegation(self, request_id: str,
                              task_instruction: str) -> None:
        """The one live-watcher delegation loop (standard multimodal ReAct).

        Every round runs the SAME per-batch delegation
        (WatcherWorker._spawn_delegation = multi-round ReAct on a frame batch),
        logs each round to analyse/watch_<rid>.md, pushes an incremental report,
        and ends with ONE final LLM summary — batch after batch from the video
        head, as new frames accumulate. There is no query-type classification.

        The run stops when: the video source stops AND the remaining frames are
        consumed, the worker conclusively signals the task's explicit finite
        completion condition, the user/main-agent calls stop_delegation, or
        op=delete arrives on the registry. No progress (no new frames / no
        findings) does NOT stop the run — it just keeps waiting. cfg.watch_max_rounds defaults
        to 0 = unlimited; only a value >0 acts as a runaway hard cap.

        Round pacing is a TTL + frame-count gate (see _round_pacing): a round
        fires once it has target_frames new frames or ttl_sec elapses. New frames
        are the buffer's frames with ts past a cursor (dedup already done at
        FrameBuffer entry); congested rounds are evenly downsampled to
        target_frames. Search sub-queries are deduped across rounds via a shared
        seen set. Sets stop_reason (deleted / disabled / source_end /
        task_complete / normal) for terminal state and completion-hook policy.
        """
        from . import watch_file as _df
        emit = self._emit_cb
        rid = request_id

        poll = float(getattr(self.cfg, "watch_poll_interval", 2.0) or 2.0)
        static_tail_flush_sec = max(0.1, float(getattr(
            self.cfg, "watch_static_tail_flush_sec", 2.0) or 2.0))
        task_lower = str(task_instruction or "").lower()
        finite_completion_requested = any(token in task_lower for token in (
            "结束", "播放完", "播完", "看完", "完成后", "结束后",
            "until", "ends", "ended", "finish", "finished", "complete",
        ))
        max_rounds = int(getattr(self.cfg, "watch_max_rounds", 0) or 0)  # 0/负=不限
        # ★ 帧降采样已上移到 FrameBuffer 入口 (dHash 去重), watcher 不再 stride。
        #   buffer 里已是去重后的稀疏帧, 直接取即可。

        # ★ TTL + 帧数双门: 每轮攒到 (帧数达 target_frames) 或 (ttl 到) 就执行,
        #   帧数不足但 ttl 到 → 取手头全部。pacing 每轮从 registry(set_live_watcher
        #   给的 ttl)刷新, registry 没有则从 FrameBuffer.current_scene(自动检测的
        #   场景)兜底, 再兜底 medium。四档: 200s/100 · 60s/60 · 30s/40 · 10s/15。
        def _round_pacing(reg) -> tuple:
            """Return (ttl_sec, target_frames) for this round."""
            def _clamp(ttl, tf):
                # Defensive floor: a bad config / registry value must never make
                # the round gate 0 (→ ttl=0 busy-spin / target=0 empty batches).
                # Tiny positive floor (not a product min) so tests can use 0.1s.
                return max(0.05, float(ttl)), max(1, int(tf))
            # 1) explicit ttl from set_live_watcher (stored on the registry entry)
            if isinstance(reg, dict) and reg.get("ttl_sec") and reg.get("target_frames"):
                return _clamp(reg["ttl_sec"], reg["target_frames"])
            # 2) auto-detected current scene on the FrameBuffer
            try:
                scene = self.frame_buffer.current_scene
                if isinstance(scene, dict) and scene.get("ttl_sec") and scene.get("target_frames"):
                    return _clamp(scene["ttl_sec"], scene["target_frames"])
            except Exception:
                pass
            # 3) explicit cfg override (tests / advanced tuning): a bare
            #    watch_min_batch (+ optional watch_round_ttl_sec) sets the gate
            #    without a scene. Production always has (1) or (2).
            _mb = getattr(self.cfg, "watch_min_batch", None)
            if _mb:
                _ttl = float(getattr(self.cfg, "watch_round_ttl_sec", 120.0) or 120.0)
                return _clamp(_ttl, _mb)
            # 4) medium default
            return 120.0, 64

        # Ensure the analyse file exists (set_live_watcher normally creates
        # it first, but be defensive so the log is never lost).
        try:
            _df.init_file(rid, query=task_instruction)
        except Exception as _exc:
            log.warning("[watcher] init watch file failed (%s): %s", rid, _exc)

        # Register a stop event so stop_delegation() can end a run early.
        stop_ev = asyncio.Event()
        self._stop_events[rid] = stop_ev
        stop_reasons = getattr(self, "_stop_reasons", None)
        if stop_reasons is None:
            stop_reasons = {}
            self._stop_reasons = stop_reasons
        stop_reasons.pop(rid, None)

        # Some focused tests construct a minimal WatcherAgent via ``__new__``.
        # Keep the lifecycle fields defensive without weakening the production
        # lock used by normal ``__init__`` instances.
        state_lock = getattr(self, "_state_lock", None)
        if state_lock is None:
            state_lock = threading.RLock()
            self._state_lock = state_lock
        with state_lock:
            run_source_epoch = int(getattr(self, "_source_epoch", 0) or 0)
            self._source_epoch = run_source_epoch
            source_stopped_at_start = bool(
                getattr(self, "_source_stopped", False))

        def _source_replaced() -> bool:
            """Whether a newer capture generation superseded this run.

            A plain source stop keeps the epoch stable so the run may drain its
            already-buffered tail.  A subsequent source start advances the epoch;
            from that point this run must never inspect or report frames belonging
            to the replacement stream.
            """
            with state_lock:
                return int(getattr(self, "_source_epoch", 0) or 0) != run_source_epoch

        def _ended() -> bool:
            """Stop waiting after source close/replacement or an explicit stop."""
            with state_lock:
                source_ended = (
                    bool(getattr(self, "_source_stopped", False))
                    or int(getattr(self, "_source_epoch", 0) or 0)
                    != run_source_epoch
                )
            return source_ended or stop_ev.is_set()

        async def _run_one_batch(cur_task_instruction: str,
                                 frames, seen_subqueries: set,
                                 seg_idx: int = 0, frame_range=None,
                                 prev_segment=None,
                                 static_tail_check: bool = False):
            """Run one delegation batch.

            Returns the round answer, success flag, new subqueries, conclusive
            completion signal/reason, and an optional ambiguous completion
            candidate/reason.  A candidate never ends the task directly; the
            outer loop waits for a no-new-scene grace period and verifies it.

            seg_idx / frame_range: stamped onto every forwarded bg event so the
            DeepPanel can group a round's progress under one readable segment
            card (segment number + mm:ss–mm:ss range) rather than a flat log.
            """
            parts: list[str] = []
            batch_subqueries: list[str] = []
            # Authoritative full answer captured from the answer_ready event, so
            # the round report doesn't depend on the _sink side-channel (BUG 5).
            batch_answer = {"text": ""}
            batch_completion = {"complete": False, "reason": ""}
            batch_candidate = {"candidate": False, "reason": ""}
            # Frame time-range for this segment's header (mm:ss–mm:ss).
            _fr = frame_range
            _seg_meta = {"seg": int(seg_idx)}
            if _fr and len(_fr) == 2:
                _seg_meta["frame_ts_range"] = [float(_fr[0]), float(_fr[1])]
            # Segment header marker — one readable card per round.
            if emit:
                try:
                    emit("multimodal.bg", {
                        "request_id": rid, "channel": "bg",
                        "type": "segment_start", **_seg_meta,
                    })
                except Exception:
                    pass

            async def _sink(token: str) -> None:
                # Collect tokens to assemble this round's full answer (for the
                # round report), AND stream them live to the watcher panel so the
                # user sees the interpretation appear token-by-token (not all at
                # once). The stream is stamped with `seg` so the frontend appends
                # into THIS segment card's answer; it does not touch the center chat.
                if not token or _source_replaced():
                    return
                parts.append(token)
                if emit is not None:
                    try:
                        emit("multimodal.bg", {
                            "request_id": rid, "channel": "bg",
                            "type": "answer_delta", "delta": token, **_seg_meta,
                        })
                    except Exception:
                        pass

            async def _on_event(ev: dict) -> None:
                if _source_replaced():
                    return
                # Capture dispatched sub-query briefs for the analyse log + dedup,
                try:
                    if ev.get("type") == "router_react":
                        # ★ 段场景标记: 首次拿到非空 thought 时廉价提取一个简短标签,
                        #   补发一条 segment_start 让前端标题行刷出"场景标记"。只发一次。
                        if "scene_label" not in _seg_meta:
                            _lbl = _scene_label_from(str(ev.get("thought", "") or ""))
                            if _lbl:
                                _seg_meta["scene_label"] = _lbl
                                if emit is not None:
                                    try:
                                        emit("multimodal.bg", {
                                            "request_id": rid, "channel": "bg",
                                            "type": "segment_start", **_seg_meta,
                                        })
                                    except Exception:
                                        pass
                        for st in (ev.get("search_tasks") or []):
                            b = str((st or {}).get("brief", "")).strip()
                            if b:
                                batch_subqueries.append(b)
                        for rt in (ev.get("recall_tasks") or []):
                            b = str((rt or {}).get("brief", "")).strip()
                            if b:
                                batch_subqueries.append(b)
                    elif ev.get("type") == "answer_ready":
                        # Authoritative full answer for the round report (BUG 5).
                        _af = ev.get("answer_full")
                        if isinstance(_af, str) and _af.strip():
                            batch_answer["text"] = _af
                        batch_completion["complete"] = bool(
                            ev.get("task_complete"))
                        batch_completion["reason"] = str(
                            ev.get("completion_reason") or "").strip()
                        batch_candidate["candidate"] = bool(
                            ev.get("completion_candidate"))
                        batch_candidate["reason"] = str(
                            ev.get("completion_candidate_reason") or "").strip()
                except Exception:
                    pass
                if emit is None:
                    return
                try:
                    # Stamp seg (+ frame range) so the panel groups this round's
                    # events under its segment card. Strip answer_full from the
                    # wire payload (frontend doesn't need the full text here).
                    _out = {k: v for k, v in ev.items() if k != "answer_full"}
                    emit("multimodal.bg", {**_out, "request_id": rid, **_seg_meta})
                except Exception:
                    pass

            # NOTE: no message.start/complete (source=watcher) here — the round
            # answer is delivered ONLY via on_round_report → watcher.report_append
            # in the worker panel. Right-panel progress rides multimodal.bg.
            try:
                task = await self.responder._spawn_delegation(
                    task_instruction=cur_task_instruction, prelude="",
                    sink=_sink, on_event=_on_event,
                    ask_frames_override=(frames or None),
                    # Cross-batch dedup: share the run-level seen set so the same
                    # search sub-query isn't re-issued across batches either.
                    seen_search_briefs=seen_subqueries,
                    # ★ 上一段 (时间窗+报告) → 本段 system prompt 的增量提示。
                    prev_segment=prev_segment,
                    static_tail_check=static_tail_check)
                try:
                    await task
                except Exception as exc:
                    log.warning("[watcher] batch task error (%s): %s",
                                rid, exc, exc_info=True)
            except Exception as exc:
                log.warning("[watcher] batch failed (%s): %s", rid, exc,
                            exc_info=True)
                _reason = str(exc)[:200]
                parts.append(f"\n[深度分析本轮失败: {_reason}]")
            if _source_replaced():
                return "", False, [], False, "", False, ""
            # Prefer the authoritative answer captured from answer_ready; fall back
            # to the assembled sink tokens (BUG 5: the report must not depend on the
            # _fake_stream side channel, which is skipped for some answers).
            full_text = batch_answer["text"].strip() or "".join(parts)
            ok = bool(full_text.strip())
            # Real cross-round/cross-batch dedup now happens INSIDE
            # _spawn_delegation (it filters step.search_tasks against the shared
            # seen_subqueries set and skips already-searched ones). Here we only
            # record, for the analyse file, the DISTINCT sub-queries this batch
            # planned (deduped for display; the actual skip already occurred).
            new_sqs: list[str] = []
            _local_seen: set = set()
            for b in batch_subqueries:
                key = " ".join(b.lower().split())
                if key and key not in _local_seen:
                    _local_seen.add(key)
                    new_sqs.append(b)
            return (
                full_text,
                ok,
                new_sqs,
                bool(batch_completion["complete"]),
                str(batch_completion["reason"] or ""),
                bool(batch_candidate["candidate"]),
                str(batch_candidate["reason"] or ""),
            )

        # ★ Startup snapshot: decide one-shot vs continuous.
        #   AUTHORITATIVE signal = self._source_stopped (set by the frontend's
        #   explicit multimodal.source_stopped). We do NOT use a frame-idle
        #   heuristic here: a live stream can have a >idle_stop gap right at
        #   launch (tab throttle, a slow first analysis, a static scene) and the
        #   old `(now - _last_push_wall) < idle_stop` check then wrongly declared
        #   the source "stopped" and forced a one-shot — the exact "明明有视频流
        #   却报启动时无视频流" bug. Treat as live UNLESS the source was
        #   explicitly stopped OR nothing was ever captured (buffer empty AND no
        #   push ever), in which case there's genuinely nothing to keep waiting
        #   for → one-shot.
        _last_push0 = getattr(self.frame_buffer, "_last_push_wall", None)
        _never_captured = (self.frame_buffer.size == 0 and _last_push0 is None)
        source_live_at_start = (
            not source_stopped_at_start and not _never_captured
        )
        if not source_live_at_start:
            try:
                _df.append_note(
                    rid, "启动时视频源已停止/从未开启 → 一次性分析已缓冲的帧后结束。")
            except Exception:
                pass

        def _gather(cursor):
            """All frames after `cursor` (strictly >). cursor=None → from the
            buffer HEAD (earliest retained frame). Frames are already dHash-deduped
            at the FrameBuffer entry. NOT capped here — the ttl/frames gate + the
            congestion downsample below bound how many actually go to the LLM."""
            if cursor is None:
                return self.frame_buffer.latest(self.frame_buffer.size)  # head→tail
            return [f for f in self.frame_buffer.all_after(cursor) if f.ts > cursor]

        def _even_downsample(frames, n):
            """Keep n frames evenly across `frames` (inclusive of first + last).
            Used when a congested round accumulated more than target_frames."""
            if n <= 0 or not frames or len(frames) <= n:
                return frames
            if n == 1:
                return [frames[-1]]
            step = (len(frames) - 1) / (n - 1)
            idxs = sorted({int(round(i * step)) for i in range(n)})
            return [frames[i] for i in idxs]

        def _static_tail_remaining(frames, *, cursor=None) -> Optional[float]:
            """Seconds until a finite-task partial batch should be flushed.

            The dHash buffer contains only novel scenes, while monitor_latest_ts
            advances for every raw capture. Their timestamp gap therefore measures
            a visually static tail without consulting wall-clock frame silence.
            A dead capture cannot open this gate because its raw timestamp stops.

            After at least one batch has been consumed, a completely static tail
            legitimately has *zero* novel frames.  In that case the exclusive
            dHash cursor is the last analyzed visual timestamp and is the correct
            baseline for the raw-capture gap.  Requiring ``frames`` here used to
            strand finite video tasks at 0/N until the normal TTL expired and then
            in the indefinite "waiting for new frames" state.
            """
            if not finite_completion_requested:
                return None
            try:
                raw_ts = self.frame_buffer.monitor_latest_ts
                if raw_ts is None:
                    return None
                if frames:
                    baseline_ts = float(frames[-1].ts)
                elif cursor is not None:
                    baseline_ts = float(cursor)
                else:
                    # The first batch has no trustworthy analyzed boundary yet.
                    # Keep its existing accumulation semantics rather than
                    # treating an arbitrary raw-buffer age as a static ending.
                    return None
                static_for = max(
                    0.0, float(raw_ts) - baseline_ts)
                return max(0.0, static_tail_flush_sec - static_for)
            except Exception:
                return None

        try:
            seen_subqueries: set = _df.read_seen_subqueries(rid)
        except Exception:
            seen_subqueries = set()
        # Cursor: only frames with ts > cursor_ts are "new" (unconsumed).
        # ★ 首轮只看【最近一段】(不是从 buffer HEAD 吞整段积压): 视频流可能已跑很久,
        #   HEAD 取全部会让首轮输入巨大 / 时序被均匀稀释成无意义的概览。所以首轮把
        #   cursor 定位到"末尾往前 target_frames 帧"的起点, 有多少最近帧就用多少,
        #   立即分析 (不等累积), 旧积压丢弃。之后逐轮往前追新帧 (ttl+帧数双门)。
        cursor_ts: Optional[float] = None   # 首轮下方特殊处理; 之后 = 上轮末帧 ts
        # ★ 段号跨"中断→重启"连续: 从注册表 _seg_base 起算, 而非每次重启都从 0/1 重来。
        #   否则重启后新段 seg=1,2… 会跟中断前的旧段同号 → 前端按 seg 路由把新段合并/
        #   覆盖进旧段卡片 (看起来"不追加/被掩盖"), 主 agent 报告计数也错。
        _seg_base = 0
        persisted_state = {}
        try:
            persisted_state = _df.read_state(rid) or {}
            _seg_base = max(
                int(persisted_state.get("seg_base") or 0),
                int(_df.count_rounds(rid) or 0),
            )
        except Exception:
            persisted_state = {}
            _seg_base = 0
        try:
            cb0 = getattr(self, "_research_registry_cb", None)
            _reg0 = cb0(rid) if cb0 else None
            if isinstance(_reg0, dict):
                _seg_base = max(
                    _seg_base, int(_reg0.get("_seg_base") or 0))
        except Exception:
            pass
        runtime_id = str(getattr(self, "_runtime_id", "") or "")
        if (runtime_id
                and str(persisted_state.get("runtime_id") or "") == runtime_id):
            try:
                saved_cursor = persisted_state.get("cursor_ts")
                cursor_ts = (float(saved_cursor)
                             if saved_cursor is not None else None)
            except (TypeError, ValueError):
                cursor_ts = None
        round_idx = _seg_base
        any_round_ok = False

        # ★ v7 优化状态:
        #  #9 累积报告: running_report 跨批累积每批解读, 收尾定稿 + 周期增量推送。
        #  #10 无进展【不再自动收尾】(用户要求: 无进展保持继续状态)。no_progress 仅作日志
        #     计数; 结束只靠 源显式停止(source_stopped) / 用户停(stop_ev) / max_rounds 硬上限。
        #     ——避免网络卡顿导致的"暂时无帧"被误判为流结束而收尾。
        #  (画面级 dHash 去重已上移到 FrameBuffer 入口, watcher 不再自己做。)
        from ._memory import fmt_ts
        _report_every = int(getattr(self.cfg, "watch_report_every_rounds", 3) or 3)
        running_report: list[str] = []       # 每批一条解读 (#9)
        # ★ 上一段 (时间窗+报告) → 传给下一段做增量提示 (只保留紧邻上一段, N=1)。
        prev_segment: Optional[dict] = None
        # A segment may mark an ambiguous ending without terminating the task.
        # It is verified once only if no dHash-novel scene follows during the
        # configured grace period.
        pending_completion_candidate: Optional[dict] = None
        # A rejected rule-based static check remains latched until a genuinely
        # novel frame arrives. Without this latch the same frozen pixels trigger
        # another expensive VLM request every poll cycle forever.
        static_tail_checked_anchor_ts: Optional[float] = None
        terminal_completion_observation = ""
        no_progress = 0                      # 连续无进展周期数 (仅日志)
        batches_since_report = 0             # 距上次增量推送的批数 (#2)

        # ★ Watcher label for the UI (deep-analysis card title). Read from the
        #   tool-layer registry via the callback; fall back to task_instruction.
        #   Kept as a single lookup here so every bg emit can carry it.
        def _reg_label() -> str:
            cb = getattr(self, "_research_registry_cb", None)
            if cb is not None:
                try:
                    ent = cb(rid)
                    if ent:
                        lbl = str(ent.get("label") or "").strip()
                        if lbl:
                            return lbl
                except Exception:
                    pass
            b = (task_instruction or "").strip()
            return (b[:18] + "…") if len(b) > 18 else b

        _label = _reg_label()

        def _emit_progress_report() -> None:
            """Push the accumulated report so far to the frontend as an
            incremental progress_report bg event (without waiting for the run
            to finish)."""
            if emit is None or not running_report:
                return
            try:
                emit("multimodal.bg", {
                    "request_id": rid, "channel": "bg",
                    "type": "progress_report",
                    "label": _label,
                    "report": "\n\n".join(running_report),
                    "batches": len(running_report),
                })
            except Exception:
                pass

        # ★ Immediate right-rail feedback: emit a "waiting" bg event the instant a
        #   delegation is dispatched, so the DeepPanel appears NOW instead of only
        #   after the first ReAct round (which can be minutes away while frames
        #   accumulate). Without this the panel looks broken ("使能后右栏始终不出现").
        #   Frame-enough runs move past waiting immediately.
        if emit is not None:
            try:
                # Initial target from the current scene's pacing (registry not yet
                # readable on the engine thread here → scene/default).
                _ttl0, _need0 = _round_pacing(None)
                _have0 = len(_gather(None))
                if _have0 >= _need0:
                    emit("multimodal.bg", {
                        "request_id": rid, "channel": "bg", "type": "batch_ready",
                        "label": _label, "have": _have0, "need": _need0,
                    })
                else:
                    emit("multimodal.bg", {
                        "request_id": rid, "channel": "bg", "type": "waiting",
                        "label": _label, "have": _have0, "need": _need0,
                        "ttl_sec": _ttl0, "ttl_remaining": _ttl0,
                    })
            except Exception:
                pass

        # ★ op=update/delete take effect at the batch boundary (per spec: current
        #   round finishes, next round reflects the change). deleted=True also
        #   suppresses the completion hook regardless of hook_main_agent.
        deleted = False
        # ★ 停止原因 (决定完成回调是否触发 hook_main_agent):
        #   "deleted"   op=delete → 不触发 hook, 从注册表移除
        #   "disabled"  用户 off (stop_event) → 中断态, 不触发 hook
        #   "source_end" 视频流结束 → 在主 agent 执行一次完成 hook
        #   "task_complete" 画面明确满足任务结束条件 → 执行一次完成 hook
        #   "normal"    防跑飞上限等正常收尾 → 不触发 hook
        stop_reason = "normal"

        def _read_registry():
            cb = getattr(self, "_research_registry_cb", None)
            if cb is None:
                return None
            try:
                return cb(rid)
            except Exception:
                return None

        def _requested_stop_reason() -> str:
            """Resolve a reason-bearing stop request without conflating delete
            with a reversible disable. Registry tombstones are a compatibility
            fallback for callers created before reason-bearing stops existed."""
            reason = str(stop_reasons.get(rid) or "").strip().lower()
            if reason in {"deleted", "disabled"}:
                return reason
            reg = _read_registry()
            if isinstance(reg, dict) and reg.get("_deleted"):
                return "deleted"
            return "disabled"

        def _sync_state(status: str) -> None:
            # 每轮把执行状态 (status + 真实轮数) 就地写进本任务的日志文件头部, 让
            # list_live_watcher / get_live_watcher 随时看到真实执行态 (不再单独存)。
            try:
                _df.set_status(rid, status, round_idx=int(round_idx))
            except Exception:
                pass

        try:
            while True:
                # A new capture generation is a hard ownership boundary.  Check
                # before writing ``running`` or touching the shared FrameBuffer so
                # an old watcher can never consume the replacement stream.
                if _source_replaced():
                    stop_reason = "source_end"
                    break
                round_idx += 1
                _sync_state("running")

                # ---- 停止指令 (用户 op=delete / stop_delegation): 收尾 ----
                #   用户显式停止 (stop_ev) 立即退出。但"启动时源已停止"的一次性分析
                #   (source_live_at_start=False) 必须先跑完首轮把已缓冲的帧分析一次
                #   —— 否则这里的 _source_stopped 会在第一轮就 break, 让一次性分析变成
                #   空转 (零批次)。首轮之后, 源停由底部 drain / not-source-live 分支收尾。
                _first_iter = (cursor_ts is None)
                if stop_ev.is_set():
                    stop_reason = _requested_stop_reason()
                    deleted = stop_reason == "deleted"
                    break
                # Do not break merely because the source-stopped flag is set.
                # Continue through the gather path once more so frames captured
                # during the previous analysis round are drained. The empty-fresh
                # branch below then records source_end (not user-disabled).

                # ---- batch-boundary registry read (update/delete) ----
                _reg = _read_registry()
                if _reg is not None:
                    if _reg.get("_deleted"):
                        deleted = True
                        stop_reason = "deleted"
                        try:
                            _df.append_note(rid, "收到删除指令, 跑完当前轮后结束, 不再触发主 Agent。")
                        except Exception:
                            pass
                        break
                    if _reg.get("_pending_update"):
                        new_txt = str(_reg.get("task_instruction") or "").strip()
                        if new_txt:
                            task_instruction = new_txt
                            try:
                                _df.append_note(rid, f"研究任务已更新, 本轮起采用新目标: {new_txt[:80]}")
                            except Exception:
                                pass
                        # Clear the flag so we don't re-log every round (best-effort;
                        # the tool thread set it, we consume it here).
                        try:
                            _reg["_pending_update"] = False
                        except Exception:
                            pass

                # ---- this round's pacing (ttl + target frames), refreshed each
                #      round from the registry (set_live_watcher ttl) / current scene.
                ttl_sec, target_frames = _round_pacing(_reg)

                # ★ 首轮特殊: 只取【最近 target_frames 帧】(有多少最近的就用多少),
                #   不从 HEAD 吞整段积压, 也不等累积——立即分析。旧积压丢弃, cursor 从
                #   这段末尾开始往后追。若首轮时 buffer 里一帧都没有, 落到下面的等待逻辑
                #   (source_live 时等到有帧 / ttl; 已停则 drained 收尾)。
                _is_first = (cursor_ts is None)
                # ★ 首轮"帧够多才立刻分析"门槛: 只有已攒到 >= 60% target 才跳过累积、
                #   立即出第一批调研; 否则(第一波帧太少)落入下面的正常累积等待, 攒到
                #   target / ttl 到再做第一波 —— 避免用寥寥几帧仓促出首批结论。
                _FIRST_MIN_RATIO = 0.6
                _first_enough = False
                if _is_first:
                    fresh = self.frame_buffer.latest(target_frames)  # 最近 N 帧
                    _first_enough = len(fresh) >= max(1, int(target_frames * _FIRST_MIN_RATIO))
                    # fresh 够多 → 直接进入本轮分析; 不够 → 保留 fresh, 进累积循环续攒。
                else:
                    fresh = _gather(cursor_ts)

                if (fresh and static_tail_checked_anchor_ts is not None
                        and float(fresh[-1].ts)
                        > float(static_tail_checked_anchor_ts) + 1e-6):
                    # Playback (or another visible scene) resumed. Re-arm the
                    # two-second rule for the new visual boundary.
                    static_tail_checked_anchor_ts = None

                if _source_replaced():
                    stop_reason = "source_end"
                    break

                # ---- Two-stage embedded-player completion detection ---------
                # A video ending inside a shared tab does not end the screen-share
                # MediaStreamTrack.  A normal segment therefore marks a plausible
                # ending first; only a later no-new-scene grace period permits one
                # independent check over raw, non-dHash-deduped tail captures.
                if pending_completion_candidate is not None:
                    if fresh:
                        log.info(
                            "[watcher] completion candidate cleared: %d novel "
                            "frame(s) followed (%s)", len(fresh), rid)
                        pending_completion_candidate = None
                    elif not _ended():
                        attempts_done = max(0, int(
                            pending_completion_candidate.get("attempts") or 0))
                        max_attempts = max(1, int(getattr(
                            self.cfg, "watch_completion_confirm_max_attempts", 1)
                            or 1))
                        check_started = float(
                            pending_completion_candidate.get("check_started_mono")
                            or pending_completion_candidate.get("started_mono")
                            or time.monotonic())
                        first_seen = float(
                            pending_completion_candidate.get("first_seen_mono")
                            or check_started)
                        if attempts_done == 0:
                            # The first strict check has a short grace period.
                            gate_started = check_started
                            gate_duration = max(0.05, float(getattr(
                                self.cfg,
                                "watch_completion_confirm_delay_sec",
                                3.0,
                            ) or 3.0))
                        else:
                            # The follow-up threshold is TOTAL time since the
                            # candidate was first observed.  The initial wait
                            # and verifier latency therefore count toward it;
                            # this is not an additional eight-second sleep.
                            gate_started = first_seen
                            gate_duration = max(0.05, float(getattr(
                                self.cfg,
                                "watch_completion_confirm_retry_total_sec",
                                8.0,
                            ) or 8.0))
                        while not fresh and not _ended():
                            elapsed_gate = time.monotonic() - gate_started
                            if elapsed_gate >= gate_duration:
                                break
                            if emit is not None:
                                try:
                                    emit("multimodal.bg", {
                                        "request_id": rid, "channel": "bg",
                                        "type": "completion_confirm_wait",
                                        "seg": int(round_idx),
                                        "attempt": attempts_done + 1,
                                        "max_attempts": max_attempts,
                                        "ttl_remaining": round(
                                            gate_duration - elapsed_gate, 1),
                                    })
                                except Exception:
                                    pass
                            try:
                                await asyncio.wait_for(
                                    stop_ev.wait(),
                                    timeout=min(
                                        poll,
                                        max(0.05, gate_duration - elapsed_gate)),
                                )
                            except asyncio.TimeoutError:
                                pass
                            fresh = _gather(cursor_ts)

                        if _source_replaced():
                            stop_reason = "source_end"
                            break

                        if fresh:
                            log.info(
                                "[watcher] completion candidate rejected by %d "
                                "new frame(s) during grace (%s)", len(fresh), rid)
                            pending_completion_candidate = None
                        elif not _ended():
                            n_confirm = max(2, int(getattr(
                                self.cfg, "watch_completion_confirm_frames", 8)
                                or 8))
                            raw_tail = []
                            try:
                                raw_ts = getattr(
                                    self.frame_buffer, "monitor_latest_ts", None)
                                if raw_ts is not None and hasattr(
                                        self.frame_buffer, "raw_all_le"):
                                    raw_tail = self.frame_buffer.raw_all_le(
                                        raw_ts, n=n_confirm)
                            except Exception:
                                raw_tail = []
                            latest_raw_ts = max(
                                (float(frame.ts) for frame in raw_tail),
                                default=float("-inf"),
                            )
                            previous_raw_ts = pending_completion_candidate.get(
                                "last_confirm_raw_ts")
                            if (attempts_done > 0
                                    and previous_raw_ts is not None
                                    and latest_raw_ts <= float(previous_raw_ts)):
                                # A follow-up verdict is meaningful only if the
                                # capture pipeline itself kept producing raw
                                # samples. Without that proof, keep the candidate
                                # pending rather than mistaking a dead capture for
                                # a persistently ended player.
                                if emit is not None:
                                    try:
                                        emit("multimodal.bg", {
                                            "request_id": rid, "channel": "bg",
                                            "type": "completion_confirm_wait",
                                            "seg": int(round_idx),
                                            "attempt": attempts_done + 1,
                                            "max_attempts": max_attempts,
                                            "ttl_remaining": 0.0,
                                            "waiting_for_raw_capture": True,
                                        })
                                    except Exception:
                                        pass
                                # The total-duration gate is already open. Sleep
                                # only for the normal poll interval while waiting
                                # for proof that raw capture is still alive; do
                                # not busy-loop or restart the eight-second gate.
                                try:
                                    await asyncio.wait_for(
                                        stop_ev.wait(), timeout=poll)
                                except asyncio.TimeoutError:
                                    pass
                                round_idx -= 1
                                continue
                            candidate_frames = list(
                                pending_completion_candidate.get("frames") or [])
                            by_ts = {
                                float(frame.ts): frame
                                for frame in candidate_frames + list(raw_tail or [])
                                if getattr(frame, "jpeg_b64", "")
                            }
                            confirm_frames = sorted(
                                by_ts.values(), key=lambda frame: frame.ts)
                            if len(confirm_frames) > n_confirm:
                                confirm_frames = _even_downsample(
                                    confirm_frames, n_confirm)

                            confirmed = False
                            confidence = 0.0
                            confirm_reason = ""
                            confirm_observation = ""
                            verifier = getattr(
                                self.responder, "confirm_visual_completion", None)
                            if verifier is not None:
                                try:
                                    (confirmed, confidence, confirm_reason,
                                     confirm_observation) = await verifier(
                                        task_instruction=task_instruction,
                                        candidate_reason=str(
                                            pending_completion_candidate.get(
                                                "reason") or ""),
                                        last_segment_report=str(
                                            pending_completion_candidate.get(
                                                "report") or ""),
                                        idle_sec=max(
                                            0.0,
                                            time.monotonic() - check_started),
                                        attempt=attempts_done + 1,
                                        total_idle_sec=max(
                                            0.0, time.monotonic() - first_seen),
                                        prior_confirmation_reason=str(
                                            pending_completion_candidate.get(
                                                "prior_confirmation_reason") or ""),
                                        frames=confirm_frames,
                                    )
                                except Exception as exc:
                                    log.warning(
                                        "[watcher] completion confirmation failed "
                                        "(%s): %s", rid, exc)
                            if confirmed:
                                stop_reason = "task_complete"
                                # Confirmation consumes no analysis segment.
                                round_idx -= 1
                                try:
                                    log.info(
                                        "[watcher] %s: completion confirmed after a "
                                        "static-tail visual check #%d (confidence=%.2f): %s%s",
                                        rid, attempts_done + 1, confidence,
                                        confirm_reason or str(
                                            pending_completion_candidate.get(
                                                "reason") or "confirmed"),
                                        (f" | final observation: {confirm_observation}"
                                         if confirm_observation else ""),
                                    )
                                except Exception:
                                    pass
                                if emit is not None:
                                    try:
                                        emit("multimodal.bg", {
                                            "request_id": rid, "channel": "bg",
                                            "type": "completion_confirmed",
                                            "attempt": attempts_done + 1,
                                            "confidence": confidence,
                                            "reason": confirm_reason,
                                        })
                                    except Exception:
                                        pass
                                break

                            attempt_no = attempts_done + 1
                            if attempt_no < max_attempts:
                                pending_completion_candidate.update({
                                    "attempts": attempt_no,
                                    "check_started_mono": time.monotonic(),
                                    "prior_confirmation_reason": (
                                        confirm_reason or "insufficient evidence"),
                                    "last_confirm_raw_ts": (
                                        latest_raw_ts
                                        if latest_raw_ts != float("-inf") else
                                        previous_raw_ts),
                                })
                                try:
                                    log.info(
                                        "[watcher] %s: completion check %d/%d "
                                        "inconclusive; preserving the candidate "
                                        "for an extended static-tail follow-up: %s",
                                        rid, attempt_no, max_attempts,
                                        confirm_reason or "insufficient evidence")
                                except Exception:
                                    pass
                                # This loop iteration performed no analysis
                                # segment. Reuse its number after the follow-up.
                                round_idx -= 1
                                continue

                            try:
                                log.info(
                                    "[watcher] %s: completion candidate was not "
                                    "confirmed after %d check(s) "
                                    "(confidence=%.2f): %s",
                                    rid, attempt_no, confidence,
                                    confirm_reason or "insufficient evidence")
                            except Exception:
                                pass
                            pending_completion_candidate = None

                # ★ TTL + 帧数双门 (LIVE source, 首轮已有帧时跳过等待). 一轮结束条件:
                #   (a) 帧数 >= target_frames    (b) 距本轮开始 >= ttl_sec   (c) 源停/用户停。
                #   拥塞: 本轮一进来 fresh 就已 >= target_frames (或时间已过 ttl) → while
                #        立即退出, 连续执行; 帧数超标下面等间隔降采样。
                #   等待期间每 poll 推一次进度 (帧数 + ttl 剩余), 让面板动起来。
                # ★ 攒帧条前缀的段号 = 当前正在攒的这一段 = 已分析最大段号 + 1 = round_idx。
                _wait_seg = int(round_idx)
                round_start = time.monotonic()
                _static_tail_flushed = False
                _static_tail_anchor_ts: Optional[float] = None
                # 首轮仅当帧已够多(_first_enough)才跳过累积; 帧太少则和后续轮一样进累积等待。
                if source_live_at_start and not (_is_first and _first_enough):
                    def _elapsed():
                        return time.monotonic() - round_start
                    _emitted_wait = False
                    while (len(fresh) < target_frames and _elapsed() < ttl_sec
                           and not _ended()):
                        static_remaining = _static_tail_remaining(
                            fresh, cursor=cursor_ts)
                        if static_remaining is not None and static_remaining <= 0:
                            _static_tail_anchor_ts = float(
                                fresh[-1].ts if fresh else cursor_ts)
                            if (static_tail_checked_anchor_ts is not None
                                    and _static_tail_anchor_ts
                                    <= float(static_tail_checked_anchor_ts) + 1e-6):
                                # This unchanged visual boundary already received
                                # its one strict VLM verdict. Wait for a novel scene
                                # (or source/user stop) instead of rechecking it.
                                if emit is not None:
                                    try:
                                        emit("multimodal.bg", {
                                            "request_id": rid,
                                            "channel": "bg",
                                            "type": "completion_confirm_wait",
                                            "seg": _wait_seg,
                                            "paused": True,
                                            "waiting_for_scene_change": True,
                                            "ttl_remaining": 0.0,
                                        })
                                    except Exception:
                                        pass
                                try:
                                    await asyncio.wait_for(
                                        stop_ev.wait(), timeout=poll)
                                except asyncio.TimeoutError:
                                    pass
                                fresh = (self.frame_buffer.latest(target_frames)
                                         if _is_first else _gather(cursor_ts))
                                continue
                            try:
                                log.info(
                                    "[watcher] %s: finite-task static tail reached "
                                    "%.1fs; starting a before/after raw-frame "
                                    "completion check before the normal %.0fs "
                                    "segment TTL.",
                                    rid, static_tail_flush_sec, ttl_sec)
                            except Exception:
                                pass
                            if emit is not None:
                                try:
                                    emit("multimodal.bg", {
                                        "request_id": rid, "channel": "bg",
                                        "type": "static_tail_flush",
                                        "seg": _wait_seg,
                                        "have": len(fresh),
                                        "static_sec": static_tail_flush_sec,
                                    })
                                except Exception:
                                    pass
                            _static_tail_flushed = True
                            break
                        _remain = max(0.0, ttl_sec - _elapsed())
                        try:
                            log.info(
                                "[watcher] %s: 攒帧中… (第%d段, 帧 %d/%d, "
                                "ttl 余 %.0fs)",
                                rid, _wait_seg, len(fresh), target_frames, _remain)
                        except Exception:
                            pass
                        if emit is not None:
                            _emitted_wait = True
                            try:
                                emit("multimodal.bg", {
                                    "request_id": rid, "channel": "bg",
                                    "type": "waiting", "seg": _wait_seg,
                                    "have": len(fresh), "need": target_frames,
                                    "ttl_sec": ttl_sec,
                                    "ttl_remaining": round(_remain, 1),
                                })
                            except Exception:
                                pass
                        try:
                            wait_for = poll
                            if static_remaining is not None:
                                wait_for = min(
                                    wait_for, max(0.05, static_remaining))
                            await asyncio.wait_for(
                                stop_ev.wait(), timeout=wait_for)
                        except asyncio.TimeoutError:
                            pass
                        # First round re-gather stays anchored to the RECENT tail
                        # (latest N), not the buffer HEAD, so a mid-wait backlog
                        # doesn't suddenly pull in old积压.
                        fresh = (self.frame_buffer.latest(target_frames)
                                 if _is_first else _gather(cursor_ts))

                    # Rule-based end check: compare raw captures from before and
                    # after the two-second no-change boundary in one dedicated VLM
                    # call. This is independent of the segment model choosing a
                    # completion tool and therefore also covers zero novel frames.
                    if _static_tail_flushed:
                        anchor_ts = float(
                            _static_tail_anchor_ts
                            if _static_tail_anchor_ts is not None
                            else (fresh[-1].ts if fresh else cursor_ts))
                        n_confirm = max(2, int(getattr(
                            self.cfg, "watch_completion_confirm_frames", 8) or 8))
                        raw_window = []
                        raw_latest_ts = getattr(
                            self.frame_buffer, "monitor_latest_ts", None)
                        try:
                            window_start = anchor_ts - static_tail_flush_sec
                            window_end = anchor_ts + static_tail_flush_sec
                            before = self.frame_buffer.raw_all_le(
                                anchor_ts, n=n_confirm)
                            before = [
                                frame for frame in before
                                if float(frame.ts) >= window_start - 1e-6
                            ] or self.frame_buffer.raw_all_le(anchor_ts, n=1)
                            raw_after = getattr(
                                self.frame_buffer, "monitor_all_after", None)
                            after = (
                                raw_after(anchor_ts)
                                if callable(raw_after) else [])
                            after = [
                                frame for frame in after
                                if float(frame.ts) <= window_end + 1e-6
                            ]
                            by_ts = {
                                float(frame.ts): frame
                                for frame in list(before) + list(after)
                                if getattr(frame, "jpeg_b64", "")
                            }
                            raw_window = sorted(
                                by_ts.values(), key=lambda frame: frame.ts)
                            if len(raw_window) > n_confirm:
                                raw_window = _even_downsample(
                                    raw_window, n_confirm)
                        except Exception:
                            raw_window = []

                        confirmed = False
                        confidence = 0.0
                        confirm_reason = ""
                        confirm_observation = ""
                        verifier = getattr(
                            self.responder, "confirm_visual_completion", None)
                        if verifier is not None:
                            try:
                                static_for = max(
                                    0.0,
                                    float(raw_latest_ts) - anchor_ts
                                    if raw_latest_ts is not None else
                                    static_tail_flush_sec,
                                )
                                (confirmed, confidence, confirm_reason,
                                 confirm_observation) = await verifier(
                                    task_instruction=task_instruction,
                                    candidate_reason=(
                                        "Raw capture kept advancing while no "
                                        f"dHash-novel scene appeared for "
                                        f"{static_for:.1f}s; compare the frames "
                                        "before and after the static boundary."),
                                    last_segment_report=str(
                                        (prev_segment or {}).get("report") or ""),
                                    idle_sec=static_for,
                                    total_idle_sec=static_for,
                                    attempt=1,
                                    prior_confirmation_reason="",
                                    frames=raw_window,
                                )
                            except Exception as exc:
                                confirm_reason = str(exc)
                                log.warning(
                                    "[watcher] rule-based completion check failed "
                                    "(%s): %s", rid, exc)

                        if confirmed:
                            stop_reason = "task_complete"
                            terminal_completion_observation = (
                                confirm_observation or confirm_reason or
                                "The video reached its ended state.")
                            try:
                                log.info(
                                    "[watcher] %s: completion confirmed by the "
                                    "before/after static-boundary check "
                                    "(confidence=%.2f): %s",
                                    rid, confidence, confirm_reason or "confirmed")
                            except Exception:
                                pass
                            if emit is not None:
                                try:
                                    emit("multimodal.bg", {
                                        "request_id": rid,
                                        "channel": "bg",
                                        "type": "completion_confirmed",
                                        "attempt": 1,
                                        "confidence": confidence,
                                        "reason": confirm_reason,
                                    })
                                except Exception:
                                    pass
                            break

                        static_tail_checked_anchor_ts = anchor_ts
                        try:
                            log.info(
                                "[watcher] %s: static-boundary completion check "
                                "was not confirmed; waiting for a novel scene "
                                "before re-arming: %s",
                                rid,
                                confirm_reason or "insufficient visual ending evidence")
                        except Exception:
                            pass
                        # The dedicated verifier already judged this boundary.
                        # Do not ask the normal segment model to judge the same
                        # raw pixels again. If there are novel frames, preserve
                        # them for ordinary analysis; otherwise pause below.
                        _static_tail_flushed = False
                        if not fresh:
                            round_idx -= 1
                            continue

                    if _source_replaced():
                        stop_reason = "source_end"
                        break

                    # ★ ttl 到但一帧都没攒到 → 【暂停攒帧, 不再倒计时】(用户要求): 挂起在这里
                    #   等新帧, 期间发 paused=true 的 waiting (前端显示"等待新画面…", 不倒时)。
                    #   一旦有新帧到达 → 重开一轮攒帧 (重置 round_start + ttl 倒计时)。源停/用户停则退出。
                    while (not fresh and source_live_at_start and not _ended()):
                        if emit is not None:
                            try:
                                emit("multimodal.bg", {
                                    "request_id": rid, "channel": "bg",
                                    "type": "waiting", "seg": _wait_seg,
                                    "have": 0, "need": target_frames, "paused": True,
                                })
                            except Exception:
                                pass
                        try:
                            await asyncio.wait_for(stop_ev.wait(), timeout=poll)
                        except asyncio.TimeoutError:
                            pass
                        fresh = (self.frame_buffer.latest(target_frames)
                                 if _is_first else _gather(cursor_ts))
                        if fresh:
                            # 有新帧了 → 重开一轮攒帧: 重置计时, 回到上面的 while 继续倒计时攒。
                            round_start = time.monotonic()
                            while (len(fresh) < target_frames
                                   and (time.monotonic() - round_start) < ttl_sec
                                   and not _ended()):
                                _remain = max(0.0, ttl_sec - (time.monotonic() - round_start))
                                if emit is not None:
                                    try:
                                        emit("multimodal.bg", {
                                            "request_id": rid, "channel": "bg",
                                            "type": "waiting", "seg": _wait_seg,
                                            "have": len(fresh), "need": target_frames,
                                            "ttl_sec": ttl_sec,
                                            "ttl_remaining": round(_remain, 1),
                                        })
                                    except Exception:
                                        pass
                                try:
                                    await asyncio.wait_for(stop_ev.wait(), timeout=poll)
                                except asyncio.TimeoutError:
                                    pass
                                fresh = (self.frame_buffer.latest(target_frames)
                                         if _is_first else _gather(cursor_ts))
                    # Gate crossed → push a "满额/开始分析" marker so the panel's
                    # waiting line resolves instead of freezing mid-count.
                    if emit is not None and _emitted_wait and not _ended():
                        try:
                            emit("multimodal.bg", {
                                "request_id": rid, "channel": "bg",
                                "type": "batch_ready",
                                "have": min(len(fresh), target_frames),
                                "need": target_frames,
                            })
                        except Exception:
                            pass

                # ★ 兜底: 走到这里仍无帧 = 上面的暂停 while 因 _ended() 退出 (源停/用户停),
                #   或首轮就无帧。跳过本轮不空调 LLM; 顶部 _ended() 会终止循环。
                #   (正常"无帧"已在上面暂停等新帧, 不会持续倒计时。)
                if not fresh and source_live_at_start and not _ended():
                    no_progress += 1
                    try:
                        log.info(
                            "[watcher] %s: 本 ttl 周期(%.0fs)内无新画面, 跳过本轮继续等。",
                            rid, ttl_sec)
                    except Exception:
                        pass
                    # A skipped (no-frame) cycle must not consume a real round number
                    # (else a long-static scene would burn through max_rounds).
                    round_idx -= 1
                    continue

                # ★ 拥塞降采样: 攒到的帧超过 target_frames (上一轮跑太久积压) → 在当前帧
                #   和目标帧之间等间隔降采样到 target_frames, 保证时序覆盖又不超预算。
                if len(fresh) > target_frames:
                    _before = len(fresh)
                    fresh = _even_downsample(fresh, target_frames)
                    try:
                        log.info(
                            "[watcher] %s: 拥塞: 积压 %d 帧 → 等间隔降采样到 %d 帧。",
                            rid, _before, len(fresh))
                    except Exception:
                        pass

                if _source_replaced():
                    stop_reason = "source_end"
                    break

                # No new frames left AND (source closed / user stopped) → drained,
                # finish. 若非用户 off (stop_ev), 这就是"视频流结束"→ 触发主 agent hook。
                if not fresh:
                    if not stop_ev.is_set():
                        stop_reason = "source_end"
                    try:
                        _df.append_note(rid, "视频源已停止且剩余帧处理完毕, 结束持续分析。")
                    except Exception:
                        pass
                    break

                fr = ((fresh[0].ts, fresh[-1].ts) if fresh else None)

                # ★ 画面级去重已上移到 FrameBuffer 入口 (dHash, 阈值由场景理解动态调),
                #   所以这里不再做"本批和上批太像就跳过"的批级 dHash。fresh 已是去重后的
                #   稀疏帧。无进展不再自动收尾 (见上 no_progress 说明)。
                # ★ 攒帧顶栏"分析期间也在攒下一段"心跳 (方案A): 本段 (第 round_idx 段) 在
                #   _run_one_batch 里分析时, 攒帧主循环被 await 阻塞不发 waiting → 顶栏消失。
                #   但视频帧仍持续进 FrameBuffer, 下一段其实已在攒。这里挂一个后台心跳协程,
                #   分析进行中每 poll 秒发一次 waiting: have=分析开始后新进的帧数 (all_after
                #   本段末帧 ts), need/ttl=下一段 pacing (_round_pacing 现值), ttl 从本段分析
                #   开始时刻倒数。分析一结束就取消心跳。段与段之间自然重置 —— 不改真正的
                #   攒帧/分析串行节奏, 只让 UI 反映"下一段已经在攒"这一事实。
                async def _accum_heartbeat(anchor_ts):
                    if emit is None:
                        return
                    _next_ttl, _next_need = _round_pacing(_reg)
                    # ★ 心跳攒的是【下一段】= 当前分析段 round_idx + 1 (= 已分析最大段号+1)。
                    _next_seg = int(round_idx) + 1
                    _hb_start = time.monotonic()
                    _anchor = anchor_ts if anchor_ts is not None else self.frame_buffer.latest_ts
                    try:
                        while not _ended():
                            try:
                                await asyncio.sleep(poll)
                            except asyncio.CancelledError:
                                break
                            if _ended():
                                break
                            _have = (len(self.frame_buffer.all_after(_anchor))
                                     if _anchor is not None else self.frame_buffer.size)
                            # 下一段真正开始前, 攒够也只是"就绪等待", have 封顶到 need。
                            _have = min(_have, _next_need)
                            _remain = max(0.0, _next_ttl - (time.monotonic() - _hb_start))
                            # ★ 无帧且 ttl 已到 → 暂停 (paused, 不倒时); 有新帧到达则重开倒计时。
                            _paused = _have <= 0 and _remain <= 0
                            if _paused and _hb_start is not None:
                                pass  # 保持暂停; 下面发 paused 事件
                            elif _have > 0 and _remain <= 0:
                                # 暂停期间来了新帧 → 重置 ttl, 重新倒计时。
                                _hb_start = time.monotonic()
                                _remain = _next_ttl
                            try:
                                emit("multimodal.bg", {
                                    "request_id": rid, "channel": "bg",
                                    "type": "waiting", "seg": _next_seg,
                                    "have": _have, "need": _next_need,
                                    **({"paused": True}
                                       if _paused
                                       else {"ttl_sec": _next_ttl, "ttl_remaining": round(_remain, 1)}),
                                })
                            except Exception:
                                pass
                    except asyncio.CancelledError:
                        pass

                _hb_task = asyncio.ensure_future(
                    _accum_heartbeat(fresh[-1].ts if fresh else None))
                try:
                    if _source_replaced():
                        stop_reason = "source_end"
                        break
                    (answer, ok, new_sqs, task_done, task_done_reason,
                     completion_candidate,
                     completion_candidate_reason) = await _run_one_batch(
                        task_instruction, fresh, seen_subqueries,
                        seg_idx=round_idx, frame_range=fr,
                        prev_segment=prev_segment,
                        static_tail_check=_static_tail_flushed)
                finally:
                    _hb_task.cancel()
                    try:
                        await _hb_task
                    except (asyncio.CancelledError, Exception):
                        pass
                if _source_replaced():
                    # The old model call may have been in flight when the user
                    # restarted capture.  Its sink/events were suppressed above;
                    # do not persist its result or write this run back to running.
                    stop_reason = "source_end"
                    break
                any_round_ok = any_round_ok or ok
                # ---- #9 累积报告 + #10 统一"无进展"计数 ----
                if ok and (answer or "").strip():
                    fr_txt = f"[{fmt_ts(fr[0])}–{fmt_ts(fr[1])}] " if fr else ""
                    _round_report = fr_txt + answer.strip()
                    running_report.append(_round_report)
                    # ★ 记录本段为"上一段", 供下一段做增量提示 (只留紧邻上一段)。
                    prev_segment = {
                        "range": (f"{fmt_ts(fr[0])}–{fmt_ts(fr[1])}" if fr else "N/A"),
                        "report": answer.strip(),
                    }
                    no_progress = 0     # 有有效发现 = 真进展, 清零统一计数
                    batches_since_report += 1
                    # ★ Live-watcher streaming: publish THIS round's report to the
                    #   watcher panel/sidechannel only, so the user sees the report
                    #   grow before completion. This callback must never wake the
                    #   main agent or break the loop.
                    _rr_cb = getattr(self, "_on_round_report", None)
                    if _rr_cb is not None:
                        try:
                            _rr_cb(rid, round_idx, _round_report)
                        except Exception as _rr_exc:
                            log.warning("[watcher] on_round_report cb failed (%s): %s",
                                        rid, _rr_exc)
                else:
                    no_progress += 1    # 跑了但无有效发现 = 无进展
                # ---- #2 周期增量推送: 每 N 批把累积报告推给前端 (不等结束) ----
                if batches_since_report >= _report_every:
                    _emit_progress_report()
                    batches_since_report = 0
                log.info("[watcher] round %d done (%s) ok=%s "
                         "frames=%d new_sq=%d answer_len=%d no_progress=%d",
                         round_idx, rid, ok, len(fresh), len(new_sqs),
                         len(answer or ""), no_progress)

                # Log this round to the analyse file (findings + frame range +
                # deduped sub-queries).
                try:
                    _df.append_round(
                        rid, round_idx=round_idx, frame_range=fr,
                        sub_queries=new_sqs,
                        findings=(answer if ok else "(本轮无有效发现)"),
                        wall_epoch=getattr(self.frame_buffer, "_wall_epoch", None))
                except Exception as _exc:
                    log.warning("[watcher] append_round failed (%s): %s", rid, _exc)

                # advance the cursor past this batch
                if fresh:
                    cursor_ts = fresh[-1].ts
                try:
                    _df.update_state(
                        rid,
                        task_instruction=task_instruction,
                        seg_base=int(round_idx),
                        cursor_ts=cursor_ts,
                        runtime_id=runtime_id,
                        status="running",
                    )
                except Exception:
                    pass

                if completion_candidate and not task_done:
                    # Providers sometimes conservatively select the candidate
                    # tool despite citing an exact end-progress equality. That
                    # evidence is deterministic and the prompt contract marks it
                    # conclusive, so do not spend another visual request on it.
                    completion_evidence = " ".join((
                        completion_candidate_reason or "", answer or ""))
                    progress_pairs = re.findall(
                        r"(?<!\d)(\d{1,2}):(\d{2})\s*/\s*"
                        r"(\d{1,2}):(\d{2})(?!\d)",
                        completion_evidence,
                    )
                    if any(
                        int(em) * 60 + int(es) == int(dm) * 60 + int(ds)
                        and int(dm) * 60 + int(ds) > 0
                        for em, es, dm, ds in progress_pairs
                    ):
                        task_done = True
                        task_done_reason = (
                            "player elapsed time equals total duration")
                        stop_reason = "task_complete"
                        completion_candidate = False
                        try:
                            log.info(
                                "[watcher] %s: accepted exact end-progress "
                                "equality as conclusive completion without an "
                                "additional visual confirmation call.", rid)
                        except Exception:
                            pass

                if completion_candidate and not task_done:
                    candidate_now = time.monotonic()
                    # A static-tail flush has already supplied the normal grace
                    # period using continuously arriving raw captures. Credit
                    # that elapsed evidence so confirmation starts immediately
                    # instead of sleeping another three seconds after the model.
                    candidate_started = (
                        candidate_now - static_tail_flush_sec
                        if _static_tail_flushed else candidate_now)
                    pending_completion_candidate = {
                        "reason": completion_candidate_reason,
                        "report": (answer or "").strip(),
                        "frames": list(fresh[-4:]),
                        "started_mono": candidate_started,
                        "first_seen_mono": candidate_started,
                        "check_started_mono": candidate_started,
                        "attempts": 0,
                        "prior_confirmation_reason": "",
                        "last_confirm_raw_ts": None,
                    }
                    try:
                        log.info(
                            "[watcher] %s: marked a possible completion state; "
                            "waiting for the no-new-scene grace period before "
                            "visual confirmation: %s",
                            rid, completion_candidate_reason or "unspecified cue")
                    except Exception:
                        pass

                if task_done:
                    stop_reason = "task_complete"
                    try:
                        log.info(
                            "[watcher] %s: observed the task completion "
                            "condition: %s",
                            rid, task_done_reason or "confirmed by the current segment")
                    except Exception:
                        pass
                    break

                # ---- termination ----
                if not source_live_at_start:
                    # Source was already stopped at launch → we analyzed the
                    # buffered frames this round; next loop would find no new frames
                    # and drain-exit anyway, so end now (don't busy-spin).
                    if not stop_ev.is_set():
                        stop_reason = "source_end"
                    break
                if stop_ev.is_set():
                    stop_reason = _requested_stop_reason()
                    deleted = stop_reason == "deleted"
                    try:
                        _df.append_note(
                            rid,
                            "收到删除指令,结束持续分析。"
                            if deleted else "收到停止指令,结束持续分析。",
                        )
                    except Exception:
                        pass
                    break
                if _ended():
                    # Source closed mid-run: loop once more so any frames buffered
                    # after this batch get drained, then the empty-fresh check at
                    # the top ends the run. (Don't break here — a few final frames
                    # may still be unconsumed.)
                    pass
                # ★ 无进展【不再】自动收尾 (用户要求: 无进展保持继续状态)。无论无新帧还是
                #   跑了但无有效发现, 都继续等下一轮 —— 只有源显式停止 / 用户停结束。
                # ★ max_rounds<=0 = 不限轮数 (默认): 完全不设硬上限。仅当配置 >0 时才作为
                #   "防跑飞"上限 (代码保留能力, 暂不提供开启旋钮)。
                if max_rounds > 0 and round_idx >= max_rounds:
                    log.warning("[watcher] %s hit max_rounds=%d, ending", rid, max_rounds)
                    try:
                        _df.append_note(rid, f"达到最大轮数 {max_rounds}, 结束。")
                    except Exception:
                        pass
                    break
                # else loop: next round gathers the next batch of new frames.

            # ★ 段号连续: 把本次跑到的 round_idx 写回注册表, 供"中断→重启"时下一次
            #   run 从此接着往后编号 (不复用旧段号 → 前端追加而非覆盖)。
            try:
                _regf = _read_registry()
                if isinstance(_regf, dict):
                    _regf["_seg_base"] = int(round_idx)
            except Exception:
                pass

            # ★ 五态统一: 收尾状态落文件。
            #   deleted(用户删) → deleted (存档保留但不展示);
            #   disabled(用户 off/暂停) → interrupted (UI 手动关闭 = 中断态);
            #   source_end / normal (视频结束/到轮数/正常收尾) → done。
            terminal_status = (
                "deleted" if stop_reason == "deleted"
                else "interrupted" if stop_reason == "disabled"
                else "done")
            try:
                _df.set_status(
                    rid, terminal_status, round_idx=int(round_idx),
                    stop_reason=stop_reason)
            except Exception:
                pass

            # #9 收尾前推一次最终累积报告 (让前端拿到完整增量)。
            _emit_progress_report()

            # Build the complete chronological report directly from persisted
            # segment results. Do not call a second Watcher summarizer here: the
            # completion hook already invokes the main agent once to organize and
            # present this evidence. A second serial LLM pass added ~10 seconds and
            # duplicated the same synthesis work.
            complete_report_parts: list[str] = []
            try:
                structured = _df.read_structured(rid) or {}
                for row in structured.get("rounds") or []:
                    findings = str((row or {}).get("findings") or "").strip()
                    if not findings:
                        continue
                    frame_range = str(
                        (row or {}).get("frame_range") or "").strip()
                    complete_report_parts.append(
                        (f"[{frame_range}] " if frame_range else "") + findings
                    )
            except Exception:
                complete_report_parts = []
            if (terminal_completion_observation
                    and terminal_completion_observation
                    not in complete_report_parts):
                complete_report_parts.append(terminal_completion_observation)
            running_fallback = "\n\n".join(complete_report_parts).strip()
            if not running_fallback:
                running_fallback = "\n\n".join(running_report).strip()
            if not running_fallback:
                running_fallback = (
                    f"Deep analysis finished after {round_idx} segment(s), but no "
                    "reliable observations were produced."
                )
            successful_completion = stop_reason not in {"disabled", "deleted"}
            summary = running_fallback
            # A delete/disable can arrive while the bounded final-summary call
            # is in flight. Re-check the reason-bearing signal before writing a
            # successful terminal marker or invoking the completion callback.
            if stop_ev.is_set():
                stop_reason = _requested_stop_reason()
                deleted = stop_reason == "deleted"
                successful_completion = False
                try:
                    _df.set_status(
                        rid,
                        "deleted" if deleted else "interrupted",
                        round_idx=int(round_idx),
                        stop_reason=stop_reason,
                    )
                except Exception:
                    pass
            if successful_completion:
                try:
                    _df.mark_finished(
                        rid, rounds=round_idx, summary_preview=summary)
                except Exception:
                    pass
            log.info(
                "[watcher] %s finished (rounds=%d, reason=%s, summary_len=%d)",
                rid,
                round_idx,
                stop_reason,
                len(summary),
            )
            if self._on_delegation_complete is not None:
                try:
                    self._on_delegation_complete(
                        rid, task_instruction, summary, stop_reason)
                except Exception as _cb_exc:
                    log.warning(
                        "[watcher] on_delegation_complete cb failed (%s): %s",
                        rid,
                        _cb_exc,
                    )
        finally:
            self._stop_events.pop(rid, None)
            getattr(self, "_stop_reasons", {}).pop(rid, None)
            # ★ Terminal signal so the right-column deep-research sub-window closes
            #   / collapses once the whole delegation is finished. Without this the
            #   window stays "active" forever: ridIsActive() keeps returning true
            #   because the router-level "bg" progress item (router_react /
            #   delegate_start lines, channel="bg", empty task_id) never received a
            #   phase:"done". Emit a delegation-level done for the rid; the frontend
            #   marks ALL of the rid's bg items done → ridIsActive → false → close.
            if emit is not None:
                try:
                    emit("multimodal.bg", {"request_id": rid, "channel": "bg",
                                           "task_id": "", "phase": "done",
                                           "delegation_done": True})
                except Exception:
                    pass

    def stop_delegation(self, request_id: str, *, reason: str = "disabled") -> bool:
        """Thread-safe: ask a continuous deep-analysis run to stop after the
        current round. Called from the gateway RPC thread (multimodal.
        stop_analysis). Returns True if a run was still active for this rid.

        Runs the check-and-set on the engine loop so it can't race the run's own
        teardown."""
        loop = self._loop
        if loop is None:
            return False
        if self._stop_events.get(request_id) is None:
            return False

        normalized_reason = str(reason or "disabled").strip().lower()
        if normalized_reason not in {"disabled", "deleted"}:
            normalized_reason = "disabled"

        async def _set() -> bool:
            ev = self._stop_events.get(request_id)
            if ev is None:
                return False
            reasons = getattr(self, "_stop_reasons", None)
            if reasons is None:
                reasons = {}
                self._stop_reasons = reasons
            reasons[request_id] = normalized_reason
            ev.set()
            return True
        # 立即把日志文件头部标 stopping, 让 get_live_watcher 马上看到"正在停止"。
        try:
            from . import watch_file as _df
            _df.set_status(
                request_id, "stopping", stop_reason=normalized_reason)
        except Exception:
            pass
        try:
            fut = asyncio.run_coroutine_threadsafe(_set(), loop)
            return bool(fut.result(timeout=5.0))
        except Exception as exc:
            log.debug("[watcher] stop_delegation failed: %s", exc)
            return False

    def mark_source_stopped(self) -> None:
        """Thread-safe: the video source (screen share / camera) was closed.

        Sets the session-scoped flag so any continuous deep-analysis loop stops
        waiting for new frames and finishes (after draining what's buffered).
        Called from the gateway RPC thread (multimodal.source_stopped)."""
        state_lock = getattr(self, "_state_lock", None)
        if state_lock is None:
            state_lock = threading.RLock()
            self._state_lock = state_lock
        with state_lock:
            self._source_stopped = True

    def mark_source_started(self) -> None:
        """Thread-safe: a new video source began — clear the stopped flag so a
        freshly-launched continuous analysis treats the stream as live, AND drop
        the previous source's leftover frames so the new deep-analysis run
        (which starts from the buffer HEAD) doesn't analyze the old camera/screen
        frames as the new task's opening (cross-source frame bleed)."""
        state_lock = getattr(self, "_state_lock", None)
        if state_lock is None:
            state_lock = threading.RLock()
            self._state_lock = state_lock
        with state_lock:
            # Advance only for a genuine new source.  A stop alone keeps the old
            # epoch stable so an existing run can drain its tail; this increment
            # permanently replaces that run before the shared buffer is reused.
            self._source_epoch = int(getattr(self, "_source_epoch", 0) or 0) + 1
            self._source_stopped = False
            try:
                buf = self.frame_buffer
                if buf is not None and hasattr(buf, "clear"):
                    dropped = buf.clear()
                    if dropped:
                        log.info(
                            "[watcher] source changed → cleared %d stale frames",
                            dropped,
                        )
            except Exception:
                pass

    def is_source_live(self) -> bool:
        """Authoritative per-session verdict: is the video stream currently on?

        ``_source_stopped`` is the AUTHORITATIVE UI-switch flag, set by the
        frontend's ``multimodal.source_stopped {started}`` RPC (share start/stop)
        — it does NOT drift on tab-switch or a static scene (the browser keeps the
        share running; only an explicit stop flips it). We deliberately do NOT use
        the ``_last_push_wall`` freshness gate here (that's only for the main
        agent's single-frame Q&A); a background monitor / research run must not be
        declared dead just because the scene was static for a few seconds.

        Returns False when the source was explicitly stopped, OR nothing was ever
        captured (buffer empty AND no frame ever pushed) — genuinely no stream.
        """
        state_lock = getattr(self, "_state_lock", None)
        if state_lock is None:
            state_lock = threading.RLock()
            self._state_lock = state_lock
        with state_lock:
            source_stopped = self._source_stopped
        if source_stopped:
            return False
        buf = self.frame_buffer
        if buf is None:
            return False
        try:
            never_captured = (getattr(buf, "size", 0) == 0
                              and getattr(buf, "_last_push_wall", None) is None)
        except Exception:
            never_captured = False
        return not never_captured
