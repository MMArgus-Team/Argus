"""Lightweight, independent memory backend for the main-agent multimodal flow.

Phase-3 follow-up. MemoryWriter + MemoryReviewer (+ their MemoryStore / FrameStore
/ SearchFactStore / ConversationLog) stay an INDEPENDENT subsystem — own SQLite, own
wake loops — exactly as in the standalone DualAgent. The only change vs DualAgent
is the frame source: this backend reads the SAME ``FrameBuffer`` the main chat
agent fills via the ``multimodal.frame`` RPC, so memory is built from the live
frames the user is actually streaming into the main agent.

It runs the two async wake loops on a dedicated daemon-thread event loop, so it
coexists with the gateway's synchronous, thread-per-turn agent model without any
sync↔async bridge plumbing in the hot path.

Lifecycle (driven by the gateway):
    backend = MemoryBackend(agent.frame_buffer)
    backend.start()
    ...
    backend.stop()
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import math
import os
import re
import shutil
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

log = logging.getLogger("hermes.multimodal.memory_backend")


_ENV_AUDIO_PCM_RATE = 16_000
_ENV_AUDIO_DECODE_TIMEOUT_SEC = 15.0
_ENV_AUDIO_TEXT_TOKEN_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]+")
_ENV_AUDIO_TEXT_STRIP_RE = re.compile(
    r"[\s\u3000，。！？、；：,.!?;:~…“”\"'‘’（）()\[\]【】<>《》\-—_]+"
)
_ENV_AUDIO_FILLER_WORDS = {
    "嗯", "嗯嗯", "嗯哼", "呃", "呃呃", "啊", "哦", "噢", "唔",
    "诶", "哎", "唉", "额", "呀", "哈", "哼",
    "um", "uh", "er", "ah", "oh", "mm", "hmm", "mhm",
}
_ENV_AUDIO_FILLER_CHARS = set("嗯呃啊哦噢唔诶哎唉额呀哈哼")


def _env_audio_meaningful_text(text: str) -> str:
    return "".join(_ENV_AUDIO_TEXT_TOKEN_RE.findall(text or "")).lower()


def _env_audio_transcript_filter_reason(text: str, cfg: Any) -> str:
    text = (text or "").strip()
    if not text:
        return "empty_transcript"
    min_chars = max(0, int(getattr(cfg, "env_audio_min_text_chars", 2) or 0))
    if len(text) < min_chars:
        return "text_too_short"

    meaningful = _env_audio_meaningful_text(text)
    if len(meaningful) < min_chars:
        return "low_information_text"

    if not getattr(cfg, "env_audio_filter_fillers", True):
        return ""
    compact = _ENV_AUDIO_TEXT_STRIP_RE.sub("", text).lower()
    if not compact:
        return "low_information_text"
    if compact in _ENV_AUDIO_FILLER_WORDS:
        return "low_information_text"
    if len(compact) <= 4 and all(ch in _ENV_AUDIO_FILLER_CHARS for ch in compact):
        return "low_information_text"
    return ""


def _pcm16_signal_metrics(pcm: bytes, *, sample_rate: int = _ENV_AUDIO_PCM_RATE
                          ) -> Dict[str, Any]:
    """Return deterministic signal measurements for mono signed-16-bit PCM.

    The values are normalized to the full int16 range, so ``rms`` and ``peak``
    are in ``[0, 1]`` and can be compared directly with
    ``Config.env_audio_min_rms``.  Keeping this calculation independent from
    ffmpeg makes the gate both cheap to unit-test and backend-agnostic.
    """
    usable = len(pcm) - (len(pcm) % 2)
    if usable <= 0:
        return {
            "decode_ok": False,
            "decode_reason": "no_pcm",
            "sample_rate": int(sample_rate),
            "samples": 0,
            "duration_sec": 0.0,
            "rms": 0.0,
            "peak": 0.0,
            "dbfs": -120.0,
        }

    sample_count = usable // 2
    sum_squares = 0
    peak_int = 0
    # ``int.from_bytes`` is deliberately used instead of audioop (removed in
    # Python 3.13) or numpy (not a required Hermes dependency).
    view = memoryview(pcm)
    for offset in range(0, usable, 2):
        sample = int.from_bytes(view[offset:offset + 2], "little", signed=True)
        magnitude = abs(sample)
        if magnitude > peak_int:
            peak_int = magnitude
        sum_squares += sample * sample
    rms = math.sqrt(sum_squares / sample_count) / 32768.0
    peak = min(1.0, peak_int / 32768.0)
    dbfs = 20.0 * math.log10(rms) if rms > 0.0 else -120.0
    return {
        "decode_ok": True,
        "decode_reason": "ok",
        "sample_rate": int(sample_rate),
        "samples": sample_count,
        "duration_sec": sample_count / float(sample_rate),
        "rms": rms,
        "peak": peak,
        # Avoid -Infinity in trajectory JSON while retaining a useful floor.
        "dbfs": max(-120.0, dbfs),
    }


async def _decode_env_audio_signal(audio_bytes: bytes) -> Dict[str, Any]:
    """Decode an encoded browser audio chunk and measure its signal energy.

    ffmpeg is already used by Hermes' speech bridge.  Decoding through stdin /
    stdout avoids temporary files and runs asynchronously on the memory backend
    loop, so a five-second MediaRecorder chunk never blocks worker wake-ups.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return {
            **_pcm16_signal_metrics(b""),
            "decode_reason": "ffmpeg_not_found",
            "decoder_stderr": "ffmpeg is not available on PATH",
        }
    try:
        proc = await asyncio.create_subprocess_exec(
            ffmpeg,
            "-hide_banner", "-loglevel", "error",
            "-i", "pipe:0", "-vn",
            "-ac", "1", "-ar", str(_ENV_AUDIO_PCM_RATE),
            "-acodec", "pcm_s16le", "-f", "s16le", "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            pcm, stderr = await asyncio.wait_for(
                proc.communicate(audio_bytes),
                timeout=_ENV_AUDIO_DECODE_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return {
                **_pcm16_signal_metrics(b""),
                "decode_reason": "decode_timeout",
                "decoder_stderr": (
                    f"ffmpeg exceeded {_ENV_AUDIO_DECODE_TIMEOUT_SEC:.0f}s"
                ),
            }
        stderr_text = stderr.decode("utf-8", errors="replace").strip()[:500]
        if proc.returncode != 0 or not pcm:
            return {
                **_pcm16_signal_metrics(b""),
                "decode_reason": (
                    f"ffmpeg_exit_{proc.returncode}"
                    if proc.returncode else "no_pcm"
                ),
                "decoder_stderr": stderr_text,
            }
        return {
            **_pcm16_signal_metrics(pcm),
            "pcm_bytes": len(pcm),
            "decoder_stderr": stderr_text,
        }
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return {
            **_pcm16_signal_metrics(b""),
            "decode_reason": "decoder_exception",
            "decoder_stderr": f"{type(exc).__name__}: {exc}"[:500],
        }


def _safe_float(value: Any, default: Optional[float] = None
                ) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _env_audio_metadata(audio_bytes: bytes, *, mime: str,
                        rel_ts: Optional[float], window_sec: float,
                        metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize gateway chunk metadata while preserving legacy callers."""
    src = dict(metadata or {})
    digest = str(src.get("sha256") or src.get("sha") or "").strip().lower()
    if not digest:
        digest = hashlib.sha256(audio_bytes).hexdigest()
    t_start = _safe_float(
        src.get("t_start", src.get(
            "server_start_ts", src.get("window_start_ts", rel_ts))),
        rel_ts,
    )
    t_end = _safe_float(
        src.get("t_end", src.get("server_end_ts", src.get("window_end_ts"))),
        None,
    )
    if t_end is None and t_start is not None:
        t_end = t_start + max(0.0, float(window_sec))
    container = str(src.get("container") or "").strip().lower()
    if not container:
        content_type = (mime or "").split(";", 1)[0].strip().lower()
        container = content_type.rsplit("/", 1)[-1] if "/" in content_type else content_type
    chunk_id = str(src.get("chunk_id") or "").strip()
    if not chunk_id:
        chunk_id = f"aud_{digest[:12]}"
    seq = src.get("seq", src.get("chunk_seq"))
    try:
        seq = int(seq) if seq is not None else None
    except (TypeError, ValueError):
        seq = None
    normalized = {
        "capture_id": str(src.get("capture_id") or "").strip(),
        "chunk_id": chunk_id,
        "seq": seq,
        "sha256": digest,
        "sha": digest[:12],
        "container": container or "unknown",
        "mime": mime,
        "bytes": len(audio_bytes),
        "t_start": t_start,
        "t_end": t_end,
        # Keep the legacy name in trajectory payloads while consumers migrate.
        "rel_ts": t_start,
    }
    # These are produced by the browser/gateway diagnostics path.  Copy only
    # the known scalar fields rather than merging arbitrary client metadata into
    # logs/trajectory events.
    for key in (
        "client_start_ts", "client_end_ts", "server_start_ts", "server_end_ts",
        "client_t_start", "client_t_end", "server_t_start", "server_t_end",
        "client_window_start_ts", "client_window_end_ts",
        "server_window_start_ts", "server_window_end_ts",
        "client_duration_sec", "blob_timecode", "sha256_short",
        "standalone_header", "header_hex",
    ):
        if key in src:
            normalized[key] = src[key]
    return normalized


@dataclass
class TokenMeter:
    """Best-effort running total of the memory model's token usage (read from
    the endpoint's returned ``usage``)."""
    prompt: int = 0
    completion: int = 0
    thoughts: int = 0
    total: int = 0
    calls: int = 0
    by_kind: Dict[str, Dict[str, int]] = field(default_factory=dict)

    @staticmethod
    def _normalize_kind(kind: Any) -> str:
        raw = str(kind or "").strip() or "unknown"
        aliases = {
            "mem_l2_agg": "aggregator_l2",
            "mem_l3_agg": "aggregator_l3",
        }
        return aliases.get(raw, raw)

    def add(self, *, prompt=0, completion=0, thoughts=0, total=0,
            kind: str = "") -> None:
        prompt_i = int(prompt or 0)
        completion_i = int(completion or 0)
        thoughts_i = int(thoughts or 0)
        total_i = int(total or 0)

        self.prompt += prompt_i
        self.completion += completion_i
        self.thoughts += thoughts_i
        self.total += total_i
        self.calls += 1

        k = self._normalize_kind(kind)
        bucket = self.by_kind.setdefault(
            k, {"calls": 0, "prompt": 0, "thoughts": 0,
                "completion": 0, "total": 0})
        bucket["calls"] += 1
        bucket["prompt"] += prompt_i
        bucket["thoughts"] += thoughts_i
        bucket["completion"] += completion_i
        bucket["total"] += total_i

    def report(self) -> str:
        lines = [
            f"calls={self.calls} prompt={self.prompt} thoughts={self.thoughts} "
            f"completion={self.completion} TOTAL={self.total}"
        ]
        for kind in sorted(self.by_kind):
            b = self.by_kind[kind]
            lines.append(
                f"kind={kind} calls={b['calls']} prompt={b['prompt']} "
                f"thoughts={b['thoughts']} completion={b['completion']} "
                f"TOTAL={b['total']}")
        return "\n".join(lines)


def _llm_endpoint_identity(client: Any) -> str:
    """Return a stable endpoint string through the lightweight client wrappers."""
    current = client
    seen: set[int] = set()
    for _ in range(4):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        for attr in ("base_url", "api_url", "_base_url"):
            raw = getattr(current, attr, None)
            if raw:
                return str(raw).strip().rstrip("/").casefold()
        current = getattr(current, "client", None)
    return ""


def _clients_share_llm_channel(memory_client: Any, recall_client: Any,
                               cfg: Any) -> bool:
    """Whether Writer and Recall should share the backend's fairness lock.

    Client identity/endpoint inspection is authoritative.  Config fallback is
    needed for test doubles and providers whose adapters intentionally hide the
    transport URL.  When neither role has a dedicated URL, both follow the main
    Hermes endpoint and therefore share its concurrency budget.
    """
    memory_underlying = getattr(memory_client, "client", memory_client)
    if memory_underlying is recall_client:
        return True
    memory_endpoint = _llm_endpoint_identity(memory_client)
    recall_endpoint = _llm_endpoint_identity(recall_client)
    if memory_endpoint and recall_endpoint:
        return memory_endpoint == recall_endpoint

    memory_base = str(getattr(cfg, "memory_base_url", "") or "").strip()
    recall_base = str(getattr(cfg, "recall_base_url", "") or "").strip()
    if memory_base and recall_base:
        return memory_base.rstrip("/").casefold() == recall_base.rstrip("/").casefold()
    if bool(memory_base) != bool(recall_base):
        return False
    memory_provider = str(
        getattr(cfg, "memory_provider", "") or "").strip().casefold()
    if memory_provider in {"gemini", "gemini_omni"}:
        return False
    return True


class MemoryBackend:
    """Owns MemoryWriter/Reviewer + stores, sharing an external FrameBuffer."""

    STATE_NEW = "new"
    STATE_STARTING = "starting"
    STATE_READY = "ready"
    STATE_FAILED = "failed"
    STATE_STOPPING = "stopping"
    STATE_STOPPED = "stopped"

    def __init__(self, frame_buffer: Any, hermes_cfg: Optional[dict] = None,
                 emit_cb=None, session_id: str = "", runtime_config: Any = None):
        self.frame_buffer = frame_buffer
        self._hermes_cfg = hermes_cfg
        self._session_id = (session_id or "").strip()
        # A runtime Config may be built by the gateway while its profile-home
        # context is still active.  Reusing that exact object on the backend
        # thread both preserves the profile path and guarantees every store in
        # this backend sees the same, once-generated SQLite identity.
        self._runtime_config = runtime_config
        self._memory_run_id = uuid.uuid4().hex
        # emit_cb(event:str, payload:dict) — thread-safe push to the dashboard
        # (gateway binds it to _emit(event, sid, payload)). Used to surface the
        # observation panels (画面观察/音频观察/搜索事实) on the main-agent page.
        self._emit_cb = emit_cb
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._main_task: Optional[asyncio.Task] = None
        self._stop = threading.Event()
        self._state_lock = threading.RLock()
        self._state = self.STATE_NEW
        # _startup_done is terminal for startup: READY, FAILED, or an explicit
        # stop before construction completed.  _stopped is terminal for the
        # owning thread and therefore safe for bounded teardown waits.
        self._startup_done = threading.Event()
        self._stopped = threading.Event()
        self._stopped_callbacks = []
        self._stopped_callbacks_fired = False
        self._startup_error: Optional[BaseException] = None
        self._runtime_error: Optional[BaseException] = None
        # Resource identities already processed by the async teardown.  The
        # backend normally tears down once, but keeping this set makes cleanup
        # safe if a startup failure and a later defensive cleanup path meet.
        self._closed_llm_client_ids = set()
        # De-dup signature for _push_ctx (skip emit when payload identical to
        # the previous push — see _push_ctx for the full explanation).
        self._last_ctx_sig: Optional[tuple] = None
        # SearchFactStore listeners run on the Watcher thread while Writer and
        # audio updates normally run on this backend's thread.  Serialize the
        # whole snapshot→dedupe→emit sequence so an older snapshot can never be
        # emitted after a newer one.
        self._ctx_push_lock = threading.RLock()
        # Built lazily on the backend thread (stores/clients bind to that loop).
        self.cfg = None
        self.mem = None
        self.search_fact_store = None
        self.store = None
        self.conversation = None
        self.frame_store = None
        self.memory_client = None
        self.memory_writer = None
        self.reviewer_client = None
        self.reviewer_clients = []
        self.screen_ocr_worker = None
        self.memory_reviewers = []   # ★ P1: 3 专项 Reviewer (Entity/Event/Edge)
        self._reviewer_run_lock: Optional[asyncio.Lock] = None
        self._reviewer_llm_semaphore: Optional[asyncio.Semaphore] = None
        self._reviewer_endpoint_limiters: Dict[int, Any] = {}
        self.recall_agent = None   # ★ RecallAgent (记忆召回子 agent)
        self._recall_client_pending = None
        self._recall_verify_client_pending = None
        self.meter = TokenMeter()    # ★ #6: token 累计, 写 <db>.tokens.txt
        self._tokens_dirty = 0       # 节流: 每 N 次 add 落一次盘
        self._db_renamed = False     # ★ #6: 库是否已回填摘要 (只回填一次)
        self.stt = None
        self.whisperx = None
        self._offline = False        # ★ mm-memory-eval: 离线(喂帧节奏驱动)模式
        # ★ FIX (A): 多模态 LLM 通道锁 —— 保护 writer/recall 共享同一 upstream
        #   provider 时的串行 (recall 未配 recall_base_url 时会 fallback 到主
        #   client, 与 memory_client 争抢同一 upstream 端点; 生产表现是 recall
        #   一开跑, writer 段被拉到 60-89s 才 seal)。使用 FIFO Lock, 由 writer
        #   每 wake_interval 秒 acquire 一次, recall 每次 LLM 调用前也 acquire;
        #   writer 只要有段就一定能在 recall 一步 LLM (~10s) 结束后立刻拿到锁,
        #   把 writer 段跨度控制回 30s cap 内。真正建构在启动 backend 事件循环
        #   之后 (_run), 避免绑到主线程 loop 上。
        self._llm_channel_lock: Optional[asyncio.Lock] = None
        # ``None`` means an older/minimal build did not classify the endpoints;
        # _main keeps the conservative shared-channel behaviour in that case.
        # The real _build sets an exact bool after both clients are resolved.
        self._recall_shares_writer_channel: Optional[bool] = None
        # Idempotency dedupe only.  A chunk identity includes the browser's
        # per-share capture id plus sequence; pairing it with the audio SHA
        # catches an RPC retry without erasing a legitimate replay of identical
        # audio later in the session.  The bounded queue prevents unbounded
        # growth in long-running sessions.
        self._env_audio_seen_keys = set()
        self._env_audio_seen_order = deque(maxlen=512)

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
        return bool(self.is_ready and thread is not None and thread.is_alive())

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
        self._startup_done.wait(timeout=None if timeout is None else max(0.0, timeout))
        return self.is_ready

    def wait_stopped(self, timeout: Optional[float] = None) -> bool:
        """Wait until the owning thread has fully unwound and released resources."""
        return self._stopped.wait(
            timeout=None if timeout is None else max(0.0, timeout))

    def add_stopped_callback(self, callback) -> None:
        """Invoke ``callback(self)`` once the owning thread is fully stopped.

        Gateway ownership guards use this instead of polling.  Registration is
        race-safe with teardown: a callback added after stop is invoked
        immediately, while callbacks registered during startup are fired by the
        thread's finalizer only after stores/clients/the event loop are closed.
        """
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
                log.debug("[mm-memory] stopped callback failed", exc_info=True)

    def _publish_stopped(self) -> None:
        """Publish terminal thread teardown and fire callbacks exactly once."""
        callbacks = []
        with self._state_lock:
            self._startup_done.set()
            self._stopped.set()
            if not self._stopped_callbacks_fired:
                self._stopped_callbacks_fired = True
                callbacks = list(self._stopped_callbacks)
                self._stopped_callbacks.clear()
        for callback in callbacks:
            try:
                callback(self)
            except Exception:
                log.debug("[mm-memory] stopped callback failed", exc_info=True)

    def _build_stop_requested(self, after_stage: str) -> bool:
        """Cooperative cancellation checkpoint between expensive build stages."""
        if not self._stop.is_set():
            return False
        log.info(
            "[mm-memory] startup cancelled after %s; skipping remaining build",
            after_stage,
        )
        return True


    def _mark_startup_failed(self, exc: BaseException) -> None:
        with self._state_lock:
            if self._startup_error is None:
                self._startup_error = exc
            self._state = self.STATE_FAILED
        self._startup_done.set()

    def _mark_ready(self) -> bool:
        with self._state_lock:
            if self._stop.is_set() or self._state == self.STATE_STOPPING:
                self._state = self.STATE_STOPPING
                self._startup_done.set()
                return False
            self._state = self.STATE_READY
        self._startup_done.set()
        return True

    def start(self, *, offline: bool = False,
              timeout: Optional[float] = None) -> bool:
        """Start the backend daemon loop.

        offline=False (default, ONLINE): runs the timed _writer_loop / _reviewer_loop
        exactly as the live gateway does — writer wakes every writer_wake_interval.

        offline=True (CLI mm-memory-eval): does NOT run the timed _writer_loop
        (its wall-clock cadence would race the caller's feed-paced pump). The loop
        stays alive & idle so the caller drives writer wakes via pump_one_wake()
        at the feed pace, then triggers a final reviewer pass + aggregation drain.
        Everything else (wake_once / aggregation / recall) is the SAME code path
        as online — only the wake CADENCE is caller-driven instead of timed.

        With ``timeout=None`` this preserves the legacy non-blocking launch
        contract and returns once the thread has been accepted.  Supplying a
        timeout waits for the complete runtime (Config, stores, RecallAgent and
        loop-owned primitives) and returns ``True`` only for READY.  Callers
        that pass this backend to another resident worker must use the bounded
        form so a half-built object never escapes.
        """
        start_error: Optional[BaseException] = None
        with self._state_lock:
            if self._state == self.STATE_READY:
                return True
            if self._state in {
                self.STATE_FAILED, self.STATE_STOPPING, self.STATE_STOPPED,
            }:
                return False
            if self._state == self.STATE_NEW:
                self._offline = bool(offline)
                self._state = self.STATE_STARTING
                self._thread = threading.Thread(
                    target=self._run, name="mm-memory-backend", daemon=True)
                # Publish and start the Thread atomically with respect to
                # stop().  If the lock were released between assigning
                # ``_thread`` and Thread.start(), a concurrent stop would try
                # to join a never-started Thread and raise RuntimeError.
                try:
                    self._thread.start()
                except BaseException as exc:
                    # A Thread whose start() raised cannot be joined.  Clear the
                    # published handle before stop() can observe this terminal
                    # state, then fire gateway callbacks outside the state lock
                    # to preserve lifecycle-lock ordering.
                    self._thread = None
                    self._mark_startup_failed(exc)
                    start_error = exc
        if start_error is not None:
            self._publish_stopped()
            return False
        if timeout is None:
            return True
        return self.wait_ready(timeout)

    def stop(self, *, timeout: float = 5.0) -> bool:
        """Cooperatively cancel owned tasks and wait boundedly for the thread.

        The old implementation called ``loop.stop`` while ``run_until_complete``
        still owned live tasks, which could close the loop underneath Writer,
        Recall, or env-audio work.  Cancelling the root task lets ``finally``
        blocks run; the backend thread then drains any remaining tasks before it
        closes the loop.
        """
        stopped_before_launch = False
        with self._state_lock:
            if self._state == self.STATE_STOPPED:
                return True
            if self._state == self.STATE_NEW:
                self._state = self.STATE_STOPPED
                stopped_before_launch = True
            elif self._state != self.STATE_FAILED:
                self._state = self.STATE_STOPPING
        if stopped_before_launch:
            self._stop.set()
            self._publish_stopped()
            return True
        self._stop.set()
        self._startup_done.set()
        loop = self._loop
        task = self._main_task
        if loop is not None and not loop.is_closed():
            try:
                def _cancel_root() -> None:
                    if task is not None and not task.done():
                        task.cancel()
                loop.call_soon_threadsafe(_cancel_root)
            except Exception:
                pass
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, float(timeout)))
        stopped = bool(thread is None or not thread.is_alive())
        if not stopped:
            log.warning(
                "[mm-memory] backend stop timed out after %.1fs (state=%s)",
                max(0.0, float(timeout)), self.state,
            )
        return stopped

    # ------------------------------------------------------------------ #
    def recall(self, brief: str, user_text: str = "",
               timeout: Optional[float] = None,
               on_progress: Optional[Any] = None) -> Dict[str, Any]:
        """BLOCKING: run the RecallAgent's full ReAct memory recall and return the
        distilled findings.

        The main agent's recall_mm_memory tool and the multimodal WatcherWorker
        both enter the unified memory-recall sub-agent through here. Marshals the
        coroutine onto this backend's loop (recall_agent is bound to it) and blocks
        for the result. Never raises; returns:
          {ok, found, findings, clues, frame_ids, rounds, elapsed_sec}
        ``ok=False`` + ``error`` means not-ready / timed-out / crashed.

        ``timeout`` is the caller's (orchestration layer's) wall clock for "how long
        to wait at most"; the memory system sets no default itself. ``None`` means no
        wait wall — block until the underlying LLM request times out on its own (that
        per-request timeout follows the main agent's
        providers.<id>.request_timeout_seconds; see hermes_glue._submodule_http_client).
        That layer, not this one, decides the per-call timeout.
        """
        import asyncio as _asyncio
        from concurrent.futures import TimeoutError as _FTimeout
        brief = (brief or "").strip()
        if (not self.is_ready or self._loop is None
                or self.recall_agent is None):
            return {"ok": False, "error": "memory backend not ready (recall unavailable)"}
        if not brief:
            return {"ok": False, "error": "brief is required"}
        try:
            ask_ts = self.frame_buffer.latest_ts if self.frame_buffer else None
        except Exception:
            ask_ts = None
        # ★ Resume 老 session 时 FrameBuffer 是新建的空 buffer, latest_ts=None →
        #   若 fallback 到 0.0, 后续 vector_search_micro/entity 的
        #   "WHERE t_end <= ask_ts" 会把老库里全部事件过滤掉 (它们的 t_end
        #   都是几十秒起步), 导致 vector pool 空, 只能 keyword 全表扫且
        #   entity 也搜不到 → recall 全线失败。所以此时用一个"够大的哨兵值"
        #   表示"不设时间上限, 全库都算提问前"。用 1e18 而不是 float('inf')
        #   —— SQLite 对 inf 的 bind + 比较行为在不同版本上不稳定, 1e18
        #   已经比任何真实 monotonic ts (通常 <1e5 秒) 大 13 个数量级, 安全。
        #   新 session 有帧时才走 D3 anti-dirty-read (只看提问时刻前的事件)。
        _ASK_TS_UNBOUNDED = 1e18
        if ask_ts is None:
            ask_ts = _ASK_TS_UNBOUNDED
        else:
            ask_ts = float(ask_ts)
        # timeout=None → 不设等待墙 (阻塞到底层 LLM 请求自己超时)。调用方显式给值
        # 才启用墙; 记忆系统不替它预设。
        _wall = None if timeout is None else max(1.0, float(timeout))
        _partial: Dict[str, Any] = {
            "clues": [],
            "frame_ids": [],
            "frame_ids_seen": set(),
            "rounds": 0,
            "elapsed_sec": 0.0,
            "trace": [],
        }

        async def _capture_progress(ev: Dict[str, Any]) -> None:
            """Keep partial recall findings even if the outer wall times out."""
            try:
                if isinstance(ev, dict):
                    phase = str(ev.get("phase") or "")
                    trace = _partial["trace"]
                    if phase.startswith("r") and phase.endswith("_decision"):
                        trace.append({
                            "phase": phase,
                            "round": ev.get("round"),
                            "can_answer": bool(ev.get("can_answer")),
                            "next_tool_calls": list(ev.get("next_tool_calls") or []),
                            "decision_summary": str(
                                ev.get("decision_summary") or "")[:300],
                            "useful_info": str(ev.get("useful_info") or "")[:300],
                        })
                    elif phase == "tool_obs":
                        obs_items = []
                        for item in ev.get("observations") or []:
                            if not isinstance(item, dict):
                                continue
                            obs_items.append({
                                "name": item.get("name"),
                                "args": item.get("args") or {},
                                "obs_len": item.get("obs_len"),
                                "elapsed_sec": item.get("elapsed_sec"),
                                "obs_summary": str(item.get("obs_summary") or "")[:300],
                                "frame_ids": list(item.get("frame_ids") or [])[:10],
                                "evidence_segments": [
                                    dict(segment)
                                    for segment in (
                                        item.get("evidence_segments") or [])[:12]
                                    if isinstance(segment, dict)
                                ],
                            })
                        trace.append({
                            "phase": phase,
                            "round": ev.get("round"),
                            "tools": obs_items,
                            "parallel_elapsed_sec": ev.get("parallel_elapsed_sec"),
                        })
                    elif phase == "distill":
                        trace.append({
                            "phase": phase,
                            "round": ev.get("round"),
                            "clue": str(ev.get("clue") or "")[:500],
                        })
                    elif phase == "fast_table":
                        trace.append({
                            "phase": phase,
                            "tool_name": ev.get("tool_name"),
                            "args": dict(ev.get("args") or {}),
                            "query": ev.get("query"),
                            "obs_len": ev.get("obs_len"),
                            "obs_summary": str(
                                ev.get("obs_summary") or "")[:1200],
                            "findings_len": ev.get("findings_len"),
                            "findings_preview": str(
                                ev.get("findings_preview") or "")[:1200],
                            "elapsed_sec": ev.get("elapsed_sec"),
                            "frame_ids": list(ev.get("frame_ids") or [])[:10],
                            "evidence_segments": [
                                dict(segment)
                                for segment in (
                                    ev.get("evidence_segments") or [])[:12]
                                if isinstance(segment, dict)
                            ],
                        })
                    elif phase == "error":
                        trace.append({
                            "phase": phase,
                            "round": ev.get("round"),
                            "stage": ev.get("stage"),
                            "error": str(ev.get("error") or "")[:1000],
                            "elapsed_sec": ev.get("elapsed_sec"),
                        })
                    if len(trace) > 40:
                        del trace[:-40]
                    if phase == "distill":
                        clue = str(ev.get("clue") or "").strip()
                        if clue and clue not in _partial["clues"]:
                            _partial["clues"].append(clue)
                    if "round" in ev:
                        try:
                            _partial["rounds"] = max(
                                int(_partial.get("rounds") or 0),
                                int(ev.get("round") or 0) + 1,
                            )
                        except Exception:
                            pass
                    if "elapsed_sec" in ev:
                        try:
                            _partial["elapsed_sec"] = float(ev.get("elapsed_sec") or 0.0)
                        except Exception:
                            pass
                    for fid in ev.get("frame_ids") or ev.get("new_frame_ids") or []:
                        fid_s = str(fid or "").strip()
                        if fid_s and fid_s not in _partial["frame_ids_seen"]:
                            _partial["frame_ids_seen"].add(fid_s)
                            _partial["frame_ids"].append(fid_s)
            finally:
                await self._emit_worker_progress("RecallWorker", ev)
                if on_progress is not None:
                    maybe_awaitable = on_progress(ev)
                    if hasattr(maybe_awaitable, "__await__"):
                        await maybe_awaitable

        try:
            fut = _asyncio.run_coroutine_threadsafe(
                self.recall_agent.run(
                    initial_calls=[], brief=brief,
                    user_text=(user_text or brief), ask_ts=ask_ts,
                    # ★ 把墙透传给 ReAct 循环: 让它在 75% 预算处用已有 clues
                    #   提前收尾, 避免撞墙后结果被全部丢弃.
                    time_budget_sec=_wall,
                    on_progress=_capture_progress),
                self._loop)
        except Exception as exc:
            return {"ok": False, "error": f"submit failed: {exc}"}
        try:
            res = fut.result(timeout=_wall)
        except _FTimeout:
            try:
                fut.cancel()
            except Exception:
                pass
            partial_findings = "\n".join(
                str(c).strip() for c in (_partial.get("clues") or [])
                if str(c).strip()
            ).strip()
            if partial_findings:
                return {
                    "ok": True,
                    "found": True,
                    "findings": partial_findings,
                    "partial_findings": partial_findings,
                    "clues": list(_partial.get("clues") or []),
                    "frame_ids": list(_partial.get("frame_ids") or []),
                    "rounds": int(_partial.get("rounds") or 0),
                    "elapsed_sec": float(_partial.get("elapsed_sec") or 0.0),
                    "recall_trace": list(_partial.get("trace") or []),
                    "timed_out": True,
                    "partial": True,
                    "error": f"recall timed out after {_wall:.0f}s with partial findings",
                }
            return {"ok": False, "error": f"recall timed out after {_wall:.0f}s",
                    "timed_out": True}
        except Exception as exc:
            return {
                "ok": False,
                "error": f"recall failed: {exc}",
                "recall_trace": list(_partial.get("trace") or []),
                "rounds": int(_partial.get("rounds") or 0),
                "elapsed_sec": float(_partial.get("elapsed_sec") or 0.0),
            }
        findings = (getattr(res, "findings", "") or "").strip()
        from agent.multimodal._sentinels import RECALL_NO_CLUES
        return {
            "ok": True,
            "found": bool(findings) and RECALL_NO_CLUES not in findings,
            "findings": findings,
            "clues": list(getattr(res, "clues", []) or []),
            "frame_ids": list(getattr(res, "frame_ids", []) or []),
            "rounds": int(getattr(res, "rounds", 0) or 0),
            "elapsed_sec": float(getattr(res, "elapsed_sec", 0.0) or 0.0),
            "recall_trace": list(_partial.get("trace") or []),
        }

    # ---- offline (mm-memory-eval) 驱动接口 -------------------------------- #
    # 这些只在 start(offline=True) 时用: 由喂帧方按"喂够一拍 → pump 一次"节奏驱动,
    # 调的是跟在线 _writer_loop / _reviewer_loop 完全相同的 wake_once / _run_reviewers,
    # 仅"触发时机"改为喂帧驱动。marshal 到 backend loop, 阻塞等完 (跟 recall 同款)。
    def _run_on_loop(self, coro, *, timeout: float = 600.0):
        """Submit ``coro`` cross-thread onto the backend loop and block until done.
        Returns (True, "") on success, (False, err) on failure."""
        import asyncio as _asyncio
        if self._loop is None:
            return False, "backend loop not ready"
        try:
            fut = _asyncio.run_coroutine_threadsafe(coro, self._loop)
            fut.result(timeout=timeout)
            return True, ""
        except Exception as exc:  # noqa: BLE001 - 汇报给调用方, 不抛
            return False, repr(exc)

    def pump_one_wake(self, *, timeout: float = 600.0) -> bool:
        """Trigger one real writer wake (the same wake_once the online _writer_loop
        runs). Called by the frame feeder after it has fed enough for one tick. The
        writer internally reads the recent frames from the buffer; the logic is
        identical to online. Returns whether it succeeded."""
        if self.memory_writer is None:
            return False
        ok, err = self._run_on_loop(self.memory_writer.wake_once(), timeout=timeout)
        if not ok:
            log.warning("[mm-memory] offline pump_one_wake failed: %s", err)
        return ok

    def pump_ocr_once(self, *, timeout: float = 60.0) -> int:
        """Trigger one OCR batch in offline eval.

        Online OCR runs from ``ScreenOCRWorker.run()``. In offline mode the main
        backend loop is intentionally idle, so the evaluator drives OCR on the
        video timeline and calls this before writer wakes.
        """
        ow = getattr(self, "screen_ocr_worker", None)
        if ow is None:
            return 0

        async def _run_once():
            return await ow.process_once()

        if self._loop is None:
            return 0
        import asyncio as _asyncio
        try:
            fut = _asyncio.run_coroutine_threadsafe(_run_once(), self._loop)
            return int(fut.result(timeout=max(1.0, float(timeout))) or 0)
        except Exception as exc:  # noqa: BLE001
            log.warning("[mm-memory] offline pump_ocr_once failed: %s", exc)
            return 0

    def pump_scene_once(self, *, timeout: float = 60.0) -> bool:
        """Trigger one scene/dHash probe in offline eval."""
        sc = getattr(self, "scene_controller", None)
        if sc is None or self._loop is None:
            return False
        if not bool(getattr(self.cfg, "scene_probe_use_llm", True)):
            return False
        if getattr(sc, "client", None) is None:
            return False
        ok, err = self._run_on_loop(sc._probe_once(), timeout=timeout)
        if not ok:
            log.warning("[mm-memory] offline pump_scene_once failed: %s", err)
        return ok

    def append_audio_observations(self, items: Any, *, timeout: float = 10.0) -> int:
        """Append pre-transcribed audio observations in offline eval.

        This reuses the same ``audio_observation`` path as online env-audio ASR,
        so MemoryWriter/recall see VTT subtitles exactly like streaming speech
        transcripts instead of a separate eval-only side channel.
        """
        loop = self._loop
        conversation = self.conversation
        if loop is None or conversation is None:
            return 0
        if self._stop.is_set() or loop.is_closed():
            return 0
        batch: list = []
        for item in items or []:
            if isinstance(item, dict):
                text = str(item.get("text") or "").strip()
                rel_ts = item.get("rel_ts")
                speaker = item.get("speaker")
            else:
                text = str(item or "").strip()
                rel_ts = None
                speaker = None
            if not text:
                continue
            try:
                rel_ts = None if rel_ts is None else float(rel_ts)
            except Exception:
                rel_ts = None
            batch.append({"text": text, "rel_ts": rel_ts, "speaker": speaker})
        if not batch:
            return 0

        async def _append_many():
            from ._dual_agent import append_audio_observation
            n = 0
            for it in batch:
                await append_audio_observation(
                    conversation,
                    it["text"],
                    rel_ts=it["rel_ts"],
                    speaker=it["speaker"],
                )
                n += 1
            return n

        try:
            fut = asyncio.run_coroutine_threadsafe(_append_many(), loop)
            n = int(fut.result(timeout=max(1.0, float(timeout))) or 0)
        except Exception as exc:  # noqa: BLE001
            log.warning("[mm-memory] offline append_audio_observations failed: %s", exc)
            return 0
        if n:
            try:
                self._push_ctx()
            except Exception:
                pass
        return n

    def wait_aggregations_done(self, *, timeout: float = 600.0) -> bool:
        """Wait for the writer's pending L2/L3 aggregation tasks to run to
        COMPLETION (not cancel — writer.close() cancels; here we want natural
        completion so macros/supers are built before QA). Loops because finishing an
        L2 task may spawn a new L3 task, until none remain."""
        if self.memory_writer is None:
            return True

        async def _await_aggs():
            import asyncio as _a
            w = self.memory_writer
            # 反复等: 聚合 task 完成可能又触发新的 (L2 完 → 可能触发 L3), 直到清空。
            for _ in range(20):
                tasks = [t for t in getattr(w, "_agg_tasks", set()) if not t.done()]
                if not tasks:
                    break
                await _a.gather(*tasks, return_exceptions=True)
        ok, err = self._run_on_loop(_await_aggs(), timeout=timeout)
        if not ok:
            log.warning("[mm-memory] offline wait_aggregations failed: %s", err)
        return ok

    def finalize_offline(self, *, timeout: float = 600.0) -> bool:
        """Finalize after offline frame feeding: (1) one last pump (flush tail
        frames), (2) wait for aggregations to finish, (3) run one reviewer pass
        (online it is timer/macro-hook driven; offline it is triggered once here at
        finalize). When this returns, memory is fully built and QA can begin."""
        self.pump_one_wake(timeout=timeout)
        self.wait_aggregations_done(timeout=timeout)
        try:
            if getattr(self, "memory_reviewers", None):
                self._run_on_loop(self._run_reviewers(triggered_by="offline_finalize"),
                                  timeout=timeout)
                self.wait_aggregations_done(timeout=timeout)
        except Exception as exc:
            log.warning("[mm-memory] offline reviewer finalize failed: %s", exc)
        # 收尾把 token / ctx 落一下
        try:
            self._write_tokens_txt()
        except Exception:
            pass
        return True

    # ------------------------------------------------------------------ #
    def _build(self) -> bool:
        """Construct stores + workers on the backend thread. Returns success."""
        try:
            if self._build_stop_requested("thread launch"):
                return False
            from .hermes_glue import build_config, HermesClientFactory
            from ._memory import (
                MemoryStore, SearchFactStore, ConversationLog, FrameStore,
                ScreenTextStore, ScreenTableStore, TaskStateStore,
            )
            from ._workers import (
                MemoryWriter, MemoryReviewer, ScreenOCRWorker,
                EventReviewer, EntityReviewer, EdgeReviewer,
            )
            from ._dual_agent import STTClient, WhisperXClient

            cfg = self._runtime_config or build_config(
                self._hermes_cfg,
                session_id=self._session_id,
                runtime_id=self._memory_run_id,
            )
            if self._build_stop_requested("runtime config"):
                return False
            # ★ Vision is a HARD prerequisite for memory extraction (it's a visual
            #   task). A memory model declared vision_ability=false can't see the
            #   frames → fail loudly at startup instead of silently running a blind
            #   model. This makes writer_vision_ability a real control knob.
            if not bool(getattr(cfg, "writer_vision_ability", True)):
                log.error(
                    "[mm-memory] model.memory.vision_ability=false — the memory "
                    "model cannot see frames, which memory extraction requires. "
                    "Backend NOT started. Set a vision-capable memory model with "
                    "vision_ability=true.")
                self._startup_error = RuntimeError(
                    "multimodal memory requires model.memory.vision_ability=true")
                return False
            self.cfg = cfg
            self.mem = MemoryStore(cfg)
            if self._build_stop_requested("MemoryStore"):
                return False
            if self._session_id:
                self.mem.set_session_id(self._session_id)
            self.screen_text_store = ScreenTextStore(cfg, db_path=self.mem.db_path)
            self.screen_table_store = ScreenTableStore(cfg, db_path=self.mem.db_path)
            self.task_state_store = TaskStateStore(cfg, db_path=self.mem.db_path)
            if self._build_stop_requested("SQLite companion stores"):
                return False
            self.search_fact_store = SearchFactStore(cfg)
            # Compatibility alias for worker constructors that still name the
            # dependency ``store``. Both attributes are the same single cache;
            # no second SearchFactStore is constructed.
            self.store = self.search_fact_store
            self.conversation = ConversationLog(
                max_chars=cfg.conv_max_chars, min_turns=cfg.conv_min_turns,
                max_bg_obs=cfg.conv_max_bg_obs,
                audio_store=self.mem)
            # Search facts are committed by WatcherWorker, which runs on the
            # sibling Watcher loop.  The store listener is synchronous and is
            # invoked outside its lock, so it is safe to project the fresh
            # snapshot to the UI immediately instead of waiting for the next
            # MemoryWriter wake (or another frame) to happen.
            self.search_fact_store.add_listener(
                lambda _snapshot: self._schedule_ctx_push())
            self.frame_store = FrameStore(cfg)
            if self._build_stop_requested("session stores"):
                return False
            # Memory LLM client via the same Hermes resolution the main agent uses.
            self.memory_client = HermesClientFactory(cfg).memory_client(None)
            if self._build_stop_requested("memory LLM client"):
                return False
            # ★ #6 TokenMeter: 接 on_usage 回调, 累计 memory 模型 token → .tokens.txt.
            if hasattr(self.memory_client, "on_usage"):
                self.memory_client.on_usage = self._on_usage
            # Reviewer can override model.memory via model.memory.reviewer.*.
            # Use a separate client instance so concurrent writer/reviewer usage
            # callbacks do not cross streams.
            self.reviewer_clients = HermesClientFactory(cfg).reviewer_clients()
            self.reviewer_client = self.reviewer_clients[0]
            if self._build_stop_requested("reviewer LLM client"):
                return False
            for reviewer_client in self.reviewer_clients:
                if hasattr(reviewer_client, "on_usage"):
                    reviewer_client.on_usage = self._on_usage
            # Env-audio ASR clients (port from streaming_demo; HTTP services).
            # The qwen backend points at the env_* endpoint (falling back to
            # asr_* when env_* is unset, so env + user speech can share one
            # service). WhisperXClient reads env_url/env_api_key internally.
            env_ep = (getattr(cfg, "env_url", "") or "").strip() or cfg.asr_url
            env_key = ((getattr(cfg, "env_api_key", "") or "").strip()
                       or (getattr(cfg, "asr_api_key", "") or "").strip()
                       or (getattr(cfg, "dashscope_api_key", "") or "").strip())
            env_model = ((getattr(cfg, "env_asr_model", "") or "").strip()
                         or (getattr(cfg, "asr_model", "") or "").strip())
            self.stt = STTClient(cfg, endpoint=env_ep, api_key=env_key,
                                 model=env_model)
            self.whisperx = WhisperXClient(cfg)
            if self._build_stop_requested("audio clients"):
                return False

            # ★ Shared frame source: read the MAIN agent's live FrameBuffer.
            buf = self.frame_buffer
            # OMNI 原始音频路径已删 (omni 模型效果未 ready) — 不建 AudioBuffer。
            #   音频统一走外部 ASR 转录 → audio_observation。显式置 None 供下游兜底。
            self.audio_buffer = None
            self.memory_writer = MemoryWriter(
                cfg, self.store, self.mem, self.memory_client, buf,
                self.conversation, self.frame_store,
                screen_text_store=self.screen_text_store,
                screen_table_store=self.screen_table_store,
                task_state_store=self.task_state_store)
            if self._build_stop_requested("MemoryWriter"):
                return False
            # ScreenOCRWorker builds its OCR client here. If use_local=true
            # (local_backend=rapidocr) but the package is missing, build_ocr_client
            # raises — fail loudly (like vision_ability) instead of silently
            # degrading to a cloud backend the user never chose.
            try:
                self.screen_ocr_worker = ScreenOCRWorker(
                    cfg, buf, self.frame_store, self.screen_text_store,
                    self.screen_table_store, self._stop)
            except RuntimeError as _ocr_exc:
                log.warning("[mm-memory] OCR disabled: %s", _ocr_exc)
                self.screen_ocr_worker = None
            if self._build_stop_requested("ScreenOCRWorker"):
                return False
            # ★ P1: 3 专项 Reviewer (Entity/Event/Edge) if enabled.
            #   调度: Wave1 gather(Entity, Event) → Wave2 Edge (见 _reviewer_loop).
            #   memory_reviewers[0] 作为 hook 去重锚点 (_last_wake_wall 共享语义).
            self.memory_reviewers: list = []
            if cfg.reviewer_enabled:
                def _reviewer_client(index: int):
                    return self.reviewer_clients[
                        index % len(self.reviewer_clients)]
                if getattr(cfg, "reviewer_entity_enabled", True):
                    self.memory_reviewers.append(EntityReviewer(
                        cfg, self.store, self.mem, _reviewer_client(0), buf,
                        self.conversation, self.frame_store))
                if getattr(cfg, "reviewer_event_enabled", True):
                    self.memory_reviewers.append(EventReviewer(
                        cfg, self.store, self.mem, _reviewer_client(1), buf,
                        self.conversation, self.frame_store))
                if getattr(cfg, "reviewer_edge_enabled", True):
                    self.memory_reviewers.append(EdgeReviewer(
                        cfg, self.store, self.mem, _reviewer_client(2), buf,
                        self.conversation, self.frame_store))
            if self._build_stop_requested("memory reviewers"):
                return False
            # macro finalize hook: 触发 reviewer (若有) + ★ #6 首个 macro 回填库名.
            #   即便没 reviewer 也挂 (回填库名不依赖 reviewer). _trigger_reviewer 内部
            #   自己判断 reviewer 是否存在.
            self.memory_writer._on_macro_finalized = self._trigger_reviewer

            # ★ 记忆召回子 agent (原 RecallWorker → RecallAgent, 归 MemoryBackend)。
            #   decide/distill 跟随 model.memory; verify 可配独立 endpoint。
            #   主 Agent 的 recall_mm_memory 工具 + 多模态 WatcherWorker 都经它召回。
            from ._workers import RecallAgent
            recall_factory = HermesClientFactory(cfg)
            recall_client, recall_model = recall_factory.recall_client()
            self._recall_client_pending = recall_client
            if self._build_stop_requested("recall LLM client"):
                return False
            verify_client, verify_model = recall_factory.recall_verify_client(
                recall_client=recall_client,
                recall_model=recall_model,
            )
            self._recall_verify_client_pending = verify_client
            if hasattr(recall_client, "on_usage"):
                recall_client.on_usage = self._on_usage
            if hasattr(verify_client, "on_usage"):
                verify_client.on_usage = self._on_usage
            if self._build_stop_requested("recall verifier LLM client"):
                return False
            self._recall_shares_writer_channel = _clients_share_llm_channel(
                self.memory_client, recall_client, cfg)
            log.info(
                "[mm-memory] writer/recall LLM channel shared=%s; "
                "dedicated verifier=%s model=%s",
                self._recall_shares_writer_channel,
                verify_client is not recall_client,
                verify_model,
            )
            self.recall_agent = RecallAgent(
                cfg, self.mem, recall_client, self.conversation,
                model=recall_model,
                verify_client=verify_client,
                verify_model=verify_model,
                buf=buf, frame_store=self.frame_store,
                screen_text_store=self.screen_text_store,
                screen_table_store=self.screen_table_store,
                task_state_store=self.task_state_store)
            self._recall_client_pending = None
            self._recall_verify_client_pending = None
            if self._build_stop_requested("RecallAgent"):
                return False

            # ★ Scene → FrameBuffer entry-dedup dHash threshold controller.
            #   Uses the auxiliary.vision model to classify the scene every
            #   scene_probe_interval_s and tune how aggressively the buffer drops
            #   near-duplicate frames on push. Best-effort; None client → no-op.
            self.scene_controller = None
            try:
                from .scene_dhash import SceneDhashController
                # ★ v33: scene probe 直接用 model.memory 的端点+模型 (memory 模型本就
                #   vision_ability=true, 能看图)。取 memory_client 的底层 raw client +
                #   model —— 不再走 auxiliary.vision。仅 OpenAI 兼容 client 支持
                #   chat.completions; 非兼容 (如 gemini) 则 scene 静默停用。
                _sv_client = getattr(self.memory_client, "client", None)
                _sv_model = getattr(self.memory_client, "model", "") or ""
                if _sv_client is not None and hasattr(_sv_client, "chat"):
                    self.scene_controller = SceneDhashController(
                        cfg, buf, _sv_client, _sv_model, self._stop)
                else:
                    log.info("[mm-scene] memory client not OpenAI-compatible; "
                             "scene probe disabled")
            except Exception as _se:
                log.info("[mm-scene] controller init failed: %s", _se)
            if self._build_stop_requested("scene controller"):
                return False
            return True
        except Exception as exc:
            self._startup_error = exc
            log.warning("[mm-memory] build failed: %s", exc)
            return False

    def _schedule_ctx_push(self) -> None:
        """Marshal cross-thread SearchFact notifications onto the backend loop."""
        loop = self._loop
        if loop is not None and not loop.is_closed() and loop.is_running():
            try:
                loop.call_soon_threadsafe(self._push_ctx)
                return
            except Exception:
                pass
        # Offline/unit-test paths may not have a running owner loop.  The push
        # remains safe because _push_ctx serializes its full critical section.
        self._push_ctx()

    def _push_ctx(self) -> None:
        lock = getattr(self, "_ctx_push_lock", None)
        if lock is None:
            self._push_ctx_unlocked()
            return
        with lock:
            self._push_ctx_unlocked()

    def _push_ctx_unlocked(self) -> None:
        """Push visual/audio observations and external search facts to the UI.

        Mirrors the legacy server.py::_ui_compat_ctx_dict shape so the frontend
        renderCtx can consume {summary, audio_summary, facts, version}. Safe to
        call from the backend loop; emit_cb marshals cross-thread to the WS.
        """
        cb = self._emit_cb
        if cb is None or self.conversation is None:
            return
        try:
            from ._memory import fmt_ts
            def _stamp(t):
                return (fmt_ts(t.rel_ts) if t.rel_ts is not None
                        else fmt_ts(t.wall_ts % 3600))
            def _fmt(turns, cap):
                lines = []
                for t in turns:
                    spk = f" {t.speaker}" if getattr(t, "speaker", None) else ""
                    lines.append(f"[{_stamp(t)}{spk}] {t.content}")
                joined = "\n".join(lines)
                return joined[-cap:] if len(joined) > cap else joined
            def _items(turns):
                # Structured entries for timeline-card rendering. Newest last
                # (frontend reverses for newest-on-top).
                out = []
                for t in turns:
                    out.append({
                        "ts": _stamp(t),
                        "speaker": getattr(t, "speaker", None) or "",
                        "text": t.content,
                    })
                return out
            obs = self.conversation.latest_obs(6)
            aobs = self.conversation.latest_audio_obs(12)
            snap = self.search_fact_store.snapshot() \
                if self.search_fact_store else None
            fact_values = snap.display_values() if snap else {}
            payload = {
                # Structured arrays for the timeline cards…
                "obs": _items(obs),
                "audio_obs": _items(aobs),
                # …plus the joined strings for backward compat.
                "summary": _fmt(obs, 4000),
                "audio_summary": _fmt(aobs, 4000),
                "facts": fact_values,
                "search_facts": (snap.to_dict()
                                 if snap else {"version": 0, "facts": {}}),
                "version": getattr(snap, "version", 0) if snap else 0,
            }
            # De-dup: MemoryWriter.wake_once fires every writer_wake_interval
            # (~5s), but often the LLM returns an empty obs_text (screen barely
            # changed) so the underlying ConversationLog isn't extended. This
            # method still fired before, re-pushing an IDENTICAL payload every
            # 5s — that's the "same message repeated" you saw in the event log
            # while sharing a static screen. Cheap content signature: skip
            # emit when it matches the last push.
            sig = (payload["summary"], payload["audio_summary"],
                   tuple(sorted(payload["facts"].items())),
                   repr(payload["search_facts"]),
                   payload["version"])
            if getattr(self, "_last_ctx_sig", None) == sig:
                return
            self._last_ctx_sig = sig
            cb("multimodal.ctx", payload)
        except Exception as exc:
            log.debug("[mm-memory] push_ctx failed: %s", exc)

    def _on_usage(self, u: Dict[str, Any]) -> None:
        """Callback invoked after each memory-client call: accumulate into the
        TokenMeter and throttle-flush to .tokens.txt (every 5 calls). ``u`` is in
        Gemini usageMetadata style (OpenAIMemoryClient normalizes to the same field
        names)."""
        try:
            self.meter.add(
                prompt=u.get("promptTokenCount", 0),
                completion=u.get("candidatesTokenCount", 0),
                thoughts=u.get("thoughtsTokenCount", 0),
                total=u.get("totalTokenCount", 0),
                kind=u.get("usage_kind") or u.get("kind") or "")
            self._tokens_dirty += 1
            if self._tokens_dirty >= 5:      # 每 5 次 call 落一次盘
                self._tokens_dirty = 0
                self._write_tokens_txt()
        except Exception as exc:
            log.debug("[mm-memory] on_usage failed: %s", exc)

    def _write_tokens_txt(self) -> None:
        """Write the TokenMeter's current totals to the .tokens.txt file that sits
        alongside the sqlite DB (path from MemoryStore.tokens_txt_path)."""
        try:
            if self.mem is None:
                return
            path = self.mem.tokens_txt_path()
            if not path:
                return
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.meter.report() + "\n")
        except Exception as exc:
            log.debug("[mm-memory] write tokens.txt failed: %s", exc)

    async def _trigger_reviewer(self, macro) -> None:
        try:
            import time
            # ★ #6: 首个 macro → 把它的 label/summary 写进库内 meta 表 (只一次)。
            #   不再改名活库 (并发 race → no such table), 库名固定 <ts>.sqlite。
            #   放在 reviewer 早返回之前, 保证即便没 reviewer 也会写摘要。
            if not self._db_renamed and self.mem is not None:
                summary = (str(getattr(macro, "label", "") or "").strip()
                           or str(getattr(macro, "summary", "") or "").strip())
                if summary:
                    self._db_renamed = True
                    try:
                        self.mem.set_summary(summary)
                    except Exception as exc:
                        log.debug("[mm-memory] set_summary failed: %s", exc)
            reviewers = getattr(self, "memory_reviewers", None) or []
            if not reviewers:
                return
            # De-dup anchor: use the FIRST reviewer's last wake time. getattr
            # default: _last_wake_wall may not be set until the first wake —
            # otherwise this hook silently no-ops.
            last = getattr(reviewers[0], "_last_wake_wall", 0.0) or 0.0
            gap = time.time() - last
            if gap < 30.0:
                return
            await self._run_reviewers(
                anchor_ts=getattr(macro, "t_end", None),
                triggered_by=f"macro:{getattr(macro, 'id', '')}")
        except Exception as exc:
            log.debug("[mm-memory] reviewer hook failed: %s", exc)

    async def _run_reviewers(self, *, anchor_ts=None,
                             triggered_by: str = "interval") -> None:
        """Run the reviewers in two waves: Wave1 gathers all non-Edge reviewers
        (Entity/Event) concurrently, then Wave2 runs the EdgeReviewer(s) serially.
        Edge is deferred to Wave2 because it depends on entity merges settling first
        (currently a no-op skeleton)."""
        reviewers = getattr(self, "memory_reviewers", None) or []
        if not reviewers:
            return
        run_lock = self._reviewer_run_lock
        if run_lock is not None and run_lock.locked():
            log.info(
                "[mm-memory] reviewer run coalesced trigger=%s anchor=%s; "
                "another reviewer wave is still active",
                triggered_by, anchor_ts)
            return
        if run_lock is not None:
            async with run_lock:
                await self._run_reviewers_unlocked(
                    anchor_ts=anchor_ts, triggered_by=triggered_by)
            return
        await self._run_reviewers_unlocked(
            anchor_ts=anchor_ts, triggered_by=triggered_by)

    async def _run_reviewers_unlocked(self, *, anchor_ts=None,
                                      triggered_by: str = "interval") -> None:
        reviewers = getattr(self, "memory_reviewers", None) or []
        if not reviewers:
            return
        from ._workers import EdgeReviewer
        wave1 = [r for r in reviewers if not isinstance(r, EdgeReviewer)]
        wave2 = [r for r in reviewers if isinstance(r, EdgeReviewer)]
        def _progress_for(reviewer):
            role = str(getattr(reviewer, "ROLE_NAME", "Reviewer") or "Reviewer")
            async def _cb(ev):
                await self._emit_worker_progress(f"Memory{role}", ev)
            return _cb
        if wave1:
            await asyncio.gather(*[
                r.wake_once(
                    anchor_ts=anchor_ts,
                    triggered_by=triggered_by,
                    on_progress=_progress_for(r),
                )
                for r in wave1
            ], return_exceptions=True)
        for r in wave2:
            try:
                await r.wake_once(
                    anchor_ts=anchor_ts,
                    triggered_by=triggered_by,
                    on_progress=_progress_for(r),
                )
            except Exception as exc:
                log.debug("[mm-memory] edge reviewer wake failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Env-audio ingest: gateway (a different thread) hands raw audio bytes;
    # we transcribe + write audio_observation into the memory ConversationLog,
    # which MemoryWriter then folds into the "看+听" memory alongside frames.
    # ------------------------------------------------------------------ #
    def submit_env_audio(self, audio_bytes: bytes, *, mime: str = "audio/webm",
                         window_ts: Optional[float] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Thread-safe entry: schedule env-audio ingest on the backend loop."""
        loop = self._loop
        if not self.is_ready or loop is None or not audio_bytes:
            return False
        # ★ C20: Backend is shutting down / already down: the loop is stopped or
        # closed, so run_coroutine_threadsafe would either raise or drop the
        # coroutine into a dead loop and the Future would never resolve. Return
        # quietly instead of silently discarding the audio into a ghost Future.
        if self._stop.is_set() or loop.is_closed():
            log.debug("[mm-memory] submit_env_audio ignored: backend stopped")
            return False
        try:
            fut = asyncio.run_coroutine_threadsafe(
                self._ingest_env_audio(
                    audio_bytes,
                    mime=mime,
                    rel_ts=window_ts,
                    metadata=dict(metadata or {}),
                ),
                loop)

            # Surface exceptions that would otherwise vanish into the Future
            # (env-audio ASR silently failing = memory stops hearing).
            def _done(f):
                try:
                    exc = f.exception()
                except Exception:
                    exc = None
                if exc is not None:
                    log.warning("[mm-memory] env-audio ingest crashed: %s", exc)
            fut.add_done_callback(_done)
            return True
        except Exception as exc:
            log.debug("[mm-memory] submit_env_audio failed: %s", exc)
            return False

    async def _ingest_env_audio(self, audio_bytes: bytes, *,
                                mime: str = "audio/webm",
                                rel_ts: Optional[float] = None,
                                metadata: Optional[Dict[str, Any]] = None) -> str:
        """Transcribe env audio → append audio_observation(s). Ported from
        DualAgent.ingest_env_audio."""
        from ._dual_agent import append_audio_observation
        cfg = self.cfg
        emit = self._emit_cb
        started_at = time.perf_counter()
        win = float(getattr(cfg, "env_audio_window_sec", 5.0) or 5.0)
        chunk = _env_audio_metadata(
            audio_bytes,
            mime=mime,
            rel_ts=rel_ts,
            window_sec=win,
            metadata=metadata,
        )

        def _emit_phase(phase: str, **fields: Any) -> None:
            if emit is None:
                return
            try:
                emit("multimodal.trajectory", {
                    "worker": "EnvASRWorker",
                    "phase": phase,
                    **chunk,
                    **fields,
                })
            except Exception:
                pass

        _emit_phase("started")
        log.info(
            "[mm-memory] env-audio started chunk=%s seq=%s sha=%s "
            "bytes=%d container=%s mime=%s t=%.3f~%.3f standalone=%s header=%s",
            chunk["chunk_id"], chunk["seq"], chunk["sha"], chunk["bytes"],
            chunk["container"], chunk["mime"],
            float(chunk["t_start"] or 0.0), float(chunk["t_end"] or 0.0),
            chunk.get("standalone_header"), chunk.get("header_hex", ""),
        )

        # Lazily initialize for tests/legacy objects constructed via __new__.
        seen = getattr(self, "_env_audio_seen_keys", None)
        order = getattr(self, "_env_audio_seen_order", None)
        if seen is None or order is None:
            seen = set()
            order = deque(maxlen=512)
            self._env_audio_seen_keys = seen
            self._env_audio_seen_order = order
        digest = chunk["sha256"]
        dedupe_key = f"{chunk['chunk_id']}:{digest}"
        if dedupe_key in seen:
            _emit_phase(
                "duplicate",
                reason="duplicate_chunk",
                write_reason="not_written_duplicate_audio",
                total_latency_sec=round(time.perf_counter() - started_at, 4),
            )
            log.info(
                "[mm-memory] env-audio duplicate skipped chunk=%s seq=%s sha=%s",
                chunk["chunk_id"], chunk["seq"], chunk["sha"],
            )
            return ""
        if order.maxlen and len(order) >= order.maxlen:
            seen.discard(order[0])
        order.append(dedupe_key)
        seen.add(dedupe_key)

        if not getattr(cfg, "env_audio_enabled", True):
            _emit_phase(
                "disabled",
                reason="env_audio_disabled",
                write_reason="not_written_disabled",
            )
            return ""

        signal = await _decode_env_audio_signal(audio_bytes)
        signal_payload = {
            key: signal.get(key)
            for key in (
                "decode_ok", "decode_reason", "sample_rate", "samples",
                "duration_sec", "pcm_bytes", "rms", "peak", "dbfs",
                "decoder_stderr",
            )
            if key in signal
        }
        min_rms = max(0.0, float(getattr(cfg, "env_audio_min_rms", 0.0) or 0.0))
        _emit_phase("signal", signal=signal_payload, min_rms=min_rms)
        signal_reason = str(signal.get("decode_reason") or "decode_failed")
        # ffmpeg is useful for the RMS gate but was not historically required
        # by the direct DashScope env-ASR path.  Preserve that capability on
        # minimal installations: make the missing probe explicit, then call
        # ASR without an energy decision.  Actual decode failures still stop —
        # sending a malformed container is exactly what caused the short-text
        # hallucination regression this guard fixes.
        signal_unavailable = (
            not signal.get("decode_ok") and signal_reason == "ffmpeg_not_found"
        )
        if signal_unavailable:
            _emit_phase(
                "signal_unavailable",
                reason=signal_reason,
                signal=signal_payload,
                min_rms=min_rms,
                asr_will_run=True,
            )
            log.warning(
                "[mm-memory] env-audio RMS probe unavailable chunk=%s seq=%s "
                "sha=%s reason=%s; continuing without energy gate",
                chunk["chunk_id"], chunk["seq"], chunk["sha"], signal_reason,
            )
        elif not signal.get("decode_ok"):
            reason = signal_reason
            _emit_phase(
                "decode_failed",
                reason=reason,
                signal=signal_payload,
                asr_called=False,
                write_reason="not_written_decode_failed",
                total_latency_sec=round(time.perf_counter() - started_at, 4),
            )
            log.warning(
                "[mm-memory] env-audio decode failed chunk=%s seq=%s sha=%s "
                "reason=%s stderr=%r",
                chunk["chunk_id"], chunk["seq"], chunk["sha"], reason,
                signal.get("decoder_stderr", ""),
            )
            return ""
        if (not signal_unavailable
                and float(signal.get("rms") or 0.0) < min_rms):
            _emit_phase(
                "filtered",
                reason="low_rms",
                signal=signal_payload,
                min_rms=min_rms,
                asr_called=False,
                raw_transcript="",
                filter_reason="low_rms",
                write_reason="not_written_low_rms",
                total_latency_sec=round(time.perf_counter() - started_at, 4),
            )
            log.info(
                "[mm-memory] env-audio low-rms skipped chunk=%s seq=%s sha=%s "
                "duration=%.3fs rms=%.6f peak=%.6f dbfs=%.2f threshold=%.6f",
                chunk["chunk_id"], chunk["seq"], chunk["sha"],
                float(signal.get("duration_sec") or 0.0),
                float(signal.get("rms") or 0.0),
                float(signal.get("peak") or 0.0),
                float(signal.get("dbfs")
                      if signal.get("dbfs") is not None else -120.0), min_rms,
            )
            return ""

        # (OMNI 原始音频 push 已删 — omni 模型未 ready, env 音频只走 ASR 转录
        #  → audio_observation。上面的 decode + RMS 能量门仍保留, 用于 ASR 前过滤。)

        backend = (cfg.env_asr_backend or "qwen").strip().lower()
        all_texts = []
        raw_texts = []
        filtered_reasons = []
        filter_reason = ""
        asr_started_at = time.perf_counter()
        provider_client = self.whisperx if backend == "whisperx" else self.stt
        _emit_phase(
            "transcribing",
            backend=backend,
            signal=signal_payload,
            min_rms=min_rms,
        )
        try:
            if backend == "whisperx":
                segs = await self.whisperx.transcribe(audio_bytes, mime=mime)
                for seg in (segs or []):
                    seg_text = str(getattr(seg, "text", "") or "").strip()
                    if seg_text:
                        raw_texts.append(seg_text)
                    transcript_filter = _env_audio_transcript_filter_reason(
                        seg_text, cfg)
                    if transcript_filter:
                        filtered_reasons.append(transcript_filter)
                        continue
                    seg_rel = ((chunk["t_start"] or 0.0) + seg.start
                               if chunk["t_start"] is not None else None)
                    await append_audio_observation(
                        self.conversation, seg_text, rel_ts=seg_rel,
                        speaker=seg.speaker)
                    all_texts.append(seg_text)
            else:
                text = str(await self.stt.transcribe(
                    audio_bytes, mime=mime) or "").strip()
                if text:
                    raw_texts.append(text)
                transcript_filter = _env_audio_transcript_filter_reason(text, cfg)
                if transcript_filter:
                    filtered_reasons.append(transcript_filter)
                else:
                    await append_audio_observation(
                        self.conversation, text, rel_ts=chunk["t_start"])
                    all_texts.append(text)
        except Exception as exc:
            asr_latency = time.perf_counter() - asr_started_at
            provider_diag = getattr(provider_client, "last_diagnostics", None)
            if not isinstance(provider_diag, dict):
                provider_diag = {}
            log.warning(
                "[mm-memory] env-audio ASR failed chunk=%s seq=%s sha=%s "
                "backend=%s latency=%.3fs error=%s provider=%s",
                chunk["chunk_id"], chunk["seq"], chunk["sha"], backend,
                asr_latency, exc, provider_diag,
            )
            if emit is not None:
                try:
                    _emit_phase(
                        "failed",
                        reason="asr_exception",
                        error=str(exc),
                        backend=backend,
                        provider=provider_diag,
                        signal=signal_payload,
                        asr_called=True,
                        asr_latency_sec=round(asr_latency, 4),
                        raw_transcript=" ".join(raw_texts),
                        write_reason="not_written_asr_failed",
                        total_latency_sec=round(
                            time.perf_counter() - started_at, 4),
                    )
                    now = time.time()
                    last = float(getattr(
                        self, "_last_env_asr_error_emit", 0.0) or 0.0)
                    if now - last >= 30.0:
                        self._last_env_asr_error_emit = now
                        emit("multimodal.toast", {
                            "level": "warning",
                            "text": f"共享音频 ASR 失败: {exc}",
                        })
                except Exception:
                    pass
            return ""
        asr_latency = time.perf_counter() - asr_started_at
        raw_transcript = " ".join(raw_texts)
        provider_diag = getattr(provider_client, "last_diagnostics", None)
        if not isinstance(provider_diag, dict):
            provider_diag = {}
        provider_reason = str(provider_diag.get("reason") or "").strip()
        provider_failed = (
            provider_diag.get("ok") is False
            and provider_reason in {
                "http_error", "request_exception", "missing_api_key",
            }
        )
        # STT clients historically returned "" both for real silence and for
        # transport/auth failures.  When structured provider diagnostics say
        # the request failed, keep that distinction all the way to the worker
        # trajectory instead of mislabelling an HTTP 400 as a silent scene.
        if provider_failed and not all_texts:
            _emit_phase(
                "failed",
                reason=provider_reason,
                backend=backend,
                provider=provider_diag,
                signal=signal_payload,
                asr_called=True,
                asr_latency_sec=round(asr_latency, 4),
                total_latency_sec=round(time.perf_counter() - started_at, 4),
                raw_transcript=raw_transcript,
                filter_reason="asr_failed",
                write_reason="not_written_asr_failed",
            )
            log.warning(
                "[mm-memory] env-audio ASR provider failed chunk=%s seq=%s "
                "sha=%s backend=%s reason=%s latency=%.3fs provider=%s",
                chunk["chunk_id"], chunk["seq"], chunk["sha"], backend,
                provider_reason, asr_latency, provider_diag,
            )
            return ""
        if not all_texts:
            filter_reason = (
                filtered_reasons[0] if filtered_reasons
                else ("text_too_short" if raw_transcript else "empty_transcript")
            )
        # Trim audio_observation turns to the cap.
        if all_texts and cfg.conv_max_audio_obs > 0:
            try:
                # ★ C5: conversation._lock 现为 threading.RLock (跨 loop 共享), 用同步
                #   with; 块内仅纯内存 list 操作, 无 await, 改同步既安全又正确.
                with self.conversation._lock:
                    idxs = [i for i, t in enumerate(self.conversation._turns)
                            if t.kind == "audio_observation"]
                    if len(idxs) > cfg.conv_max_audio_obs:
                        drop = set(idxs[:len(idxs) - cfg.conv_max_audio_obs])
                        self.conversation._turns = [
                            t for i, t in enumerate(self.conversation._turns)
                            if i not in drop]
            except Exception:
                pass
        write_reason = (
            "audio_observation_appended" if all_texts
            else f"not_written_{filter_reason}"
        )
        _emit_phase(
            "completed" if all_texts else "silence",
            backend=backend,
            provider=provider_diag,
            signal=signal_payload,
            min_rms=min_rms,
            asr_called=True,
            asr_latency_sec=round(asr_latency, 4),
            total_latency_sec=round(time.perf_counter() - started_at, 4),
            raw_transcript=raw_transcript,
            transcripts=all_texts,
            text_chars=sum(len(text) for text in all_texts),
            filter_reason=filter_reason or None,
            write_reason=write_reason,
        )
        log.info(
            "[mm-memory] env-audio ASR completed chunk=%s seq=%s sha=%s "
            "backend=%s t=%.3f~%.3f duration=%.3fs rms=%.6f peak=%.6f "
            "dbfs=%.2f threshold=%.6f asr_latency=%.3fs filter=%s write=%s "
            "provider=%s raw=%r",
            chunk["chunk_id"], chunk["seq"], chunk["sha"], backend,
            float(chunk["t_start"] or 0.0), float(chunk["t_end"] or 0.0),
            float(signal.get("duration_sec") or 0.0),
            float(signal.get("rms") or 0.0),
            float(signal.get("peak") or 0.0),
            float(signal.get("dbfs")
                  if signal.get("dbfs") is not None else -120.0),
            min_rms, asr_latency,
            filter_reason or "none", write_reason, provider_diag,
            raw_transcript,
        )
        if all_texts:
            self._push_ctx()  # refresh 音频观察 panel immediately
        return " ".join(all_texts)

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            if not self._build():
                if self._stop.is_set():
                    with self._state_lock:
                        self._state = self.STATE_STOPPING
                    self._startup_done.set()
                else:
                    self._mark_startup_failed(
                        self._startup_error
                        or RuntimeError("multimodal memory backend build failed"))
                return
            self._main_task = loop.create_task(
                self._main(), name="mm-memory-main")
            loop.run_until_complete(self._main_task)
        except asyncio.CancelledError:
            # Normal cooperative stop: stop() cancels the root task so all
            # gather children receive cancellation before loop teardown.
            pass
        except Exception as exc:
            if not self._startup_done.is_set():
                self._mark_startup_failed(exc)
            else:
                with self._state_lock:
                    self._runtime_error = exc
                    if not self._stop.is_set():
                        self._state = self.STATE_FAILED
                log.warning("[mm-memory] runtime failed: %s", exc, exc_info=True)
        finally:
            # MemoryWriter owns fire-and-forget L2/L3 aggregation tasks. Give
            # it the first chance to unwind them, then cancel any other tasks
            # still registered on this loop before closing it.
            if not loop.is_closed():
                async def _drain_owned_tasks() -> None:
                    writer = getattr(self, "memory_writer", None)
                    if writer is not None and hasattr(writer, "close"):
                        try:
                            await writer.close()
                        except Exception as exc:
                            log.debug("[mm-memory] writer close failed: %s", exc)
                    current = asyncio.current_task()
                    pending = [
                        task for task in asyncio.all_tasks()
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
                    log.debug("[mm-memory] async teardown failed: %s", exc)
            # ★ #6: 收尾把 TokenMeter 最终值落盘 (补上最后 <5 次未落盘的 call).
            try:
                self._write_tokens_txt()
                log.info("[mm-memory] token usage: %s", self.meter.report())
            except Exception:
                pass
            try:
                if self.mem is not None:
                    self.mem.cleanup()
            except Exception:
                pass
            try:
                loop.close()
            except Exception:
                pass
            self._main_task = None
            self._loop = None
            with self._state_lock:
                if self._state != self.STATE_FAILED:
                    self._state = self.STATE_STOPPED
            self._publish_stopped()

    async def _close_owned_llm_clients(self) -> None:
        """Close this backend's LLM transports exactly once on their owner loop.

        MemoryWriter/Reviewer/Recall adapters expose ``aclose()``. Recall may
        also receive a raw AsyncOpenAI client, which is closed only when the
        unified resolver marked it as owned. Identity de-duping covers role
        reuse and wrappers around the same transport.
        """
        closed_ids = getattr(self, "_closed_llm_client_ids", None)
        if closed_ids is None:
            closed_ids = set()
            self._closed_llm_client_ids = closed_ids

        recall_agent = getattr(self, "recall_agent", None)
        recall_client = (
            getattr(recall_agent, "client", None)
            or getattr(self, "_recall_client_pending", None)
        )
        if recall_client is not None:
            resource = getattr(recall_client, "client", recall_client)
            client_id = id(resource)
            if client_id not in closed_ids:
                close = getattr(recall_client, "aclose", None)
                if close is None and bool(getattr(
                        recall_client, "_hermes_submodule_owned", False)):
                    close = getattr(recall_client, "close", None)
                if close is not None:
                    closed_ids.add(client_id)
                    try:
                        await close()
                    except Exception as exc:
                        log.debug(
                            "[mm-memory] recall client close failed: %s", exc)
        self._recall_client_pending = None

        verify_client = (
            getattr(recall_agent, "verify_client", None)
            or getattr(self, "_recall_verify_client_pending", None)
        )
        if verify_client is not None and verify_client is not recall_client:
            resource = getattr(verify_client, "client", verify_client)
            client_id = id(resource)
            if client_id not in closed_ids:
                closed_ids.add(client_id)
                close = getattr(verify_client, "aclose", None)
                if close is None:
                    close = getattr(verify_client, "close", None)
                if close is not None:
                    try:
                        await close()
                    except Exception as exc:
                        log.debug(
                            "[mm-memory] recall verifier client close failed: %s",
                            exc,
                        )
        self._recall_verify_client_pending = None

        # Reviewer commonly reuses the exact Writer adapter.  De-duplicate by
        # the underlying transport where present, then let aclose's ownership
        # contract decide whether closing is appropriate.
        seen_this_pass = set()
        clients_to_close = [
            ("memory", getattr(self, "memory_client", None)),
        ]
        reviewer_clients = getattr(self, "reviewer_clients", None) or []
        if reviewer_clients:
            clients_to_close.extend(
                (f"reviewer[{idx}]", client)
                for idx, client in enumerate(reviewer_clients)
            )
        else:
            clients_to_close.append(
                ("reviewer", getattr(self, "reviewer_client", None)))
        for label, client in clients_to_close:
            if client is None:
                continue
            resource = getattr(client, "client", client)
            client_id = id(resource)
            if client_id in closed_ids or client_id in seen_this_pass:
                continue
            seen_this_pass.add(client_id)
            closed_ids.add(client_id)
            close = getattr(client, "aclose", None)
            if close is None:
                continue
            try:
                await close()
            except Exception as exc:
                log.debug("[mm-memory] %s client close failed: %s", label, exc)

        # Remote QwenVLOCR owns an AsyncOpenAI transport on this backend loop.
        # The default local RapidOCR client has no ``client`` field, so this is
        # naturally a no-op for local OCR.  Identity de-duping keeps defensive
        # teardown idempotent.
        ocr_worker = getattr(self, "screen_ocr_worker", None)
        ocr_client = getattr(ocr_worker, "ocr_client", None)
        ocr_transport = getattr(ocr_client, "client", None)
        if ocr_transport is not None and id(ocr_transport) not in closed_ids:
            closed_ids.add(id(ocr_transport))
            close = getattr(ocr_transport, "close", None)
            if close is not None:
                try:
                    result = close()
                    if inspect.isawaitable(result):
                        await result
                except Exception as exc:
                    log.debug("[mm-memory] OCR client close failed: %s", exc)

    async def _main(self) -> None:
        # ★ offline (mm-memory-eval): 不跑定时 writer/reviewer 循环 (墙钟节奏会跟
        #   喂帧节奏打架)。只让 loop 保活 & idle, 由 pump_one_wake / drain_and_finalize
        #   / recall 通过 run_coroutine_threadsafe 驱动。wake_once / 聚合 / 召回 全走
        #   跟在线一样的代码, 仅"触发时机"改由喂帧方控制。
        # ★ FIX (A): 在 backend 事件循环内创建 LLM 通道锁, 并把它注入到 recall_agent,
        #   让 recall 每次 LLM 调用前 acquire, 与 writer 抢锁; 保证 writer 有段就能
        #   在下一次 recall 释放锁瞬间抢到, 段跨度控制回 30s cap。
        # Only serialize when Writer and Recall resolve to the same endpoint.
        # Dedicated endpoints are intentionally independent; putting them behind
        # one local lock would add latency without protecting any shared limit.
        shares_channel = self._recall_shares_writer_channel is not False
        self._llm_channel_lock = asyncio.Lock() if shares_channel else None
        self._reviewer_run_lock = asyncio.Lock()
        reviewer_concurrency = max(
            1, int(getattr(self.cfg, "reviewer_max_concurrency", 1) or 1))
        reviewer_clients = getattr(self, "reviewer_clients", None) or []
        single_endpoint = len(reviewer_clients) <= 1
        endpoint_concurrency = 1 if single_endpoint else reviewer_concurrency
        endpoint_interval = (
            max(0.0, float(getattr(
                self.cfg, "reviewer_single_endpoint_interval_sec", 0.5) or 0.0))
            if single_endpoint else 0.0
        )
        from ._workers import ReviewerEndpointLimiter
        self._reviewer_endpoint_limiters = {}
        for reviewer in getattr(self, "memory_reviewers", None) or []:
            client_key = id(getattr(reviewer, "client", None))
            limiter = self._reviewer_endpoint_limiters.get(client_key)
            if limiter is None:
                limiter = ReviewerEndpointLimiter(
                    max_concurrency=endpoint_concurrency,
                    min_start_interval_sec=endpoint_interval,
                )
                self._reviewer_endpoint_limiters[client_key] = limiter
            reviewer.endpoint_limiter = limiter
            reviewer.llm_semaphore = None
        log.info(
            "[mm-memory] reviewer endpoints=%d max_concurrency_per_endpoint=%d "
            "single_endpoint_interval=%.2fs overload_retries=%d",
            max(1, len(reviewer_clients)), endpoint_concurrency,
            endpoint_interval,
            max(0, int(getattr(self.cfg, "reviewer_overload_retries", 0) or 0)))
        if getattr(self, "recall_agent", None) is not None:
            try:
                self.recall_agent.llm_channel_lock = self._llm_channel_lock
            except Exception as e:
                log.warning("[mm-memory] 注入 recall llm_channel_lock 失败: %s", e)
        # READY is deliberately published only after every store/client/agent
        # exists *and* loop-owned synchronization has been created. Watcher may
        # now safely snapshot the resource bundle or submit work through the
        # backend proxy; before this point it must receive no backend at all.
        if not self._mark_ready():
            return
        if self._offline:
            while not self._stop.is_set():
                await asyncio.sleep(0.2)
            return
        # return_exceptions=True so a fatal error in one wake loop doesn't
        # cancel the sibling (writer + reviewer + scene controller are independent).
        coros = [self._writer_loop(), self._reviewer_loop()]
        ow = getattr(self, "screen_ocr_worker", None)
        if ow is not None:
            coros.append(self._ocr_loop())
        sc = getattr(self, "scene_controller", None)
        if sc is not None:
            coros.append(sc.run())
        results = await asyncio.gather(*coros, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                log.warning("[mm-memory] wake loop exited with error: %s", r)

    async def _emit_worker_progress(self, worker: str, ev: Any) -> None:
        cb = self._emit_cb
        if cb is None:
            return
        payload = dict(ev) if isinstance(ev, dict) else {"detail": str(ev)}
        payload.setdefault("worker", worker)
        payload.setdefault("source_type", str(
            getattr(self.frame_buffer, "current_source_type", "") or "unknown"))
        try:
            cb("multimodal.trajectory", payload)
        except Exception as exc:
            log.debug("[mm-memory] trajectory emit failed: %s", exc)

    async def _ocr_loop(self) -> None:
        """Run screen OCR with an explicit, visible per-tick trajectory."""
        worker = self.screen_ocr_worker
        if worker is None:
            return
        interval = max(0.2, float(
            getattr(self.cfg, "ocr_worker_interval", 1.0) or 1.0))
        model = str(getattr(worker.ocr_client, "model", "unknown") or "unknown")
        if not getattr(worker.ocr_client, "enabled", False):
            await self._emit_worker_progress("OCRWorker", {
                "phase": "disabled",
                "model": model,
                "reason": getattr(worker.ocr_client, "_missing_reason", "") or "disabled",
            })
            while not self._stop.is_set():
                await asyncio.sleep(1.0)
            return
        await self._emit_worker_progress("OCRWorker", {
            "phase": "started", "model": model, "interval_sec": interval,
            "scope": "screen_only",
        })
        last_idle_emit = 0.0
        last_idle_key = ""
        while not self._stop.is_set():
            t0 = time.time()
            source = str(getattr(
                self.frame_buffer, "current_source_type", "") or "unknown")
            try:
                written = await worker.process_once()
                is_screen = worker._source_is_screen()
                phase = "tick" if is_screen else "source_skipped"
                now = time.time()
                idle_key = f"{phase}:{source}"
                # Keep every productive OCR tick. Repeated no-op ticks are a
                # heartbeat, not a distinct trajectory step; sampling them keeps
                # a stress-test trace readable and prevents 1 Hz DOM growth.
                idle_period = 10.0 if is_screen else 30.0
                if (int(written) > 0 or idle_key != last_idle_key
                        or (now - last_idle_emit) >= idle_period):
                    await self._emit_worker_progress("OCRWorker", {
                        "phase": phase,
                        "source_type": source,
                        "scope": "screen_only",
                        "written": int(written),
                        "elapsed_sec": now - t0,
                        "model": model,
                        "heartbeat_sec": idle_period if not written else 0,
                    })
                    last_idle_emit = now
                    last_idle_key = idle_key
            except asyncio.CancelledError:
                break
            except Exception as exc:
                await self._emit_worker_progress("OCRWorker", {
                    "phase": "failed", "source_type": source,
                    "error": str(exc)[:500], "elapsed_sec": time.time() - t0,
                })
                log.warning("[ocr-worker] loop error: %s", exc)
            await asyncio.sleep(interval)

    async def _writer_loop(self) -> None:
        cfg = self.cfg
        wake = float(cfg.writer_wake_interval or 5.0)
        await asyncio.sleep(min(2.0, wake))
        max_fail = max(1, int(cfg.memory_max_consecutive_failures or 5))
        fails = 0

        # ★ FIX (watchdog): 只要段挂在 accumulator 里超过 cap, 就在 _writer_loop
        #   层强制 seal —— 覆盖三种 wake_once 完全够不着的情况:
        #     (a) frame_buffer 空 → wake_once 根本不会被调
        #     (b) wake_once 抛异常 → 直接进 except, 不走 finalize
        #     (c) 连续多拍 LLM 失败 → wake_once 走的是 return WriterResult(...) 早退
        #   watchdog 内部会双维度检查 (挂钟 + 视频帧 ts), 二者都超 cap 才 seal,
        #   避免时钟漂移误伤。
        async def _run_watchdog() -> None:
            try:
                mid = await self.memory_writer.try_watchdog_seal()
                if mid:
                    self._push_ctx()
            except Exception as e:
                log.warning("[mm-memory] watchdog seal 内部异常: %s", e)

        # ★ FIX (A): writer 每次 wake 前抢通道锁; recall 每步 LLM 之前也 acquire
        #   同一把锁 → writer 只需等 recall 当前一步 (~10s) 就能拿到通道, 段跨度
        #   回落到 30s cap 内。锁在 wake_once 结束或异常时严格释放。
        async def _acquire_channel():
            lock = self._llm_channel_lock
            if lock is None:
                return None
            await lock.acquire()
            return lock

        while not self._stop.is_set():
            cycle_started = time.monotonic()
            wrote = False
            channel = None
            try:
                if self.frame_buffer.size == 0:
                    # ★ FIX: buffer 空也先 watchdog 一次 —— 用户对着黑屏 30s+,
                    #   dHash 完全没新帧入 buffer, wake_once 永远不会被调, 老代码
                    #   段就会永远悬空。watchdog 只查本地状态, 不占 LLM 通道。
                    await _run_watchdog()
                else:
                    channel = await _acquire_channel()
                    try:
                        await self.memory_writer.wake_once(
                            on_progress=lambda ev: self._emit_worker_progress(
                                "MemoryWriter", ev))
                    finally:
                        if channel is not None:
                            channel.release()
                            channel = None
                    fails = 0
                    wrote = True
            except asyncio.CancelledError:
                if channel is not None:
                    try: channel.release()
                    except Exception: pass
                break
            except Exception as exc:
                if channel is not None:
                    try: channel.release()
                    except Exception: pass
                fails += 1
                log.debug("[mm-memory] writer wake failed %d/%d: %s", fails, max_fail, exc)
                if fails >= max_fail:
                    log.warning("[mm-memory] writer stopped (consecutive failures)")
                    break
                # ★ FIX: wake_once 抛异常也走一次 watchdog, 避免段被异常"卡死"。
                await _run_watchdog()
            # Push observation panels OUTSIDE the fail-count try — a UI-push
            # blip must not count as a memory-write failure. (_push_ctx also
            # swallows its own errors internally.)
            if wrote:
                self._push_ctx()
            # ★ FIX: 常规成功路径也顺手 watchdog 一次 —— wake_once 内已有前置
            #   cap 检查覆盖 "有帧但 LLM 失败" 的情况, 这里主要防御 wake_once
            #   返回后到下次 wake 之间的时段。开销极小 (无段时秒退)。
            await _run_watchdog()
            # Keep wake_interval measured from cycle start. A 35s model call
            # with a 10s interval should start the next backlog batch
            # immediately, not add another fixed 10s blind spot.
            cycle_elapsed = time.monotonic() - cycle_started
            sleep_for = max(0.0, wake - cycle_elapsed)
            if sleep_for:
                await asyncio.sleep(sleep_for)

    async def _reviewer_loop(self) -> None:
        cfg = self.cfg
        if not cfg.reviewer_enabled or not getattr(self, "memory_reviewers", None):
            return
        await asyncio.sleep(max(cfg.reviewer_wake_interval / 2,
                                cfg.writer_wake_interval * 3))
        max_fail = max(1, int(cfg.reviewer_max_consecutive_failures))
        fails = 0
        while not self._stop.is_set():
            try:
                if self.frame_buffer.size == 0:
                    await asyncio.sleep(cfg.reviewer_wake_interval)
                    continue
                await self._run_reviewers(triggered_by="interval")
                fails = 0
            except asyncio.CancelledError:
                break
            except Exception as exc:
                fails += 1
                log.debug("[mm-memory] reviewer wake failed %d/%d: %s", fails, max_fail, exc)
                if fails >= max_fail:
                    log.warning("[mm-memory] reviewer stopped (consecutive failures)")
                    break
            await asyncio.sleep(cfg.reviewer_wake_interval)
