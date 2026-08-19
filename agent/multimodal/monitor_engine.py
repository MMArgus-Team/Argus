# -*- coding: utf-8 -*-
"""MonitorEngine — async job container for always-on video monitors.

Replaces the old single per-session sync daemon
(``tui_gateway/server.py:_multimodal_monitor_loop``) with a WatcherAgent-style
architecture:

  * one background thread hosting a private asyncio event loop (``run_forever``);
  * EACH monitor is a long-lived async job (its own ``while`` tick loop, one T
    per iteration) scheduled on that loop;
  * ``add_monitor`` / ``remove_monitor`` (called from the gateway thread via
    ``run_coroutine_threadsafe``) spawn / cancel per-monitor tasks — the engine
    thread is just the container that owns and manages these jobs.

Monitors are long-lived periodic watchers ("alert me when X appears"), unlike
WatcherAgent's one-shot deep-analysis jobs, so a job here is a forever tick loop
rather than a bounded run. LLM calls go through an AsyncOpenAI client
(``HermesClientFactory.monitor_client``), so multiple monitors' vision calls can
overlap instead of serializing on one thread.

Side effects the engine cannot do itself (they touch gateway/session state) are
injected as thread-safe callbacks, mirroring WatcherAgent's ``emit_cb`` /
``on_delegation_complete``:

  * ``speak_cb(mid, entry, text) -> bool`` — deliver a hit through the gateway.
    UI-only notifications can appear beside a foreground answer; an explicit
    main-agent hook is serialized through the session FIFO instead of dropped.
  * ``notify_cb(kind, mid, entry, text)`` — surface an error / circuit-break
    notice (write to history, push to frontend, refresh the monitors panel).
  * ``emit_cb(event, payload)`` — raw dashboard push (rarely needed directly).

The monitor registry (``agent.mm_monitors``) is shared by reference with
``set_monitor`` so create/update/enable/disable/delete are seen live by the jobs.
"""

from __future__ import annotations

import asyncio
import base64
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
import contextvars
from io import BytesIO
import inspect
import json
import logging
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger("hermes.multimodal.monitor")


# --------------------------------------------------------------------------- #
# Pure helpers (moved verbatim from tui_gateway/server.py)
# --------------------------------------------------------------------------- #
# 每轮 eval 新帧上限: ≤ 此值全送, 超过即判拥塞降采样。采集 2fps → 32 帧 ≈ 16s 画面量,
# 即"攒够 16 秒还没评估完"就算拥塞 (原为 64=32s, 对单请求图量偏大)。
_MM_MONITOR_MAX_WINDOW = 32
# A SPEAK alert is durable UI evidence, not another copy of the full model
# payload. Keep a uniformly sampled strip small enough to hydrate safely on
# session reopen while still showing the beginning, middle and end of the
# exact frame batch the model evaluated.
_MM_MONITOR_EVIDENCE_MAX_FRAMES = 6
_MM_MONITOR_EVIDENCE_MAX_SIDE = 320
_MM_MONITOR_EVIDENCE_JPEG_QUALITY = 58
_MM_MONITOR_EVIDENCE_MAX_B64_CHARS = 600_000
# 持续拥塞时抽样步长几何翻倍 (stride 1→2→4→…); 达到此翻倍次数仍未缓解 → 放弃, 只送尾 2 帧。
# cap=32 时 stride 到 32 (=2^5) 即已把整窗抽成 1 帧, 所以 5 次翻倍就够; 取 5 与 cap 对齐。
_MM_MONITOR_MAX_DOUBLINGS = 5


def _uniform_sample(items: list, limit: int) -> list:
    """Return at most *limit* evenly-spaced items, including both ends."""
    if limit <= 0 or not items:
        return []
    if len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[-1]]
    indexes = [round(i * (len(items) - 1) / (limit - 1)) for i in range(limit)]
    return [items[i] for i in indexes]


def _monitor_evidence_thumb(jpeg_b64: str) -> str:
    """Create a bounded UI-only thumbnail; failures never retain the original."""
    try:
        from PIL import Image

        raw = base64.b64decode(jpeg_b64, validate=True)
        image = Image.open(BytesIO(raw)).convert("RGB")
        if max(image.size) > _MM_MONITOR_EVIDENCE_MAX_SIDE:
            resampling = getattr(Image, "Resampling", Image)
            image.thumbnail(
                (_MM_MONITOR_EVIDENCE_MAX_SIDE, _MM_MONITOR_EVIDENCE_MAX_SIDE),
                resampling.LANCZOS,
            )
        out = BytesIO()
        image.save(
            out,
            format="JPEG",
            quality=_MM_MONITOR_EVIDENCE_JPEG_QUALITY,
            optimize=True,
        )
        return base64.b64encode(out.getvalue()).decode("ascii")
    except Exception:
        return ""


def build_monitor_evidence(frames: list) -> dict:
    """Build the durable evidence strip from the exact model-input frames."""
    rows = []
    used_chars = 0
    for frame in _uniform_sample(frames, _MM_MONITOR_EVIDENCE_MAX_FRAMES):
        thumb = _monitor_evidence_thumb(str(getattr(frame, "jpeg_b64", "") or ""))
        if not thumb or used_chars + len(thumb) > _MM_MONITOR_EVIDENCE_MAX_B64_CHARS:
            continue
        used_chars += len(thumb)
        rows.append({
            "ts": float(getattr(frame, "ts", 0.0) or 0.0),
            "source_type": str(getattr(frame, "source_type", "") or ""),
            "thumb_b64": thumb,
        })
    return {
        "input_count": len(frames),
        "shown_count": len(rows),
        "frames": rows,
    }


def merge_monitor_evidence(current: Optional[dict], incoming: Optional[dict]) -> dict:
    """Merge aggregation-window evidence without allowing image growth."""
    left = current if isinstance(current, dict) else {}
    right = incoming if isinstance(incoming, dict) else {}
    combined = [
        row for row in list(left.get("frames") or []) + list(right.get("frames") or [])
        if isinstance(row, dict) and row.get("thumb_b64")
    ]
    rows = _uniform_sample(combined, _MM_MONITOR_EVIDENCE_MAX_FRAMES)
    return {
        "input_count": int(left.get("input_count") or 0) + int(right.get("input_count") or 0),
        "shown_count": len(rows),
        "frames": rows,
    }


def _model_prefers_portable_chat_params(model: str) -> bool:
    # Normalize spaces/hyphens so config names like "GPT-5.6 Luna" match the
    # gateway's canonical "gpt-5.6-luna" form.
    ml = (model or "").lower().replace(" ", "-")
    return "gpt-5.6-luna" in ml or "kimi-k3" in ml


def _model_is_kimi_k3(model: str) -> bool:
    return "kimi-k3" in (model or "").lower().replace(" ", "-")


def _model_is_kimi_k26(model: str) -> bool:
    return (model or "").strip().lower() == "kimi-k2.6"


# 从模型输出里抠出 JSON 对象 (容忍前后杂字)。
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
# 代码围栏: 模型常把 JSON 放进 ```json ... ``` / ``` ... ``` / ~~~ ... ~~~。
#   捕获围栏【内部】内容; 开栏可带语言标注 (json/JSON/…)。DOTALL 跨行。
_CODE_FENCE_RE = re.compile(
    r"(?:```|~~~)[ \t]*[a-zA-Z0-9_-]*[ \t]*\r?\n(?P<body>.*?)\r?\n?(?:```|~~~)",
    re.DOTALL,
)
_SILENT_RE = re.compile(r"^SILENT(?:\s*[:：].*)?$", re.IGNORECASE | re.DOTALL)
_SPEAK_RE = re.compile(
    r"^SPEAK\s*[:：]\s*(?P<reason>.+?)\s*$",
    re.IGNORECASE | re.DOTALL,
)
_MAX_SPEAK_REASON_CHARS = 160
_MM_MONITOR_MAX_OUTPUT_TOKENS = 256
_MM_MONITOR_KIMI_K26_MAX_OUTPUT_TOKENS = 128
# GPT-5-family ``max_completion_tokens`` includes any hidden reasoning budget.
# 256 is enough for a one-image scene probe but can be exhausted by a 10+ image
# monitor window before the one-line verdict is emitted. Keep the ordinary
# no-thinking Qwen path at 256 and give portable reasoning routes modest headroom.
_MM_MONITOR_PORTABLE_MAX_OUTPUT_TOKENS = 1024


def _strip_code_fence(s: str) -> str:
    """If the string is wrapped (fully or partly) in a ``` ``` or ~~~ code fence,
    return the fence's inner content; otherwise return it unchanged.

    Handles the common case where the model puts its JSON inside a code block.
    Returns the first fenced block (a monitor verdict has only one JSON). No fence
    found → returned unchanged for the downstream JSON extraction."""
    if not s:
        return s
    m = _CODE_FENCE_RE.search(s)
    if m:
        return (m.group("body") or "").strip()
    return s


def _brief_monitor_reason(value: Any) -> str:
    """Normalise a hit description and keep it safe for logs/notifications."""
    text = " ".join(str(value or "").split())
    return text[:_MAX_SPEAK_REASON_CHARS].rstrip()


def parse_monitor_verdict(raw) -> "tuple[bool, str]":
    """Parse the SPEAK/SILENT verdict into ``(hit, reason)``.

    The live protocol is deliberately tiny: ``SILENT`` for every miss and
    ``SPEAK: <brief reason>`` for a hit. During rolling upgrades we still accept
    the former JSON ``{"status": true/false, "reason": "..."}`` shape, including
    fenced/surrounded JSON and loose bool encodings.

    A miss *always* returns an empty reason, even when a legacy or malformed
    response supplies one. Invalid output is a conservative miss.
    """
    s = (raw or "").strip()
    if not s:
        return False, ""
    # ① 先显式剥掉代码围栏 (若有), 拿到内部内容。
    inner = _strip_code_fence(s)

    # ② 先读新的单行协议。兼容模型偶尔把整行包成 JSON string。
    decoded = None
    try:
        decoded = json.loads(inner)
    except Exception:
        pass
    protocol_text = decoded.strip() if isinstance(decoded, str) else inner.strip()
    if (len(protocol_text) >= 2
            and protocol_text[0] == protocol_text[-1]
            and protocol_text[0] in ("'", '"')):
        protocol_text = protocol_text[1:-1].strip()
    if _SILENT_RE.fullmatch(protocol_text):
        return False, ""
    speak = _SPEAK_RE.fullmatch(protocol_text)
    if speak:
        reason = _brief_monitor_reason(speak.group("reason"))
        return (True, reason) if reason else (False, "")

    # ③ 旧 JSON 兼容: 剥栏后内容多为纯 JSON, 先复用 decoded。
    obj = decoded if isinstance(decoded, dict) else None
    # ④ 兜底: 从内容里抠首个 {...} 再 loads (容忍前后仍有杂字)。
    if not isinstance(obj, dict):
        m = _JSON_OBJ_RE.search(inner)
        if not m:
            return False, ""
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return False, ""
    if not (isinstance(obj, dict) and "status" in obj):
        return False, ""
    st = obj.get("status")
    # status 可能是 bool, 也可能被模型写成 "true"/"是"/1 等。
    if isinstance(st, bool):
        hit = st
    elif isinstance(st, (int, float)):
        hit = bool(st)
    else:
        _sv = str(st).strip().lower()
        hit = _sv in ("true", "1", "yes", "是", "命中", "出现", "speak")
    if not hit:
        return False, ""
    return True, _brief_monitor_reason(obj.get("reason"))


def fmt_agg(buf: list) -> str:
    """Join a monitor's buffered SPEAKs into one digest, one line per event,
    each prefixed with an ABSOLUTE wall-clock timestamp.

    ``buf`` is a list of ``(wall_ts, text)`` tuples in chronological order.
    Returns "" if the buffer is empty.
    """
    lines = []
    for ts, text in buf:
        try:
            stamp = time.strftime("%H:%M:%S", time.localtime(float(ts)))
        except Exception:
            stamp = "--:--:--"
        lines.append(f"[{stamp}] {str(text).strip()}")
    return "\n".join(lines)


def should_flush(report_interval, agg_buf: list,
                 window_start: float, now: float) -> bool:
    """True when an aggregation window is due to flush.

    Only in aggregation mode (positive ``report_interval``), only with at least
    one buffered SPEAK (empty windows never emit), and only once the interval has
    elapsed since ``window_start``.
    """
    if not report_interval or report_interval <= 0:
        return False
    if not agg_buf:
        return False
    return (now - float(window_start or 0.0)) >= float(report_interval)


def _trigger_mode(monitor: dict) -> str:
    """Return the persisted trigger contract for a monitor.

    Entries restored from event files created before trigger modes existed have
    no key. Those monitors were historically long-lived, so missing/invalid
    values deliberately retain ``continuous`` semantics.
    """
    return (
        "once"
        if str(monitor.get("trigger_mode", "continuous")).strip().lower() == "once"
        else "continuous"
    )


def _monitor_revision(monitor: dict) -> int:
    """Return the live semantic-config revision used by verdict CAS checks."""
    try:
        return int(monitor.get("_config_revision", 0) or 0)
    except (TypeError, ValueError):
        return 0


def pick_window(new_frames_count: int, overload_ticks: int,
                base_period: int = 4):
    """Decide (target_frames, stride, tail_only) for a congested tick.

    Sub-cap: send all frames. At-cap: geometric stride based on sustained
    overload; after too many doublings, tail_only (just the last 2 frames).
    """
    target = 4
    while target < new_frames_count and target < _MM_MONITOR_MAX_WINDOW:
        target *= 2
    target = min(target, _MM_MONITOR_MAX_WINDOW)
    if target < _MM_MONITOR_MAX_WINDOW:
        return target, 1, False
    period = max(1, int(base_period))
    doublings = max(0, int(overload_ticks)) // period
    if doublings >= _MM_MONITOR_MAX_DOUBLINGS:
        return _MM_MONITOR_MAX_WINDOW, 0, True
    return _MM_MONITOR_MAX_WINDOW, (1 << doublings), False


def sample_frames(frames, target_frames: int, stride: int, tail_only: bool):
    """Downsample a chronological frame list to fit the target window."""
    if not frames:
        return []
    if tail_only:
        return frames[-2:]
    sampled = frames[::stride] if stride > 1 else frames
    if len(sampled) > target_frames:
        step = len(sampled) / target_frames
        sampled = [sampled[int(i * step)] for i in range(target_frames)]
    return sampled


def _monitor_completion_kwargs(model: str, *, messages: list) -> dict:
    """Build model-compatible chat.completions kwargs for monitor verdicts."""
    model = (model or "").strip()
    if _model_is_kimi_k26(model):
        # The Kimi K2.6 OpenAI-compatible endpoint uses ``thinking`` (rather
        # than vLLM's usual ``enable_thinking`` spelling) for instant mode.
        # Keep the short one-line verdict budget while disabling hidden
        # reasoning so it cannot consume that budget before content is emitted.
        return {
            "model": model,
            "messages": messages,
            "max_tokens": _MM_MONITOR_KIMI_K26_MAX_OUTPUT_TOKENS,
            "stream": False,
            "timeout": 30.0,
            "extra_body": {"chat_template_kwargs": {"thinking": False}},
        }
    if _model_prefers_portable_chat_params(model):
        max_completion_tokens = 3072 if _model_is_kimi_k3(model) else 4096
        kw = {
            "model": model or None,
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
            "stream": False,
            "timeout": 30.0,
        }
        if _model_is_kimi_k3(model):
            kw["reasoning_effort"] = "low"
        return kw
    return {
        "model": model or None,
        "messages": messages,
        # 2048: 文字密集画面下模型会把较长内容写进 reason, 200 会截断 JSON →
        # json.loads 失败 → 命中被误吞成"未触发"。给足空间让 JSON 写完整。
        "max_tokens": 2048,
        "temperature": 0.2,
        "stream": False,
        "timeout": 30.0,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }


# --------------------------------------------------------------------------- #
# MonitorEngine
# --------------------------------------------------------------------------- #
class MonitorEngine:
    """Async job container for per-session video monitors. See module docstring."""

    def __init__(
        self,
        frame_buffer: Any,
        *,
        monitors_ref: Dict[str, Dict[str, Any]],
        hermes_cfg: Optional[dict] = None,
        sid: Optional[str] = None,
        emit_cb: Optional[Callable[[str, dict], None]] = None,
        speak_cb: Optional[Callable[[str, dict, str], bool]] = None,
        notify_cb: Optional[Callable[[str, str, dict, str], None]] = None,
        is_source_off: Optional[Callable[[], bool]] = None,
        is_session_busy: Optional[Callable[[], bool]] = None,
        # Back-compat alias for is_source_off (older callers passed is_stream_stale).
        is_stream_stale: Optional[Callable[[], bool]] = None,
    ):
        self.frame_buffer = frame_buffer
        # SHARED by reference with agent.mm_monitors — set_monitor mutates it,
        # jobs read it live.
        self.monitors = monitors_ref
        self._hermes_cfg = hermes_cfg
        self._sid = sid
        self._emit_cb = emit_cb
        self._speak_cb = speak_cb
        self._notify_cb = notify_cb
        # Gate hook: True ⇒ the video source is OFF (UI switch stopped / never
        # captured) ⇒ skip NEW-frame evaluation this tick (a due aggregation
        # flush still runs). NOT a frame-freshness gate. Gateway supplies a
        # session-aware version; default "source on" so the engine works
        # standalone (tests).
        self._is_source_off = is_source_off or is_stream_stale or (lambda: False)
        self._is_session_busy = is_session_busy or (lambda: False)

        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._healthy = False
        self._lifecycle_lock = threading.Lock()
        self._closed_client_ids: set[int] = set()

        # Built lazily on the engine thread.
        self.cfg = None
        self.client = None
        self.model: str = ""

        # Per-monitor async jobs keyed by monitor_id.
        self._jobs: Dict[str, asyncio.Task] = {}
        # Frame-driven wakeups. One Event per monitor lets every independent job
        # observe the same new-frame notification; a single shared Event would
        # let the first waiter clear it and starve its siblings.
        self._frame_events: Dict[str, asyncio.Event] = {}

        # Cross-monitor delivery policy lives in speak_cb: UI notices are a
        # side channel, while main-agent hooks enter the foreground FIFO.

    # ------------------------------------------------------------------ #
    # Lifecycle (mirrors WatcherAgent.start/_run/stop)
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        with self._lifecycle_lock:
            if self._thread is None:
                # ContextVar values (notably the per-profile HERMES_HOME
                # override) are not inherited by a new Python thread.  Capture
                # the caller's session context so lazy config loads and monitor
                # event files stay in the owning profile instead of drifting
                # into the backend's launch profile.
                runtime_context = contextvars.copy_context()
                self._thread = threading.Thread(
                    target=runtime_context.run,
                    args=(self._run,),
                    name="mm-monitor-engine",
                    daemon=True,
                )
                self._thread.start()
        self._ready.wait(timeout=10.0)

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            if not self._stop.is_set() and self._build():
                # Spawn jobs for monitors restored as enabled before startup.
                try:
                    for mid, monitor in list(self.monitors.items()):
                        if bool(monitor.get("enabled", True)):
                            self._spawn_job(mid)
                except Exception as exc:
                    log.debug("[mm-monitor] initial spawn failed: %s", exc)
                # Reconcile only this session and only jobs actually spawned.
                try:
                    from . import monitor_agent as _ma
                    _fixed = _ma.reconcile_stale(
                        active_ids=list(self._jobs.keys()), session_id=self._sid)
                    if _fixed:
                        log.info(
                            "[mm-monitor] startup reconcile: %d monitor 文件被校准",
                            _fixed,
                        )
                except Exception as rec_exc:
                    log.debug("[mm-monitor] reconcile_stale failed: %s", rec_exc)
                if not self._stop.is_set():
                    # Publish readiness only from a callback dispatched by the
                    # running loop. Marking healthy before run_forever creates
                    # a window where add_monitor queues work into a loop that
                    # cannot yet acknowledge it and falsely times out.
                    def _mark_loop_ready() -> None:
                        if self._stop.is_set():
                            loop.stop()
                            self._ready.set()
                            return
                        self._healthy = True
                        self._ready.set()
                        log.info(
                            "[mm-monitor] engine ready (sid=%s model=%s)",
                            self._sid, self.model,
                        )

                    loop.call_soon(_mark_loop_ready)
                    loop.run_forever()
            else:
                self._healthy = False
                self._ready.set()
        except Exception as exc:
            log.debug("[mm-monitor] loop ended: %s", exc, exc_info=True)
        finally:
            self._healthy = False
            self._ready.set()
            if not loop.is_closed():
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
                    self._jobs.clear()
                    self._frame_events.clear()
                    await self._close_owned_client()

                try:
                    loop.run_until_complete(_drain_owned_tasks())
                except Exception as exc:
                    log.debug("[mm-monitor] async teardown failed: %s", exc)
                try:
                    loop.close()
                except Exception:
                    pass
            try:
                asyncio.set_event_loop(None)
            except Exception:
                pass
            self._loop = None

    def stop(self, timeout: float = 5.0) -> bool:
        """Stop all jobs, close owned transports, and join the engine thread.

        The bounded join makes session teardown deterministic without ever
        closing a client shared with the main agent.
        """
        self._stop.set()
        self._healthy = False
        loop = self._loop
        thread = self._thread
        if (loop is not None and not loop.is_closed() and loop.is_running()):
            async def _teardown_and_stop() -> None:
                try:
                    tasks = [t for t in self._jobs.values() if not t.done()]
                    for task in tasks:
                        task.cancel()
                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)
                    self._jobs.clear()
                    self._frame_events.clear()
                finally:
                    loop.stop()

            def _schedule_teardown() -> None:
                coro = _teardown_and_stop()
                try:
                    loop.create_task(coro)
                except Exception:
                    coro.close()
                    loop.stop()

            try:
                loop.call_soon_threadsafe(_schedule_teardown)
            except Exception as exc:
                log.debug("[mm-monitor] stop scheduling failed: %s", exc)

        if (thread is not None
                and thread is not threading.current_thread()
                and thread.ident is not None):
            thread.join(timeout=max(0.0, float(timeout)))
        stopped = bool(thread is None or not thread.is_alive())
        if not stopped:
            log.warning(
                "[mm-monitor] engine stop timed out after %.1fs",
                max(0.0, float(timeout)),
            )
        return stopped

    async def _close_owned_client(self) -> None:
        """Close the dedicated monitor transport once, on its owner loop."""
        client = self.client
        if (client is None
                or not bool(getattr(client, "_hermes_submodule_owned", False))
                or id(client) in self._closed_client_ids):
            return
        self._closed_client_ids.add(id(client))
        close = getattr(client, "close", None)
        if close is None:
            return
        try:
            result = close()
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            log.debug("[mm-monitor] client close failed: %s", exc)

    # ------------------------------------------------------------------ #
    def _build(self) -> bool:
        """Resolve cfg + the async monitor LLM client on the engine thread."""
        try:
            from .hermes_glue import build_config, HermesClientFactory
            self.cfg = build_config(self._hermes_cfg)
            self.client, self.model = HermesClientFactory(self.cfg).monitor_client()
            return self.client is not None
        except Exception as exc:
            log.warning("[mm-monitor] build failed: %s", exc, exc_info=True)
            return False

    # ------------------------------------------------------------------ #
    # Thread-safe job management (called from the gateway / set_monitor thread)
    # ------------------------------------------------------------------ #
    def is_healthy(self) -> bool:
        """Return whether this engine can accept monitor jobs right now."""
        thread = self._thread
        return bool(
            self._healthy
            and self._loop is not None
            and thread is not None
            and thread.is_alive()
            and not self._stop.is_set()
        )

    def add_monitor(self, mid: str, timeout: float = 1.0) -> bool:
        """Create/confirm a per-monitor job on the engine loop.

        Merely enqueueing ``call_soon_threadsafe`` is not sufficient: a closed
        or failing loop would otherwise make set_monitor report ``running``
        while no task exists. The bounded acknowledgement normally completes
        in the same event-loop turn (milliseconds) and fails closed on timeout.
        """
        loop = self._loop
        if not self.is_healthy() or loop is None or not mid:
            return False
        if threading.current_thread() is self._thread:
            return self._spawn_job(mid)
        acknowledgement: Future = Future()

        def _schedule() -> None:
            if not acknowledgement.set_running_or_notify_cancel():
                return
            try:
                acknowledgement.set_result(self._spawn_job(mid))
            except BaseException as exc:
                acknowledgement.set_exception(exc)

        try:
            loop.call_soon_threadsafe(_schedule)
            try:
                return bool(acknowledgement.result(
                    timeout=max(0.0, float(timeout))))
            except FutureTimeoutError:
                # cancel() succeeds only before the callback starts. The
                # queued callback then observes cancellation through
                # set_running_or_notify_cancel and cannot create a ghost job.
                if acknowledgement.cancel():
                    log.warning("[mm-monitor] add_monitor(%s) timed out", mid)
                    return False
                # The callback has started; _spawn_job is synchronous and
                # non-blocking, so consume its authoritative result instead of
                # rolling back while a task may already exist.
                return bool(acknowledgement.result())
        except Exception as exc:
            log.debug("[mm-monitor] add_monitor(%s) failed: %s", mid, exc)
            return False

    def notify_frame(self) -> None:
        """Wake all monitor jobs after a server-received capture frame.

        Monitor reads FrameBuffer's short non-dHash raw ring. Gateway ingestion
        therefore calls this for every accepted 2fps frame, while the per-job
        debounce still coalesces bursts into one vision request.
        """
        loop = self._loop
        if loop is None:
            return

        def _wake_all() -> None:
            for event in list(self._frame_events.values()):
                event.set()

        try:
            loop.call_soon_threadsafe(_wake_all)
        except Exception:
            pass

    def remove_monitor(self, mid: str) -> None:
        """Cancel a per-monitor job (idempotent)."""
        loop = self._loop
        if loop is None or not mid:
            return

        def _cancel():
            t = self._jobs.pop(mid, None)
            self._frame_events.pop(mid, None)
            if t is not None:
                try:
                    t.cancel()
                except Exception:
                    pass
        try:
            loop.call_soon_threadsafe(_cancel)
        except Exception as exc:
            log.debug("[mm-monitor] remove_monitor(%s) failed: %s", mid, exc)

    def _spawn_job(self, mid: str) -> bool:
        """Create the async job for ``mid`` if not already running. Runs ON the
        engine loop (via call_soon_threadsafe or the initial spawn)."""
        if self._stop.is_set():
            return False
        existing = self._jobs.get(mid)
        if existing is not None and not existing.done():
            return True
        try:
            self._frame_events[mid] = asyncio.Event()
            self._jobs[mid] = asyncio.ensure_future(self._monitor_job(mid))
            return True
        except Exception as exc:
            log.debug("[mm-monitor] spawn_job(%s) failed: %s", mid, exc)
            self._frame_events.pop(mid, None)
            self._jobs.pop(mid, None)
            return False

    # ------------------------------------------------------------------ #
    # The per-monitor job — one monitor = one long-lived async task.
    # ------------------------------------------------------------------ #
    def _mm(self) -> dict:
        """Fresh multimodal config sub-dict (cheap; load_config is cached).

        Uses the shared flatten so the new nested layout (settings.* / model.*)
        and the legacy flat multimodal.* both resolve to the same flat keys.
        """
        try:
            from hermes_cli.config import load_config
            from agent.multimodal.hermes_glue import flatten_mm_config
            return flatten_mm_config(load_config() or {})
        except Exception:
            return {}

    def _tick_sec(self, mm: dict) -> float:
        # New frames wake jobs immediately; this is only the no-frame heartbeat
        # for aggregation flushes/static scenes. Keep it short as well so a
        # monitor never inherits the old ~5s blind spot.
        return max(0.5, float(mm.get("monitor_tick_sec", 1.0) or 1.0))

    def _sync_state(self, mid: str, status: str) -> bool:
        # 每 tick 把执行状态就地写进本 monitor 的事件文件头部，
        # 供恢复链路和前端 registry 共享同一个权威状态源。
        try:
            from . import monitor_agent as _ma
            return bool(_ma.set_status(mid, status))
        except Exception:
            return False

    def _emit_progress(self, mid: str, phase: str, **payload: Any) -> None:
        cb = self._emit_cb
        if cb is None:
            return
        try:
            cb("multimodal.trajectory", {
                "worker": "MonitorWorker",
                "monitor_id": mid,
                "phase": phase,
                **payload,
            })
        except Exception:
            pass

    def _complete_once(
        self,
        mid: str,
        monitor: dict,
        *,
        reason: str,
        delivered: bool,
    ) -> bool:
        """Move a one-shot monitor to its durable terminal state.

        The entry remains in the shared registry so the UI can display the
        completed result; disabling it makes the owning job retire on its next
        liveness check. ``set_status`` is monotonic for ``done`` and protects
        this terminal state from the job's ``finally`` block.
        """
        from . import monitor_agent as _ma

        live = self.monitors.get(mid)
        if live is None or live is not monitor:
            return False
        with _ma.monitor_state_lock(live):
            pending = live.get("_once_pending_completion")
            if (
                self.monitors.get(mid) is not live
                or not live.get("enabled", True)
                or _trigger_mode(live) != "once"
                or not isinstance(pending, dict)
                or int(pending.get("revision", -1)) != _monitor_revision(live)
            ):
                return False
            # Durable terminal state is the commit point.  Never retire the job
            # on an in-memory-only done: a restart would otherwise resurrect it
            # as interrupted and a silent one-shot could lose its only record.
            if not self._sync_state(mid, "done"):
                if not live.get("_once_status_error_notified"):
                    live["_once_status_error_notified"] = True
                    self._notify(
                        "error",
                        mid,
                        live,
                        "单次监控已命中，但完成状态写入失败；后台将自动重试。",
                    )
                return False
            live["enabled"] = False
            live["status"] = "done"
            live["completed_at"] = time.time()
            try:
                live["last_seen_ts"] = float(pending.get("cursor_ts"))
            except (TypeError, ValueError):
                pass
            live.pop("_once_pending_completion", None)
            live.pop("_once_delivery_error_notified", None)
            live.pop("_once_status_error_notified", None)
            live.pop("_once_event_error_notified", None)
            self._emit_progress(
                mid,
                "job_done",
                status="done",
                trigger_mode="once",
                delivered=bool(delivered),
                reason=reason,
            )
            return True

    def _retry_once_completion(self, mid: str, monitor: dict) -> bool:
        """Retry a staged one-shot delivery/terminal commit without re-evaluating.

        Returns true when a pending hit existed, whether it completed this call
        or remains queued for the next heartbeat.
        """
        from . import monitor_agent as _ma

        live = self.monitors.get(mid)
        if live is None or live is not monitor:
            return False
        with _ma.monitor_state_lock(live):
            pending = live.get("_once_pending_completion")
            if not isinstance(pending, dict):
                return False
            if (
                self.monitors.get(mid) is not live
                or not live.get("enabled", True)
                or _trigger_mode(live) != "once"
                or int(pending.get("revision", -1)) != _monitor_revision(live)
            ):
                live.pop("_once_pending_completion", None)
                return True
            if (pending.get("delivery_required")
                    and not pending.get("delivery_accepted")):
                if not self._speak(
                    mid,
                    live,
                    str(pending.get("reason") or ""),
                    evidence=pending.get("evidence"),
                ):
                    if not live.get("_once_delivery_error_notified"):
                        live["_once_delivery_error_notified"] = True
                        self._notify(
                            "error",
                            mid,
                            live,
                            "单次监控已命中，但提醒发送失败；后台将自动重试。",
                        )
                    return True
                pending["delivery_accepted"] = True
                pending["delivered"] = True
                live.pop("_once_delivery_error_notified", None)
            self._complete_once(
                mid,
                live,
                reason=str(pending.get("reason") or ""),
                delivered=bool(pending.get("delivered", False)),
            )
            return True

    async def _monitor_job(self, mid: str) -> None:
        log.info("[mm-monitor] job start %s (sid=%s)", mid, self._sid)
        self._emit_progress(mid, "job_start")
        last_eval_mono = 0.0
        try:
            while not self._stop.is_set():
                monitor = self.monitors.get(mid)
                if not monitor or not bool(monitor.get("enabled", True)):
                    break
                from . import monitor_agent as _ma
                with _ma.monitor_state_lock(monitor):
                    if (self.monitors.get(mid) is not monitor
                            or not bool(monitor.get("enabled", True))):
                        break
                    self._sync_state(mid, "running")
                mm = self._mm()
                tick = self._tick_sec(mm)
                debounce = max(0.2, float(
                    mm.get("monitor_debounce_sec", 0.8) or 0.8))
                try:
                    remaining = debounce - (time.monotonic() - last_eval_mono)
                    if last_eval_mono and remaining > 0:
                        await asyncio.sleep(remaining)
                    # Rate-limit request *starts*, not completions. A 4-6s
                    # vision request has already exceeded the 0.8s debounce;
                    # sleeping another 0.8s afterwards only adds blind time.
                    last_eval_mono = time.monotonic()
                    await self._tick_once(mid, mm)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    log.debug("[mm-monitor] job %s tick error: %s", mid, exc)
                    self._emit_progress(mid, "tick_error", error=str(exc)[:500])
                # Monitor gone from the registry → job retires itself.
                monitor = self.monitors.get(mid)
                if not monitor or not bool(monitor.get("enabled", True)):
                    break
                # Event-driven first, periodic timeout second. The timeout keeps
                # aggregation windows and source-state transitions progressing
                # even when a static scene produces no new deduped frame.
                event = self._frame_events.setdefault(mid, asyncio.Event())
                if event.is_set():
                    event.clear()
                    continue
                try:
                    await asyncio.wait_for(event.wait(), timeout=tick)
                except asyncio.TimeoutError:
                    pass
                finally:
                    event.clear()
        except asyncio.CancelledError:
            pass
        finally:
            self._jobs.pop(mid, None)
            self._frame_events.pop(mid, None)
            # ★ 五态统一: 普通 job 退出 = 被移除/停止/熔断 → interrupted。
            #   单次监控会自然完成为 done；删除为 deleted。两个终态都
            #   不得被迟到的 finally 覆盖。
            try:
                from . import monitor_agent as _ma
                cur = (_ma.read_status(mid) or {}).get("status")
            except Exception:
                cur = None
            if cur not in ("deleted", "done"):
                self._sync_state(mid, "interrupted")
            log.info("[mm-monitor] job stop %s", mid)
            self._emit_progress(mid, "job_stop", status=cur or "interrupted")

    async def _tick_once(self, mid: str, mm: dict) -> None:
        """One evaluation tick for a single monitor (ported 1:1 from the old
        per-tick body, scoped to one monitor)."""
        m = self.monitors.get(mid)
        if not m:
            return
        # Per-monitor runtime flag (UI toggle / failure circuit-breaker). The
        # old global monitor_enabled master switch was removed (v33): the monitor
        # subsystem is always on; individual monitors gate on their own enabled.
        if not m.get("enabled", True):
            return
        if m.get("_config_updating"):
            return
        # A previously accepted one-shot hit may only be waiting for delivery
        # or its durable done write. Retry that transaction before requiring new
        # frames or issuing another model call.
        if _trigger_mode(m) == "once" and self._retry_once_completion(mid, m):
            return
        buf = self.frame_buffer
        if buf is None:
            return
        monitor_size = getattr(buf, "monitor_size", None)
        if monitor_size is None:
            monitor_size = getattr(buf, "size", 0)
        if int(monitor_size or 0) == 0:
            return

        now = time.time()
        tick = self._tick_sec(mm)

        # ── Aggregation flush (report_interval mode) — runs BEFORE the new-frame
        # gates, so a due window flushes even with no new frames / stale stream.
        report_interval = m.get("report_interval")
        if should_flush(report_interval, m.get("_agg_buf") or [],
                        m.get("_agg_window_start", 0.0), now):
            await self._flush_aggregate(mid, m, mm)

        # New-frame evaluation is skipped only when the video source is OFF.
        # MonitorWorker owns a separate loop/client, so a foreground main-agent
        # turn is not a reason to make perception blind. Delivery is serialized
        # separately by the gateway callback when a hit needs the main agent.
        try:
            if self._is_source_off():
                return
        except Exception:
            pass

        from . import monitor_agent as _ma_state
        with _ma_state.monitor_state_lock(m):
            if (self.monitors.get(mid) is not m
                    or not m.get("enabled", True)
                    or m.get("_config_updating")):
                return
            evaluated_monitor = m
            evaluated_revision = _monitor_revision(m)
            brief = str(
                m.get("brief", "") or m.get("monitor_query", "") or ""
            ).strip()
        if not brief:
            return

        # Cursor: everything strictly after the last frame we evaluated.
        last_seen = float(m.get("last_seen_ts", 0.0) or 0.0)
        overload = int(m.get("_overload_ticks", 0) or 0)
        if last_seen <= 0:
            # Production FrameBuffer exposes a monitor-only raw cursor; old test
            # fakes/custom buffers fall back to the deduped read contract.
            lt = getattr(buf, "monitor_latest_ts", None)
            if lt is None:
                lt = buf.latest_ts  # @property, not a method
            if lt is not None:
                last_seen = float(lt) - tick
        raw_reader = getattr(buf, "monitor_all_after", None)
        candidate_frames = (
            raw_reader(last_seen) if callable(raw_reader)
            else buf.all_after(last_seen)
        )
        # Monitor sees every server-received 2fps frame; dHash remains confined
        # to memory/watcher/main-agent consumers. Evaluate whatever accumulated
        # while the previous request ran, with the existing 32-frame overload cap.
        new_frames = [f for f in candidate_frames if f.ts > last_seen]
        if not new_frames:
            return

        base_period = int(mm.get("monitor_overload_period", 4) or 4)
        if len(new_frames) <= _MM_MONITOR_MAX_WINDOW:
            picked = new_frames
            m["_overload_ticks"] = 0
        else:
            target_n, stride, tail_only = pick_window(
                len(new_frames), overload, base_period=base_period)
            picked = sample_frames(new_frames, target_n, stride, tail_only)
            log.warning(
                "[mm-monitor] CONGESTION (%s): %d new frames in one tick "
                "(> cap %d) → downsampling to %d",
                mid, len(new_frames), _MM_MONITOR_MAX_WINDOW, len(picked))
            m["_overload_ticks"] = overload + 1
        if not picked:
            return

        # Snapshot the source epoch before yielding to the vision endpoint. A
        # camera/screen switch can happen while this request is in flight; its
        # verdict must not be delivered into the new source.
        evaluated_source_generation = getattr(buf, "source_generation", None)

        from ._memory import frame_to_image_content
        from .monitor_agent import MONITOR_AGENT_SYSTEM
        # ★ 直接用 frame buffer 里的原图 (采集端已封顶 1024/1280)，不再送前二次降质。
        #   原先 ÷2 + q70 重编码会把屏幕文字压糊 (512px + 双重 JPEG artifact) → 误识别。
        #   帧数已被 ≤32 窗口 + 拥塞降采样控住，无需再靠缩图省 token。
        frame_parts = []
        model_frames = []
        for f in picked:
            try:
                frame_parts.append(frame_to_image_content(f))
                model_frames.append(f)
            except Exception:
                continue
        if not frame_parts:
            return

        messages = [
            {"role": "system", "content": MONITOR_AGENT_SYSTEM},
            {"role": "user",
             "content": [{"type": "text", "text": f"User delegation: {brief}\n"}] + frame_parts},
        ]
        log.info("[mm-monitor] eval %s (%d frames)", mid, len(frame_parts))
        self._emit_progress(
            mid,
            "eval_start",
            brief=brief,
            n_new_frames=len(new_frames),
            n_frames=len(frame_parts),
            frame_ts=[float(f.ts) for f in model_frames],
            source_type=str(getattr(buf, "current_source_type", "") or ""),
        )

        raw = None
        resp = None
        eval_started = time.monotonic()
        try:
            resp = await self.client.chat.completions.create(
                **_monitor_completion_kwargs(self.model, messages=messages),
            )
            raw = self._extract(resp)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # A failure from an older query/mode must not trip the replacement
            # monitor's circuit breaker. Serialize this mutation with CRUD and
            # apply it only if the request still owns the live revision.
            live = self.monitors.get(mid)
            if live is None or live is not evaluated_monitor:
                self._emit_progress(
                    mid,
                    "eval_failure_discarded_monitor_changed",
                    evaluated_revision=evaluated_revision,
                    current_revision=None,
                )
                return
            with _ma_state.monitor_state_lock(live):
                if (
                    self.monitors.get(mid) is not live
                    or not live.get("enabled", True)
                    or live.get("_config_updating")
                    or _monitor_revision(live) != evaluated_revision
                ):
                    self._emit_progress(
                        mid,
                        "eval_failure_discarded_monitor_changed",
                        evaluated_revision=evaluated_revision,
                        current_revision=_monitor_revision(live),
                    )
                    return
                self._emit_progress(mid, "eval_failed", error=str(exc)[:500])
                self._handle_failure(mid, live, mm, exc)
            return

        # ★ Post-await liveness re-check: the LLM call above yielded control, so
        # set_monitor (gateway thread) may have deleted or disabled this monitor
        # meanwhile. Re-fetch the live entry and bail if it's gone/disabled —
        # otherwise we'd write to a detached dict or emit a bubble for a monitor
        # the user just turned off. (The old sync daemon couldn't hit this since
        # its LLM call blocked the whole loop; async jobs can.)
        m = self.monitors.get(mid)
        if (m is None
                or m is not evaluated_monitor
                or not m.get("enabled", True)):
            return
        current_source_generation = getattr(buf, "source_generation", None)
        if (evaluated_source_generation is not None
                and current_source_generation != evaluated_source_generation):
            self._emit_progress(
                mid,
                "verdict_discarded_source_changed",
                evaluated_source_generation=evaluated_source_generation,
                current_source_generation=current_source_generation,
            )
            return

        # 解析 SPEAK/SILENT; 滚动升级期仍容忍旧 JSON。
        hit, text = parse_monitor_verdict(raw)
        # Keep verdict observability in the durable log. Previously ``raw`` was
        # only emitted through the in-memory trajectory stream, so after an app
        # restart a genuine SILENT, an empty completion, and a protocol violation
        # were indistinguishable. Valid SILENTs need no raw text; malformed output
        # gets a bounded local preview so the parser/model contract is debuggable.
        _protocol_text = _strip_code_fence((raw or "").strip()).strip()
        try:
            _decoded_protocol = json.loads(_protocol_text)
        except Exception:
            _decoded_protocol = None
        if isinstance(_decoded_protocol, str):
            _protocol_text = _decoded_protocol.strip()
        if hit:
            _protocol_kind = "speak"
        elif not _protocol_text:
            _protocol_kind = "empty_output"
        elif _SILENT_RE.fullmatch(_protocol_text):
            _protocol_kind = "silent"
        elif isinstance(_decoded_protocol, dict) and "status" in _decoded_protocol:
            _protocol_kind = "legacy_miss"
        else:
            _protocol_kind = "protocol_violation"
        _usage = getattr(resp, "usage", None)
        _completion_tokens = getattr(_usage, "completion_tokens", None)
        _details = getattr(_usage, "completion_tokens_details", None)
        _reasoning_tokens = getattr(_details, "reasoning_tokens", None)
        _invalid_preview = ""
        if _protocol_kind in ("empty_output", "protocol_violation"):
            _invalid_preview = f" raw={(' '.join((raw or '').split())[:240])!r}"
        log.info(
            "[mm-monitor] verdict %s protocol=%s hit=%s elapsed=%.2fs "
            "chars=%d completion_tokens=%s reasoning_tokens=%s%s",
            mid, _protocol_kind, bool(hit), time.monotonic() - eval_started,
            len(raw or ""), _completion_tokens, _reasoning_tokens,
            _invalid_preview,
        )
        # ★ 2026-08-19: 卸到线程池。build_monitor_evidence 对 6 帧做
        #   base64 解码 + PIL LANCZOS 缩放 + JPEG 重编码, 实测同步阻塞
        #   1080p 94ms / 1440p 157ms / MBP Retina 全屏 290ms。这里是 engine
        #   私有 loop (见模块 docstring: 一线程一 loop, 多 monitor 的 vision
        #   调用靠 await 重叠), 同步跑会让同 loop 上其它 monitor 的 tick 一起
        #   等这 0.3s —— 正好削掉"不在一个线程上串行"这个设计点。
        evidence = (
            await asyncio.to_thread(build_monitor_evidence, model_frames)
            if hit else None
        )
        verdict_payload = {
            "hit": bool(hit),
            "reason": text,
            "raw": (raw or "")[:4000],
            "frame_ts": [float(f.ts) for f in model_frames],
        }
        if evidence:
            # Memory Debug consumes the same bounded strip. SILENT evaluations
            # intentionally remain metadata-only so a long-running monitor
            # cannot accumulate image bytes every tick.
            verdict_payload["input_frame_count"] = evidence["input_count"]
            verdict_payload["frames"] = evidence["frames"]
        # Commit the verdict under the same per-monitor lock used by CRUD. The
        # model call intentionally ran without this lock; revision + identity
        # form the CAS that rejects results from an older query/mode. Holding the
        # lock through append/delivery also makes disable/delete a strict barrier:
        # once either operation returns, no stale alert can escape afterward.
        with _ma_state.monitor_state_lock(m):
            if (
                self.monitors.get(mid) is not m
                or m is not evaluated_monitor
                or not m.get("enabled", True)
                or m.get("_config_updating")
                or _monitor_revision(m) != evaluated_revision
            ):
                self._emit_progress(
                    mid,
                    "verdict_discarded_monitor_changed",
                    evaluated_revision=evaluated_revision,
                    current_revision=_monitor_revision(m),
                )
                return
            # Emit only after the revision CAS succeeds. This keeps both the
            # text verdict and its evidence images out of a replacement
            # monitor's Debug stream when set_monitor wins the post-LLM race.
            self._emit_progress(mid, "verdict", **verdict_payload)

            # Only a successful response for the currently live contract may
            # clear its error latch/circuit-breaker streak. A stale success from
            # the previous query must not make the replacement look recovered.
            m["_err_notified"] = False
            m["_fail_streak"] = 0
            cursor_ts = float(picked[-1].ts)
            trigger_mode = _trigger_mode(m)
            armed = bool(m.get("_trigger_armed", True))
            if not hit:
                m["last_seen_ts"] = cursor_ts
                # A continuous monitor represents event *edges*, not every frame
                # on which the condition remains true. One explicit SILENT is
                # the reset edge that permits the next notification.
                if (trigger_mode == "continuous"
                        and not armed
                        and _protocol_kind in {"silent", "legacy_miss"}):
                    m["_trigger_armed"] = True
                    self._emit_progress(
                        mid,
                        "rearmed",
                        trigger_mode="continuous",
                    )
                return
            if trigger_mode == "continuous" and not armed:
                m["last_seen_ts"] = cursor_ts
                self._emit_progress(
                    mid,
                    "hit_suppressed_until_silent",
                    trigger_mode="continuous",
                    reason=text,
                )
                return
            if not text:
                text = "监控事件已出现"

            event_persisted = False
            try:
                _ma_state.append_event(mid, text)
                event_persisted = True
                m.pop("_once_event_error_notified", None)
            except Exception as exc:
                log.debug("[mm-monitor] append_event failed (%s): %s", mid, exc)

            if trigger_mode == "once":
                if not event_persisted:
                    if not m.get("_once_event_error_notified"):
                        m["_once_event_error_notified"] = True
                        self._notify(
                            "error",
                            mid,
                            m,
                            "单次监控已命中，但事件记录写入失败；后台将自动重试。",
                        )
                    # Do not advance the cursor: the same visual evidence remains
                    # eligible for retry after storage recovers.
                    return
                m["_once_pending_completion"] = {
                    "revision": evaluated_revision,
                    "reason": text,
                    "cursor_ts": cursor_ts,
                    "delivery_required": not bool(m.get("silent", False)),
                    "delivery_accepted": bool(m.get("silent", False)),
                    "delivered": False,
                    "evidence": evidence,
                }
                self._retry_once_completion(mid, m)
                return

            # Continuous hits are committed once per armed visual episode.
            m["last_seen_ts"] = cursor_ts
            m["_trigger_armed"] = False
            if m.get("silent", False):
                return
            if report_interval and report_interval > 0:
                if not m.get("_agg_buf"):
                    m["_agg_window_start"] = time.time()
                m.setdefault("_agg_buf", []).append((time.time(), text))
                m["_agg_evidence"] = merge_monitor_evidence(
                    m.get("_agg_evidence"), evidence,
                )
                return
            self._speak(mid, m, text, evidence=evidence)

    # ------------------------------------------------------------------ #
    async def _flush_aggregate(self, mid: str, m: dict, mm: dict) -> None:
        """Merge the aggregation window's SPEAKs into one digest + emit.

        The digest is formatted locally with timestamped lines (fmt_agg) — a
        deterministic concat. (An async LLM-merge variant could replace this
        later if richer digests are wanted.)
        """
        buf_list = list(m.get("_agg_buf") or [])
        digest = fmt_agg(buf_list)
        evidence = m.get("_agg_evidence")
        if digest and self._speak(mid, m, digest, evidence=evidence):
            live = self.monitors.get(mid)
            if live is not None:
                live["_agg_buf"] = []
                live.pop("_agg_evidence", None)
                live["_agg_window_start"] = 0.0
                if _trigger_mode(live) == "once":
                    self._complete_once(
                        mid,
                        live,
                        reason=digest,
                        delivered=True,
                    )

    # ------------------------------------------------------------------ #
    def _handle_failure(self, mid: str, m: dict, mm: dict, exc) -> None:
        """Circuit breaker + first-failure notice (ported from the old loop)."""
        log.warning("[mm-monitor] LLM call failed (%s): %s", mid, exc)
        reason = str(exc)[:200]
        m["_fail_streak"] = int(m.get("_fail_streak", 0)) + 1
        fail_cap = int(mm.get("monitor_fail_disable_after", 5) or 5)
        if m["_fail_streak"] >= fail_cap and m.get("enabled", True):
            m["enabled"] = False
            # ★ 五态统一: 熔断 = interrupted (中断态)。同步落文件 status + notify kind 用
            #   interrupted, 文案统一。之前是 enabled=False + notify("disabled")。
            self._sync_state(mid, "interrupted")
            label = m.get("label") or m.get("brief") or mid
            msg = (f"监控「{label}」连续 {m['_fail_streak']} 次模型调用失败, "
                   f"已【自动停用】以免持续消耗额度。请检查 config.yaml 的 "
                   f"multimodal.monitor_* 模型配置后重新启用/重建。\n原因: {reason}")
            log.warning("[mm-monitor] circuit-break: 自动停用 %s (%s)", mid, label)
            self._notify("interrupted", mid, m, msg)
            return
        if not m.get("_err_notified"):
            m["_err_notified"] = True
            label = m.get("label") or m.get("brief") or mid
            msg = (f"监控「{label}」启动异常: 模型调用失败。该监控暂时无法工作, "
                   f"请检查 config.yaml 的 multimodal.monitor_* 模型配置或查看 agent.log。\n"
                   f"原因: {reason}")
            self._notify("error", mid, m, msg)

    # ------------------------------------------------------------------ #
    def _extract(self, resp) -> str:
        try:
            from agent.auxiliary_client import extract_content_or_reasoning
            return (extract_content_or_reasoning(resp) or "").strip()
        except Exception:
            try:
                return (resp.choices[0].message.content or "").strip()
            except Exception:
                return ""

    def _speak(
        self,
        mid: str,
        m: dict,
        text: str,
        *,
        evidence: Optional[dict] = None,
    ) -> bool:
        if self._speak_cb is None:
            return False
        previous = m.get("_delivery_evidence")
        if evidence:
            m["_delivery_evidence"] = evidence
        else:
            m.pop("_delivery_evidence", None)
        try:
            return bool(self._speak_cb(mid, m, text))
        except Exception as exc:
            log.debug("[mm-monitor] speak_cb failed (%s): %s", mid, exc)
            return False
        finally:
            if previous is None:
                m.pop("_delivery_evidence", None)
            else:
                m["_delivery_evidence"] = previous

    def _notify(self, kind: str, mid: str, m: dict, text: str) -> None:
        # Always record to the event file so the timeline stays truthful.
        try:
            from . import monitor_agent as _ma
            _ma.append_event(mid, f"[{kind.upper()}] {text}")
        except Exception:
            pass
        if self._notify_cb is not None:
            try:
                self._notify_cb(kind, mid, m, text)
            except Exception as exc:
                log.debug("[mm-monitor] notify_cb failed (%s): %s", mid, exc)
