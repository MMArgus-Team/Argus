import atexit
import concurrent.futures
import contextlib
import contextvars
import copy
import hashlib
import inspect
import json
import logging
import os
import queue
import re
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from hermes_constants import (
    get_hermes_home,
    get_hermes_home_override,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from hermes_cli.env_loader import load_hermes_dotenv
from utils import is_truthy_value
from tools.environments.local import hermes_subprocess_env
from agent.replay_cleanup import sanitize_replay_history
from tui_gateway import git_probe
from tui_gateway.transport import (
    StdioTransport,
    Transport,
    bind_transport,
    current_transport,
    reset_transport,
)

logger = logging.getLogger(__name__)

_MM_MEMORY_STARTUP_TIMEOUT_SEC = 60.0
_MM_MEMORY_STOP_TIMEOUT_SEC = 5.0
_MM_WATCHER_STARTUP_TIMEOUT_SEC = 10.0
_MM_WATCHER_STOP_TIMEOUT_SEC = 5.0
_MM_WATCHER_DEPENDENCY_JOIN_SEC = 3.0
# Desktop source activation waits for the ordinary agent plus optional memory,
# Watcher, and Monitor services. Keep this aligned with the renderer's source
# activation request budget; the generic 30s agent wait is too short on a cold
# local install and caused a healthy camera preview to remain at zero frames.
_MM_CAPTURE_ACTIVATION_TIMEOUT_SEC = 120.0

# A durable multimodal conversation may be rebuilt under a new transport sid
# while its previous resident runtimes are still unwinding.  This is also the
# session lifecycle lock: construction publishes resident multimodal runtimes
# under this lock before start/wait, while finalize atomically marks the session
# and captures them. Strong registry references remain until each owner thread
# reports fully stopped, so a timeout can never look like "no runtime".
_MM_ACTIVE_MEMORY_BACKENDS_LOCK = threading.RLock()
_MM_ACTIVE_MEMORY_BACKENDS: dict[tuple[str, str], Any] = {}
_MM_ACTIVE_WATCHERS: dict[tuple[str, str], Any] = {}

_hermes_home = get_hermes_home()
load_hermes_dotenv(
    hermes_home=_hermes_home, project_env=Path(__file__).parent.parent / ".env"
)


# ── Panic logger ─────────────────────────────────────────────────────
# Gateway crashes in a TUI session leave no forensics: stdout is the
# JSON-RPC pipe (TUI side parses it, doesn't log raw), the root logger
# only catches handled warnings, and the subprocess exits before stderr
# flushes through the stderr->gateway.stderr event pump. This hook
# appends every unhandled exception to ~/.argus/logs/tui_gateway_crash.log
# AND re-emits a one-line summary to stderr so the TUI can surface it in
# Activity — exactly what was missing when the voice-mode turns started
# exiting the gateway mid-TTS.
_CRASH_LOG = os.path.join(_hermes_home, "logs", "tui_gateway_crash.log")


def _panic_hook(exc_type, exc_value, exc_tb):
    import traceback

    trace = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    try:
        os.makedirs(os.path.dirname(_CRASH_LOG), exist_ok=True)
        with open(_CRASH_LOG, "a", encoding="utf-8") as f:
            f.write(
                f"\n=== unhandled exception · {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
            )
            f.write(trace)
    except Exception:
        pass
    # Stderr goes through to the TUI as a gateway.stderr Activity line —
    # the first line here is what the user will see without opening any
    # log files.  Rest of the stack is still in the log for full context.
    first = (
        str(exc_value).strip().splitlines()[0]
        if str(exc_value).strip()
        else exc_type.__name__
    )
    print(f"[gateway-crash] {exc_type.__name__}: {first}", file=sys.stderr, flush=True)
    # Chain to the default hook so the process still terminates normally.
    sys.__excepthook__(exc_type, exc_value, exc_tb)


sys.excepthook = _panic_hook


def _thread_panic_hook(args):
    # threading.excepthook signature: SimpleNamespace(exc_type, exc_value, exc_traceback, thread)
    import traceback

    trace = "".join(
        traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)
    )
    try:
        os.makedirs(os.path.dirname(_CRASH_LOG), exist_ok=True)
        with open(_CRASH_LOG, "a", encoding="utf-8") as f:
            f.write(
                f"\n=== thread exception · {time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"· thread={args.thread.name} ===\n"
            )
            f.write(trace)
    except Exception:
        pass
    first_line = (
        str(args.exc_value).strip().splitlines()[0]
        if str(args.exc_value).strip()
        else args.exc_type.__name__
    )
    print(
        f"[gateway-crash] thread {args.thread.name} raised {args.exc_type.__name__}: {first_line}",
        file=sys.stderr,
        flush=True,
    )


threading.excepthook = _thread_panic_hook

try:
    from hermes_cli.banner import prefetch_update_check

    prefetch_update_check()
except Exception:
    pass

from tui_gateway.render import make_stream_renderer, render_diff, render_message

_sessions: dict[str, dict] = {}
_methods: dict[str, callable] = {}
_pending: dict[str, tuple[str, threading.Event]] = {}
_pending_prompt_payloads: dict[str, tuple[str, dict]] = {}
_answers: dict[str, str] = {}
_db = None
_db_error: str | None = None
_stdout_lock = threading.Lock()
_cfg_lock = threading.Lock()
_sessions_lock = threading.RLock()  # reentrant: _close_session_by_id may run under callers that already hold it
_prompt_lock = threading.Lock()
_cfg_cache: dict | None = None
_cfg_mtime: float | None = None
_cfg_path = None
_session_resume_lock = threading.Lock()
try:
    _slash_timeout = float(os.environ.get("ARGUS_TUI_SLASH_TIMEOUT_S") or "45")
except (ValueError, TypeError):
    _slash_timeout = 45.0
_SLASH_WORKER_TIMEOUT_S = max(5.0, _slash_timeout)

# When a WebSocket client (the dashboard's embedded-chat tab / desktop app)
# disconnects, ``tui_gateway.ws`` detaches the transport but intentionally
# leaves the session parked so a quick reconnect can reattach it (see ws.py).
# That park is unbounded, though: a browser refresh spins up a brand-new
# ``session.create`` (new sid + a fresh _SlashWorker via _deferred_build) and
# never reattaches the OLD sid, so the old session's slash-worker subprocess
# lingers forever — one leaked python process per refresh (#38591 fallout).
# After this grace window, an orphaned (transport-detached, not-running) WS
# session is reaped: its _SlashWorker is closed and the session finalized.
# Set to 0 to disable (park forever, pre-fix behaviour).
try:
    _ws_orphan_reap_grace = float(
        os.environ.get("ARGUS_TUI_WS_ORPHAN_REAP_GRACE_S") or "20"
    )
except (ValueError, TypeError):
    _ws_orphan_reap_grace = 20.0
_WS_ORPHAN_REAP_GRACE_S = max(0.0, _ws_orphan_reap_grace)
_DETAIL_SECTION_NAMES = ("thinking", "tools", "subagents", "activity")
_DETAIL_MODES = frozenset({"hidden", "collapsed", "expanded"})

# ── Async RPC dispatch (#12546) ──────────────────────────────────────
# A handful of handlers block the dispatcher loop in entry.py for seconds
# to minutes (slash.exec, cli.exec, shell.exec, session.resume,
# session.branch, session.compress, skills.manage).  While they're running, inbound RPCs —
# notably approval.respond and session.interrupt — sit unread in the
# stdin pipe.  We route only those slow handlers onto a small thread pool;
# everything else stays on the main thread so ordering stays sane for the
# fast path.  write_json is already _stdout_lock-guarded, so concurrent
# response writes are safe.
_LONG_HANDLERS = frozenset(
    {
        "billing.step_up",
        "browser.manage",
        "cli.exec",
        # Completion RPCs run inline on the reader thread by default, but both
        # can block it for seconds: complete.path spawns `git ls-files` and
        # fuzzy-ranks the whole repo (slow on large repos / WSL2 mounts), and
        # complete.slash does first-call prompt_toolkit imports + a skill-dir
        # scan. While either runs inline, prompt.submit / session.interrupt sit
        # unread in the stdin pipe — the TUI appears frozen until the 120s RPC
        # timeout fires (#21123). Routing them to the pool keeps the fast path
        # responsive; completion is read-only and write_json is lock-guarded.
        "complete.path",
        "complete.slash",
        "llm.oneshot",
        # Starting a desktop camera/screen source may wait for the ordinary
        # main-chat agent to finish building, then attach the resident MM
        # runtime before acknowledging. Keep that wait off the dispatcher so
        # frame/interrupt/approval traffic remains responsive.
        "multimodal.source_stopped",
        # Realtime ASR start can wait for runtime promotion + upstream WS
        # readiness; manual stop intentionally waits for a bounded
        # session.finish/session.finished flush.  Both must stay off the reader
        # thread so stop-before-start cancellation, audio/event traffic, and
        # unrelated chat streaming remain responsive.
        "multimodal.asr_start",
        "multimodal.asr_stop",
        # Session-switch hydration. The dashboard fires all four of these
        # immediately after session.resume returns (fetchRegistries /
        # fetchMmSidechannel / fetchTrajectory in MultimodalChatPage). resume
        # itself is already pool-routed, but these ran inline on the reader
        # thread — the same thread that flushes streaming tokens — so switching
        # sessions stalled the loop for seconds and the restored bubbles landed
        # LATER than the resume response that was supposed to carry them.
        # Symptom: "web 端切换 session 时候内容加载速度非常的慢", with
        # `ws write slow (loop stalled >10.0s)` and max_send=9.47s in agent.log.
        # trajectory.list is the heaviest (bounded at 16 MB of base64 evidence
        # thumbnails); the other three are DB reads over the mm sidechannel
        # tables. All four are read-only and write_json is lock-guarded.
        "multimodal.list_registries",
        "multimodal.list_monitor_alerts",
        "multimodal.list_watcher_content",
        "multimodal.trajectory.list",
        # Pet RPCs hit the network (manifest fetch / spritesheet download) or do
        # per-frame PNG decode/encode (pet.cells): inline they serialize on the
        # reader thread, so picker previews trickle in one at a time and the
        # animation poll stutters. On the pool they run concurrently.
        "pet.cells",
        "pet.gallery",
        # Generation is the heaviest pet path by far — multiple image-model
        # round-trips per call — so it must never block the reader thread.
        "pet.generate",
        "pet.hatch",
        "pet.select",
        "pet.thumb",
        "plugins.manage",
        "projects.discover_repos",
        "projects.record_repos",
        "projects.for_cwd",
        "projects.tree",
        "projects.project_sessions",
        "session.branch",
        "session.compress",
        # Same _history_to_messages projection session.resume runs, but it was
        # never pool-routed — so an explicit history reload of a long monitored
        # session (its base64 tool-result images make the projection heavy) ran
        # inline on the WS reader thread. Route it with resume.
        "session.history",
        "session.resume",
        "shell.exec",
        "skills.manage",
        "slash.exec",
        # High-frequency image upload (~2 fps × ~200 KB base64 per frame).
        # Even though FrameBuffer.push itself is fast, decoding + validating
        # the base64 payload on the dispatcher thread stalls it for tens of
        # ms per frame — long enough to serialize behind qwen SSE tokens
        # coming out of the same WS. Route to the pool so the dispatcher
        # loop stays free for chat streaming.
        "multimodal.frame",
        # Voice input: multimodal.user_audio runs a SYNCHRONOUS ASR transcribe
        # (network, ~seconds) AND then _run_prompt_submit inline. On the
        # dispatcher thread that froze the whole WS for seconds — the user's
        # own message bubble (and every other event) couldn't render until the
        # transcript came back. Symptom: "发语音卡住,等右侧文字出来才画到主
        # 界面;打字不卡". Route to the pool. (prompt.submit is already
        # thread-spawned, which is why typing never had this problem.)
        "multimodal.user_audio",
        # Env-audio slices (screen-share system audio) also run inline and do a
        # base64 decode before the async ingest — keep them off the dispatcher
        # too so a burst at share start can't stack behind chat.
        "multimodal.env_audio",
    }
)

try:
    _rpc_pool_workers = max(
        2, int(os.environ.get("ARGUS_TUI_RPC_POOL_WORKERS") or "4")
    )
except (ValueError, TypeError):
    _rpc_pool_workers = 4
_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=_rpc_pool_workers,
    thread_name_prefix="tui-rpc",
)
atexit.register(lambda: _pool.shutdown(wait=False, cancel_futures=True))

# Reserve real stdout for JSON-RPC only; redirect Python's stdout to stderr
# so stray print() from libraries/tools becomes harmless gateway.stderr instead
# of corrupting the JSON protocol.
_real_stdout = sys.stdout
sys.stdout = sys.stderr


class _DropTransport:
    """Detached WS sink: keep sessions resumable without writing stale frames."""

    def write(self, obj: dict) -> bool:
        return False

    def close(self) -> None:
        return None


# Module-level stdio transport — fallback sink when no transport is bound via
# contextvar or session. Stream resolved through a lambda so runtime monkey-
# patches of `_real_stdout` (used extensively in tests) still land correctly.
_stdio_transport = StdioTransport(lambda: _real_stdout, _stdout_lock)

# Detached websocket sessions use a drop sink instead of stdio. Desktop embeds
# the gateway in-process and captures stdout into logs, so stale JSON-RPC frames
# must not fall through there while the session waits for resume or reap.
_detached_ws_transport = _DropTransport()


class _SlashWorker:
    """Persistent HermesCLI subprocess for slash commands."""

    def __init__(self, session_key: str, model: str):
        self._lock = threading.Lock()
        self._seq = 0
        self.stderr_tail: list[str] = []
        self.stdout_queue: queue.Queue[dict | None] = queue.Queue()

        argv = [
            sys.executable,
            "-m",
            "tui_gateway.slash_worker",
            "--session-key",
            session_key,
        ]
        if model:
            argv += ["--model", model]

        self._closed = False
        self.proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=os.getcwd(),
            # slash_worker runs the Argus → needs provider credentials.
            # Tier-1 secrets (gateway/GitHub/infra) are still stripped (#29157).
            env=hermes_subprocess_env(inherit_credentials=True),
        )
        threading.Thread(target=self._drain_stdout, daemon=True).start()
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def _drain_stdout(self):
        for line in self.proc.stdout or []:
            try:
                self.stdout_queue.put(json.loads(line))
            except json.JSONDecodeError:
                continue
        self.stdout_queue.put(None)

    def _drain_stderr(self):
        for line in self.proc.stderr or []:
            if text := line.rstrip("\n"):
                self.stderr_tail = (self.stderr_tail + [text])[-80:]

    def run(self, command: str) -> str:
        if self.proc.poll() is not None:
            raise RuntimeError("slash worker exited")

        with self._lock:
            self._seq += 1
            rid = self._seq
            self.proc.stdin.write(json.dumps({"id": rid, "command": command}) + "\n")
            self.proc.stdin.flush()

            while True:
                try:
                    msg = self.stdout_queue.get(timeout=_SLASH_WORKER_TIMEOUT_S)
                except queue.Empty:
                    raise RuntimeError("slash worker timed out")
                if msg is None:
                    break
                if msg.get("id") != rid:
                    continue
                if not msg.get("ok"):
                    raise RuntimeError(msg.get("error", "slash worker failed"))
                return str(msg.get("output", "")).rstrip()

            raise RuntimeError(
                f"slash worker closed pipe{': ' + chr(10).join(self.stderr_tail[-8:]) if self.stderr_tail else ''}"
            )

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        proc = self.proc
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=1)
                except Exception:
                    proc.kill()
                    try:
                        proc.wait(timeout=1)  # reap the zombie SIGKILL leaves behind
                    except Exception:
                        pass
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=1)
            except Exception:
                pass
        finally:
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    stream.close()
                except Exception:
                    pass


def _load_busy_input_mode() -> str:
    """Policy for a TYPED prompt that arrives while a turn is running.

    ``display.busy_input_mode`` owns this decision — the config file, not the
    code, decides whether typing interrupts. Default stays ``interrupt`` for
    parity with cli.py.

    Note this governs the typed path only. A voice utterance is never allowed to
    interrupt regardless of this setting: speech cannot be unsaid, two
    utterances are two separate requests, and aborting the first turn to run the
    second loses an answer the user already asked for. See _submit_main, which
    passes queue_only unconditionally.
    """
    display = _load_cfg().get("display")
    if not isinstance(display, dict):
        display = {}
    raw = str(display.get("busy_input_mode", "") or "").strip().lower()
    return raw if raw in {"queue", "steer", "interrupt"} else "interrupt"


def _notify_session_boundary(event_type: str, session_id: str | None) -> None:
    """Fire session lifecycle hooks with CLI parity."""
    try:
        from hermes_cli.plugins import invoke_hook as _invoke_hook

        _invoke_hook(event_type, session_id=session_id, platform="tui")
    except Exception:
        pass


def _claim_active_session_slot(
    session_key: str,
    *,
    live_session_id: str,
    surface: str = "tui",
) -> tuple[Any, str | None]:
    try:
        from hermes_cli.active_sessions import try_acquire_active_session

        return try_acquire_active_session(
            session_id=session_key,
            surface=surface,
            config=_load_cfg(),
            metadata={"live_session_id": live_session_id},
        )
    except Exception as exc:
        logger.warning("Failed to claim active session slot: %s", exc)
        return None, None


def _release_active_session_slot(session: dict | None) -> None:
    if not session:
        return
    lease = session.pop("active_session_lease", None)
    if lease is None:
        return
    try:
        lease.release()
    except Exception:
        logger.debug("Failed to release active session slot", exc_info=True)


def _transfer_active_session_slot(
    sid: str,
    session: dict,
    *,
    new_session_id: str,
) -> bool:
    if not new_session_id:
        return False
    lease = session.get("active_session_lease")
    if lease is None:
        return True
    try:
        from hermes_cli.active_sessions import transfer_active_session

        if transfer_active_session(
            lease,
            session_id=new_session_id,
            metadata={"live_session_id": sid},
        ):
            return True
    except Exception:
        logger.debug("Failed to transfer active session slot", exc_info=True)

    # Fallback: the in-place transfer could not move the lease (entry pruned /
    # pid-check transiently failed). Reserve the new slot BEFORE releasing the
    # old one, so a concurrent gateway at the session cap cannot grab the freed
    # slot in a release-then-reacquire window and leave this session with no
    # lease at all (#49041 review). If the reserve fails, KEEP the old lease.
    new_lease, limit_message = _claim_active_session_slot(
        new_session_id,
        live_session_id=sid,
    )
    if new_lease is not None:
        old_lease = session.pop("active_session_lease", None)
        if old_lease is not None:
            try:
                old_lease.release()
            except Exception:
                logger.debug("Failed to release stale active session slot", exc_info=True)
        session["active_session_lease"] = new_lease
        return True
    # Reserve failed — retain the existing lease rather than dropping it.
    if limit_message:
        logger.warning(
            "Compression session lease re-anchor failed (kept old lease): "
            "sid=%s new_session_id=%s reason=%s",
            sid,
            new_session_id,
            limit_message,
        )
    return False


def _stop_mm_memory_backend(memory_backend: Any) -> None:
    """Stop one MemoryBackend with the gateway's bounded compatibility API."""
    if memory_backend is None:
        return
    try:
        memory_backend.stop(timeout=_MM_MEMORY_STOP_TIMEOUT_SEC)
    except TypeError:
        try:
            memory_backend.stop()
        except Exception:
            pass
    except Exception:
        pass


def _defer_memory_stop_until_watcher(
    watcher: Any, memory_backend: Any,
) -> bool:
    """Arrange dependency-safe Memory teardown after a stuck Watcher exits.

    A Watcher stopped callback is preferred because it fires only after query
    tasks, owned clients, and its loop have drained.  Older resident engines
    without that API fall back to a daemon join coordinator so session.close
    remains bounded without ever tearing Recall's backend out from under a
    still-live Watcher thread.
    """
    callback = getattr(watcher, "add_stopped_callback", None)
    if callable(callback):
        try:
            callback(lambda _watcher: _stop_mm_memory_backend(memory_backend))
            return True
        except Exception:
            logger.debug(
                "[watcher] failed to register deferred memory stop",
                exc_info=True,
            )

    watcher_thread = getattr(watcher, "_thread", None)
    if (isinstance(watcher_thread, threading.Thread)
            and watcher_thread is not threading.current_thread()
            and watcher_thread.is_alive()):
        def _join_then_stop() -> None:
            watcher_thread.join()
            _stop_mm_memory_backend(memory_backend)

        threading.Thread(
            target=_join_then_stop,
            name="mm-watcher-memory-stop",
            daemon=True,
        ).start()
        return True
    return False


def _finalize_session(session: dict | None, end_reason: str = "tui_close") -> None:
    """Best-effort finalize hook + memory commit for a session.

    Fires ``on_session_end`` plugin hook and attempts to persist any
    unflushed messages before closing the session.  This mirrors the
    CLI's exit-path behaviour and prevents data loss when the TUI is
    force-quit (double Ctrl‑C, terminal‑close, SIGHUP) while the agent
    is mid‑turn.
    """
    if not session:
        return
    # Serialize this boundary with MemoryBackend/Watcher/Monitor construction
    # and registration. All residents are published before their threads start,
    # so finalize can never observe a constructed-but-invisible owner.
    # Either finalize marks the session first (and startup refuses to build), or
    # startup publishes the freshly constructed backend into the session first
    # (and finalize captures/stops it).  There is no "constructed but invisible"
    # interleaving.
    with _MM_ACTIVE_MEMORY_BACKENDS_LOCK:
        if session.get("_finalized"):
            return
        session["_finalized"] = True
        mm_mem = session.get("_mm_memory_backend")
        mm_re = session.get("_mm_live_watcher_agent")
        mm_mon = session.get("_mm_monitor_engine")
    _release_active_session_slot(session)
    stop_event = session.get("_notif_stop")
    if stop_event is not None:
        stop_event.set()
    try:
        _abort_active_asr_turn(
            session,
            reason="session_finalized",
            reopenable=False,
        )
    except Exception:
        logger.debug("ASR abort during session finalize failed", exc_info=True)
    # Voice can submit into the Watcher/TTS loop, so stop it before either of
    # the resident engines it references. VoiceAgent.stop is idempotent and
    # performs its own bounded join.
    mm_voice = session.get("_mm_voice_agent")
    if mm_voice is not None:
        try:
            mm_voice.stop()
        except Exception:
            pass
    # Multimodal phase 3: stop the per-session MonitorEngine (async job container).
    if mm_mon is not None:
        try:
            mm_mon.stop()
        except Exception:
            pass
    # Watcher owns in-flight QueryWorkers that may still submit Recall through
    # MemoryBackend. Stop/cancel that dependent first; only then tear down its
    # backend owner.
    watcher_stopped = True
    if mm_re is not None:
        try:
            try:
                stop_result = mm_re.stop(timeout=_MM_WATCHER_STOP_TIMEOUT_SEC)
            except TypeError:
                stop_result = mm_re.stop()
            watcher_stopped = stop_result is not False
        except Exception:
            watcher_stopped = False
        # Compatibility backstop for older engines/test doubles: the new API's
        # stop() already joins, but an explicit short join also settles a loop
        # that completed immediately after its bounded stop returned False.
        watcher_thread = getattr(mm_re, "_thread", None)
        if (isinstance(watcher_thread, threading.Thread)
                and watcher_thread is not threading.current_thread()):
            watcher_thread.join(timeout=_MM_WATCHER_DEPENDENCY_JOIN_SEC)
            watcher_stopped = not watcher_thread.is_alive()
        stopped_marker = getattr(mm_re, "is_stopped", None)
        if stopped_marker is True:
            watcher_stopped = True

    # Stop the independent layered-memory backend only after its dependent
    # Watcher has fully drained. If Watcher teardown exceeds the bounded close
    # path, leave both strong registry guards in place and transfer Memory stop
    # to Watcher's race-safe stopped callback (or a legacy join coordinator).
    if mm_mem is not None:
        if watcher_stopped or mm_re is None:
            _stop_mm_memory_backend(mm_mem)
        elif _defer_memory_stop_until_watcher(mm_re, mm_mem):
            logger.warning(
                "[watcher] watcher stop timed out; MemoryBackend teardown "
                "deferred until Watcher fully stops"
            )
        else:
            # No thread/callback means there is no observable live dependency;
            # preserve compatibility rather than leaking Memory forever.
            _stop_mm_memory_backend(mm_mem)

    agent = session.get("agent")
    lock = session.get("history_lock")
    if lock is not None:
        with lock:
            history = list(session.get("history", []))
    else:
        history = list(session.get("history", []))

    # ── Recover terminal registries without rewriting past chat ─────────
    # Tool receipts may already have participated in a provider-cached prefix.
    # Their terminal truth lives in event/sidechannel files, so reconcile only
    # a detached snapshot and leave canonical history byte-stable.
    try:
        _reconcile_stale_mm_jobs(copy.deepcopy(history), agent)
    except Exception:
        pass

    # ── Persist unflushed messages to SQLite ──────────────────────────
    # Two sources, tried in order of freshness:
    #   1. agent._session_messages — set by the last _persist_session()
    #      call inside run_conversation().  This is the most recent
    #      snapshot the agent thread wrote, and may include partial
    #      turn data that hasn't reached session["history"] yet.
    #   2. session["history"] — updated after run_conversation()
    #      returns.  Stale when the agent is mid‑turn, but correct
    #      when the turn completed before finalize.
    # Best‑effort — the agent thread may still be mid‑turn, so only
    # previously completed messages are guaranteed.
    if agent is not None and hasattr(agent, "_persist_session"):
        snapshot = (
            getattr(agent, "_session_messages", None)
            or history
        )
        if snapshot:
            try:
                agent._persist_session(snapshot, conversation_history=history)
            except Exception:
                pass

    # ── Plugin hook: on_session_end ────────────────────────────────────
    # Signals every plugin that the session is closing, with
    # interrupted=True so crash‑recovery plugins can flush buffers,
    # persist state, or close connections before the gateway exits.
    # Mirrors cli.py's atexit handler that fires the same hook when
    # the user Ctrl‑C's mid‑turn.
    if agent is not None:
        try:
            from hermes_cli.plugins import invoke_hook

            invoke_hook(
                "on_session_end",
                session_id=getattr(agent, "session_id", None)
                or session.get("session_key", ""),
                completed=False,
                interrupted=True,
                model=getattr(agent, "model", "unknown"),
                platform=getattr(agent, "platform", None) or "tui",
            )
        except Exception:
            pass

    if agent is not None and history and hasattr(agent, "commit_memory_session"):
        try:
            agent.commit_memory_session(history)
        except Exception:
            pass

    session_key = session.get("session_key")
    session_id = getattr(agent, "session_id", None) or session_key
    _notify_session_boundary("on_session_finalize", session_id)

    # Mark session ended in DB so it doesn't linger as a ghost row in /resume.
    # Use session_id (from agent.session_id) not session_key — after compression,
    # session_key may be stale (the ended parent) while session_id is the live
    # continuation. Fix for #20001.
    if session_id:
        try:
            db = _get_db()
            if db is not None:
                db.end_session(session_id, end_reason)
        except Exception:
            pass

    # Close the slash-worker subprocess as part of finalize itself, not just
    # in the callers. Defense-in-depth: every session-end path goes through
    # _finalize_session (it's the single ``_finalized``-guarded chokepoint), so
    # folding worker cleanup in here means a future code path that calls
    # _finalize_session directly — without the surrounding _teardown_session /
    # _shutdown_sessions worker.close() — can't reintroduce the #38095 leak.
    # Idempotent: _SlashWorker.close() is poll()-guarded, so the explicit
    # close() still in those callers is harmless.
    try:
        worker = session.get("slash_worker")
        if worker:
            worker.close()
    except Exception:
        pass


def _teardown_session(session: dict | None, *, end_reason: str = "tui_close") -> None:
    """Fully tear down a session: finalize, unregister, close agent + worker.

    Shared by ``session.close`` and the orphaned-WS-session reaper. The
    slash-worker subprocess is closed inside ``_finalize_session`` (the single
    finalize chokepoint); this still unregisters the approval notifier and
    closes the in-process agent. Idempotent: the ``_finalized`` guard in
    ``_finalize_session`` and the ``poll()`` guard in ``_SlashWorker.close``
    make repeat calls harmless.
    """
    if not session:
        return
    _finalize_session(session, end_reason=end_reason)
    try:
        from tools.approval import unregister_gateway_notify

        if key := session.get("session_key"):
            unregister_gateway_notify(key)
    except Exception:
        pass
    try:
        agent = session.get("agent")
        if agent is not None and hasattr(agent, "close"):
            agent.close()
    except Exception:
        pass
    # NOTE: the slash-worker is closed inside _finalize_session (the single
    # _finalized-guarded chokepoint that main folded it into), exactly once.
    # We deliberately do NOT re-close it here — _teardown_session's job beyond
    # finalize is unregistering the notifier and closing the in-process agent.


def _attach_worker(sid: str, session: dict, worker) -> None:
    """Store worker on session iff sid still maps to it, else close it — a
    concurrent teardown already popped the session and would orphan the
    worker. Closes the create/close race at every slash-worker spawn site."""
    with _sessions_lock:
        if _sessions.get(sid) is session:
            session["slash_worker"] = worker
            return
    worker.close()


def _close_session_by_id(sid: str, *, end_reason: str = "tui_close") -> bool:
    """Single idempotent teardown for one session: pop it under the sessions
    lock, then finalize, unregister notify, close agent + slash worker via the
    shared ``_teardown_session`` path. Returns True iff it closed a live
    session. The ``_finalized`` / worker ``_closed`` guards make concurrent or
    repeat calls (e.g. session.close racing the WS-orphan reaper) harmless."""
    with _sessions_lock:
        session = _sessions.pop(sid, None)
    if session is None:
        return False
    _teardown_session(session, end_reason=end_reason)
    return True



def _ws_session_is_orphaned(session: dict | None) -> bool:
    """True if a WS session has no live transport and no in-flight turn.

    After ``handle_ws`` detaches a disconnected client it points the session at
    ``_detached_ws_transport``. A session left on that transport (and not
    mid-turn) is genuinely orphaned and safe to reap.
    """
    if not session or session.get("_finalized"):
        return False
    if session.get("running"):
        return False
    return session.get("transport") is _detached_ws_transport


def _schedule_ws_orphan_reap(sid: str) -> None:
    """After a grace window, reap session ``sid`` iff it's still orphaned.

    Called from the WS-disconnect path. The grace window lets a transient
    reconnect (or a ``session.resume`` that reattaches the transport) cancel
    the reap by re-binding a live transport. Disabled when the grace is 0.
    """
    if _WS_ORPHAN_REAP_GRACE_S <= 0:
        return

    def _reap() -> None:
        # Serialize the orphan re-check against session.resume (which re-binds a
        # live transport under _session_resume_lock and would make this session
        # non-orphaned). The actual pop + teardown then goes through the shared
        # _close_session_by_id funnel so the dict mutation happens under
        # _sessions_lock — consistent with every other _sessions mutator
        # (#39591: _reap previously popped under _session_resume_lock, giving no
        # mutual exclusion against _init_session / _close_session_by_id, which
        # guard with _sessions_lock). _sessions_lock is an RLock and the global
        # ordering is always resume_lock -> sessions_lock, so nesting is safe.
        with _session_resume_lock:
            if not _ws_session_is_orphaned(_sessions.get(sid)):
                return
            _close_session_by_id(sid, end_reason="ws_orphan_reap")

    timer = threading.Timer(_WS_ORPHAN_REAP_GRACE_S, _reap)
    timer.daemon = True
    timer.start()


def _close_sessions_for_transport(
    transport, *, end_reason: str = "ws_disconnect"
) -> tuple[int, int]:
    """On transport disconnect, reap the sessions that opted into
    close_on_disconnect (sidecar/dashboard) immediately via the unified
    ``_close_session_by_id`` path, and re-point the rest back to stdio so later
    emits don't hit a dead socket.

    Non-flagged detached sessions are handed to the grace-windowed WS-orphan
    reaper (``_schedule_ws_orphan_reap``): a quick reconnect / session.resume
    that re-binds a live transport cancels the reap, otherwise the orphan is
    torn down through the same idempotent ``_teardown_session`` path. This is
    the single WS-disconnect teardown entry point — there is no second
    independent reap loop in ``handle_ws``.

    Returns ``(reaped, detached)`` counts for disconnect-path observability."""
    with _sessions_lock:
        owned = [
            (sid, s) for sid, s in _sessions.items()
            if s.get("transport") is transport
        ]
    reaped = 0
    detached = 0
    for sid, session in owned:
        if session.get("close_on_disconnect"):
            # Serialize the ownership re-check with session.resume.  The
            # disconnect callback may run after a new renderer has already
            # rebound this live session; in that case it no longer owns the
            # session and must not reap it.
            with _session_resume_lock:
                with _sessions_lock:
                    still_owned = (
                        _sessions.get(sid) is session
                        and session.get("transport") is transport
                    )
                if still_owned and _close_session_by_id(
                        sid, end_reason=end_reason):
                    reaped += 1
        else:
            # Park the dead transport *before* any potentially blocking ASR
            # teardown.  Otherwise a cold asr_start can retain A as the owner
            # while this disconnect callback waits for its capture lock, and a
            # later unconditional sentinel assignment can overwrite renderer
            # B after session.resume has already rebound it.
            #
            # Keep the capture lock after releasing the resume lock: B may
            # rebind immediately while abortive close is in flight, but its
            # asr_start cannot transfer/reuse A's logical turn until A has been
            # retired.  Once abort completes we never touch session.transport
            # again, so B's binding wins deterministically.
            capture_lock = session.setdefault(
                "_mm_capture_lock", threading.RLock())
            parked = False
            with _session_resume_lock:
                with _sessions_lock:
                    still_live = _sessions.get(sid) is session
                if not still_live:
                    continue
                history_lock = session.get("history_lock")
                if history_lock is not None:
                    with history_lock:
                        if session.get("transport") is transport:
                            session["transport"] = _detached_ws_transport
                            parked = True
                elif session.get("transport") is transport:
                    session["transport"] = _detached_ws_transport
                    parked = True
                # Acquire only after the synchronous park.  A start already in
                # promotion will observe the sentinel in its final ownership
                # check and refuse to publish a late ASR connection.
                capture_lock.acquire()
            try:
                turn = session.get("_mm_asr_turn")
                # If B won the resume + start race before this disconnect
                # callback acquired the resume lock, its turn ownership is no
                # longer A's and must be preserved.  A mere resume (without a
                # new start) leaves owner_transport=A, so the stale upstream
                # microphone is still cancelled and made reopenable for B.
                if (isinstance(turn, dict)
                        and turn.get("owner_transport") is transport):
                    try:
                        _abort_active_asr_turn(
                            session,
                            reason="transport_disconnected",
                            reopenable=True,
                        )
                    except Exception:
                        logger.debug(
                            "ASR abort on transport detach failed",
                            exc_info=True,
                        )
            finally:
                capture_lock.release()
            if parked:
                detached += 1
                try:
                    _schedule_ws_orphan_reap(sid)
                except Exception:
                    pass
    return reaped, detached


def _shutdown_sessions() -> None:
    with _sessions_lock:
        sids = list(_sessions)
    for sid in sids:
        _close_session_by_id(sid, end_reason="tui_shutdown")


# Last-resort net for any disconnect path that slips past the WS finally. TTL is
# hours-scale because last_active freezes during a long turn and on passive
# viewing — running/pending/starting/live-transport are hard exemptions instead.
try:
    _SESSION_TTL_S = float(os.environ.get("ARGUS_TUI_SESSION_TTL_S") or 6 * 3600)
except (TypeError, ValueError):
    _SESSION_TTL_S = float(6 * 3600)
_SESSION_TTL_S = max(0.0, _SESSION_TTL_S)
_REAPER_SCAN_S = 300.0


def _transport_is_dead(transport) -> bool:
    # _detached_ws_transport is the post-WS-disconnect drop sentinel; a session
    # parked on it has no live client. _stdio_transport is the REAL transport
    # for a standalone `hermes --tui`, so it must NOT count as dead here (doing
    # so let the idle reaper evict healthy standalone TUI sessions).
    if transport is _detached_ws_transport:
        return True
    return getattr(transport, "_closed", None) is True


def _session_is_evictable(sid: str, session: dict, now: float) -> bool:
    if session.get("running") or _session_pending_kind(sid):
        return False
    ready = session.get("agent_ready")
    # Lazy watch sessions (subagent spectator windows) never start a build,
    # so their forever-unset agent_ready must not make them immortal.
    if ready is not None and not ready.is_set() and not session.get("lazy"):
        return False
    if not _transport_is_dead(session.get("transport")):
        return False
    last_active = float(session.get("last_active") or 0.0)
    created_at = float(session.get("created_at") or 0.0)
    return (now - last_active) > _SESSION_TTL_S and (now - created_at) > _SESSION_TTL_S


def _reap_idle_sessions() -> None:
    now = time.time()
    with _sessions_lock:
        victims = [sid for sid, s in _sessions.items() if _session_is_evictable(sid, s, now)]
    for sid in victims:
        _close_session_by_id(sid, end_reason="idle_timeout")
    _enforce_session_cap()


# Soft LRU cap on in-memory sessions. The 6h TTL reaper above only frees
# sessions that have been idle for hours; a heavy user who reconnects often
# accumulates detached sessions (the report's ``detached_sessions=5``) whose
# agents sit resident for the full TTL. The cap evicts the least-recently-active
# DETACHED sessions sooner so live agents don't pile up under memory pressure.
# Default-on but provably safe: it only touches sessions with no live client
# (reopening re-resumes them from the DB) and never a running / pending /
# mid-build / live-transport one. 0/null disables.
def _max_live_sessions() -> int:
    try:
        from hermes_cli.active_sessions import coerce_max_concurrent_sessions

        cfg = _load_cfg() or {}
        raw = cfg.get("max_live_sessions")
        if raw is None:
            gateway_cfg = cfg.get("gateway")
            if isinstance(gateway_cfg, dict):
                raw = gateway_cfg.get("max_live_sessions")
        coerced = coerce_max_concurrent_sessions(raw, key="max_live_sessions")
        return int(coerced) if coerced else 0
    except Exception:
        return 0


def _session_is_lru_evictable(sid: str, session: dict) -> bool:
    # Same hard exemptions as the TTL reaper (never evict a session mid-turn,
    # awaiting input, or still building), but WITHOUT the hours-scale age gate:
    # a detached session is eligible the moment it loses its client.
    if session.get("running") or _session_pending_kind(sid):
        return False
    ready = session.get("agent_ready")
    if ready is not None and not ready.is_set() and not session.get("lazy"):
        return False
    return _transport_is_dead(session.get("transport"))


def _enforce_session_cap() -> None:
    cap = _max_live_sessions()
    if cap <= 0:
        return
    with _sessions_lock:
        total = len(_sessions)
        if total <= cap:
            return
        evictable = [
            (sid, s) for sid, s in _sessions.items() if _session_is_lru_evictable(sid, s)
        ]
    # Oldest-touched first; only evict down to the cap (live/focused sessions on
    # a live transport are never eligible, so we may stop short of the cap).
    evictable.sort(key=lambda kv: float(kv[1].get("last_active") or 0.0))
    overflow = total - cap
    for sid, _s in evictable[:overflow]:
        _close_session_by_id(sid, end_reason="lru_evict")


def _schedule_session_cap_enforcement() -> None:
    """Run the LRU sweep off the response path (eviction can call agent.close)."""

    def _run():
        try:
            _enforce_session_cap()
        except Exception:
            logger.debug("session cap enforcement failed", exc_info=True)

    timer = threading.Timer(0.1, _run)
    timer.daemon = True
    timer.start()


def _start_idle_reaper() -> None:
    def _loop():
        while True:
            time.sleep(_REAPER_SCAN_S)
            try:
                _reap_idle_sessions()
            except Exception:
                pass

    threading.Thread(target=_loop, daemon=True).start()


atexit.register(_shutdown_sessions)
_start_idle_reaper()


# ── Plumbing ──────────────────────────────────────────────────────────


def _get_db():
    global _db, _db_error
    if _db is None:
        from hermes_state import SessionDB

        try:
            _db = SessionDB()
            _db_error = None
        except Exception as exc:
            _db_error = str(exc)
            logger.warning(
                "TUI session store unavailable — continuing without state.db features: %s",
                exc,
            )
            return None
    return _db


def _db_unavailable_error(rid, *, code: int):
    detail = _db_error or "state.db unavailable"
    return _err(rid, code, f"state.db unavailable: {detail}")


# ── per-session profile scoping (global remote mode) ───────────────────────────
# One dashboard normally serves its launch profile. But the desktop's app-global
# remote mode points every profile at this single backend, so resume/prompt must
# be able to act on ANOTHER local profile's state.db + home. The desktop passes
# ``profile`` on those calls; we open that profile's db and bind its HERMES_HOME
# (a ContextVar override) for the duration of the call so config/skills/model and
# message persistence all resolve to the right profile. Omitted/own profile → the
# launch profile (unchanged for single-profile and per-profile-remote setups).
def _profile_home(profile: str | None) -> Path | None:
    """Resolve a named profile's home on THIS host, or None for the launch profile."""
    name = (profile or "").strip()
    if not name:
        return None
    try:
        from hermes_cli import profiles as profiles_mod

        home = Path(profiles_mod.get_profile_dir(name))
    except Exception:
        return None
    # Already the launch profile? No override needed.
    if home.resolve() == Path(_hermes_home).resolve():
        return None
    return home if (home / "state.db").exists() or home.exists() else None


def _profile_scoped(handler):
    """Bind ``params['profile']``'s HERMES_HOME around a pet RPC handler.

    Pets are per-profile: ``display.pet.*`` lives in the profile's config.yaml and
    sprites install under its ``pets/`` dir (both resolve via ``get_hermes_home``).
    The desktop sends ``profile`` on pet calls so config + pets dir resolve to the
    focused profile even in app-global remote mode, where one backend serves every
    profile. No-op for the launch profile (own-profile backends already resolve it).
    """

    def wrapper(rid, params):
        home = _profile_home(params.get("profile") if isinstance(params, dict) else None)
        if home is None:
            return handler(rid, params)
        token = set_hermes_home_override(home)
        try:
            return handler(rid, params)
        finally:
            reset_hermes_home_override(token)

    return wrapper


# Placeholder ``terminal.cwd`` values that don't name a real directory — the
# gateway resolves these to the home dir at runtime, so they must NOT be treated
# as an explicit workspace (mirrors gateway/run.py's config bridge).
_CWD_PLACEHOLDERS = {".", "auto", "cwd"}


def _profile_configured_cwd(profile_home: Path | None) -> str | None:
    """Resolve a non-launch profile's ``terminal.cwd`` from its own config.yaml.

    The desktop's app-global remote mode serves every profile from one backend,
    so the process-global ``TERMINAL_CWD`` belongs to the *launch* profile. A new
    session bound to another profile must take its workspace from THAT profile's
    config, not the stale env var (issue #40334). Returns an absolute, existing
    directory, or None for placeholders / missing / invalid paths.
    """
    if profile_home is None:
        return None
    try:
        import yaml

        p = Path(profile_home) / "config.yaml"
        if not p.exists():
            return None
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        raw = str((data.get("terminal") or {}).get("cwd") or "").strip()
        if not raw or raw in _CWD_PLACEHOLDERS:
            return None
        resolved = os.path.abspath(os.path.expanduser(raw))
        return resolved if os.path.isdir(resolved) else None
    except Exception:
        return None


def write_json(obj: dict) -> bool:
    """Emit one JSON frame. Routes via the most-specific transport available.

    Precedence:

    1. Event frames with a session id → the transport stored on that session,
       so async events land with the client that owns the session even if
       the emitting thread has no contextvar binding.
    2. Otherwise the transport bound on the current context (set by
       :func:`dispatch` for the lifetime of a request).
    3. Otherwise the module-level stdio transport, matching the historical
       behaviour and keeping tests that monkey-patch ``_real_stdout`` green.
    """
    if obj.get("method") == "event":
        sid = ((obj.get("params") or {}).get("session_id")) or ""
        if sid and (t := (_sessions.get(sid) or {}).get("transport")) is not None:
            return t.write(obj)

    return (current_transport() or _stdio_transport).write(obj)


_MM_TRAJECTORY_EVENTS = {
    "multimodal.anchor",
    "multimodal.asr_final",
    "multimodal.asr_partial",
    "multimodal.bg",
    "multimodal.ctx",
    "multimodal.diag",
    "multimodal.monitors",
    "multimodal.toast",
    "multimodal.watchers",
}

# Keep the inspector useful without letting recalled thumbnails dominate the
# resident gateway session.  Metadata remains available for the usual 2,000
# entries; full image payloads are retained only for a small recent window and
# within a process-memory budget.  The newest image-bearing entry is always
# retained as one atomic evidence group, even when that group alone is larger
# than the normal budget.
_MM_TRAJECTORY_MAX_ENTRIES = 2000
_MM_TRAJECTORY_IMAGE_GROUPS_MAX = 8
_MM_TRAJECTORY_IMAGE_B64_BUDGET_CHARS = 16_000_000
_MM_TRAJECTORY_IMAGE_KEYS = frozenset({"jpeg_b64", "thumb_b64"})
_MM_TRAJECTORY_FRAMES_PER_ENTRY_MAX = 12

# Structured trajectory payloads are rendered by the desktop Debug inspector.
# Unlike prose redaction, a key that explicitly declares a credential is an
# unambiguous safety boundary: mask its value even when it is too short or too
# oddly-shaped for a token regex.  Keep this matcher deliberately narrower than
# a raw substring check so useful counters/identity fields such as
# ``prompt_tokens``, ``token_count``, ``session_id``, and ``client_request_id``
# remain available for trajectory grouping.
_MM_TRAJECTORY_SENSITIVE_KEYS = frozenset({
    "access_key",
    "access_token",
    "api_key",
    "apikey",
    "auth",
    "auth_header",
    "auth_token",
    "authorization",
    "bearer",
    "bearer_token",
    "client_secret",
    "cookie",
    "cookie_header",
    "cookies",
    "credential",
    "credentials",
    "csrf_token",
    "id_token",
    "jwt",
    "key_material",
    "oauth_token",
    "passphrase",
    "passwd",
    "password",
    "private_key",
    "proxy_authorization",
    "raw_secret",
    "refresh_token",
    "secret",
    "secret_access_key",
    "secret_key",
    "secret_value",
    "session_cookie",
    "set_cookie",
    "signature",
    "token",
    "x_amz_signature",
})
_MM_TRAJECTORY_SENSITIVE_KEY_SUFFIXES = (
    "_api_key",
    "_apikey",
    "_auth",
    "_authorization",
    "_cookie",
    "_cookies",
    "_credential",
    "_credentials",
    "_passphrase",
    "_passwd",
    "_password",
    "_private_key",
    "_secret",
    "_token",
)


def _trajectory_key_name(key: str) -> str:
    """Normalize snake/kebab/camel credential field names for matching."""
    camel_split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(key))
    return re.sub(r"[^A-Za-z0-9]+", "_", camel_split).strip("_").lower()


def _trajectory_sensitive_key(key: str) -> bool:
    normalized = _trajectory_key_name(key)
    return (
        normalized in _MM_TRAJECTORY_SENSITIVE_KEYS
        or normalized.endswith(_MM_TRAJECTORY_SENSITIVE_KEY_SUFFIXES)
    )


def _trajectory_redact_text(value: str) -> str:
    """Force-redact one Debug-inspector string, failing closed on errors."""
    try:
        from agent.redact import redact_sensitive_text, redact_url_userinfo

        redacted = redact_sensitive_text(value, force=True)
        return redact_url_userinfo(redacted)
    except Exception:
        # Trajectory capture is a debug aid, never a reason to leak the raw
        # value when the central redactor is unavailable or fails unexpectedly.
        return "<redacted: unavailable>"


def _trajectory_image_chars(value: Any, *, key: str = "") -> int:
    """Count retained base64 image characters in one safe trajectory value."""
    k = key.lower()
    if isinstance(value, str):
        if k in _MM_TRAJECTORY_IMAGE_KEYS and not value.startswith("<omitted "):
            return len(value)
        return 0
    if isinstance(value, dict):
        return sum(
            _trajectory_image_chars(vv, key=str(kk))
            for kk, vv in value.items()
        )
    if isinstance(value, (list, tuple)):
        return sum(_trajectory_image_chars(item, key=key) for item in value)
    return 0


def _omit_trajectory_images(value: Any) -> None:
    """Replace retained image strings in ``value`` while preserving metadata."""
    if isinstance(value, dict):
        for raw_key, child in value.items():
            child_key = str(raw_key)
            if (
                child_key.lower() in _MM_TRAJECTORY_IMAGE_KEYS
                and isinstance(child, str)
                and not child.startswith("<omitted ")
            ):
                value[raw_key] = (
                    f"<omitted {child_key}: trajectory image retention budget; "
                    f"{len(child)} chars>"
                )
            else:
                _omit_trajectory_images(child)
    elif isinstance(value, list):
        for child in value:
            _omit_trajectory_images(child)


def _trim_mm_trajectory_entries(rows: list[dict]) -> None:
    """Drop old trajectory rows without scanning their nested payloads."""
    entry_cap = max(1, int(_MM_TRAJECTORY_MAX_ENTRIES))
    if len(rows) > entry_cap:
        del rows[:-entry_cap]


def _bound_mm_trajectory_images(rows: list[dict]) -> None:
    """Bound the aggregate image footprint of already entry-bounded rows.

    Image-bearing entries are evidence groups.  Keep a contiguous recent suffix
    that satisfies both the group count and character budget, except that the
    latest group is never stripped.  Older entries remain in ``rows`` with ids,
    frame ids, timestamps, and all other debug metadata intact.
    """
    image_groups = [
        (idx, _trajectory_image_chars(entry))
        for idx, entry in enumerate(rows)
    ]
    image_groups = [(idx, size) for idx, size in image_groups if size > 0]
    if not image_groups:
        return

    group_cap = max(1, int(_MM_TRAJECTORY_IMAGE_GROUPS_MAX))
    budget = max(0, int(_MM_TRAJECTORY_IMAGE_B64_BUDGET_CHARS))
    keep = image_groups[-group_cap:]
    total = sum(size for _, size in keep)
    # Preserve the latest complete evidence group.  Remove older groups from
    # oldest to newest until the remaining suffix fits the normal budget.
    while len(keep) > 1 and total > budget:
        _, removed_size = keep.pop(0)
        total -= removed_size
    keep_indices = {idx for idx, _ in keep}

    for idx, _ in image_groups:
        if idx not in keep_indices:
            _omit_trajectory_images(rows[idx])


def _bound_mm_trajectory(rows: list[dict]) -> None:
    """Apply both entry and aggregate image bounds (used by list RPC audit)."""
    _trim_mm_trajectory_entries(rows)
    _bound_mm_trajectory_images(rows)


def _trajectory_safe(value: Any, *, depth: int = 0, key: str = "") -> Any:
    """Redact and bound debug payloads without hiding useful worker evidence.

    Recall JPEG thumbnails are intentionally retained so a user can inspect the
    exact recalled frames. Raw PCM/audio chunks and unrelated binary blobs are
    omitted; they would turn a stress-test trace into hundreds of megabytes.
    Every other string passes through the central force-on secret redactor, and
    explicitly credential-shaped structure keys are masked unconditionally.
    """
    if depth > 6:
        return "<max-depth>"
    k = key.lower()
    if k in {"pcm_b64", "data_b64", "audio_b64", "content_base64"}:
        return f"<omitted {k}>"
    if _trajectory_sensitive_key(key):
        return "***"
    if isinstance(value, str):
        # Recall events already carry 480px JPEG thumbnails. Give those a
        # larger bounded allowance and never return a truncated (therefore
        # invalid) base64 image.
        if k in _MM_TRAJECTORY_IMAGE_KEYS:
            cap = 500_000
            if len(value) <= cap:
                return value
            return f"<omitted oversized {k}: {len(value)} chars>"
        value = _trajectory_redact_text(value)
        cap = 16_000
        if len(value) <= cap:
            return value
        return value[:cap] + f"… <truncated {len(value) - cap} chars>"
    if isinstance(value, dict):
        return {
            str(kk): _trajectory_safe(vv, depth=depth + 1, key=str(kk))
            for kk, vv in list(value.items())[:120]
        }
    if isinstance(value, (list, tuple)):
        item_cap = (
            _MM_TRAJECTORY_FRAMES_PER_ENTRY_MAX
            if k == "frames" else 120
        )
        out = [_trajectory_safe(v, depth=depth + 1, key=key)
               for v in list(value)[:item_cap]]
        if len(value) > item_cap:
            out.append(f"<truncated {len(value) - item_cap} items>")
        return out
    if value is None or isinstance(value, (int, float, bool)):
        return value
    text = _trajectory_redact_text(str(value))
    cap = 16_000
    if len(text) <= cap:
        return text
    return text[:cap] + f"… <truncated {len(text) - cap} chars>"


def _trajectory_worker(event: str, payload: dict) -> str:
    if payload.get("worker"):
        return str(payload["worker"])
    if event == "multimodal.bg":
        typ = str(payload.get("type") or payload.get("phase") or "").lower()
        channel = str(payload.get("channel") or "").lower()
        if "recall" in typ or channel == "recall":
            return "RecallWorker"
        if "search" in typ or channel == "search":
            return "SearchWorker"
        return "WatcherRouter"
    if event == "multimodal.ctx":
        return "MemoryWriter"
    if event == "multimodal.anchor":
        return "MainVision"
    if event in {"multimodal.asr_final", "multimodal.asr_partial"}:
        return "VoiceAgent"
    if event == "multimodal.diag":
        return str(payload.get("kind") or payload.get("component") or "LLM")
    if event in {"tool.start", "tool.complete"}:
        name = str(payload.get("name") or payload.get("tool_name") or "tool")
        if name == "query_multimodal":
            return "QueryWorker"
        if name == "set_live_watcher":
            return "WatcherRouter"
        if name == "set_monitor":
            return "MonitorWorker"
        return f"MainTool:{name}"
    if event in {"multimodal.monitors", "multimodal.toast"}:
        return "MonitorWorker"
    if event == "multimodal.watchers":
        return "WatcherRouter"
    return "Multimodal"


def _record_mm_trajectory(event: str, sid: str, payload: dict | None) -> dict:
    raw = dict(payload or {})
    safe = _trajectory_safe(raw)
    if not isinstance(safe, dict):
        safe = {"detail": safe}
    with _sessions_lock:
        session = _sessions.get(sid)
        if session is None:
            seq = 1
        else:
            seq = int(session.get("_mm_trajectory_seq", 0)) + 1
            session["_mm_trajectory_seq"] = seq
    entry = {
        "id": f"tr_{seq}_{uuid.uuid4().hex[:6]}",
        "seq": seq,
        "ts": time.time(),
        "event": event,
        "worker": _trajectory_worker(event, safe),
        "phase": str(safe.get("phase") or safe.get("type") or event),
        "payload": safe,
    }
    if session is not None:
        with _sessions_lock:
            live = _sessions.get(sid)
            if live is not None:
                rows = live.setdefault("_mm_trajectory", [])
                # The emitted entry stays independent and complete.  Only this
                # storage copy is eligible for later image-budget omission.
                stored_entry = copy.deepcopy(entry)
                rows.append(stored_entry)
                _trim_mm_trajectory_entries(rows)
                # Ordinary ASR/progress/diagnostic entries are high-frequency.
                # Avoid an O(2,000) nested scan unless this append can actually
                # change image retention. The list RPC performs a full audit as
                # a defensive backstop for legacy/injected session state.
                if _trajectory_image_chars(stored_entry) > 0:
                    _bound_mm_trajectory_images(rows)
    return entry


def _vtrace(stage: str, **fields) -> None:
    """对话模式语音链路专用日志 → ``~/.argus/logs/voice_chain.log``。

    实现见 ``agent/multimodal/voice_trace.py``；默认关闭，``ARGUS_TRACE=1`` 或
    config ``logging.voice_trace: true`` 打开。lazy import：gateway 启动路径不为
    一条默认关闭的日志付 import 代价。永不抛异常。
    """
    try:
        from agent.multimodal.voice_trace import vtrace
        vtrace(stage, **fields)
    except Exception:
        pass


def _emit(event: str, sid: str, payload: dict | None = None):
    trajectory = None
    session_source = str((_sessions.get(sid) or {}).get("source") or "")
    mm_tool_event = (
        event in {"tool.start", "tool.complete"}
        and session_source in {"multimodal", "tool"}
    )
    if (event == "multimodal.trajectory"
            or event in _MM_TRAJECTORY_EVENTS
            or mm_tool_event):
        trajectory = _record_mm_trajectory(event, sid, payload)
    params = {"type": event, "session_id": sid}
    if event == "multimodal.trajectory" and trajectory is not None:
        params["payload"] = trajectory
    elif payload is not None:
        params["payload"] = payload
    write_json({"jsonrpc": "2.0", "method": "event", "params": params})
    # Existing events keep their public contract. A second normalized event
    # feeds the inspector without making the UI subscribe to every worker's
    # bespoke payload shape. Direct trajectory events are already normalized.
    if trajectory is not None and event != "multimodal.trajectory":
        write_json({
            "jsonrpc": "2.0",
            "method": "event",
            "params": {
                "type": "multimodal.trajectory",
                "session_id": sid,
                "payload": trajectory,
            },
        })


def _normalize_mm_monitor_trigger_mode(value: Any) -> str:
    """Normalize a restored Monitor trigger contract.

    Event files and receipts created before trigger modes existed represented
    persistent monitors, so missing/invalid values retain ``continuous``.
    """
    return "once" if str(value or "").strip().lower() == "once" else "continuous"


def _ensure_client_request_id(value: Any = None) -> str:
    """Return one bounded, stable correlation id for a foreground turn.

    Browser callers normally allocate the id before submitting so they can
    pre-create the matching answer slot.  Voice, ASR, and backend-originated
    turns do not always have a caller-provided id, so allocate one here rather
    than letting their message events fall back to the shared ``__main__``
    stream key.
    """
    supplied = str(value or "").strip()[:128]
    return supplied or f"turn_{uuid.uuid4().hex}"


def _recompute_mm_monitor_active(agent) -> bool:
    """Refresh the legacy active gate from the live monitor registry."""
    if agent is None:
        return False
    monitors = getattr(agent, "mm_monitors", None) or {}
    active = any(
        bool(m.get("enabled", True))
        and str(m.get("status") or "running").strip().lower()
        not in {"done", "complete", "deleted", "interrupted"}
        for m in monitors.values()
        if isinstance(m, dict)
    )
    try:
        agent.mm_monitor_active = active
    except Exception:
        pass
    return active


def _push_mm_registries(sid: str, agent) -> None:
    """Push the monitor + research registries to the frontend (best-effort).

    Reuses the tool-side emitters so the payload shape stays in one place. Used
    after a resume re-registers interrupted jobs, so the panel lists them (as
    off) the moment the session is ready — without waiting for a tool call."""
    if agent is None:
        return
    try:
        from tools.monitor_tool import _push_monitors_event
        _push_monitors_event(sid, agent)
    except Exception:
        pass
    try:
        from tools.live_watcher_tool import _push_watchers_event
        _push_watchers_event(sid, agent)
    except Exception:
        pass


def _emit_approval_request(sid: str, data: dict | None) -> None:
    """Emit an ``approval.request`` event to the TUI client with the command
    redacted. The approval payload is built from the RAW command string, so a
    credential-shaped value Tirith flagged would otherwise be echoed verbatim
    to the TUI client (#48456 — third egress transport alongside the chat
    platforms and the SSE/API stream fixed in #50767). Reuse the shared gateway
    seam so all approval transports redact consistently."""
    payload = dict(data or {})
    if "command" in payload:
        from gateway.run import _redact_approval_command

        payload["command"] = _redact_approval_command(payload.get("command"))
    _emit("approval.request", sid, payload)


def _status_update(sid: str, kind: str, text: str | None = None):
    body = (text if text is not None else kind).strip()
    if not body:
        return
    out_kind = kind if text is not None else "status"
    # Auto-compaction reaches us as a generic "lifecycle" status. Re-tag it so
    # drivers (desktop app) can show an explicit "Summarizing…" indicator —
    # otherwise a mid-turn compaction looks like the transcript reset itself.
    if out_kind == "lifecycle":
        from agent.conversation_compression import COMPACTION_STATUS_MARKER

        if COMPACTION_STATUS_MARKER in body:
            out_kind = "compacting"
    _emit("status.update", sid, {"kind": out_kind, "text": body})


def _estimate_image_tokens(width: int, height: int) -> int:
    """Very rough UI estimate for image prompt cost.

    Uses 512px tiles at ~85 tokens/tile as a lightweight cross-provider hint.
    This is intentionally approximate and only used for attachment display.
    """
    if width <= 0 or height <= 0:
        return 0
    return max(1, (width + 511) // 512) * max(1, (height + 511) // 512) * 85


def _image_meta(path: Path) -> dict:
    meta = {"name": path.name}
    try:
        from PIL import Image

        with Image.open(path) as img:
            width, height = img.size
        meta["width"] = int(width)
        meta["height"] = int(height)
        meta["token_estimate"] = _estimate_image_tokens(int(width), int(height))
    except Exception:
        pass
    return meta


def _ok(rid, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def _optional_finite_float(value: Any) -> Optional[float]:
    """Best-effort finite float parsing for browser timing diagnostics."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        return None
    return parsed


def _audio_container_signature(data: bytes, media_type: str) -> tuple[str, bool]:
    """Return ``(container, has_standalone_header)`` for an audio upload.

    A WebM MediaRecorder timeslice after the first blob commonly starts with a
    Cluster rather than an EBML header.  It may be valid after concatenation,
    but it is not a safe independent file for a single-shot ASR request.
    """
    if data.startswith(b"\x1a\x45\xdf\xa3"):
        return "webm_ebml", True
    if data.startswith(b"\x1f\x43\xb6\x75"):
        return "webm_cluster", False
    if data.startswith(b"OggS"):
        return "ogg", True
    if data.startswith(b"RIFF") and data[8:12] == b"WAVE":
        return "wav", True
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return "mp4", True
    short_mime = (media_type or "").split(";", 1)[0].strip().lower()
    return short_mime or "unknown", False


def _resolve_env_audio_window(
    *,
    server_end_ts: Optional[float],
    client_start_ts: Optional[float],
    client_end_ts: Optional[float],
    client_duration_sec: Optional[float],
    default_duration_sec: float,
) -> tuple[Optional[float], Optional[float], float]:
    """Resolve one audio window on the frame buffer's authoritative clock.

    ``server_end_ts`` is sampled when a completed chunk arrives.  Downstream
    stores ``rel_ts`` as the *start* of the chunk, so subtract the measured
    duration here rather than saving the receipt/end time as the start.
    """
    duration = client_duration_sec
    if duration is None and client_start_ts is not None and client_end_ts is not None:
        duration = client_end_ts - client_start_ts
    if duration is None or not 0.05 <= duration <= 60.0:
        duration = default_duration_sec
    duration = max(0.05, min(float(duration), 60.0))
    end_ts = server_end_ts if server_end_ts is not None else client_end_ts
    start_ts = (
        max(0.0, float(end_ts) - duration)
        if end_ts is not None else client_start_ts
    )
    return start_ts, end_ts, duration


def _err(rid, code: int, msg: str) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": msg}}


def method(name: str):
    def dec(fn):
        _methods[name] = fn
        return fn

    return dec


def _normalize_request(req: Any) -> tuple[Any, str, dict] | dict:
    """Validate a JSON-RPC request enough for safe local dispatch."""
    if not isinstance(req, dict):
        return _err(None, -32600, "invalid request: expected an object")

    rid = req.get("id")
    method = req.get("method")
    if not isinstance(method, str) or not method:
        return _err(rid, -32600, "invalid request: method must be a non-empty string")

    params = req.get("params", {})
    if params is None:
        params = {}
    elif not isinstance(params, dict):
        return _err(rid, -32602, "invalid params: expected an object")

    return rid, method, params


def handle_request(req: dict) -> dict | None:
    normalized = _normalize_request(req)
    if isinstance(normalized, dict):
        return normalized

    rid, method, params = normalized
    fn = _methods.get(method)
    if not fn:
        return _err(rid, -32601, f"unknown method: {method}")
    return fn(rid, params)


def dispatch(req: dict, transport: Optional[Transport] = None) -> dict | None:
    """Route inbound RPCs — long handlers to the pool, everything else inline.

    Returns a response dict when handled inline. Returns None when the
    handler was scheduled on the pool; the worker writes its own response
    via the bound transport when done.

    *transport* (optional): pins every write produced by this request —
    including any events emitted by the handler — to the given transport.
    Omitting it falls back to the module-level stdio transport, preserving
    the original behaviour for ``tui_gateway.entry``.
    """
    t = transport or _stdio_transport
    token = bind_transport(t)
    try:
        normalized = _normalize_request(req)
        if isinstance(normalized, dict):
            return normalized

        _rid, method, _params = normalized
        # JSON-RPC 2.0 notifications (no `id`) MUST NOT receive a response.
        # High-frequency channels like multimodal.frame notify at ~2 fps to
        # avoid a per-frame ACK roundtrip — writing back a null-id response
        # would waste half the bandwidth savings and clutter the client's
        # inbound stream with messages it just drops.
        is_notify = "id" not in req or req.get("id") is None
        if method not in _LONG_HANDLERS:
            resp = handle_request(req)
            return None if is_notify else resp

        # Snapshot the context so the pool worker sees the bound transport.
        ctx = contextvars.copy_context()

        def run():
            try:
                resp = handle_request(req)
            except Exception as exc:
                resp = _err(req.get("id"), -32000, f"handler error: {exc}")
            if resp is not None and not is_notify:
                t.write(resp)

        _pool.submit(lambda: ctx.run(run))

        return None
    finally:
        reset_transport(token)


def _wait_agent(session: dict, rid: str, timeout: float = 30.0) -> dict | None:
    ready = session.get("agent_ready")
    if ready is not None and not ready.wait(timeout=timeout):
        return _err(rid, 5032, "agent initialization timed out")
    err = session.get("agent_error")
    return _err(rid, 5032, err) if err else None


_MM_CAPTURE_RETIRED_OWNER_LIMIT = 64


def _capture_owner_is_retired(
    session: dict,
    client_id: str,
    generation: int,
    attempt_id: str,
) -> bool:
    """Whether an exact capture owner was explicitly stopped.

    Modern renderers may reuse a generation while replacing a reconnect
    attempt, so retirement is keyed by (client, generation, attempt).  Legacy
    owners have no attempt id and therefore retain their historical
    same-generation wildcard semantics.
    """
    candidates = list(session.get("_mm_capture_retired_owners") or [])
    legacy = session.get("_mm_capture_stop_tombstone")
    if isinstance(legacy, (tuple, list)):
        candidates.append(legacy)
    for raw in candidates:
        if not isinstance(raw, (tuple, list)) or len(raw) < 2:
            continue
        retired_client = str(raw[0] or "").strip()
        try:
            retired_generation = int(raw[1])
        except (TypeError, ValueError):
            continue
        retired_attempt = str(raw[2] or "").strip() if len(raw) >= 3 else ""
        if retired_client != client_id or retired_generation != int(generation):
            continue
        if not attempt_id or not retired_attempt or retired_attempt == attempt_id:
            return True
    return False


def _retire_capture_owner(
    session: dict,
    client_id: str,
    generation: int,
    attempt_id: str,
) -> None:
    """Retain a bounded monotonic set of stopped capture attempts."""
    owner = (str(client_id or "").strip(), int(generation), str(attempt_id or "").strip())
    rows = [
        tuple(raw)
        for raw in list(session.get("_mm_capture_retired_owners") or [])
        if isinstance(raw, (tuple, list)) and tuple(raw) != owner
    ]
    rows.append(owner)
    session["_mm_capture_retired_owners"] = rows[-_MM_CAPTURE_RETIRED_OWNER_LIMIT:]
    # Keep the last owner for compatibility with live session state created by
    # an older gateway version; stale checks consult both representations.
    session["_mm_capture_stop_tombstone"] = owner


def _is_multimodal_runtime_session(session: dict | None) -> bool:
    """Return whether this live session owns the dashboard MM runtime.

    ``tool`` is the durable source for delegated/sub-agent conversations.  It
    used to be (incorrectly) treated as the multimodal page marker in a few
    call sites, which both started empty memory databases for workers and left
    real ``source=multimodal`` sessions without frames, memory, or env ASR.
    Keep the provenance check centralized so those paths cannot drift again.
    """
    return str((session or {}).get("source") or "").strip().lower() == "multimodal"


def _multimodal_runtime_ready(session: dict | None) -> bool:
    """Return whether capture, QueryWorker, and Monitor are actually usable.

    Provenance alone is not readiness: a failed partial promotion deliberately
    leaves ``source=multimodal``. Source-start is an ACK barrier, so it must not
    report success until the concrete resident services needed by subsequent
    frames/tools are healthy.
    """
    current = session or {}
    agent = current.get("agent")
    frame_buffer = getattr(agent, "frame_buffer", None) if agent is not None else None
    memory = current.get("_mm_memory_backend")
    watcher = current.get("_mm_live_watcher_agent")
    monitor = current.get("_mm_monitor_engine")
    if frame_buffer is None or memory is None or watcher is None or monitor is None:
        return False
    try:
        memory_ready = getattr(memory, "is_ready", False)
        if callable(memory_ready):
            memory_ready = memory_ready()
        memory_healthy = getattr(memory, "healthy", True)
        if callable(memory_healthy):
            memory_healthy = memory_healthy()
        watcher_healthy = getattr(watcher, "healthy", True)
        if callable(watcher_healthy):
            watcher_healthy = watcher_healthy()
        return bool(
            memory_ready is True
            and memory_healthy is True
            and _mm_watcher_is_ready(watcher)
            and watcher_healthy is True
            and monitor.is_healthy() is True
        )
    except Exception:
        return False


def _promote_session_to_multimodal(sid: str, session: dict) -> bool:
    """Attach the MM runtime to an ALREADY-LIVE session that was built without it.

    ``session.resume`` has a fast path that reuses a still-live session as-is.
    That path deliberately ignores ``params["source"]``, so a conversation first
    opened by the TUI/desktop (``source=tui``) or by a delegated worker
    (``source=tool``) stayed non-multimodal forever — even after the dashboard's
    /multimodal page reopened it with ``source=multimodal``. The agent then had
    no frame_buffer and no MonitorEngine, so set_monitor failed with
    "Monitor backend 未就绪" and get_current_frame/recall had nothing to read.

    Promotion is the same sequence _start_agent_build runs for a natively-built
    MM session (frame_buffer → memory backend → watcher → monitor engine), minus
    the parts that only make sense at construction time (context-file skipping is
    already baked into the built prompt; the deep-thinking flag is re-read per
    turn anyway). Every helper it calls is idempotent and registry-guarded, so a
    concurrent/duplicate promotion cannot produce two runtimes for one session.

    Returns True when the session is multimodal on exit (already was, or was
    successfully promoted). Best-effort: a failure leaves the session usable as a
    plain chat rather than breaking the resume.
    """
    if _is_multimodal_runtime_session(session) and _multimodal_runtime_ready(session):
        return True
    if session.get("_finalized"):
        return False
    prev_source = str(session.get("source") or "")
    agent = session.get("agent")
    if agent is None:
        # No agent yet. Two very different cases:
        #   * build not started (lazy/deferred) — flipping the provenance is
        #     enough, _start_agent_build will read it and build the MM runtime.
        #   * build IN FLIGHT — it already captured is_mm_session=False, so the
        #     flip alone would be silently ignored. Wait for the build to finish
        #     (bounded), then attach the runtime explicitly below.
        session["source"] = "multimodal"
        if not session.get("agent_build_started"):
            logger.info(
                "[mm] session marked multimodal before agent build (sid=%s)", sid)
            return True
        ready = session.get("agent_ready")
        if ready is not None:
            ready.wait(timeout=30.0)
        agent = session.get("agent")
        if agent is None:
            logger.warning(
                "[mm] promote: agent build did not finish in time (sid=%s)", sid)
            return False
        if session.get("_finalized"):
            return False

    # Flip provenance FIRST: every helper below (and _emit's trajectory routing,
    # and the per-turn gates in _run_prompt_submit) keys off it.
    session["source"] = "multimodal"
    runtime_tokens: list = []
    runtime_home_token = None
    try:
        # Promotion can happen on a shared global-remote backend long after the
        # ordinary agent was built. Bind every config/file/registry helper to
        # this session's own profile exactly like the normal build/turn paths.
        runtime_tokens = _set_session_context(
            str(session.get("session_key") or sid),
            str(session.get("cwd") or ""),
        )
        if profile_home := session.get("profile_home"):
            runtime_home_token = set_hermes_home_override(profile_home)
        from agent.multimodal._memory import FrameBuffer as _MMFrameBuffer
        from agent.multimodal.hermes_glue import build_config as _mm_build_config

        if getattr(agent, "frame_buffer", None) is None:
            agent.frame_buffer = _MMFrameBuffer(_mm_build_config())
            logger.info("[mm] frame_buffer created on promote (sid=%s)", sid)
        if getattr(agent, "mm_monitors", None) is None:
            agent.mm_monitors = {}
        try:
            agent._multimodal_session = True
        except Exception:
            pass

        _frame_buffer = getattr(agent, "frame_buffer", None)
        _ready_backend = _maybe_start_memory_backend(
            sid, str(session.get("session_key") or sid), _frame_buffer,
            session=session,
        )
        session.setdefault("_mm_live_watcher_agent", None)
        _maybe_start_live_watcher_agent(
            sid, _frame_buffer, _ready_backend, session)
        _maybe_start_monitor_engine(sid, session, _frame_buffer)

        # Re-register monitors/watchers that this conversation left behind, then
        # push so the dashboard's panels list them immediately (same tail the
        # native build path runs).
        try:
            _hist = session.get("history")
            if isinstance(_hist, list):
                # Promotion happens inside an already-live conversation. The
                # stale-job reconciler repairs legacy tool receipts in place,
                # which is correct during restore but would mutate the cached
                # provider prefix here. Recover registries from a detached
                # snapshot so every prior message remains byte-stable.
                _reconcile_stale_mm_jobs(
                    copy.deepcopy(_hist), agent,
                    session_id=str(session.get("session_key") or ""))
            _push_mm_registries(sid, agent)
        except Exception:
            logger.debug("[mm] registry push failed on promote (sid=%s)", sid,
                         exc_info=True)

        ready = _multimodal_runtime_ready(session)
        if ready:
            logger.info("[mm] session promoted to multimodal (sid=%s from=%s)",
                        sid, prev_source or "?")
        else:
            logger.warning(
                "[mm] promotion incomplete: capture/query/monitor runtime not ready "
                "(sid=%s)", sid)
        return ready
    except Exception as exc:
        # Leave source flipped: a partially-attached runtime is still the MM
        # session the caller asked for, and the per-monitor guards fail loudly
        # with their own cause rather than silently doing nothing.
        logger.warning("[mm] promote to multimodal failed (sid=%s): %s",
                       sid, exc, exc_info=True)
        return False
    finally:
        if runtime_home_token is not None:
            reset_hermes_home_override(runtime_home_token)
        _clear_session_context(runtime_tokens)


def _start_agent_build(sid: str, session: dict) -> None:
    """Start building the real AIAgent for a TUI session, once.

    Classic `hermes` shows the prompt before constructing AIAgent; the TUI used
    to eagerly build it during session.create, making startup feel blocked on
    tool discovery/model metadata even though the composer was visible.  Keep
    the shell responsive by deferring this work until the first prompt (or any
    command that actually needs the agent), while retaining the same ready/error
    event contract for the frontend.
    """
    ready = session.get("agent_ready")
    if ready is None:
        return
    # A lazy watch session spectating an in-flight child must stay lazy so the
    # subagent live-mirror keeps flowing. Incidental RPCs (session.info, model
    # metadata, etc.) resolve through _sess(), which would otherwise upgrade it
    # to a full agent mid-stream and silently kill the mirror (the mirror bails
    # once agent is set). Once the child completes, the guard lifts and the next
    # prompt/RPC builds the agent normally so the user can talk to the session.
    if session.get("lazy") and _child_run_active(str(session.get("session_key") or "")):
        return
    lock = session.setdefault("agent_build_lock", threading.Lock())
    with lock:
        if ready.is_set() or session.get("agent_build_started"):
            return
        session["agent_build_started"] = True
        # An upgrading lazy session is now genuinely mid-construction — restore
        # its "still starting" eviction exemption.
        session.pop("lazy", None)
    key = session["session_key"]

    def _build() -> None:
        with _sessions_lock:
            current = _sessions.get(sid)
        if current is None:
            ready.set()
            return

        worker = None
        notify_registered = False
        home_token = None
        profile_home = current.get("profile_home")
        # The dashboard's real multimodal chat uses the explicit
        # ``source=multimodal`` provenance. ``source=tool`` belongs to internal
        # sidebar/worker sessions and must not start video/audio memory runtimes.
        is_mm_session = _is_multimodal_runtime_session(current)
        try:
            tokens = _set_session_context(key)
            # Build against the session's profile (global-remote): bind its
            # HERMES_HOME so config/skills/model resolve to it, and hand the
            # agent that profile's db so turns persist to the right state.db.
            session_db = None
            if profile_home:
                home_token = set_hermes_home_override(profile_home)
                try:
                    from hermes_state import SessionDB

                    session_db = SessionDB(db_path=Path(profile_home) / "state.db")
                except Exception:
                    session_db = None
            try:
                # Lazy-resumed (watch) sessions carry the stored conversation
                # id — pass it through so the upgrade continues that session
                # instead of starting a fresh one under the same key.
                kw = {"session_db": session_db}
                if resume_sid := current.get("resume_session_id"):
                    kw["session_id"] = resume_sid
                resume_overrides = current.get("resume_runtime_overrides")
                if isinstance(resume_overrides, dict) and resume_overrides:
                    # Cold deferred resume: restore the full persisted runtime
                    # identity (model/provider/base_url/api_mode/reasoning/tier)
                    # exactly as the eager resume path's _stored_session_runtime_
                    # overrides splat did, so a deferred build can't drop the
                    # provider and fail with "No LLM provider configured".
                    kw.update(resume_overrides)
                else:
                    # Model/effort/fast the desktop picked for a brand-new chat
                    # ride in as per-session overrides so the first build uses
                    # them directly (no global config, no build-then-switch).
                    if override := current.get("model_override"):
                        kw["model_override"] = override
                    if (reasoning := current.get("create_reasoning_override")) is not None:
                        kw["reasoning_config_override"] = reasoning
                    if (tier := current.get("create_service_tier_override")) is not None:
                        kw["service_tier_override"] = tier
                # Multimodal sessions (source="multimodal", /multimodal page)
                # do video
                # Q&A / monitoring — the repo's AGENTS.md dev-guide (~8k tokens
                # after truncation) is irrelevant and inflates every turn's
                # input, slowing responses. Skip context files for these; CLI
                # and other sessions keep AGENTS.md.
                if is_mm_session:
                    kw["skip_context_files"] = True
                agent = _make_agent(sid, key, **kw)
            finally:
                _clear_session_context(tokens)

            # Session DB row deferred to first run_conversation() call.
            # pending_title applied post-first-message (see cli.exec handler).
            current["agent"] = agent
            # ★ Resume 去重修复: 恢复的历史已在 DB 里, 落盘游标必须钉到末尾, 否则
            #   首个 turn 的 _flush_messages_to_session_db 会把整段历史当"新消息"
            #   再 append 一遍 → reopen 后消息翻倍 (DB 实证: 同内容+同时间戳+新 row id)。
            #   落盘去重靠 id() 身份, 但 mm turn 每轮用 _strip_* 新建 conversation_
            #   history, 且 _last_flushed_db_idx==0 会触发身份集重置 → 身份对不上。
            #   这里同步游标 + 预种身份集 (对齐 /undo 15943、cli_commands_mixin 742)。
            #   仅"有恢复历史"的 resume 进此分支; 全新 session(history 空)行为不变。
            try:
                _restored_hist = current.get("history")
                if isinstance(_restored_hist, list) and _restored_hist:
                    agent._last_flushed_db_idx = len(_restored_hist)
                    agent._flushed_db_message_ids = {
                        id(m) for m in _restored_hist if isinstance(m, dict)
                    }
                    agent._flushed_db_message_session_id = getattr(
                        agent, "session_id", None)
            except Exception:
                logger.debug("resume flush-cursor sync failed", exc_info=True)
            # Multimodal phase 2: attach a shared, session-persistent FrameBuffer
            # so the browser can stream camera/screen frames (via the
            # ``multimodal.frame`` RPC) into the SAME main agent that answers chat.
            # Best-effort: a missing multimodal package must never break agent build.
            # This runtime is exclusive to the dashboard's multimodal source.
            # Starting it for ordinary TUI/desktop sessions creates timestamped,
            # empty memory databases that can later mask the active session in
            # the debug picker.
            if is_mm_session:
                try:
                    from agent.multimodal._memory import FrameBuffer as _MMFrameBuffer
                    from agent.multimodal.hermes_glue import build_config as _mm_build_config
                    if not hasattr(agent, "frame_buffer") or agent.frame_buffer is None:
                        agent.frame_buffer = _MMFrameBuffer(_mm_build_config())
                        logger.info("[mm] frame_buffer created (sid=%s)", sid)
                    # Phase 10: multi-monitor registry on the agent. Each entry:
                    # {id, brief, created_at, last_speak_ts}. mm_monitor_active is
                    # kept as the daemon's master gate (=True iff len(mm_monitors)>0).
                    if not hasattr(agent, "mm_monitors"):
                        agent.mm_monitors = {}
                except Exception as _fb_exc:
                    logger.warning("[mm] frame_buffer creation failed (sid=%s): %s",
                                   sid, _fb_exc, exc_info=True)
                    agent.frame_buffer = None

            # ★ 尽早推一次注册表 → 右侧深度/监控面板能和主界面几乎同时出现, 不用等后面的
            #   memory backend / watcher / monitor 引擎启动 (那些慢, 但面板只需"有哪些任务"
            #   的列表, 不需要引擎跑起来)。reconcile 只读 history (已加载) + agent 的 mm_*
            #   dict, 此刻都已就绪。末尾还有一次 push 兜底 (引擎起完后状态可能更新)。
            try:
                _hist0 = current.get("history")
                if isinstance(_hist0, list):
                    # session_id → 磁盘为权威: 先扫本 session 磁盘事件, history 里磁盘
                    #   没有的 monitor/watcher 视为孤儿 (不 re-register)。
                    _reconcile_stale_mm_jobs(
                        copy.deepcopy(_hist0), agent,
                        session_id=str(current.get("session_key") or ""))
                _push_mm_registries(sid, agent)
            except Exception:
                pass

            # ★ Multimodal main-agent thinking switch. Turn thinking OFF: on
            # this deployment (aliyun MaaS qwen3.5-plus, compatible-mode) an
            # unconfigured turn ran ~44s with 2300+ reasoning tokens; disabling
            # drops it to ~1.4s. IMPORTANT: this deployment wants the TOP-LEVEL
            # extra_body `enable_thinking` flag — the nested
            # `chat_template_kwargs.enable_thinking` form is IGNORED here (probed
            # via _tmp_qwen_bench.py). So send both: top-level (what works on
            # aliyun MaaS) AND nested (for vLLM/other stacks that read that form).
            if is_mm_session:
                # Dashboard /multimodal: skip STEER_CHANNEL_NOTE in system prompt
                # (see agent/system_prompt.py) — steer is unused and the example
                # markers leak into assistant output on weaker models.
                try:
                    agent._multimodal_session = True
                except Exception:
                    pass
                # ★ Deep-thinking toggle: set the INITIAL value at build time.
                # The authoritative per-turn refresh lives in _run_prompt_submit
                # .run() (the agent is built once per session, so reading the
                # flag only here froze it at the first-turn value — toggling 🧠
                # later did nothing). Both call the same helper.
                _apply_mm_deep_thinking(agent, current)

                # ★ Diagnostic bridge: chat_completion_helpers._mm_diag_before/
                # _after pushes SEND/RECV events HERE via this callback, and we
                # forward them to the frontend as `multimodal.diag` so the
                # dashboard can `console.log(payload)` them in F12. This is the
                # canonical way to see per-turn LLM cost (msgs / imgs / latency
                # / prompt_tokens / reasoning_tokens) in real time.
                try:
                    _diag_sid = sid
                    def _mm_diag_emit(payload, _sid=_diag_sid):
                        try:
                            _emit("multimodal.diag", _sid, payload)
                        except Exception:
                            pass
                    agent._mm_diag_emit = _mm_diag_emit
                except Exception:
                    pass
            # Phase-3 follow-up: optional independent layered-memory backend that
            # reads the SAME frame_buffer. Off unless multimodal.memory_enabled.
            if is_mm_session:
                # _maybe_start_memory_backend publishes a newly constructed
                # backend into ``current`` before starting/waiting for it. Keep
                # its return value separate: None means "not ready for Watcher",
                # while the session may intentionally retain a timed-out backend
                # so finalize can still stop it.
                _ready_mm_memory_backend = _maybe_start_memory_backend(
                    sid,
                    str(current.get("session_key") or sid),
                    getattr(agent, "frame_buffer", None),
                    session=current,
                )
                # Sibling resident module for on-demand complex multimodal queries
                # (driven by the main agent's set_live_watcher tool). Shares
                # FrameBuffer + MemoryStore with the memory backend.
                # The helper publishes a constructed Watcher into the session
                # before start/wait. Do not assign its return value here: on a
                # startup timeout ``None`` deliberately coexists with the
                # still-stopping, registry-guarded engine stored in the session.
                current.setdefault("_mm_live_watcher_agent", None)
                _maybe_start_live_watcher_agent(
                    sid, getattr(agent, "frame_buffer", None),
                    _ready_mm_memory_backend, current)
            else:
                current["_mm_memory_backend"] = None
                current["_mm_live_watcher_agent"] = None
            # Baseline for the per-turn config sync; the profile home
            # override is still active here.
            current["config_model_seen"] = _config_model_target()

            try:
                worker = _SlashWorker(key, getattr(agent, "model", _resolve_model()))
                _attach_worker(sid, current, worker)
            except Exception:
                pass

            try:
                from tools.approval import (
                    register_gateway_notify,
                    load_permanent_allowlist,
                )

                register_gateway_notify(
                    key, lambda data: _emit_approval_request(sid, data)
                )
                notify_registered = True
                load_permanent_allowlist()
            except Exception:
                pass

            _wire_callbacks(sid)
            # Surface the self-improvement review's "💾 …" summary as an event
            # the TUI/desktop render in-transcript, honoring
            # display.memory_notifications. _init_session wires this for the
            # eager/branch paths; deferred-built sessions (session.create and the
            # default cold resume) build through here, so without this their
            # review summaries would leak to stdout instead of the chat.
            try:
                agent.background_review_callback = lambda message, _sid=sid: _emit(
                    "review.summary", _sid, {"text": str(message)}
                )
                agent.memory_notifications = _load_memory_notifications()
            except Exception:
                pass
            # Hydrate credits notices at session OPEN (not just on the first
            # message), so depletion / usage-band warnings show at "ready". Runs
            # off the build thread, after the notice_callback is wired. Fail-open.
            try:
                from agent.credits_tracker import seed_credits_at_session_start

                seed_credits_at_session_start(agent)
            except Exception:
                pass
            with _sessions_lock:
                if sid in _sessions:
                    _sessions[sid]["_notif_stop"] = _start_notification_poller(sid, _sessions[sid])
                    _maybe_start_monitor_engine(
                        sid, _sessions[sid], getattr(agent, "frame_buffer", None))
            # Re-register any interrupted monitor / research jobs from the resumed
            # history into the (now-built) agent registries — disabled, no engine
            # job — so the panel lists them for on/off. The user flips one back on
            # only AFTER (re)starting the video stream (enable is stream-guarded).
            # Then push the registries so the frontend shows them immediately.
            # ★ 无条件推送 (不再只在 reconcile 翻转了 ≥1 个任务时才推): 若某任务在上次
            #   关闭时已被标 interrupted, 本次 reconcile 返回 0, 但它仍在 mm_watchers 里,
            #   需要推给前端才能自动开面板。reconcile 照常跑 (重新注册), 推送与其返回值解耦。
            try:
                _sess_entry = _sessions.get(sid, {}) if sid in _sessions else {}
                _hist = _sess_entry.get("history") if _sess_entry else None
                if isinstance(_hist, list):
                    _reconcile_stale_mm_jobs(
                        copy.deepcopy(_hist), agent,
                        session_id=str(_sess_entry.get("session_key") or ""))
                _push_mm_registries(sid, agent)
            except Exception:
                pass
            _notify_session_boundary("on_session_reset", key)

            info = _session_info(agent, current)
            cfg_warn = _probe_config_health(_load_cfg())
            if cfg_warn:
                info["config_warning"] = cfg_warn
                logger.warning(cfg_warn)
            _emit("session.info", sid, info)
            # If MCP discovery is still in flight (a server slower than the
            # bounded wait_for_mcp_discovery join in _make_agent), the agent
            # was built without those tools. Catch up once they land — see
            # _schedule_mcp_late_refresh. Cache-safe (pre-first-turn only).
            _schedule_mcp_late_refresh(sid, agent)
        except Exception as e:
            current["agent_error"] = str(e)
            _emit("error", sid, {"message": f"agent init failed: {e}"})
        finally:
            if home_token is not None:
                reset_hermes_home_override(home_token)
            # _attach_worker already closed the worker if this session was
            # reaped mid-build; only the late notify registration can still
            # leak (session.close unregistered before _build registered it).
            with _sessions_lock:
                replaced = _sessions.get(sid) is not current
            if replaced and notify_registered:
                try:
                    from tools.approval import unregister_gateway_notify

                    unregister_gateway_notify(key)
                except Exception:
                    pass
            ready.set()

    threading.Thread(target=_build, daemon=True).start()


def _sess_nowait(params, rid):
    s = _sessions.get(params.get("session_id") or "")
    return (s, None) if s else (None, _err(rid, 4001, "session not found"))


def _sess(params, rid):
    s, err = _sess_nowait(params, rid)
    if err:
        return (None, err)
    _start_agent_build(params.get("session_id") or "", s)
    return (s, _wait_agent(s, rid))


def _normalize_completion_path(path_part: str) -> str:
    expanded = os.path.expanduser(path_part)
    if os.name != "nt":
        normalized = expanded.replace("\\", "/")
        if (
            len(normalized) >= 3
            and normalized[1] == ":"
            and normalized[2] == "/"
            and normalized[0].isalpha()
        ):
            return f"/mnt/{normalized[0].lower()}/{normalized[3:]}"
    return expanded


def _completion_cwd(params: dict | None = None) -> str:
    params = params or {}
    raw = (
        params.get("cwd")
        or _sessions.get(params.get("session_id") or "", {}).get("cwd")
        # A session bound to another profile resolves its workspace from THAT
        # profile's config before falling back to the launch profile's env var.
        or _profile_configured_cwd(_profile_home(params.get("profile")))
        or os.environ.get("TERMINAL_CWD")
        or os.getcwd()
    )
    try:
        resolved = os.path.abspath(os.path.expanduser(str(raw)))
        if os.path.isdir(resolved):
            return resolved
    except Exception:
        pass
    return os.getcwd()


def _terminal_task_cwd(session: dict | None) -> str:
    """Return the cwd that terminal_tool should use for this TUI session.

    ``_completion_cwd`` validates paths on the host so file completion does not
    point at nonsense.  Non-local terminal backends are different: their cwd is
    inside the target environment, so an SSH path like /home/user/workspace may
    not exist on the local macOS host but is still the correct execution cwd.
    """
    backend = (os.environ.get("TERMINAL_ENV") or "").strip().lower()
    if backend and backend != "local":
        raw = os.environ.get("TERMINAL_CWD", "").strip()
        if not raw:
            try:
                terminal_cfg = _load_cfg().get("terminal", {})
                if isinstance(terminal_cfg, dict):
                    raw = str(terminal_cfg.get("cwd") or "").strip()
            except Exception:
                raw = ""
        if raw and raw not in {".", "auto", "cwd"}:
            return raw

    return _session_cwd(session)


# Git working-tree probing (run git, resolve roots, fold worktrees) lives in a
# focused, single-flight-cached module; these stay as the in-server names every
# call site already uses.
_git = git_probe.run_git
_git_branch_for_cwd = git_probe.branch
_git_repo_root_for_cwd = git_probe.repo_root
_git_common_repo_root_for_cwd = git_probe.common_repo_root
_resolve_cwd_git = git_probe.resolve


def _session_cwd(session: dict | None) -> str:
    if session and session.get("cwd"):
        return str(session["cwd"])
    return _completion_cwd()


def _heal_dead_cwd(cwd: str) -> str:
    """Resolve a session cwd that points at a now-deleted directory.

    A session anchored to a linked worktree (``<repo>/.worktrees/<name>``) keeps
    that path after the worktree is removed (branch merged, `git worktree
    remove`, etc). The literal dir is gone, so a probe of it returns nothing and
    the composer shows no branch — while the sidebar still folds the path up to
    the repo's main lane. Heal the mismatch: walk up to the first existing
    ancestor, then resolve its common git root, so a dead-worktree cwd collapses
    to the live repo root (and its real current branch).

    Only meaningful for local backends; a remote/SSH cwd may legitimately not
    exist on the host, so callers must skip healing there.
    """
    raw = (cwd or "").strip()
    if not raw or os.path.isdir(raw):
        return raw

    probe = raw
    # Climb to the first ancestor that still exists on disk.
    for _ in range(64):
        parent = os.path.dirname(probe)
        if not parent or parent == probe:
            break
        probe = parent
        if os.path.isdir(probe):
            break

    if not os.path.isdir(probe):
        return raw

    try:
        root = _git_common_repo_root_for_cwd(probe) or _git_repo_root_for_cwd(probe)
    except Exception:
        root = ""

    return root or probe


def _is_local_terminal_backend() -> bool:
    backend = (os.environ.get("TERMINAL_ENV") or "").strip().lower()
    return not backend or backend == "local"


def _display_session_cwd(session: dict | None) -> str:
    """Session cwd for display/probe surfaces, healed past deleted worktrees.

    Persists the healed value back to the session row (best-effort, local only)
    so the next load is already coherent and the sidebar lane stops showing a
    session pinned to a vanished path.
    """
    cwd = _session_cwd(session)
    if not _is_local_terminal_backend():
        return cwd

    healed = _heal_dead_cwd(cwd)
    if healed and healed != cwd and session is not None:
        session["cwd"] = healed
        try:
            with _session_db(session) as db:
                if db is not None:
                    db.update_session_cwd(session.get("session_key", ""), healed)
        except Exception:
            logger.debug("failed to persist healed session cwd", exc_info=True)
        _persist_session_git_meta(session, healed)

    return healed


def _session_source(session: dict | None) -> str:
    if session:
        source = str(session.get("source") or "").strip()
        if source:
            return source
    return "tui"


def _register_session_cwd(session: dict | None) -> None:
    if not session:
        return
    try:
        from tools.terminal_tool import register_task_env_overrides

        register_task_env_overrides(
            session["session_key"], {"cwd": _terminal_task_cwd(session)}
        )
    except Exception:
        pass


def _ensure_session_db_row(session: dict) -> None:
    """Idempotently persist the session's DB row on first real activity.

    Called from prompt.submit so a row only exists once the user actually sends
    a message — abandoned drafts never leave an empty "Untitled" session behind.
    Uses INSERT OR IGNORE under the hood, so re-calls (and the AIAgent's own
    lazy create) are no-ops.

    Only an *explicitly chosen* workspace is persisted as the session's cwd.
    The agent still runs in the auto-detected directory (session["cwd"]), but
    we don't stamp that onto the row — otherwise every session the user never
    picked a folder for gets grouped under whatever directory the desktop
    happened to launch in (e.g. "desktop"). Leaving it null groups them under
    "No workspace", which is the desired default.
    """
    key = session.get("session_key")
    if not key:
        return
    # Persist into the session's own profile db (global remote mode), not the
    # launch profile's — otherwise the row lands in the wrong state.db, the
    # unified list mis-tags it, and resume 404s ("session not found").
    profile_home = session.get("profile_home")
    if profile_home:
        from hermes_state import SessionDB

        try:
            db = SessionDB(db_path=Path(profile_home) / "state.db")
        except Exception:
            logger.debug("failed to open profile db for session row", exc_info=True)
            return
        close_db = True
    else:
        db = _get_db()
        close_db = False
    if db is None:
        return
    # The session's own model/effort/fast pick — the composer override shipped on
    # session.create, or a restored /model switch — must own the row's model +
    # model_config. The agent isn't built yet at first prompt.submit, so derive
    # the row from the live override dict; fall back to the global resolved model
    # only when this chat made no explicit pick. Writing the global default here
    # used to win the INSERT-OR-IGNORE race against the agent's own correct
    # lazy-create, so a reconnect/resume rebuilt from the global model and
    # silently reverted the chat (e.g. picked gpt-5.5, reconnect snapped back to
    # the profile default). model_config carries provider/reasoning/service_tier
    # so resume restores effort + fast too, not just the model name.
    override = session.get("model_override")
    override = override if isinstance(override, dict) else {}
    row_model = str(override.get("model") or "").strip() or _resolve_model()
    model_config: dict = {}
    for src_key, cfg_key in (
        ("model", "model"),
        ("provider", "provider"),
        ("base_url", "base_url"),
        ("api_mode", "api_mode"),
    ):
        if val := override.get(src_key):
            model_config[cfg_key] = str(val)
    # The composer override may carry the RESOLVED provider "custom" for a named
    # ``providers:`` / ``custom_providers:`` entry. Persisting bare "custom" here
    # (the very first DB write for a fresh desktop session, before the agent is
    # built) is the origin of the recurring "No LLM provider configured" rows:
    # on the next resume bare "custom" routes to OpenRouter with no key. Recover
    # the durable ``custom:<name>`` identity from the override's base_url, else
    # the configured provider, so a routable identity is persisted from the
    # start (matches _runtime_model_config's normalization).
    if str(model_config.get("provider") or "").strip().lower() == "custom":
        try:
            from hermes_cli.runtime_provider import canonical_custom_identity

            healed = canonical_custom_identity(
                base_url=model_config.get("base_url") or None
            )
            if healed:
                model_config["provider"] = healed
        except Exception:
            logger.debug(
                "custom provider identity recovery failed (db row)", exc_info=True
            )
    if (reasoning := session.get("create_reasoning_override")) is not None:
        model_config["reasoning_config"] = reasoning
    if tier := session.get("create_service_tier_override"):
        model_config["service_tier"] = tier
    # Branch lineage: stamp the same ``_branched_from`` marker the TUI /branch
    # uses so list_sessions_rich keeps the branch listed and the desktop sidebar
    # can nest it under its parent.
    parent_session_id = session.get("parent_session_id") or None
    if parent_session_id:
        model_config["_branched_from"] = parent_session_id
    try:
        db.create_session(
            key,
            source=_session_source(session),
            model=row_model,
            model_config=model_config or None,
            parent_session_id=parent_session_id,
            cwd=_session_cwd(session) if session.get("explicit_cwd") else None,
        )
    except Exception:
        logger.debug("failed to persist desktop session row", exc_info=True)
    finally:
        if close_db:
            try:
                db.close()
            except Exception:
                pass


def _persist_branch_seed(session: dict) -> None:
    """First-turn persist of a branch's copied transcript.

    A branch is a draft until its first submit: the parent's messages live only
    in ``session["history"]`` (they ride into the agent as ``conversation_history``,
    which ``_flush_messages_to_session_db`` skips by identity). Without this the
    branch row would resume missing its pre-branch context. Runs once; the row +
    parent link are written by ``_ensure_session_db_row`` just before this.
    """
    if not session.get("parent_session_id") or session.get("_branch_seed_persisted"):
        return
    key = session.get("session_key")
    if not key:
        return
    with session["history_lock"]:
        seed = [dict(msg) for msg in (session.get("history") or [])]
    if not seed:
        return
    with _session_db(session) as db:
        if db is None:
            return
        try:
            for msg in seed:
                db.append_message(session_id=key, role=msg.get("role", "user"), content=msg.get("content"))
            session["_branch_seed_persisted"] = True
        except Exception:
            logger.debug("branch seed persist failed", exc_info=True)


@contextlib.contextmanager
def _session_db(session: dict):
    """Yield the SessionDB that owns this session's row (profile-aware).

    Mirrors :func:`_ensure_session_db_row`: a remote/profile session persists
    into its own profile's ``state.db`` (a fresh handle we close on exit);
    everything else borrows the shared ``_get_db()`` handle (left open). Yields
    None when the db is unavailable.
    """
    db, close_db = None, False
    profile_home = session.get("profile_home")
    if profile_home:
        from hermes_state import SessionDB

        try:
            db, close_db = SessionDB(db_path=Path(profile_home) / "state.db"), True
        except Exception:
            logger.debug("failed to open profile db for session", exc_info=True)
    else:
        db = _get_db()
    try:
        yield db
    finally:
        if close_db and db is not None:
            with contextlib.suppress(Exception):
                db.close()


def _persist_session_git_meta(session: dict, cwd: str) -> None:
    """Resolve + persist a session's git branch / repo root WITHOUT blocking.

    Branch and root come from ``git`` subprocess probes; running them inline on
    the session-init / cwd-set path would stall startup whenever ``cwd`` is slow
    or on an unreachable mount. Run them on a short-lived daemon thread instead
    and persist via the same profile-aware db the caller writes ``cwd`` to.

    Best-effort: ``cwd`` itself is persisted synchronously by the caller, so a
    probe failure just leaves these enrichment columns unset (the project tree
    falls back to its live resolver / lazy backfill). Daemon, so a mid-flight
    probe never delays gateway shutdown.
    """
    session_key = session.get("session_key", "")
    if not session_key or not cwd:
        return
    # Snapshot the routing fields now; the live session dict may be gone by the
    # time the thread runs. `_session_db` reopens the profile-correct db inside.
    db_session = {"session_key": session_key, "profile_home": session.get("profile_home")}

    def _run() -> None:
        try:
            branch = _git_branch_for_cwd(cwd)
            root = _git_common_repo_root_for_cwd(cwd)
            if not (branch or root):
                return
            with _session_db(db_session) as db:
                if db is not None:
                    db.update_session_cwd(session_key, cwd, branch, root)
        except Exception:
            logger.debug("failed to persist session git metadata", exc_info=True)

    threading.Thread(target=_run, name="git-meta", daemon=True).start()


def _set_session_cwd(session: dict, cwd: str) -> str:
    resolved = os.path.abspath(os.path.expanduser(str(cwd)))
    if not os.path.isdir(resolved):
        raise ValueError(f"working directory does not exist: {cwd}")
    session["cwd"] = resolved
    # An explicit user choice — persist it as the workspace (and let a later
    # lazy row creation persist it too, not the launch-dir fallback).
    session["explicit_cwd"] = True
    _register_session_cwd(session)
    with _session_db(session) as db:
        if db is not None:
            try:
                db.update_session_cwd(session.get("session_key", ""), resolved)
            except Exception:
                logger.debug("failed to persist session cwd", exc_info=True)
    # Branch/repo-root probes are git subprocesses — capture them off the hot path.
    _persist_session_git_meta(session, resolved)
    try:
        from tools.terminal_tool import cleanup_vm

        cleanup_vm(session["session_key"])
    except Exception:
        pass
    return resolved


# ── Config I/O ────────────────────────────────────────────────────────


# Keep aligned with `INDICATOR_STYLES` / `DEFAULT_INDICATOR_STYLE` in
# ``ui-tui/src/app/interfaces.ts`` — both ends validate against the
# same shape so `config.get indicator` and the live TUI render agree.
_INDICATOR_STYLES: tuple[str, ...] = ("ascii", "emoji", "kaomoji", "unicode")
_INDICATOR_DEFAULT = "kaomoji"


def _load_cfg() -> dict:
    global _cfg_cache, _cfg_mtime, _cfg_path
    try:
        import yaml

        # Honor a per-session profile override (see session.resume) so a resumed
        # remote profile loads ITS config (model, skills, prompt); otherwise the
        # launch profile's _hermes_home. Cache is keyed on the resolved path, so
        # profiles don't clobber each other.
        override = get_hermes_home_override()
        home = override if isinstance(override, str) and override else _hermes_home
        p = Path(home) / "config.yaml"
        mtime = p.stat().st_mtime if p.exists() else None
        with _cfg_lock:
            if _cfg_cache is not None and _cfg_mtime == mtime and _cfg_path == p:
                return _apply_managed(copy.deepcopy(_cfg_cache))
        if p.exists():
            with open(p, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}
        with _cfg_lock:
            # Cache the RAW user config (no managed overlay) so _save_cfg, which
            # writes _cfg_cache back to disk, never persists managed values into
            # the user's file. The managed overlay is applied on every return
            # path instead (read-side only).
            _cfg_cache = copy.deepcopy(data)
            _cfg_mtime = mtime
            _cfg_path = p
        return _apply_managed(data)
    except Exception:
        pass
    return {}


def _apply_managed(cfg: dict) -> dict:
    """Overlay administrator-pinned managed-scope values on a config dict.

    The TUI/desktop backend builds config independently of
    hermes_cli.config.load_config, so without this a managed skin / reasoning_effort
    / service_tier / provider_routing would be silently ignored here. Read-side
    only — the raw user config is what gets cached and saved. Fail-open.
    """
    try:
        from hermes_cli import managed_scope

        return managed_scope.apply_managed_overlay(cfg if isinstance(cfg, dict) else {})
    except Exception:
        return cfg


def _save_cfg(cfg: dict):
    global _cfg_cache, _cfg_mtime, _cfg_path

    from utils import atomic_yaml_write

    path = _hermes_home / "config.yaml"
    atomic_yaml_write(path, cfg)
    with _cfg_lock:
        _cfg_cache = copy.deepcopy(cfg)
        _cfg_path = path
        try:
            _cfg_mtime = path.stat().st_mtime
        except Exception:
            _cfg_mtime = None


def _cwd_for_session_key(session_key: str) -> str:
    """Reverse-map session_key to the session's logical cwd.

    Snapshots ``_sessions`` first: concurrent RPC handlers mutate it from the
    thread pool, so iterating the live view risks ``RuntimeError: dictionary
    changed size during iteration``.
    """
    if not session_key:
        return ""
    with _sessions_lock:
        for sess in list(_sessions.values()):
            if sess.get("session_key") == session_key:
                return str(sess.get("cwd") or "")
    return ""


def _set_session_context(session_key: str, cwd: str | None = None) -> list:
    try:
        from gateway.session_context import set_session_vars

        # Ephemeral task IDs (background, preview) aren't in `_sessions`, so the
        # reverse-map returns "" and would clear the cwd override. Callers that
        # know the parent workspace pass it explicitly so spawned agents inherit
        # it instead of falling back to the gateway launch dir.
        resolved = cwd if cwd is not None else _cwd_for_session_key(session_key)
        source = "tui"
        with _sessions_lock:
            for sess in list(_sessions.values()):
                if sess.get("session_key") == session_key:
                    source = _session_source(sess)
                    break
        return set_session_vars(session_key=session_key, source=source, cwd=resolved)
    except Exception:
        return []


def _clear_session_context(tokens: list) -> None:
    if not tokens:
        return
    try:
        from gateway.session_context import clear_session_vars

        clear_session_vars(tokens)
    except Exception:
        pass


def _enable_gateway_prompts() -> None:
    """Route approvals through gateway callbacks instead of CLI input()."""
    os.environ["ARGUS_GATEWAY_SESSION"] = "1"
    os.environ["ARGUS_EXEC_ASK"] = "1"
    os.environ["ARGUS_INTERACTIVE"] = "1"


# ── Blocking prompt factory ──────────────────────────────────────────


def _block(event: str, sid: str, payload: dict, timeout: int = 300) -> str:
    rid = uuid.uuid4().hex[:8]
    ev = threading.Event()
    with _prompt_lock:
        _pending[rid] = (sid, ev)
        payload["request_id"] = rid
        _pending_prompt_payloads[rid] = (event, dict(payload))
    try:
        _emit(event, sid, payload)
        ev.wait(timeout=timeout)
    finally:
        with _prompt_lock:
            _pending.pop(rid, None)
            _pending_prompt_payloads.pop(rid, None)
    with _prompt_lock:
        return _answers.pop(rid, "")


def _clear_pending(sid: str | None = None) -> None:
    """Release pending prompts with an empty answer.

    When *sid* is provided, only prompts owned by that session are
    released — critical for session.interrupt, which must not
    collaterally cancel clarify/sudo/secret prompts on unrelated
    sessions sharing the same tui_gateway process.  When *sid* is
    None, every pending prompt is released (used during shutdown).
    """
    with _prompt_lock:
        for rid, (owner_sid, ev) in list(_pending.items()):
            if sid is None or owner_sid == sid:
                _answers[rid] = ""
                ev.set()


# ── Agent factory ────────────────────────────────────────────────────


def resolve_skin() -> dict:
    try:
        from hermes_cli.skin_engine import init_skin_from_config, get_active_skin

        init_skin_from_config(_load_cfg())
        skin = get_active_skin()
        return {
            "name": skin.name,
            "colors": skin.colors,
            "branding": skin.branding,
            "banner_logo": skin.banner_logo,
            "banner_hero": skin.banner_hero,
            "tool_prefix": skin.tool_prefix,
            "help_header": (skin.branding or {}).get("help_header", ""),
        }
    except Exception:
        return {}


def _resolve_model() -> str:
    env = (
        os.environ.get("ARGUS_MODEL", "")
        or os.environ.get("ARGUS_INFERENCE_MODEL", "")
    ).strip()
    if env:
        return env
    m = _load_cfg().get("model", "")
    if isinstance(m, dict):
        return str(m.get("default", "") or "").strip()
    if isinstance(m, str) and m:
        return m.strip()
    return "anthropic/claude-sonnet-4"


def _config_model_target() -> tuple[str, str]:
    """(model, provider) currently selected by config (env as fallback).

    config.yaml wins over ARGUS_MODEL / ARGUS_INFERENCE_MODEL here, the
    reverse of `_resolve_model()`'s startup order. Those env vars are a
    provision-time seed (hosted instances set ARGUS_INFERENCE_MODEL in the
    container env); if they outranked config.yaml, the per-turn sync would
    stay pinned to the seed forever and dashboard/CLI model changes would
    never reach an open chat — the exact bug this sync exists to fix.
    """
    cfg_model = _load_cfg().get("model")
    model = ""
    provider = ""
    if isinstance(cfg_model, dict):
        model = str(cfg_model.get("default", "") or "").strip()
        provider = str(cfg_model.get("provider") or "").strip()
        if provider.lower() == "auto":
            provider = ""
    elif isinstance(cfg_model, str):
        model = cfg_model.strip()
    if not model:
        model = _resolve_model()
    return model, provider


def _resolve_startup_runtime() -> tuple[str, str | None]:
    model = _resolve_model()
    explicit_provider = os.environ.get("ARGUS_TUI_PROVIDER", "").strip()
    if explicit_provider:
        return model, explicit_provider

    explicit_model = (
        os.environ.get("ARGUS_MODEL", "")
        or os.environ.get("ARGUS_INFERENCE_MODEL", "")
    ).strip()
    if not explicit_model:
        return model, None

    try:
        from hermes_cli.models import detect_static_provider_for_model

        cfg = _load_cfg().get("model") or {}
        current_provider = (
            (
                str(cfg.get("provider") or "").strip().lower()
                if isinstance(cfg, dict)
                else ""
            )
            or os.environ.get("ARGUS_INFERENCE_PROVIDER", "").strip().lower()
            or "auto"
        )
        detected = detect_static_provider_for_model(explicit_model, current_provider)
        if detected:
            provider, detected_model = detected
            return detected_model, provider
    except Exception:
        pass
    return model, None


# Bare billing buckets are not routable provider identities (kept in parity with the
# provider gate in agent_init). Restoring one as a session provider override breaks resume.
_BARE_BILLING_PROVIDERS = {"auto", "openrouter", "custom"}


def _stored_session_runtime_overrides(row: dict | None) -> dict:
    """Return resumable preferences persisted with a stored session.

    Model/provider/base-URL identity deliberately comes from the current
    process configuration when a chat is resumed. Only identity-independent
    preferences such as reasoning configuration and service tier survive.
    Historical per-message model metadata remains available for display.
    """
    if not row:
        return {}

    raw_config = row.get("model_config")
    model_config: dict = {}
    if isinstance(raw_config, dict):
        model_config = raw_config
    elif isinstance(raw_config, str) and raw_config.strip():
        try:
            parsed = json.loads(raw_config)
            if isinstance(parsed, dict):
                model_config = parsed
        except Exception:
            logger.debug("failed to parse stored session model_config", exc_info=True)

    overrides: dict = {}
    model = str(row.get("model") or model_config.get("model") or "").strip()
    # ``billing_provider`` is only the billing bucket — for a custom endpoint it is the
    # bare class ``"custom"``, which agent_init treats as non-routable, so restoring it as
    # the provider override makes ``session.resume`` fail with "No LLM provider configured".
    # Only restore an explicit provider; otherwise leave it unset so resume falls back to
    # the configured default, matching the working CLI path.
    explicit_provider = str(model_config.get("provider") or "").strip()
    billing_provider = str(
        model_config.get("billing_provider") or row.get("billing_provider") or ""
    ).strip()
    provider = explicit_provider
    if not provider and billing_provider.lower() not in _BARE_BILLING_PROVIDERS:
        provider = billing_provider
    base_url = str(model_config.get("base_url") or "").strip()
    api_mode = str(model_config.get("api_mode") or "").strip()
    reasoning_config = model_config.get("reasoning_config")
    service_tier = str(model_config.get("service_tier") or "").strip()

    # Heal a bare ``"custom"`` provider stored by an older build (or any leak
    # site that bypassed _runtime_model_config's normalization). Bare custom is
    # the resolved billing class, not a routable identity — restoring it as the
    # session's provider override routes the resume to the OpenRouter default
    # URL with no api_key, surfacing as "No LLM provider configured". Recover
    # the durable ``custom:<name>`` menu key from the stored base_url, falling
    # back to the configured provider when the row has no base_url (the
    # recurring Desktop/TUI regression vector). If neither names a real entry,
    # drop the bare provider entirely so resume falls back to the configured
    # default rather than the broken OpenRouter route.
    if provider.strip().lower() == "custom":
        healed = None
        try:
            from hermes_cli.runtime_provider import canonical_custom_identity

            healed = canonical_custom_identity(base_url=base_url or None)
        except Exception:
            logger.debug(
                "custom provider identity recovery failed", exc_info=True
            )
        provider = healed or ("" if not base_url else provider)

    # Model identity is intentionally not restored. Mixing a historical model
    # with credentials or an endpoint from the current configuration can route
    # an invalid model/endpoint pair. Resume therefore uses one coherent
    # identity from the current config while preserving identity-independent
    # session preferences below.
    _ = (model, provider, base_url, api_mode)
    if isinstance(reasoning_config, dict):
        overrides["reasoning_config_override"] = reasoning_config
    if service_tier:
        overrides["service_tier_override"] = service_tier

    return overrides


def _runtime_model_config(agent, existing: dict | None = None) -> dict:
    config = dict(existing or {})
    model = str(getattr(agent, "model", "") or "").strip()
    provider = str(getattr(agent, "provider", "") or "").strip()
    base_url = str(getattr(agent, "base_url", "") or "").strip()
    api_mode = str(getattr(agent, "api_mode", "") or "").strip()
    reasoning_config = getattr(agent, "reasoning_config", None)
    service_tier = getattr(agent, "service_tier", None)

    if model:
        config["model"] = model
    if provider:
        if provider.strip().lower() == "custom":
            # ``agent.provider`` is the RESOLVED provider, and for any named
            # ``providers:`` / ``custom_providers:`` entry that is the literal
            # string "custom" — persisting it loses the entry identity, so a
            # later resume/rebuild cannot re-resolve the entry's credentials
            # (the api_key is deliberately never persisted; see
            # _stored_session_runtime_overrides). Recover the canonical
            # ``custom:<name>`` menu key from the endpoint URL when present,
            # else from the configured provider — this second fallback is the
            # fix for sessions built WITHOUT a base_url on the override (the
            # recurring Desktop/TUI "No LLM provider configured" regression:
            # bare "custom" with no base_url was persisted verbatim and routed
            # to OpenRouter with no key on the next resume).
            try:
                from hermes_cli.runtime_provider import (
                    canonical_custom_identity,
                )

                provider = (
                    canonical_custom_identity(base_url=base_url) or provider
                )
            except Exception:
                logger.debug(
                    "custom provider identity lookup failed", exc_info=True
                )
        config["provider"] = provider
    if base_url:
        config["base_url"] = base_url
    else:
        config.pop("base_url", None)
    if api_mode:
        config["api_mode"] = api_mode
    else:
        config.pop("api_mode", None)
    if isinstance(reasoning_config, dict):
        config["reasoning_config"] = reasoning_config
    else:
        config.pop("reasoning_config", None)
    if service_tier:
        config["service_tier"] = service_tier
    else:
        config.pop("service_tier", None)

    return config


def _persist_live_session_runtime(session: dict | None) -> None:
    """Persist active session runtime so future resumes restore the same footer."""
    if not session:
        return
    agent = session.get("agent")
    session_key = str(session.get("session_key") or "").strip()
    if agent is None or not session_key:
        return

    db = getattr(agent, "_session_db", None) or _get_db()
    if db is None:
        return

    try:
        row = db.get_session(session_key) or {}
        raw_config = row.get("model_config")
        existing_config = {}
        if isinstance(raw_config, dict):
            existing_config = raw_config
        elif isinstance(raw_config, str) and raw_config.strip():
            parsed = json.loads(raw_config)
            if isinstance(parsed, dict):
                existing_config = parsed
        model_config = _runtime_model_config(agent, existing_config)
        model = str(getattr(agent, "model", "") or "").strip()
        if hasattr(db, "update_session_meta"):
            db.update_session_meta(session_key, json.dumps(model_config), model or None)
        elif model and hasattr(db, "update_session_model"):
            db.update_session_model(session_key, model)
    except Exception:
        logger.debug("failed to persist live session runtime", exc_info=True)


def _persist_live_session_system_prompt(session: dict | None) -> None:
    """Refresh the stored system prompt after a live runtime identity change."""
    if not session:
        return
    agent = session.get("agent")
    session_key = str(session.get("session_key") or "").strip()
    if agent is None or not session_key or not hasattr(agent, "_build_system_prompt"):
        return

    db = getattr(agent, "_session_db", None) or _get_db()
    if db is None or not hasattr(db, "update_system_prompt"):
        return

    try:
        prompt = agent._build_system_prompt(None)
        agent._cached_system_prompt = prompt
        db.update_system_prompt(getattr(agent, "session_id", None) or session_key, prompt)
    except Exception:
        logger.debug("failed to persist live session system prompt", exc_info=True)


def _append_model_switch_marker(session: dict | None, *, model: str, provider: str) -> None:
    """Record a real system-history pivot after a live model switch."""
    if not session:
        return
    session_key = str(session.get("session_key") or "").strip()
    if not session_key:
        return

    provider_part = f" via provider {provider}" if provider else ""
    marker = (
        "[System: The active model for this chat has changed to "
        f"{model}{provider_part}. From this point forward, use this runtime "
        "metadata when answering questions about what model/provider is active.]"
    )
    entry = {"role": "system", "content": marker}

    lock = session.get("history_lock")
    if lock is not None:
        with lock:
            session.setdefault("history", []).append(entry)
            session["history_version"] = int(session.get("history_version", 0)) + 1
    else:
        session.setdefault("history", []).append(entry)
        session["history_version"] = int(session.get("history_version", 0)) + 1

    try:
        agent = session.get("agent")
        db = getattr(agent, "_session_db", None) if agent is not None else None
        if db is not None:
            db.append_message(session_id=session_key, role="system", content=marker)
            return

        _ensure_session_db_row(session)
        with _session_db(session) as scoped_db:
            if scoped_db is not None:
                scoped_db.append_message(
                    session_id=session_key, role="system", content=marker
                )
    except Exception:
        logger.debug("failed to persist model switch marker", exc_info=True)


def _write_config_key(key_path: str, value):
    cfg = _load_cfg()
    current = cfg
    keys = key_path.split(".")
    for key in keys[:-1]:
        if key not in current or not isinstance(current.get(key), dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value
    _save_cfg(cfg)


_STATUSBAR_MODES = frozenset({"off", "top", "bottom"})


def _coerce_statusbar(raw) -> str:
    if raw is False:
        return "off"
    if isinstance(raw, str) and (s := raw.strip().lower()) in _STATUSBAR_MODES:
        return s
    return "top"


_MOUSE_TRACKING_ALIASES = {
    "0": "off",
    "1": "all",
    "all": "all",
    "any": "all",
    "button": "buttons",
    "buttons": "buttons",
    "click": "buttons",
    "false": "off",
    "full": "all",
    "no": "off",
    "off": "off",
    "on": "all",
    "scroll": "wheel",
    "true": "all",
    "wheel": "wheel",
    "yes": "all",
}


def _display_mouse_tracking(display: dict) -> str:
    """Resolve display.mouse_tracking to one of ``off|wheel|buttons|all``.

    Boolean values keep their legacy meaning (``True`` → ``all``, ``False`` →
    ``off``). The ``wheel`` preset (DEC 1000+1006) is the tmux-friendly
    subset — wheel + click only, no hover events to trigger prompt-row
    clipboard probes. Legacy ``tui_mouse`` is honored only when
    ``mouse_tracking`` is absent.
    """
    if not isinstance(display, dict):
        return "all"
    if "mouse_tracking" in display:
        raw = display.get("mouse_tracking")
    else:
        raw = display.get("tui_mouse", True)
    if raw is False or raw == 0:
        return "off"
    if raw is True or raw is None:
        return "all"
    if isinstance(raw, (int, float)):
        return "all"
    if isinstance(raw, str):
        return _MOUSE_TRACKING_ALIASES.get(raw.strip().lower(), "all")
    return "all"


def _load_reasoning_config() -> dict | None:
    from hermes_constants import parse_reasoning_effort

    effort = str(
        (_load_cfg().get("agent") or {}).get("reasoning_effort", "") or ""
    ).strip()
    return parse_reasoning_effort(effort)


def _load_service_tier() -> str | None:
    raw = (
        str((_load_cfg().get("agent") or {}).get("service_tier", "") or "")
        .strip()
        .lower()
    )
    if not raw or raw in {"normal", "default", "standard", "off", "none"}:
        return None
    if raw in {"fast", "priority", "on"}:
        return "priority"
    return None


def _load_provider_routing() -> dict:
    """OpenRouter provider-routing prefs from config.yaml (``provider_routing``).

    Parity with the messaging gateway (``gateway/run.py::_load_provider_routing``)
    and the classic CLI: without this the desktop/TUI backend builds agents with
    no routing prefs, so OpenRouter falls back to its default (effectively random)
    provider selection even when the user configured ``provider_routing``.
    """
    try:
        return _load_cfg().get("provider_routing", {}) or {}
    except Exception:
        return {}


def _load_show_reasoning() -> bool:
    return bool((_load_cfg().get("display") or {}).get("show_reasoning", False))


def _load_memory_notifications() -> str:
    """Self-improvement review notification mode from config.yaml.

    Parity with the messaging gateway (``gateway/run.py``) and the classic CLI:
    ``display.memory_notifications`` controls whether the background review's
    "💾 Self-improvement review: …" summary is surfaced. Without this the
    TUI/desktop backend always behaved as ``"on"`` and silently ignored a user
    who set ``off``. Accepts ``off`` / ``on`` (default) / ``verbose``; a bool is
    normalized for back-compat.
    """
    raw = (_load_cfg().get("display") or {}).get("memory_notifications")
    if isinstance(raw, bool):
        return "on" if raw else "off"
    return str(raw).lower() if raw else "on"


def _load_tool_progress_mode() -> str:
    env = os.environ.get("ARGUS_TUI_TOOL_PROGRESS", "").strip().lower()
    if env in {"off", "new", "all", "verbose"}:
        return env
    raw = (_load_cfg().get("display") or {}).get("tool_progress", "all")
    if raw is False:
        return "off"
    if raw is True:
        return "all"
    mode = str(raw or "all").strip().lower()
    return mode if mode in {"off", "new", "all", "verbose"} else "all"


def _load_enabled_toolsets() -> list[str] | None:
    explicit = [
        item.strip()
        for item in os.environ.get("ARGUS_TUI_TOOLSETS", "").split(",")
        if item.strip()
    ]
    cfg = None
    fallback_notice = None

    # Coding posture (base Hermes): with no explicit pin, collapse to the
    # coding toolset (+ enabled MCP servers) when sitting in a code workspace.
    # The desktop app and `hermes --tui` both land here. See
    # agent/coding_context.py. No config is loaded yet at this point, so we let
    # coding_selection() load it lazily (cli.py passes its already-resolved
    # CLI_CONFIG instead, purely to avoid a redundant read).
    if not explicit:
        try:
            from agent.coding_context import coding_selection

            selection = coding_selection(platform="tui")
            if selection is not None:
                # Fold in `project` here too: this is a GUI-only resolver, and
                # the focus-mode coding posture returns before the fallback path
                # that normally adds it — without this the desktop loses the
                # project tools exactly when sitting in a repo (see below).
                return sorted({*selection, "project"})
        except Exception:
            pass

    try:
        from toolsets import validate_toolset
    except Exception:
        validate_toolset = None

    if explicit and validate_toolset is not None:
        built_in = [name for name in explicit if validate_toolset(name)]
        unresolved = [name for name in explicit if name not in built_in]

        if unresolved:
            try:
                from hermes_cli.plugins import discover_plugins

                discover_plugins()
                plugin_valid = [name for name in unresolved if validate_toolset(name)]
            except Exception:
                plugin_valid = []

            if plugin_valid:
                built_in.extend(plugin_valid)
                unresolved = [name for name in unresolved if name not in plugin_valid]

        if any(name in {"all", "*"} for name in built_in):
            ignored = [name for name in explicit if name not in {"all", "*"}]
            if ignored:
                print(
                    "[tui] ARGUS_TUI_TOOLSETS=all enables every toolset; "
                    f"ignoring additional entries: {', '.join(ignored)}",
                    file=sys.stderr,
                    flush=True,
                )
            return None

        if not unresolved:
            return built_in

        mcp_names: set[str] = set()
        mcp_disabled: set[str] = set()
        try:
            from hermes_cli.config import read_raw_config
            from hermes_cli.tools_config import _parse_enabled_flag

            raw_cfg = read_raw_config()
            mcp_servers = (
                raw_cfg.get("mcp_servers")
                if isinstance(raw_cfg.get("mcp_servers"), dict)
                else {}
            )
            for name, server_cfg in mcp_servers.items():
                if not isinstance(server_cfg, dict):
                    continue
                if _parse_enabled_flag(server_cfg.get("enabled", True), default=True):
                    mcp_names.add(str(name))
                else:
                    mcp_disabled.add(str(name))
        except Exception:
            mcp_names = set()
            mcp_disabled = set()

        mcp_valid = [name for name in unresolved if name in mcp_names]
        disabled = [name for name in unresolved if name in mcp_disabled]
        unknown = [
            name
            for name in unresolved
            if name not in mcp_names and name not in mcp_disabled
        ]
        valid = built_in + mcp_valid

        if unknown:
            print(
                f"[tui] ignoring unknown ARGUS_TUI_TOOLSETS entries: {', '.join(unknown)}",
                file=sys.stderr,
                flush=True,
            )
        if disabled:
            print(
                "[tui] ignoring disabled MCP servers in ARGUS_TUI_TOOLSETS "
                "(set enabled: true in config.yaml to use): "
                f"{', '.join(disabled)}",
                file=sys.stderr,
                flush=True,
            )

        if valid:
            return valid

        fallback_notice = (
            "[tui] no valid ARGUS_TUI_TOOLSETS entries; using configured CLI toolsets"
        )

    try:
        from hermes_cli.config import load_config
        from hermes_cli.tools_config import _get_platform_tools

        cfg = cfg if cfg is not None else load_config()

        # Runtime toolset resolution must include default MCP servers so the
        # agent can actually call them. Passing ``False`` here is the
        # config-editing variant — used when we need to persist a toolset
        # list without baking in implicit MCP defaults. Using the wrong
        # variant at agent creation time makes MCP tools silently missing
        # from the TUI. See PR #3252 for the original design split.
        enabled = _get_platform_tools(cfg, "cli", include_default_mcp_servers=True)
        if fallback_notice is not None:
            print(fallback_notice, file=sys.stderr, flush=True)
        if not enabled:
            return None
        # The desktop Project tools are off _HERMES_CORE_TOOLS (every other
        # platform would carry their schema for nothing), so the platform
        # recovery above — which keys off hermes-cli's tool universe — can't
        # surface them. This resolver runs ONLY in the desktop/TUI gateway, so
        # folding in the `project` toolset here is the gate that exposes them on
        # exactly the surface that can follow a project move.
        return sorted(enabled | {"project"})
    except Exception:
        if fallback_notice is not None:
            print(
                "[tui] no valid ARGUS_TUI_TOOLSETS entries and configured CLI toolsets could not be loaded; enabling all toolsets",
                file=sys.stderr,
                flush=True,
            )
        return None


def _session_tool_progress_mode(sid: str) -> str:
    return str(_sessions.get(sid, {}).get("tool_progress_mode", "all") or "all")


def _session_verbose(sid: str) -> bool:
    return _session_tool_progress_mode(sid) == "verbose"


def _tool_progress_enabled(sid: str) -> bool:
    return _session_tool_progress_mode(sid) != "off"


def _restart_slash_worker(sid: str, session: dict):
    worker = session.get("slash_worker")
    if worker:
        try:
            worker.close()
        except Exception:
            pass
    try:
        new_worker = _SlashWorker(
            session["session_key"],
            getattr(session.get("agent"), "model", _resolve_model()),
        )
    except Exception:
        session["slash_worker"] = None
        return
    # Route through the same store-iff-still-mapped guard as the spawn sites:
    # the post-turn restart runs as `running` flips false, exactly when a
    # close_on_disconnect reap can pop this session — a bare store would orphan
    # the fresh worker (it self-heals only on gateway exit via the watchdog).
    _attach_worker(sid, session, new_worker)


def _persist_model_switch(result) -> None:
    # Use targeted, atomic key writes (comment/ordering-preserving) instead of
    # rewriting the whole `model:` block. A full-block rewrite via save_config()
    # destroys sibling keys the user set under `model:` — `model_slots`,
    # `model_fallback`, etc. — when switching models from the TUI (#48305).
    from cli import save_config_value

    save_config_value("model.default", result.new_model)
    save_config_value("model.provider", result.target_provider)
    if result.base_url:
        save_config_value("model.base_url", result.base_url)
    else:
        # Clear any stale base_url when switching to a provider that doesn't use
        # one (e.g. custom endpoint -> native provider). Reads coalesce null to
        # absent (`model_cfg.get("base_url") or ""`), so a null is equivalent to
        # removal without needing a key-delete. Leaving the old value would
        # route the new model at the previous custom host (#48305).
        save_config_value("model.base_url", None)


def _apply_model_switch(
    sid: str,
    session: dict,
    raw_input: str,
    *,
    confirm_expensive_model: bool = False,
    pin_session_override: bool = True,
    parsed_flags: tuple[str, str, bool, bool, bool] | None = None,
) -> dict:
    from hermes_cli.model_switch import (
        parse_model_flags,
        resolve_persist_behavior,
        switch_model,
    )
    from hermes_cli.runtime_provider import resolve_runtime_provider

    if parsed_flags is None:
        parsed_flags = parse_model_flags(raw_input)
    (
        model_input,
        explicit_provider,
        is_global_flag,
        _force_refresh,
        is_session,
    ) = parsed_flags
    persist_global = resolve_persist_behavior(is_global_flag, is_session)
    if not model_input:
        raise ValueError("model value required")

    agent = session.get("agent")
    if agent:
        current_provider = getattr(agent, "provider", "") or ""
        current_model = getattr(agent, "model", "") or ""
        current_base_url = getattr(agent, "base_url", "") or ""
        current_api_key = getattr(agent, "api_key", "") or ""
    else:
        current_model = _resolve_model()
        current_provider = explicit_provider.strip()
        current_base_url = ""
        current_api_key = ""
        if not explicit_provider:
            runtime = resolve_runtime_provider(requested=None)
            current_provider = str(runtime.get("provider", "") or "")
            current_base_url = str(runtime.get("base_url", "") or "")
            # Preserve a callable api_key (Azure Foundry Entra ID bearer
            # provider) unchanged — ``str(...)`` would produce
            # ``"<function ...>"`` and poison downstream switch_model
            # validation. Match the agent-present branch's behavior at the
            # top of this block.
            _runtime_key = runtime.get("api_key", "")
            if callable(_runtime_key) and not isinstance(_runtime_key, str):
                current_api_key = _runtime_key
            else:
                current_api_key = str(_runtime_key or "")

    # Load user-defined providers so switch_model can resolve named custom
    # endpoints (e.g. "ollama-launch") and validate against saved model lists.
    user_provs = None
    custom_provs = None
    try:
        from hermes_cli.config import get_compatible_custom_providers, load_config

        cfg = load_config()
        user_provs = cfg.get("providers")
        custom_provs = get_compatible_custom_providers(cfg)
    except Exception:
        pass

    result = switch_model(
        raw_input=model_input,
        current_provider=current_provider,
        current_model=current_model,
        current_base_url=current_base_url,
        current_api_key=current_api_key,
        is_global=persist_global,
        explicit_provider=explicit_provider,
        user_providers=user_provs,
        custom_providers=custom_provs,
    )
    if not result.success:
        raise ValueError(result.error_message or "model switch failed")

    if agent:
        try:
            from hermes_cli.context_switch_guard import merge_preflight_compression_warning

            _cfg_ctx = None
            if isinstance(cfg, dict):
                _mc = cfg.get("model", {})
                if isinstance(_mc, dict) and _mc.get("context_length") is not None:
                    _cfg_ctx = int(_mc["context_length"])
            merge_preflight_compression_warning(
                result,
                agent=agent,
                # ★ 功能1: token 预估也要排除 mm_notice — 它们不会真的发给 LLM。
                messages=_strip_mm_context(list(session.get("history", []))),
                custom_providers=custom_provs,
                config_context_length=_cfg_ctx,
            )
        except Exception as exc:
            logger.debug("preflight-compression switch warning failed: %s", exc)

    if not confirm_expensive_model:
        try:
            from hermes_cli.model_cost_guard import expensive_model_warning

            warning = expensive_model_warning(
                result.new_model,
                provider=result.target_provider,
                base_url=result.base_url or current_base_url,
                api_key=result.api_key or current_api_key,
                model_info=result.model_info,
            )
        except Exception:
            warning = None
        if warning is not None:
            confirm_msg = warning.message
            if result.warning_message:
                confirm_msg = f"{confirm_msg}\n\n{result.warning_message}"
            return {
                "value": result.new_model,
                "warning": confirm_msg,
                "confirm_required": True,
                "confirm_message": confirm_msg,
            }

    if agent:
        try:
            agent.switch_model(
                new_model=result.new_model,
                new_provider=result.target_provider,
                api_key=result.api_key,
                base_url=result.base_url,
                api_mode=result.api_mode,
            )
        except Exception as exc:
            # The in-place swap rolled the agent back to the old working
            # model/client and re-raised.  Abort the commit: do NOT restart the
            # slash worker, persist runtime, append the switch marker, set a
            # session model_override, or persist to config — all of which would
            # otherwise leave the session pinned to a broken model and kill the
            # conversation on the next turn (#50163).  A failed switch is a
            # no-op; surface a clean error to the client.
            logger.warning("In-place model switch failed for TUI agent: %s", exc)
            raise ValueError(
                f"Model switch to {result.new_model} failed ({exc}); "
                f"staying on {getattr(agent, 'model', current_model)}."
            ) from exc
        _restart_slash_worker(sid, session)
        _persist_live_session_runtime(session)
        _persist_live_session_system_prompt(session)
        _append_model_switch_marker(
            session, model=result.new_model, provider=result.target_provider
        )
        _emit("session.info", sid, _session_info(agent, session))

    # Record the switch as a PER-SESSION override so a later rebuild of THIS
    # session (e.g. /new via _reset_session_agent, or resume) re-derives the
    # user's chosen model/provider instead of falling back to global config.
    #
    # We deliberately do NOT write process-global env vars (ARGUS_MODEL /
    # ARGUS_INFERENCE_MODEL / ARGUS_TUI_PROVIDER / ARGUS_INFERENCE_PROVIDER)
    # here. The desktop backend hosts every same-profile session in ONE process,
    # so mutating os.environ on a /model switch leaked the new model/provider
    # into every OTHER live session's next agent rebuild — switching the model
    # in one session silently changed it in the others (the cross-session
    # contamination bug). agent.switch_model() above already mutated the right
    # agent in place; the override dict makes that choice survive a rebuild
    # without touching shared process state.
    if pin_session_override and isinstance(session, dict):
        session["model_override"] = {
            "model": result.new_model,
            "provider": result.target_provider,
            "base_url": result.base_url,
            "api_key": result.api_key,
            "api_mode": result.api_mode,
        }
    if persist_global:
        _persist_model_switch(result)
    return {
        "value": result.new_model,
        "warning": result.warning_message or "",
        "confirm_required": False,
    }


def _apply_mm_deep_thinking(agent, session) -> None:
    """Set the MAIN agent's extra_body thinking flag derived from the agent's
    ``reasoning_config``. Called at build time (initial) AND per-turn (in
    _run_prompt_submit.run()), because the agent is built once per session so
    the flag must be re-read every turn to actually take effect.

    Single source of truth: ``agent.reasoning_config`` (produced from
    ``agent.reasoning_effort`` in config.yaml via ``parse_reasoning_effort``).
      * ``{"enabled": False}`` → thinking OFF (was: legacy deep_thinking=False)
      * anything else (missing / ``{"enabled": True, "effort": ...}``) → ON

    The per-vendor wire spelling is NOT decided here — it comes from the
    endpoint's ``ProviderProfile.build_api_kwargs_extras`` (see
    ``providers/README.md``), resolved by name AND hostname so a vendor reached
    through a hand-configured ``custom`` endpoint still gets its own quirks.

    ★ This used to be an inline ``if is_kimi / elif is_qwenish / else`` chain, and
      that is how GLM broke: it matched neither branch, fell into the catch-all
      that emitted ``{}``, and since Zhipu defaults ``thinking`` to ``enabled``,
      "off" silently became "on". Three separate copies of the vendor table
      existed (here, the transport, the multimodal submodule glue) and GLM was
      missing from all three. Add a provider profile, never a branch here.

    The remaining special case is genuinely session-scoped rather than
    vendor-scoped: a multimodal session forces thinking ON for its own turns when
    the user has not explicitly turned it off, because the visual workers depend
    on it. That is why this function exists at all.
    """
    if agent is None:
        return
    rc = getattr(agent, "reasoning_config", None)
    think = not (isinstance(rc, dict) and rc.get("enabled") is False)
    try:
        from providers import resolve_provider_profile

        profile = resolve_provider_profile(
            getattr(agent, "provider", "") or "",
            getattr(agent, "base_url", "") or "",
        )
        if profile is None:
            agent._extra_body_additions = {}
            return

        extra_body, _top_level = profile.build_api_kwargs_extras(
            reasoning_config={"enabled": bool(think)},
            model=getattr(agent, "model", "") or "",
            base_url=getattr(agent, "base_url", "") or "",
        )
        agent._extra_body_additions = extra_body or {}
    except Exception:
        pass




def _mm_message_text(message: Any) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        parts: list[str] = []
        for part in message:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text") or ""))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(p for p in parts if p)
    return str(message or "")


_MM_QUERY_PROJECTION_MARKER = "_mm_query_projection"
_MM_QUERY_PROJECTION_TYPE = "mm_query_projection"


def _mm_query_projection_fields(msg: Any) -> Optional[dict]:
    """Return private projection metadata from live or DB-restored history."""
    if not isinstance(msg, dict):
        return None
    live_key = str(msg.get(_MM_QUERY_PROJECTION_MARKER) or "").strip()
    content = msg.get("content")
    if live_key:
        return {"key": live_key, "text": _content_display_text(content)}
    if isinstance(content, dict) and content.get("type") == _MM_QUERY_PROJECTION_TYPE:
        return {
            "key": str(content.get("projection_key") or "").strip(),
            "text": str(content.get("text") or ""),
        }
    return None


def _normalize_mm_query_projection_history(history: list) -> list:
    """Restore DB-tagged projections to plain Q/A messages for the model."""
    out = []
    for msg in history or []:
        fields = _mm_query_projection_fields(msg)
        content = msg.get("content") if isinstance(msg, dict) else None
        if (fields is not None and isinstance(content, dict)
                and content.get("type") == _MM_QUERY_PROJECTION_TYPE):
            out.append({
                "role": msg.get("role"),
                "content": fields.get("text") or "",
                _MM_QUERY_PROJECTION_MARKER: fields.get("key") or "",
            })
        else:
            out.append(msg)
    return out


def _reserve_mm_query_turn_projection(
    session: dict, reservation_id: str,
) -> list[dict]:
    """Reserve complete QueryWorker results as ordinary historical Q/A turns.

    The model must not know that a worker produced the answer.  Therefore the
    wire representation is exactly ``user: original query`` followed by
    ``assistant: final answer`` -- no sidecar prose, worker name, task id, or
    special role.  The private marker is stripped before provider calls and is
    used only to keep these projected turns out of the duplicate UI transcript.

    A result is reserved by one foreground turn at a time. The caller atomically
    appends the complete pair immediately before taking that turn's history
    snapshot; a failed append releases it for retry. Incomplete/error results
    never create either half of the pair.
    """
    reservation_id = str(reservation_id or "").strip()
    if not reservation_id:
        return []
    lock = session.get("history_lock")
    if lock is None:
        return []

    with lock:
        results = session.setdefault("_mm_query_results", [])
        if not isinstance(results, list):
            results = []
            session["_mm_query_results"] = results

        # On resume the in-memory ledger is absent. Query notices are the
        # durable, UI-only source of truth, so recreate pending rows from them.
        # Projected Q/A messages use a private DB content envelope. Older
        # sessions may have only the durable query notice, so recreate a pending
        # row from that notice when no projected pair with the same key exists.
        known_keys = set()
        for row in results:
            if not isinstance(row, dict):
                continue
            for field in ("projection_key", "task_id", "parent_user_message_id"):
                value = str(row.get(field) or "").strip()
                if value:
                    known_keys.add(value)
        projected_keys = {
            str(fields.get("key") or "")
            for msg in list(session.get("history") or [])
            if (fields := _mm_query_projection_fields(msg)) is not None
            and fields.get("key")
        }
        for index, msg in enumerate(list(session.get("history") or [])):
            if not _is_mm_notice(msg):
                continue
            fields = _mm_notice_fields(msg)
            if fields.get("kind") != "query":
                continue
            if str(fields.get("status") or "complete") != "complete":
                continue
            key = str(fields.get("event_id") or "").strip()
            query = str(fields.get("label") or "").strip()
            answer = str(fields.get("text") or "").strip()
            if not key or key in known_keys or key in projected_keys:
                continue
            if not query or not answer:
                continue
            results.append({
                "task_id": "",
                "parent_user_message_id": key,
                "projection_key": key,
                "query": query,
                "answer": answer,
                "status": "complete",
                "originated_at": float(msg.get("timestamp") or index),
                "completed_at": float(msg.get("timestamp") or index),
                "projection_state": "pending",
                "restored_from_notice": True,
            })
            known_keys.add(key)

        eligible = []
        for row in results:
            if not isinstance(row, dict):
                continue
            if str(row.get("status") or "") != "complete":
                continue
            if str(row.get("projection_state") or "pending") != "pending":
                continue
            query = str(row.get("query") or "").strip()
            answer = str(row.get("answer") or "").strip()
            if not query or not answer:
                continue
            eligible.append(row)

        eligible.sort(key=lambda row: (
            float(row.get("originated_at") or row.get("completed_at") or 0.0),
            float(row.get("completed_at") or 0.0),
            str(row.get("task_id") or row.get("parent_user_message_id") or ""),
        ))

        projected: list[dict] = []
        for row in eligible:
            key = str(row.get("projection_key") or row.get("task_id")
                      or row.get("parent_user_message_id") or "")
            if not key:
                continue
            row["projection_key"] = key
            row["projection_state"] = "reserved"
            row["projection_reservation_id"] = reservation_id
            projected.extend([
                {
                    "role": "user",
                    "content": str(row.get("query") or "").strip(),
                    _MM_QUERY_PROJECTION_MARKER: key,
                },
                {
                    "role": "assistant",
                    "content": str(row.get("answer") or "").strip(),
                    _MM_QUERY_PROJECTION_MARKER: key,
                },
            ])
        return projected


def _finish_mm_query_turn_projection(
    session: dict, reservation_id: str, *, committed: bool,
) -> None:
    """Commit or release every query-result row owned by one reservation."""
    reservation_id = str(reservation_id or "").strip()
    if not reservation_id:
        return
    lock = session.get("history_lock")
    if lock is None:
        return
    now = time.time()
    with lock:
        for row in session.get("_mm_query_results") or []:
            if not isinstance(row, dict):
                continue
            if row.get("projection_state") != "reserved":
                continue
            if str(row.get("projection_reservation_id") or "") != reservation_id:
                continue
            row.pop("projection_reservation_id", None)
            if committed:
                row["projection_state"] = "committed"
                row["projected_at"] = now
            else:
                row["projection_state"] = "pending"


def _commit_mm_query_turn_projection(
    session: dict, reservation_id: str, projected: list[dict],
) -> bool:
    """Atomically append a reserved Q/A batch and advance its ledger state.

    This is the projection transaction itself; it completes immediately before
    the next main-agent history snapshot. The following model call may succeed
    or fail independently -- a valid earlier Q/A answer must not disappear just
    because the user's newer question failed.
    """
    reservation_id = str(reservation_id or "").strip()
    if not reservation_id or not projected or len(projected) % 2:
        return False
    lock = session.get("history_lock")
    if lock is None:
        return False
    with lock:
        reserved_rows = [
            row for row in (session.get("_mm_query_results") or [])
            if isinstance(row, dict)
            and row.get("projection_state") == "reserved"
            and str(row.get("projection_reservation_id") or "") == reservation_id
        ]
        reserved_keys = {
            str(row.get("projection_key") or "") for row in reserved_rows
            if row.get("projection_key")
        }
        message_keys = {
            str(msg.get(_MM_QUERY_PROJECTION_MARKER) or "")
            for msg in projected if isinstance(msg, dict)
            and msg.get(_MM_QUERY_PROJECTION_MARKER)
        }
        if not reserved_keys or message_keys != reserved_keys:
            return False
        for idx in range(0, len(projected), 2):
            q_msg, a_msg = projected[idx], projected[idx + 1]
            if q_msg.get("role") != "user" or a_msg.get("role") != "assistant":
                return False
            if (q_msg.get(_MM_QUERY_PROJECTION_MARKER)
                    != a_msg.get(_MM_QUERY_PROJECTION_MARKER)):
                return False

        history = session.get("history")
        if not isinstance(history, list):
            history = []
        history.extend(projected)
        session["history"] = history
        session["history_version"] = int(session.get("history_version", 0)) + 1
        now = time.time()
        for row in reserved_rows:
            row.pop("projection_reservation_id", None)
            row["projection_state"] = "committed"
            row["projected_at"] = now
        return True


def _strip_history_image_parts(history: list) -> list:
    """Drop image parts from PAST user turns before sending history to the LLM.

    Live video frames enter the conversation as get_current_frame tool results
    (v33: the main agent is never passively injected — it must call the tool).
    Those frame-laden messages get persisted into session["history"], so without
    this every turn would replay ALL previous turns' frames too — image count
    (and prefill tokens) grow linearly and the turn gets slower and slower (the
    "越用越卡" bug). We keep ONLY the current turn's freshly-fetched frames: strip
    image_url parts from every historical user message, leaving its text.
    Assistant/tool messages and text are untouched. Returns a NEW list
    (shallow-rebuilt only where needed); the stored
    session["history"] is not mutated.
    """
    out = []
    for msg in history:
        if (isinstance(msg, dict) and msg.get("role") == "user"
                and isinstance(msg.get("content"), list)):
            text_parts = [p for p in msg["content"]
                          if not (isinstance(p, dict) and p.get("type") == "image_url")]
            if len(text_parts) != len(msg["content"]):
                # Had image parts → rebuild without them. Collapse a single
                # remaining text part to a plain string (cleaner for the model);
                # keep the list form if multiple non-image parts remain.
                if len(text_parts) == 1 and isinstance(text_parts[0], dict) \
                        and text_parts[0].get("type") == "text":
                    new_content = text_parts[0].get("text", "")
                elif not text_parts:
                    new_content = ""
                else:
                    new_content = text_parts
                out.append({**msg, "content": new_content})
                continue
        out.append(msg)
    return out


def _sync_agent_model_with_config(sid: str, session: dict) -> None:
    """Adopt a config.yaml model change at turn start, like gateways do per
    message. Sessions pinned with /model keep their choice; a failed switch
    keeps the current model and never blocks the turn.
    """
    agent = session.get("agent")
    if agent is None or session.get("model_override"):
        return
    target = _config_model_target()
    if not target[0]:
        return
    seen = session.get("config_model_seen")
    # Record first so a broken config gets one attempt per edit, not per turn.
    session["config_model_seen"] = target
    if target == seen:
        return
    model, provider = target
    # Already running the configured model (branched/resumed session before
    # its first sync, or a config revert after a failed switch): adopt the
    # baseline without a redundant switch.
    if model == getattr(agent, "model", "") and (
        not provider or provider == getattr(agent, "provider", "")
    ):
        return
    raw = f"{model} --provider {provider}" if provider else model
    try:
        _apply_model_switch(
            sid,
            session,
            raw,
            confirm_expensive_model=True,
            pin_session_override=False,
        )
    except Exception as e:
        _emit(
            "error",
            sid,
            {"message": f"Could not switch to configured model {model}: {e}"},
        )


def _compress_session_history(
    session: dict,
    focus_topic: str | None = None,
    approx_tokens: int | None = None,
    before_messages: list | None = None,
    history_version: int | None = None,
) -> tuple[int, dict]:
    from agent.model_metadata import estimate_request_tokens_rough

    agent = session["agent"]
    # Snapshot history under the lock so the LLM-bound compression call
    # below does NOT hold history_lock for the duration of the request —
    # otherwise other handlers acquiring the lock (prompt.submit etc.)
    # block on the dispatcher loop while compaction runs.
    if before_messages is None or history_version is None:
        with session["history_lock"]:
            before_messages = list(session.get("history", []))
            history_version = int(session.get("history_version", 0))
    history = before_messages
    # ★ 功能1: 压缩前摘出 monitor/watcher 通知 (mm_notice) —— 它们不参与 LLM 摘要
    #   (dict content 会让 provider 收到畸形 assistant 消息), 压缩后再原样接回, 使
    #   前端全量 context 仍保留这些气泡。压缩本就是有损, 顺序轻微变动 (摘要在前、
    #   通知在后) 对通知类内容可接受。
    _mm_notices = [m for m in history if _is_mm_notice(m)]
    if _mm_notices:
        history = [m for m in history if not _is_mm_notice(m)]
    history = _normalize_mm_query_projection_history(history)
    if len(history) < 4:
        usage = _get_usage(agent)
        return 0, usage
    if approx_tokens is None:
        # Include system prompt + tool schemas so the figure reflects real
        # request pressure, not a transcript-only underestimate (#6217).
        _sys_prompt = getattr(agent, "_cached_system_prompt", "") or ""
        _tools = getattr(agent, "tools", None) or None
        approx_tokens = estimate_request_tokens_rough(
            history, system_prompt=_sys_prompt, tools=_tools
        )
    # Pass system_message=None so AIAgent._compress_context rebuilds the
    # system prompt cleanly via _build_system_prompt(None). Passing the
    # cached prompt (which already contains the agent identity block)
    # makes the rebuild append the identity a second time. Mirrors the
    # CLI's _manual_compress fix for issue #15281.
    compressed, _ = agent._compress_context(
        history,
        None,
        approx_tokens=approx_tokens,
        focus_topic=focus_topic or None,
    )
    # 摘要真正削减的消息数 (只算参与压缩的那部分, 不含接回的 mm_notice)。
    _removed = len(history) - len(compressed)
    # ★ 功能1: 把先前摘出的 mm_notice 接回压缩结果 (保留前端全量气泡)。
    if _mm_notices:
        compressed = list(compressed) + _mm_notices
    with session["history_lock"]:
        if int(session.get("history_version", 0)) != history_version:
            # External mutation during compaction — drop the compressed
            # result so we don't clobber concurrent edits.
            usage = _get_usage(agent)
            return 0, usage
        session["history"] = compressed
        session["history_version"] = history_version + 1
    usage = _get_usage(agent)
    return _removed, usage


def _sync_session_key_after_compress(
    sid: str,
    session: dict,
    *,
    clear_pending_title: bool = True,
    restart_slash_worker: bool = True,
) -> None:
    """Re-anchor session_key when AIAgent._compress_context rotates session_id.

    AIAgent._compress_context ends the current SessionDB session and creates
    a new continuation session, rotating ``agent.session_id``.  The TUI
    gateway keeps the gateway-side ``session_key`` separate (used for
    approval routing, slash worker init, DB title/history lookups, yolo
    state).  Without this sync, those operations would target the ended
    parent session while the agent writes to the new continuation session.

    Policy flags:
        clear_pending_title: True for manual /compress (title belongs to old
            session). False for post-turn auto-compression (preserve user
            intent so pending_title can be applied to the continuation).
        restart_slash_worker: True for manual /compress and post-turn
            auto-compression (worker holds stale session key). False only
            if the caller manages the worker lifecycle separately.
    """
    agent = session.get("agent")
    new_session_id = getattr(agent, "session_id", None) or ""
    old_key = session.get("session_key", "") or ""
    if not new_session_id or new_session_id == old_key:
        return

    lease_reanchored = _transfer_active_session_slot(
        sid,
        session,
        new_session_id=new_session_id,
    )
    if not lease_reanchored:
        logger.warning(
            "Compression session lease did not re-anchor: sid=%s old_session_id=%s new_session_id=%s",
            sid,
            old_key,
            new_session_id,
        )

    try:
        from tools.approval import (
            disable_session_yolo,
            enable_session_yolo,
            is_session_yolo_enabled,
            register_gateway_notify,
            unregister_gateway_notify,
        )

        try:
            unregister_gateway_notify(old_key)
        except Exception:
            pass
        session["session_key"] = new_session_id
        try:
            yolo_was_on = is_session_yolo_enabled(old_key)
        except Exception:
            yolo_was_on = False
        if yolo_was_on:
            try:
                enable_session_yolo(new_session_id)
                disable_session_yolo(old_key)
            except Exception:
                pass
        try:
            register_gateway_notify(
                new_session_id,
                lambda data: _emit_approval_request(sid, data),
            )
        except Exception:
            pass
    except Exception:
        # Even if the approval module fails to import, still anchor the
        # session_key on the new continuation id so downstream lookups
        # don't keep targeting the ended row.
        session["session_key"] = new_session_id

    if clear_pending_title:
        session["pending_title"] = None
    if restart_slash_worker:
        try:
            _restart_slash_worker(sid, session)
        except Exception:
            pass


def _get_usage(agent) -> dict:
    g = lambda k, fb=None: getattr(agent, k, 0) or (getattr(agent, fb, 0) if fb else 0)
    usage = {
        "model": getattr(agent, "model", "") or "",
        "input": g("session_input_tokens", "session_prompt_tokens"),
        "output": g("session_output_tokens", "session_completion_tokens"),
        "reasoning": g("session_reasoning_tokens"),
        "prompt": g("session_prompt_tokens"),
        "completion": g("session_completion_tokens"),
        "total": g("session_total_tokens"),
        "calls": g("session_api_calls"),
    }
    comp = getattr(agent, "context_compressor", None)
    if comp:
        ctx_used = getattr(comp, "last_prompt_tokens", 0) or usage["total"] or 0
        ctx_max = getattr(comp, "context_length", 0) or 0
        if ctx_max:
            usage["context_used"] = ctx_used
            usage["context_max"] = ctx_max
            usage["context_percent"] = max(0, min(100, round(ctx_used / ctx_max * 100)))
        usage["compressions"] = getattr(comp, "compression_count", 0) or 0
    # Live count of background/async subagents still running (delegate_task
    # batches + background single delegations). Mirrors the classic CLI status
    # bar's ⛓ indicator; sourced from the same async_delegation registry.
    try:
        from tools.async_delegation import active_count as _async_active_count
        usage["active_subagents"] = _async_active_count()
    except Exception:
        pass
    # Dev-only live credits-spent readout (L0 usage-aware-credits). Gated on
    # ARGUS_DEV_CREDITS so the payload stays clean when the flag is off.
    if is_truthy_value(os.environ.get("ARGUS_DEV_CREDITS")):
        try:
            spent = agent.get_credits_spent_micros()
            if spent is not None:
                usage["dev_credits_spent_micros"] = int(spent)
        except Exception:
            pass
    return usage


def _probe_credentials(agent) -> str:
    """Light credential check at session creation — returns warning or ''."""
    try:
        key = getattr(agent, "api_key", "") or ""
        provider = getattr(agent, "provider", "") or ""
        if not key or key == "no-key-required":
            return f"No API key configured for provider '{provider}'. First message will fail."
    except Exception:
        pass
    return ""


def _probe_config_health(cfg: dict) -> str:
    """Flag bare YAML keys (`agent:` with no value → None) that silently
    drop nested settings. Returns warning or ''."""
    if not isinstance(cfg, dict):
        return ""
    warnings: list[str] = []
    null_keys = sorted(k for k, v in cfg.items() if v is None)
    if not null_keys:
        pass
    else:
        keys = ", ".join(f"`{k}`" for k in null_keys)
        warnings.append(
            f"config.yaml has empty section(s): {keys}. "
            f"Remove the line(s) or set them to `{{}}` — "
            f"empty sections silently drop nested settings."
        )
    display_cfg = cfg.get("display")
    agent_cfg = cfg.get("agent")
    if isinstance(display_cfg, dict):
        personality = str(display_cfg.get("personality", "") or "").strip().lower()
        if (
            personality
            and personality not in {"default", "none", "neutral"}
            and isinstance(agent_cfg, dict)
            and agent_cfg.get("personalities") is None
        ):
            warnings.append(
                "`display.personality` is set but `agent.personalities` is empty/null; "
                "personality overlay will be skipped."
            )
    return " ".join(warnings).strip()


def _current_profile_name() -> str:
    try:
        from hermes_cli.profiles import get_active_profile_name

        return get_active_profile_name() or "default"
    except Exception:
        return "default"


# Monotonic GUI<->backend contract version. The desktop app refuses to drive a
# backend reporting less than its required value (or none at all — a pre-GUI
# checkout), surfacing a one-click "update to align" prompt instead of failing
# cryptically downstream. Bump whenever the desktop's backend contract changes.
# v2: adds the file.attach RPC (remote-gateway non-image file upload).
DESKTOP_BACKEND_CONTRACT = 2


def _session_info(agent, session: dict | None = None) -> dict:
    if session is None:
        for candidate in _sessions.values():
            if candidate.get("agent") is agent:
                session = candidate
                break
    cwd = _display_session_cwd(session)
    session_key = str(
        (session or {}).get("session_key") or getattr(agent, "session_id", "") or ""
    )
    cfg_personality = ((_load_cfg().get("display") or {}).get("personality") or "")
    personality = (session or {}).get("personality", cfg_personality)
    reasoning_config = getattr(agent, "reasoning_config", None)
    # ★ Thinking-OFF must report the explicit "none", not "". Empty string is
    #   the "no explicit level, use the default" signal, and every client maps
    #   it to medium/on — so echoing "" for a disabled config made turning
    #   thinking OFF impossible: the client optimistically switched the toggle
    #   off, this event arrived with "", and the toggle snapped straight back on
    #   (desktop isThinkingEnabled('') === true; web normalizeEffort('') ===
    #   "medium"). "none" round-trips through parse_reasoning_effort, and both
    #   clients already have a label for it (desktop REASONING_LABELS.none →
    #   "Off"), it was simply never reachable.
    reasoning_effort = ""
    if isinstance(reasoning_config, dict):
        if reasoning_config.get("enabled") is False:
            reasoning_effort = "none"
        else:
            reasoning_effort = str(reasoning_config.get("effort", "") or "")
    service_tier = getattr(agent, "service_tier", None) or ""
    can_toggle_reasoning = True
    try:
        from providers import resolve_provider_profile

        profile = resolve_provider_profile(
            getattr(agent, "provider", ""), getattr(agent, "base_url", "")
        )
        if profile is not None:
            model = getattr(agent, "model", "")
            off_config = {"enabled": False}
            extra_body, top_level = profile.build_api_kwargs_extras(
                reasoning_config=off_config, model=model
            )
            body = profile.build_extra_body(reasoning_config=off_config, model=model)
            can_toggle_reasoning = bool(extra_body or top_level or body)
    except Exception:
        can_toggle_reasoning = True
    # Effective approval-bypass state — the same three sources that
    # check_all_command_guards() ORs together: persistent config
    # (approvals.mode=off), the process-scoped --yolo env, and the
    # per-session flag. Reporting only the per-session flag here would lie to
    # the desktop status bar (it would show YOLO "off" while approvals.mode=off
    # silently auto-approves every dangerous command).
    yolo = False
    try:
        from tools.approval import (
            _YOLO_MODE_FROZEN,
            _get_approval_mode,
            is_session_yolo_enabled,
        )

        session_yolo = (
            bool(is_session_yolo_enabled(session_key)) if session_key else False
        )
        yolo = bool(_YOLO_MODE_FROZEN) or session_yolo or _get_approval_mode() == "off"
    except Exception:
        yolo = False
    # ★ Whether the thinking control can actually DO anything on THIS endpoint —
    #   a different question from "does the model reason". Parity with the web
    #   dashboard, which reports the same flag from /api/model/info; the desktop
    #   path had no equivalent, so it could only ever show an ungated switch.
    #   Resolved by provider name AND base_url, because a vendor reached through a
    #   hand-configured ``custom:`` endpoint still has its own wire quirks.
    #   Unknown → True: a missing switch is worse than a no-op one (same
    #   optimistic default as hermes_cli/inventory.py::_apply_capabilities).
    try:
        from providers import resolve_provider_profile

        _rprofile = resolve_provider_profile(
            getattr(agent, "provider", "") or "",
            getattr(agent, "base_url", "") or "",
        )
        can_toggle_reasoning = bool(
            _rprofile is None
            or _rprofile.can_toggle_reasoning(getattr(agent, "model", "") or "")
        )
    except Exception:
        can_toggle_reasoning = True

    info: dict = {
        "model": getattr(agent, "model", ""),
        "provider": getattr(agent, "provider", ""),
        "reasoning_effort": reasoning_effort,
        "can_toggle_reasoning": can_toggle_reasoning,
        "service_tier": service_tier,
        "fast": service_tier == "priority",
        "yolo": yolo,
        "tools": {},
        "skills": {},
        "cwd": cwd,
        "branch": _git_branch_for_cwd(cwd),
        "personality": str(personality or ""),
        "running": bool((session or {}).get("running")),
        "title": _session_live_title(session or {}, session_key) if session_key else "",
        "desktop_contract": DESKTOP_BACKEND_CONTRACT,
        "version": "",
        "release_date": "",
        "update_behind": None,
        "update_command": "",
        "usage": _get_usage(agent),
        "profile_name": _current_profile_name(),
    }
    try:
        from hermes_cli import __version__, __release_date__

        info["version"] = __version__
        info["release_date"] = __release_date__
    except Exception:
        pass
    try:
        from model_tools import get_toolset_for_tool

        for t in getattr(agent, "tools", []) or []:
            name = t["function"]["name"]
            info["tools"].setdefault(get_toolset_for_tool(name) or "other", []).append(
                name
            )
    except Exception:
        pass
    try:
        from hermes_cli.banner import get_available_skills

        info["skills"] = get_available_skills()
    except Exception:
        pass
    try:
        from tools.mcp_tool import get_mcp_status

        info["mcp_servers"] = get_mcp_status()
    except Exception:
        info["mcp_servers"] = []
    try:
        info["system_prompt"] = getattr(agent, "_cached_system_prompt", "") or ""
    except Exception:
        pass
    try:
        from hermes_cli.banner import get_update_result
        from hermes_cli.config import recommended_update_command

        info["update_behind"] = get_update_result(timeout=0.5)
        info["update_command"] = recommended_update_command()
    except Exception:
        pass
    warn = _probe_credentials(agent)
    if warn:
        info["credential_warning"] = warn
    return info


def _tool_ctx(name: str, args: dict) -> str:
    try:
        from agent.display import build_tool_preview

        return build_tool_preview(name, args, max_len=80) or ""
    except Exception:
        return ""


def _tool_arg_fields(name: str, args: dict) -> list[dict]:
    """Privacy-classified structured args for the UI's expandable tool row.

    Unlike ``args_text`` (full JSON, verbose-only) this ships by DEFAULT: a
    tool row with nothing to expand is why "the model called a skill and I
    can't tell what it did" — see ``agent.display.describe_arg_fields`` for
    the payload-vs-intent classification that makes it safe to show.
    """
    try:
        from agent.display import describe_arg_fields, redact_tool_args_for_display

        safe = redact_tool_args_for_display(name, args) or args
        return describe_arg_fields(safe)
    except Exception:
        return []


def _emit_session_info_for_session(sid: str, session: dict) -> None:
    agent = session.get("agent")
    if agent is None:
        return
    try:
        _emit("session.info", sid, _session_info(agent, session))
    except Exception:
        pass


# Tool Args/Result text shipped to the TUI for the verbose trail line. The TUI
# renders only a small persisted preview (ui-tui VERBOSE_TRAIL_MAX_CHARS), kept
# all session and expanded by default — so shipping more than that is pure pipe
# waste AND feeds the Ink render-tree blowup that silently OOM-killed the TUI
# parent (#34095). Cap here to match the render budget (a hair more, so the
# "[omitted …]" label is still informative when output is genuinely large).
# Full output stays in the agent context and the SQLite session, untouched.
_TUI_VERBOSE_TEXT_MAX_CHARS = 1_000
_TUI_VERBOSE_TEXT_MAX_LINES = 16


def _cap_tui_verbose_text(text: str) -> str:
    if (
        len(text) <= _TUI_VERBOSE_TEXT_MAX_CHARS
        and text.count("\n") < _TUI_VERBOSE_TEXT_MAX_LINES
    ):
        return text

    idx = len(text)
    start = 0
    for _ in range(_TUI_VERBOSE_TEXT_MAX_LINES):
        idx = text.rfind("\n", 0, idx)
        if idx < 0:
            start = 0
            break
        start = idx + 1

    line_start = start
    start = max(line_start, len(text) - _TUI_VERBOSE_TEXT_MAX_CHARS)
    if start > line_start:
        next_break = text.find("\n", start)
        if 0 <= next_break < len(text) - 1:
            start = next_break + 1

    tail = text[start:].lstrip()
    omitted_chars = max(0, len(text) - len(tail))
    omitted_lines = text[:start].count("\n")
    if omitted_lines:
        label = (
            "[showing verbose tail; omitted "
            f"{omitted_lines} lines / {omitted_chars} chars]\n"
        )
    else:
        label = f"[showing verbose tail; omitted {omitted_chars} chars]\n"
    return f"{label}{tail}"


def _redact_tui_display_text(text: str) -> str:
    """Force-redact text crossing from tool results into a UI client.

    ``redact_sensitive_text`` deliberately preserves ordinary web URLs so
    tools can follow magic links and OAuth callbacks.  Tool result text is a
    display-only projection, though, and URL userinfo is always a credential
    at this boundary.  Strip it explicitly while keeping the global redactor's
    tool-facing URL semantics unchanged.
    """
    try:
        from agent.redact import redact_sensitive_text, redact_url_userinfo

        redacted = redact_sensitive_text(str(text), force=True)
        return redact_url_userinfo(redacted)
    except Exception:
        return ""


# Redaction runs the secret-scan regex over the whole input BEFORE the cap (so
# a credential straddling the truncation point can't be sliced into a fragment
# the recognizer misses — see _history_tool_result's summary note). That's fine
# for normal output but pathological on multi-MB blobs (base64 image data, a
# giant read_file): the regex is ~10s/MB and 99.9% of it is discarded by the
# cap. Pre-trim to a window FAR larger than the visible tail (last 1000 chars /
# 16 lines) so credential matching around the shown text is unaffected, while
# megabytes of never-shown prefix never reach the regex. The image path already
# strips base64 before it gets here; this is the generic backstop.
_TUI_VERBOSE_REDACT_MAX_CHARS = 64 * 1024


def _redact_tui_verbose_text(text: str) -> str:
    if len(text) > _TUI_VERBOSE_REDACT_MAX_CHARS:
        text = text[-_TUI_VERBOSE_REDACT_MAX_CHARS:]
    return _cap_tui_verbose_text(_redact_tui_display_text(text))


def _tool_args_text(args: dict) -> str:
    try:
        raw = json.dumps(args or {}, indent=2, ensure_ascii=False, default=str)
    except Exception:
        raw = str(args or {})
    return _redact_tui_verbose_text(raw)


def _tool_result_text(result: object) -> str:
    try:
        from agent.tool_dispatch_helpers import _multimodal_text_summary

        raw = _multimodal_text_summary(result)
    except Exception:
        raw = str(result)
    return _redact_tui_verbose_text(raw)


def _strip_content_image_parts(content: list) -> list:
    """Replace multimodal image parts with a text placeholder, in O(parts).

    Used on the tool-result history-projection path so a ~3.8 MB base64
    ``image_url`` never reaches ``_coerce_message_text`` (which would inline the
    whole data: URL) nor the secret redactor (which scans it for ~10s). Cheap:
    it inspects each part's ``type`` — it never touches the base64 payload
    itself. Returns a new list; non-image parts are passed through unchanged.
    """
    out: list = []
    for part in content:
        if isinstance(part, dict) and part.get("type") in {
            "image_url", "input_image", "image",
        }:
            out.append({"type": "text", "text": "[image]"})
        else:
            out.append(part)
    return out


def _history_tool_result(name: str, content: object) -> tuple[str, str]:
    """Project a persisted tool result into ``(body, summary)`` for a client.

    The full ToolResult lives in ``messages.content`` on disk (a terminal
    output can be tens of KB, a read_file 150KB+). Resume payloads must not
    carry that verbatim: a long session would ship megabytes and the clients
    hold every byte in component state. So the body is capped/redacted with the
    same budget the live verbose path uses (:func:`_redact_tui_verbose_text`),
    and structured envelopes are unwrapped to the part a human reads —
    ``{"output": ..., "exit_code": ...}`` becomes the output, not raw JSON.

    Returns ``("", "")`` when there is nothing worth showing, so the caller can
    omit the fields entirely and the client renders a single summary line
    instead of an empty disclosure.
    """
    if content is None:
        return "", ""

    # Tool results can carry multimodal parts — get_current_frame returns a
    # ~2.5-3.8 MB base64 image_url per call, and a monitored session accumulates
    # ~10 of them. Those images are never rendered from the history body (the
    # cap below keeps only a meaningless 1000-char base64 tail); worse, feeding
    # the raw data: URL into _redact_tui_display_text runs the secret-scan regex
    # over multiple megabytes of base64 — ~10s PER image, ~62s per resume of a
    # long monitored session, all on the WS loop (session.history) or delaying
    # time-to-content (session.resume). Collapse image parts to a placeholder
    # BEFORE building the string so the redactor never sees the base64.
    if isinstance(content, list):
        raw = _coerce_message_text(_strip_content_image_parts(content))
    else:
        raw = content if isinstance(content, str) else _coerce_message_text(content)
    if not raw.strip():
        return "", ""

    summary = ""
    body = raw

    # Structured tool envelopes: surface the human-facing stream and keep the
    # exit status as the summary, mirroring _preview_tool_result_preview.
    try:
        data = json.loads(raw)
    except Exception:
        data = None

    if isinstance(data, dict):
        stream = ""
        for key in ("output", "stdout", "text_summary", "content", "answer"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                stream = value
                break
        error = data.get("error")
        if isinstance(error, str) and error.strip():
            summary = error.strip().splitlines()[0]
            body = stream or error
        else:
            exit_code = data.get("exit_code")
            if exit_code not in (None, 0):
                summary = f"exit {exit_code}"
            if stream:
                body = stream
            elif data.get("session_id"):
                summary = summary or f"background process {data.get('session_id')}"
                body = ""

    # An empty envelope ({} / []) is a tool that returned nothing to show. Emit
    # no body at all so the client renders one summary line instead of a
    # disclosure that opens onto "{}".
    if isinstance(data, (dict, list)) and not data:
        body = ""

    safe_body = _redact_tui_verbose_text(body) if body.strip() else ""
    # Summary is a separate client-visible field.  Redact before truncating so
    # a credential crossing the old 200-character boundary cannot be cut into
    # a fragment that no longer matches the central token recognizer.
    safe_summary = _redact_tui_display_text(summary).strip()[:200]
    return safe_body, safe_summary


def _fmt_tool_duration(seconds: float | None) -> str:
    if seconds is None:
        return ""
    if seconds < 10:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{round(seconds)}s"
    mins, secs = divmod(int(round(seconds)), 60)
    return f"{mins}m {secs}s" if secs else f"{mins}m"


def _count_list(obj: object, *path: str) -> int | None:
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return len(cur) if isinstance(cur, list) else None


def _tool_summary(name: str, result: str, duration_s: float | None) -> str | None:
    try:
        data = json.loads(result)
    except Exception:
        data = None

    dur = _fmt_tool_duration(duration_s)
    suffix = f" in {dur}" if dur else ""
    text = None

    if name == "web_search" and isinstance(data, dict):
        n = _count_list(data, "data", "web")
        if n is not None:
            text = f"Did {n} {'search' if n == 1 else 'searches'}"

    elif name == "web_extract" and isinstance(data, dict):
        n = _count_list(data, "results") or _count_list(data, "data", "results")
        if n is not None:
            text = f"Extracted {n} {'page' if n == 1 else 'pages'}"

    if isinstance(data, dict) and data.get("fallback_warning"):
        warning = str(data.get("fallback_warning") or "").strip()
        if warning:
            text = warning

    if text:
        return _redact_tui_display_text(f"{text}{suffix}") or None
    return None


def _track_internal_query_tool_locked(
    session: dict,
    parent_id: str,
    tool_call_id: str,
) -> None:
    """Register one hidden query tool without replacing its siblings.

    A single internal Monitor/Watcher turn may issue more than one
    ``query_multimodal`` call under the same parent request.  The worker can
    finish before or after the model tool callback, so every invocation needs
    its own two-phase record.  Live records are never evicted for a size cap:
    doing so would turn a later private worker answer into ordinary main-agent
    context.  The records are instead structurally bounded by outstanding tool
    calls and are removed as each call settles.
    """
    internal_requests = session.setdefault("_mm_internal_request_origins", {})
    marker = internal_requests.get(parent_id)
    if not isinstance(marker, dict):
        marker = {
            "origin": "internal_hook",
            "created_at": time.time(),
            "calls": {},
            "early_query_completions": {},
        }
        internal_requests[parent_id] = marker
    calls = marker.setdefault("calls", {})
    calls[tool_call_id] = {
        "tool_complete": False,
        "query_complete": False,
        "deferred_reply": None,
        "task_id": "",
        "created_at": time.time(),
    }


def _find_internal_query_marker_locked(
    session: dict,
    parent_id: str,
    tool_call_id: str,
) -> tuple[str, dict | None]:
    """Find the hidden-query parent, falling back to its stable tool id."""
    internal_requests = session.get("_mm_internal_request_origins") or {}
    marker = internal_requests.get(parent_id)
    if isinstance(marker, dict):
        calls = marker.get("calls") or {}
        if tool_call_id in calls:
            return parent_id, marker
    for candidate_parent, candidate in internal_requests.items():
        if not isinstance(candidate, dict):
            continue
        if tool_call_id in (candidate.get("calls") or {}):
            return str(candidate_parent), candidate
    return "", None


def _settle_internal_query_tool_locked(
    session: dict,
    *,
    parent_id: str,
    tool_call_id: str,
    task_id: str,
    deferred_reply: bool,
) -> bool:
    """Record tool completion and retire only the matching invocation."""
    resolved_parent, marker = _find_internal_query_marker_locked(
        session, parent_id, tool_call_id)
    if not resolved_parent or not isinstance(marker, dict):
        return False

    calls = marker.get("calls") or {}
    call = calls.get(tool_call_id)
    if not isinstance(call, dict):
        return False
    call["tool_complete"] = True
    call["deferred_reply"] = bool(deferred_reply)
    call["task_id"] = str(task_id or "")

    early = marker.get("early_query_completions") or {}
    if task_id and str(task_id) in early:
        call["query_complete"] = True
        early.pop(str(task_id), None)

    if not deferred_reply or call.get("query_complete"):
        calls.pop(tool_call_id, None)

    if not calls:
        # No invocation remains that could own an early completion.  Discard
        # duplicate/invalid callback ids along with the settled parent record.
        (session.get("_mm_internal_request_origins") or {}).pop(
            resolved_parent, None)
    return True


def _settle_internal_query_worker_locked(
    session: dict,
    *,
    parent_id: str,
    task_id: str,
) -> bool:
    """Record worker completion while preserving every sibling invocation."""
    internal_requests = session.get("_mm_internal_request_origins") or {}
    marker = internal_requests.get(parent_id)
    if not isinstance(marker, dict):
        return False

    calls = marker.get("calls") or {}
    matching_tool_ids = [
        tool_id
        for tool_id, call in calls.items()
        if isinstance(call, dict)
        and str(call.get("task_id") or "") == str(task_id or "")
    ]
    if matching_tool_ids:
        for tool_id in matching_tool_ids:
            call = calls.get(tool_id)
            if not isinstance(call, dict):
                continue
            call["query_complete"] = True
            if call.get("tool_complete"):
                calls.pop(tool_id, None)
    else:
        # tool.complete has not exposed the task id yet.  At most one distinct
        # worker completion can belong to each live invocation, which gives us
        # a deterministic bound without evicting any active parent/call.
        early = marker.setdefault("early_query_completions", {})
        early[str(task_id or "")] = time.time()
        while len(early) > len(calls):
            early.pop(next(iter(early)), None)

    if not calls:
        internal_requests.pop(parent_id, None)
    return True


def _on_tool_start(sid: str, tool_call_id: str, name: str, args: dict):
    session = _sessions.get(sid)
    if session is not None:
        if name == "query_multimodal":
            agent = session.get("agent")
            parent_id = str(
                getattr(agent, "_active_parent_user_message_id", "") or "")
            if (
                parent_id
                and bool(getattr(agent, "_ephemeral_internal_turn", False))
            ):
                # tool.start precedes the handler that schedules QueryWorker.
                # Mark only actual hidden queries so their completion cannot be
                # projected even if it beats tool.complete.
                with session["history_lock"]:
                    _track_internal_query_tool_locked(
                        session, parent_id, tool_call_id)
        try:
            from agent.display import capture_local_edit_snapshot

            snapshot = capture_local_edit_snapshot(name, args)
            if snapshot is not None:
                session.setdefault("edit_snapshots", {})[tool_call_id] = snapshot
        except Exception:
            pass
        session.setdefault("tool_started_at", {})[tool_call_id] = time.time()
    if _tool_progress_enabled(sid):
        payload = {
            "tool_id": tool_call_id,
            "name": name,
            "context": _tool_ctx(name, args),
        }
        # Structured args ship by default (classified, payload values elided);
        # args_text stays verbose-only because it is the raw JSON dump.
        arg_fields = _tool_arg_fields(name, args)
        if arg_fields:
            payload["args_fields"] = arg_fields
        if _session_verbose(sid):
            args_text = _tool_args_text(args)
            if args_text:
                payload["args_text"] = args_text
        # tool.complete is the source of truth for todos (full list from the
        # tool result). args.todos here may be a partial merge update.
        _emit("tool.start", sid, payload)


def _on_tool_complete(sid: str, tool_call_id: str, name: str, args: dict, result: str):
    payload = {"tool_id": tool_call_id, "name": name, "args": args}
    session = _sessions.get(sid)
    snapshot = None
    started_at = None
    if session is not None:
        snapshot = session.setdefault("edit_snapshots", {}).pop(tool_call_id, None)
        started_at = session.setdefault("tool_started_at", {}).pop(tool_call_id, None)
    duration_s = time.time() - started_at if started_at else None
    if duration_s is not None:
        payload["duration_s"] = duration_s
    try:
        payload["result"] = json.loads(result)
    except Exception:
        payload["result"] = result
    # ★ Two-segment dispatch rendering: the FRONTEND shows the tool card as a
    #   fixed "派发" header line + the result note, in one box. Segment 1
    #   (dispatch_label) is built HERE from the tool name + the real event id
    #   (request_id / monitor_id the tool generated), so it ALWAYS matches the
    #   background event regardless of whatever free text the model wrote — the
    #   model never has to produce the id itself. Segment 2 = note (see below).
    #   (The event id is generated inside the tool via secrets, not by the model.)
    _rslt = payload.get("result")
    if isinstance(_rslt, dict):
        if _rslt.get("history_policy") == "ephemeral_control":
            # Pure Monitor control belongs to the registry/right-hand panel,
            # not provider-facing history.  Keep this per-tool marker for
            # diagnostics/rollout compatibility, but the browser waits for the
            # final turn-level message.complete marker before removing anything;
            # a model may have issued this tool in a mixed, non-ephemeral batch.
            payload["ephemeral_control"] = True
        if name == "query_multimodal":
            if (_rslt.get("control") == "handoff"
                    and _rslt.get("reply_owner") == "query_worker"):
                _task = str(_rslt.get("task_id") or "")
                _parent = str(_rslt.get("parent_user_message_id") or "")
                payload["dispatch_label"] = (
                    f"🧠 已交给 QueryWorker 异步回答"
                    + (f" · 任务 #{_task}" if _task else "")
                )
                payload["dispatch_note"] = str(
                    _rslt.get("ack") or _rslt.get("note") or "").strip()
                if _parent:
                    payload["request_id"] = _parent
                if _task:
                    payload["task_id"] = _task
                if session is not None and _task:
                    lock = session.get("history_lock")
                    if lock is not None:
                        with lock:
                            ledger = session.setdefault("_mm_worker_tasks", {})
                            worker_row = ledger.setdefault(_task, {})
                            worker_row.setdefault("task_id", _task)
                            worker_row.setdefault(
                                "parent_user_message_id", _parent)
                            worker_row.setdefault(
                                "reply_owner", "query_worker")
                            worker_row.setdefault(
                                "query",
                                str(
                                    _rslt.get("original_user_text")
                                    or _rslt.get("query") or ""
                                ),
                            )
                            worker_row.setdefault("created_at", time.time())
                            # A very fast QueryWorker may finish before the
                            # model's tool.complete callback runs.  Never let
                            # the later dispatch receipt downgrade that
                            # terminal ledger row back to ``running`` or erase
                            # its result.
                            terminal_states = {
                                "cancelled", "canceled", "complete", "done",
                                "error", "failed", "interrupted",
                            }
                            if str(worker_row.get("status") or "") not in terminal_states:
                                worker_row["status"] = str(
                                    _rslt.get("status") or "running")
                # Preserve the user's bubble for session reopen as a UI-only
                # notice. It is stripped from main-agent model history until a
                # complete answer lets us commit the ordinary hidden Q/A pair.
                _original_query = str(
                    _rslt.get("original_user_text")
                    or _rslt.get("query") or "").strip()
                _internal_query = bool(
                    session is not None
                    and (session.get("_mm_internal_request_origins") or {}).get(
                        _parent
                    )
                )
                if (
                    session is not None
                    and _parent
                    and _original_query
                    and not _internal_query
                ):
                    _append_mm_context(
                        session,
                        kind="query_user",
                        text=_original_query,
                        event_id=_parent,
                        label=_original_query,
                        status="running",
                    )
                if session is not None and _parent:
                    lock = session.get("history_lock")
                    if lock is not None:
                        with lock:
                            _settle_internal_query_tool_locked(
                                session,
                                parent_id=_parent,
                                tool_call_id=tool_call_id,
                                task_id=_task,
                                deferred_reply=(
                                    _rslt.get("handoff_mode")
                                    == "deferred_reply"
                                ),
                            )
            elif session is not None:
                # A hidden query tool that did not hand off a worker cannot
                # produce an asynchronous completion.  Retire only this call;
                # sibling deferred workers under the same internal parent stay
                # classified as private until their own two phases settle.
                agent = session.get("agent")
                parent_id = str(
                    getattr(agent, "_active_parent_user_message_id", "") or "")
                lock = session.get("history_lock")
                if lock is not None:
                    with lock:
                        _settle_internal_query_tool_locked(
                            session,
                            parent_id=parent_id,
                            tool_call_id=tool_call_id,
                            task_id=str(_rslt.get("task_id") or ""),
                            deferred_reply=False,
                        )
            _trace = _rslt.get("recall_trace") or []
            _findings = _rslt.get("findings") or _rslt.get("partial_findings") or ""
            if _trace or _findings:
                payload["recall_debug"] = {
                    "trace": _trace[:40] if isinstance(_trace, list) else [],
                    "findings": str(_findings or "")[:4000],
                    "found": bool(_rslt.get("found")),
                    "timed_out": bool(_rslt.get("timed_out")),
                }
        if name == "set_live_watcher":
            _ev = _rslt.get("request_id") or ""
            if _ev:
                payload["dispatch_label"] = f"🔬 已为你派发多模态深度研究 · 事件 #{_ev}"
                payload["dispatch_note"] = str(_rslt.get("note") or "").strip()
                # Carry the rid so the frontend can associate this dispatch card
                # with the watcher panel's progress/final report (same request_id).
                payload["request_id"] = _ev
        elif name == "set_monitor" and _rslt.get("op") == "create":
            _ev = _rslt.get("monitor_id") or ""
            if _ev:
                payload["dispatch_label"] = f"👁 已为你启动屏幕监控 · 事件 #{_ev}"
                payload["dispatch_note"] = str(_rslt.get("note") or "已创建监控").strip()
    summary = _tool_summary(name, result, duration_s)
    if summary:
        payload["summary"] = summary
    # Tool result ships by DEFAULT, not just in verbose mode. It used to be
    # verbose-only, which left `toolDetail` empty for almost every tool — and
    # the web UI only renders its expand triangle when there IS detail. Net
    # effect: a finished tool row looked clickable-ish but opened onto nothing,
    # so users could not see what a skill had actually returned.
    #
    # `_tool_result_text` already redacts and caps to 1000 chars / 16 lines
    # (_TUI_VERBOSE_TEXT_MAX_CHARS), so this is bounded per tool call. Verbose
    # mode still adds `args_text` on top; this is only the result.
    result_text = _tool_result_text(result)
    if result_text:
        payload["result_text"] = result_text
    if name == "todo":
        try:
            data = json.loads(result)
            if isinstance(data, dict) and isinstance(data.get("todos"), list):
                payload["todos"] = data.get("todos")
        except Exception:
            pass
    try:
        from agent.display import render_edit_diff_with_delta

        rendered: list[str] = []
        if render_edit_diff_with_delta(
            name,
            result,
            function_args=args,
            snapshot=snapshot,
            print_fn=rendered.append,
        ):
            payload["inline_diff"] = "\n".join(rendered)
    except Exception:
        pass
    if _tool_progress_enabled(sid) or payload.get("inline_diff"):
        _emit("tool.complete", sid, payload)


def _on_tool_progress(
    sid: str,
    event_type: str,
    name: str | None = None,
    preview: str | None = None,
    _args: dict | None = None,
    **_kwargs,
):
    if not _tool_progress_enabled(sid):
        return
    if event_type == "tool.started" and name:
        # `_on_tool_start` already emits the authoritative `tool.start` with
        # the stable tool id and args. Emitting another id-less progress row
        # here makes the desktop live view diverge from hydrated history.
        return
    if event_type == "reasoning.available" and preview:
        payload: dict[str, object] = {"text": str(preview)}
        if _session_verbose(sid):
            payload["verbose"] = True
        _emit("reasoning.available", sid, payload)
        return
    if event_type == "moa.reference" and name:
        # MoA reference-model output — relay as a labelled block the Ink/desktop
        # client renders before the aggregator's response (like a thinking
        # block, tagged with the source model). `name` is the slot label,
        # `preview` is the reference text.
        ref_payload: dict[str, object] = {
            "label": str(name),
            "text": str(preview or ""),
        }
        if _kwargs.get("moa_index") is not None:
            ref_payload["index"] = _kwargs.get("moa_index")
        if _kwargs.get("moa_count") is not None:
            ref_payload["count"] = _kwargs.get("moa_count")
        _emit("moa.reference", sid, ref_payload)
        return
    if event_type == "moa.aggregating":
        _emit("moa.aggregating", sid, {"aggregator": str(name or "")})
        return
    if event_type.startswith("subagent."):
        payload = {
            "goal": str(_kwargs.get("goal") or ""),
            "task_count": int(_kwargs.get("task_count") or 1),
            "task_index": int(_kwargs.get("task_index") or 0),
        }
        # Identity fields for the TUI spawn tree.  All optional — older
        # emitters that omit them fall back to flat rendering client-side.
        if _kwargs.get("subagent_id"):
            payload["subagent_id"] = str(_kwargs["subagent_id"])
        if _kwargs.get("parent_id"):
            payload["parent_id"] = str(_kwargs["parent_id"])
        if _kwargs.get("child_session_id"):
            payload["child_session_id"] = str(_kwargs["child_session_id"])
        if _kwargs.get("depth") is not None:
            payload["depth"] = int(_kwargs["depth"])
        if _kwargs.get("model"):
            payload["model"] = str(_kwargs["model"])
        if _kwargs.get("tool_count") is not None:
            payload["tool_count"] = int(_kwargs["tool_count"])
        if _kwargs.get("toolsets"):
            payload["toolsets"] = [str(t) for t in _kwargs["toolsets"]]
        # Per-branch rollups emitted on subagent.complete (features 1+2+4).
        for int_key in (
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "api_calls",
        ):
            val = _kwargs.get(int_key)
            if val is not None:
                try:
                    payload[int_key] = int(val)
                except (TypeError, ValueError):
                    pass
        if _kwargs.get("files_read"):
            payload["files_read"] = [str(p) for p in _kwargs["files_read"]]
        if _kwargs.get("files_written"):
            payload["files_written"] = [str(p) for p in _kwargs["files_written"]]
        if _kwargs.get("output_tail"):
            payload["output_tail"] = list(_kwargs["output_tail"])  # list of dicts
        if name:
            payload["tool_name"] = str(name)
        if preview:
            payload["text"] = str(preview)
        if _kwargs.get("status"):
            payload["status"] = str(_kwargs["status"])
        if _kwargs.get("summary"):
            payload["summary"] = str(_kwargs["summary"])
        if _kwargs.get("duration_seconds") is not None:
            payload["duration_seconds"] = float(_kwargs["duration_seconds"])
        if preview and event_type == "subagent.tool":
            payload["tool_preview"] = str(preview)
            payload["text"] = str(preview)
        # subagent.text is the child's per-token reply, relayed solely to feed a
        # watch window's live mirror. It is meaningless on the parent session
        # (which shows the child via the spawn tree, not its reply body), so
        # skip the parent emit — sending hundreds of ignored token frames there
        # is wasted traffic and a trap for any future parent-side subagent
        # catch-all. The mirror keys off the child sid and is unaffected.
        if event_type != "subagent.text":
            _emit(event_type, sid, payload)
        _mirror_subagent_to_child(event_type, payload)


# ── Child-session live mirror ────────────────────────────────────────
# A delegated child is not a live gateway session — it runs synchronously
# inside the parent's turn, and its activity reaches the gateway only as
# relayed ``subagent.*`` events on the PARENT sid. When a UI opens the child's
# own session (session.resume on ``child_session_id``, e.g. the desktop's
# open-in-new-window), that window would otherwise sit silent until the run
# persists. Translate the relayed events into the native stream events the
# window already renders — emitted on the CHILD sid, routed to its transport
# by write_json — so the window shows a real midstream turn.
_child_mirrors: dict[str, dict] = {}
_child_mirrors_lock = threading.Lock()
# Stored child session ids with a delegation run currently in flight (refreshed
# on every relayed subagent.* event, popped on subagent.complete). Lets a lazy
# watch resume report running=true so the window shows a busy indicator even
# while the child is silent inside a long tool call (no events for 25s+).
_active_child_runs: dict[str, float] = {}
# Staleness bound for the registry: entries refresh on every relayed event, so
# anything this quiet means the completion event was lost (callback raised,
# parent crashed) — don't let a leaked entry pin "running" forever.
_CHILD_RUN_STALE_S = 3600.0


def _child_run_active(child_key: str) -> bool:
    ts = _active_child_runs.get(child_key)
    return ts is not None and (time.time() - ts) < _CHILD_RUN_STALE_S


def _mirror_subagent_to_child(event_type: str, payload: dict) -> None:
    child_key = str(payload.get("child_session_id") or "")
    if not child_key:
        return
    # Liveness registry first — it must be accurate even when no window is
    # open, so a window opened mid-run can immediately know the child is busy.
    if event_type == "subagent.complete":
        _active_child_runs.pop(child_key, None)
    else:
        _active_child_runs[child_key] = time.time()
    # Mirror only into a live watch session (keyed by session_key; its live sid
    # differs from the stored id) that has NOT been upgraded to a full agent.
    # No window / closed → nothing to mirror; an upgraded session owns a real
    # native stream and mirroring on top would interleave two turns on one sid.
    # Either way drop state so a reopened window starts a fresh synthetic turn.
    live = _find_live_session_by_key(child_key)
    if live is None or live[1].get("agent") is not None:
        with _child_mirrors_lock:
            _child_mirrors.pop(child_key, None)
        return
    csid = live[0]
    with _child_mirrors_lock:
        st = _child_mirrors.setdefault(child_key, {"seq": 0, "open_tool": None, "started": False})
        if not st["started"]:
            st["started"] = True
            _emit("message.start", csid)
        if event_type == "subagent.thinking":
            if text := str(payload.get("text") or ""):
                _emit("reasoning.delta", csid, {"text": text})
        elif event_type == "subagent.text":
            # The child's streamed reply text — the actual "agent talking".
            # Relayed token-by-token from the child's run_conversation
            # stream_callback, so the watch window streams the reply live.
            if text := str(payload.get("text") or ""):
                _emit("message.delta", csid, {"text": text})
        elif event_type == "subagent.start":
            # One-time header line (the child's goal) so a freshly opened window
            # shows immediate context before the first reply token streams.
            if text := str(payload.get("text") or ""):
                _emit("message.delta", csid, {"text": f"{text}\n"})
        elif event_type == "subagent.tool":
            if st["open_tool"]:
                _emit("tool.complete", csid, st["open_tool"])
            st["seq"] += 1
            tool = {
                "name": str(payload.get("tool_name") or "tool"),
                "tool_id": f"submirror:{child_key}:{st['seq']}",
                "args": {},
            }
            if preview := str(payload.get("tool_preview") or payload.get("text") or ""):
                tool["preview"] = preview
            st["open_tool"] = tool
            _emit("tool.start", csid, tool)
        elif event_type == "subagent.complete":
            if st["open_tool"]:
                _emit("tool.complete", csid, st["open_tool"])
            summary = str(payload.get("summary") or payload.get("text") or "")
            _emit("message.complete", csid, {"text": summary})
            _child_mirrors.pop(child_key, None)


def _agent_cbs(sid: str) -> dict:
    return {
        "tool_start_callback": lambda tc_id, name, args: _on_tool_start(
            sid, tc_id, name, args
        ),
        "tool_complete_callback": lambda tc_id, name, args, result: _on_tool_complete(
            sid, tc_id, name, args, result
        ),
        "tool_progress_callback": lambda event_type, name=None, preview=None, args=None, **kwargs: _on_tool_progress(
            sid, event_type, name, preview, args, **kwargs
        ),
        "tool_gen_callback": lambda name: _tool_progress_enabled(sid)
        and _emit("tool.generating", sid, {"name": name}),
        # thinking_callback in this codebase is fed KAWAII spinner decorations
        # ("( ˘⌣˘)♡ contemplating...") from conversation_loop, NOT real model
        # reasoning content. Forwarding those to the web/desktop as thinking.delta
        # caused the "点开思考过程也没东西 / 只有一句 emoji 占位" issue. Drop the
        # decorative spinner: real reasoning still flows through reasoning_callback
        # below, and the FE has its own animate-pulse "Thinking…" placeholder for
        # the pre-first-token state.
        "thinking_callback": lambda text: None,
        "reasoning_callback": lambda text: _emit(
            "reasoning.delta",
            sid,
            {"text": text, **({"verbose": True} if _session_verbose(sid) else {})},
        ),
        "status_callback": lambda kind, text=None: _status_update(
            sid, str(kind), None if text is None else str(text)
        ),
        # Credits/notice spine (L1): an AgentNotice fired by the agent becomes a
        # notification.show WS event; a recovery clear becomes notification.clear.
        # Snake_case payload to match the existing gateway-event convention.
        "notice_callback": lambda n: _emit(
            "notification.show",
            sid,
            {
                "text": n.text,
                "level": n.level,
                "kind": n.kind,
                "ttl_ms": n.ttl_ms,
                "key": n.key,
                "id": n.id,
            },
        ),
        "notice_clear_callback": lambda key: _emit(
            "notification.clear", sid, {"key": key}
        ),
        "clarify_callback": lambda q, c: _block(
            "clarify.request", sid, {"question": q, "choices": c}
        ),
        # read_terminal tool (desktop GUI): same blocking bridge as clarify — the
        # renderer answers terminal.read.respond with the serialized buffer.
        "read_terminal_callback": lambda start=None, count=None: _block(
            "terminal.read.request",
            sid,
            {k: v for k, v in (("start", start), ("count", count)) if v is not None},
            timeout=30,
        ),
    }


def _apply_project_workspace(task_id: str, path: str, _name: str = "") -> None:
    """Intentional workspace move from the project_* tools: re-anchor the live
    session's cwd to the chosen project's folder and push session.info so the
    desktop follows (refresh tree + scope into the project). This is the ONLY
    auto-cwd path — driven by an explicit tool call, never a terminal `cd`."""
    if not path:
        return

    # The tool's task_id is the durable session_key, but _sessions is keyed by a
    # short sid uuid (and the desktop routes events by that sid). Resolve it.
    key = str(task_id or "")
    sid = ""
    session = None
    with _sessions_lock:
        if key in _sessions:
            sid, session = key, _sessions[key]
        else:
            for cand_sid, cand in _sessions.items():
                if cand.get("session_key") == key or getattr(cand.get("agent"), "session_id", None) == key:
                    sid, session = cand_sid, cand
                    break

    if session is None:
        return

    resolved = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.isdir(resolved):
        return

    session["cwd"] = resolved
    session["explicit_cwd"] = True
    _register_session_cwd(session)

    with _session_db(session) as db:
        if db is not None:
            try:
                db.update_session_cwd(session.get("session_key", ""), resolved)
            except Exception:
                logger.debug("failed to persist project workspace cwd", exc_info=True)

    _persist_session_git_meta(session, resolved)

    try:
        agent = session.get("agent")
        info = (
            _session_info(agent, session)
            if agent is not None
            else {"cwd": resolved, "branch": _git_branch_for_cwd(resolved), "lazy": True}
        )
        _emit("session.info", sid, info)
    except Exception:
        logger.debug("failed to emit session.info after project workspace move", exc_info=True)


def _wire_callbacks(sid: str):
    from tools.terminal_tool import set_sudo_password_callback
    from tools.skills_tool import set_secret_capture_callback
    from tools.project_tools import set_project_workspace_callback

    set_sudo_password_callback(lambda: _block("sudo.request", sid, {}, timeout=120))
    set_project_workspace_callback(_apply_project_workspace)

    def secret_cb(env_var, prompt, metadata=None):
        pl = {"prompt": prompt, "env_var": env_var}
        if metadata:
            pl["metadata"] = metadata
        val = _block("secret.request", sid, pl)
        if not val:
            return {
                "success": True,
                "stored_as": env_var,
                "validated": False,
                "skipped": True,
                "message": "skipped",
            }
        from hermes_cli.config import save_env_value_secure

        return {
            **save_env_value_secure(env_var, val),
            "skipped": False,
            "message": "ok",
        }

    set_secret_capture_callback(secret_cb)


def _render_personality_prompt(value) -> str:
    if isinstance(value, dict):
        parts = [value.get("system_prompt", "")]
        if value.get("tone"):
            parts.append(f'Tone: {value["tone"]}')
        if value.get("style"):
            parts.append(f'Style: {value["style"]}')
        return "\n".join(p for p in parts if p)
    return str(value)


def _available_personalities(cfg: dict | None = None) -> dict:
    try:
        from cli import load_cli_config

        return (load_cli_config().get("agent") or {}).get("personalities", {}) or {}
    except Exception:
        try:
            from hermes_cli.config import load_config as _load_full_cfg

            return (_load_full_cfg().get("agent") or {}).get("personalities", {}) or {}
        except Exception:
            cfg = cfg or _load_cfg()
            return (cfg.get("agent") or {}).get("personalities", {}) or {}


def _validate_personality(value: str, cfg: dict | None = None) -> tuple[str, str]:
    raw = str(value or "").strip()
    name = raw.lower()
    if not name or name in {"none", "default", "neutral"}:
        return "", ""

    personalities = _available_personalities(cfg)
    if name not in personalities:
        names = sorted(personalities)
        available = ", ".join(f"`{n}`" for n in names)
        base = f"Unknown personality: `{raw}`."
        if available:
            base += f"\n\nAvailable: `none`, {available}"
        else:
            base += "\n\nNo personalities configured."
        raise ValueError(base)

    return name, _render_personality_prompt(personalities[name])


def _prompt_text(value) -> str:
    """Normalize config prompt values from YAML before handing them to AIAgent."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def _apply_personality_to_session(
    sid: str, session: dict, new_prompt: str, personality: str = ""
) -> tuple[bool, dict | None]:
    """Apply a personality change to an existing session without resetting history.

    Updates the agent's ephemeral system prompt in-place so the new personality
    takes effect on the next turn.  The cached base system prompt is left intact
    (ephemeral_system_prompt is appended at API-call time, not baked into the
    cache), which preserves prompt-cache hits.

    Also injects a system-role marker into the conversation history so the model
    knows to pivot its style from this point forward (without this, LLMs tend to
    continue the tone established by earlier messages in the transcript).

    Returns (history_reset, info) — history_reset is always False since we
    preserve the conversation.
    """
    if not session:
        return False, None
    session["personality"] = personality

    agent = session.get("agent")
    if agent:
        agent.ephemeral_system_prompt = new_prompt or None
        # Inject a pivot marker into history so the model sees the change point.
        # This prevents it from pattern-matching its prior style.
        if new_prompt:
            marker = (
                "[System: The user has changed the assistant's personality. "
                "From this point forward, adopt the following persona and respond "
                f"accordingly: {new_prompt}]"
            )
        else:
            marker = (
                "[System: The user has cleared the personality overlay. "
                "From this point forward, respond in your normal default style.]"
            )
        with session["history_lock"]:
            session["history"].append({"role": "user", "content": marker})
            session["history_version"] = int(session.get("history_version", 0)) + 1
        info = _session_info(agent)
        _emit("session.info", sid, info)
        return False, info
    return False, None


def _cfg_max_turns(cfg: dict, default: int) -> int:
    try:
        env_max = int(os.environ.get("ARGUS_TUI_MAX_TURNS", "") or 0)
        if env_max > 0:
            return env_max
    except (TypeError, ValueError):
        pass
    agent_cfg = cfg.get("agent") or {}
    return int(agent_cfg.get("max_turns") or cfg.get("max_turns") or default)


def _parse_tui_skills_env() -> list[str]:
    raw = os.environ.get("ARGUS_TUI_SKILLS", "")
    skills: list[str] = []
    seen: set[str] = set()
    for part in raw.replace("\n", ",").split(","):
        item = part.strip()
        if item and item not in seen:
            seen.add(item)
            skills.append(item)
    return skills


def _load_fallback_model():
    """Return the configured fallback chain for TUI-created agents.

    Delegates to the shared ``get_fallback_chain`` helper so the TUI path
    stays in parity with ``HermesCLI.__init__`` and ``gateway/run.py``:
    ``fallback_providers`` is the primary source of truth and keeps its
    order, with legacy ``fallback_model`` entries merged in afterwards
    (deduped on provider/model/base_url).
    """
    from hermes_cli.fallback_config import get_fallback_chain

    return get_fallback_chain(_load_cfg())


def _agent_fallback_model(agent):
    """Return an agent's fallback chain without rehydrating deliberately empty chains."""
    if hasattr(agent, "_fallback_chain"):
        return getattr(agent, "_fallback_chain") or []
    if hasattr(agent, "_fallback_model"):
        return getattr(agent, "_fallback_model", None)
    return _load_fallback_model()


def _background_agent_kwargs(agent, task_id: str) -> dict:
    cfg = _load_cfg()

    return {
        "base_url": getattr(agent, "base_url", None) or None,
        "api_key": getattr(agent, "api_key", None) or None,
        "provider": getattr(agent, "provider", None) or None,
        "api_mode": getattr(agent, "api_mode", None) or None,
        "acp_command": getattr(agent, "acp_command", None) or None,
        "acp_args": getattr(agent, "acp_args", None) or None,
        "model": getattr(agent, "model", None) or _resolve_model(),
        "max_iterations": _cfg_max_turns(cfg, 25),
        "enabled_toolsets": getattr(agent, "enabled_toolsets", None)
        or _load_enabled_toolsets(),
        "quiet_mode": True,
        "verbose_logging": False,
        "ephemeral_system_prompt": getattr(agent, "ephemeral_system_prompt", None)
        or None,
        "providers_allowed": getattr(agent, "providers_allowed", None),
        "providers_ignored": getattr(agent, "providers_ignored", None),
        "providers_order": getattr(agent, "providers_order", None),
        "provider_sort": getattr(agent, "provider_sort", None),
        "provider_require_parameters": getattr(
            agent, "provider_require_parameters", False
        ),
        "provider_data_collection": getattr(agent, "provider_data_collection", None),
        "openrouter_min_coding_score": getattr(agent, "openrouter_min_coding_score", None),
        "session_id": task_id,
        "reasoning_config": getattr(agent, "reasoning_config", None)
        or _load_reasoning_config(),
        "service_tier": getattr(agent, "service_tier", None) or _load_service_tier(),
        "request_overrides": dict(getattr(agent, "request_overrides", {}) or {}),
        "platform": "tui",
        "session_db": _get_db(),
        "fallback_model": _agent_fallback_model(agent),
    }


def _ephemeral_preview_agent_kwargs(agent, task_id: str) -> dict:
    kwargs = _background_agent_kwargs(agent, task_id)
    kwargs.update(
        {
            "enabled_toolsets": ["terminal", "file"],
            "session_db": None,
            "skip_memory": True,
        }
    )
    return kwargs


def _preview_restart_history(session: dict, max_messages: int = 24, max_tool_chars: int = 1200) -> list[dict]:
    """Distill the parent session's recent history into a context the
    ephemeral preview-restart agent can actually use.

    The restart agent has no idea what app the user was building, what
    server they ran, what cwd was active, or which port belongs to which
    project. Without this, it would take the bare URL + console logs and
    guess — usually starting the wrong thing.

    We keep the last ``max_messages`` messages from the parent session so
    the restart agent sees recent user prompts, assistant replies, and
    most importantly any terminal/tool calls. Tool result payloads are
    truncated so we don't blow the context window with file dumps.
    """
    try:
        with session["history_lock"]:
            history = list(session.get("history", []) or [])
    except Exception:
        history = list(session.get("history", []) or [])

    if not history:
        return []

    # Anchor on the last user turn so we always include at least the most
    # recent request and the assistant/tool work that followed it. Then
    # extend backwards up to max_messages so we capture the prior context.
    last_user_idx = None
    for idx in range(len(history) - 1, -1, -1):
        if history[idx].get("role") == "user":
            last_user_idx = idx
            break

    start = max(0, len(history) - max_messages)
    if last_user_idx is not None:
        start = min(start, last_user_idx)

    trimmed: list[dict] = []
    for msg in history[start:]:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in ("user", "assistant", "tool", "system"):
            continue

        copy = {k: v for k, v in msg.items() if k != "reasoning"}
        # Truncate heavy tool outputs so a single 50KB file read doesn't
        # crowd out the rest of the context.
        if role == "tool":
            content = copy.get("content")
            if isinstance(content, str) and len(content) > max_tool_chars:
                copy["content"] = (
                    content[:max_tool_chars]
                    + f"\n... (truncated, original {len(content)} chars)"
                )
        trimmed.append(copy)

    return trimmed


def _preview_tool_result_preview(name: str, result: str) -> str:
    try:
        data = json.loads(result)
    except Exception:
        return ""

    if not isinstance(data, dict):
        return ""

    if name == "terminal":
        output = str(data.get("output") or "").strip()
        exit_code = data.get("exit_code")
        if output:
            return _redact_tui_display_text(output)[-1200:]
        if data.get("session_id"):
            return _redact_tui_display_text(
                f"Background process started: {data.get('session_id')}"
            )[-1200:]
        if exit_code is not None:
            return f"terminal exited with code {exit_code}"

    return _redact_tui_display_text(
        str(data.get("error") or "").strip()
    )[:1200]


def _preview_restart_callbacks(parent: str, task_id: str) -> dict:
    started_at: dict[str, float] = {}

    def progress(message: str, level: str = "info") -> None:
        text = str(message or "").strip()
        if text:
            _emit("preview.restart.progress", parent, {"task_id": task_id, "level": level, "text": text})

    def tool_start(tool_call_id: str, name: str, args: dict) -> None:
        started_at[tool_call_id] = time.time()
        ctx = _tool_ctx(name, args)
        progress(f"Running {name}{f': {ctx}' if ctx else ''}")

    def tool_complete(tool_call_id: str, name: str, _args: dict, result: str) -> None:
        duration_s = time.time() - started_at.get(tool_call_id, time.time())
        summary = _tool_summary(name, result, duration_s) or f"Finished {name}{f' in {_fmt_tool_duration(duration_s)}' if duration_s else ''}"
        output = _preview_tool_result_preview(name, result)
        progress(summary + (f"\n{output}" if output else ""))

    def tool_progress(event_type: str, name: str | None = None, preview: str | None = None, **_kwargs) -> None:
        if preview:
            progress(str(preview))
        elif name:
            progress(f"{event_type.replace('.', ' ')}: {name}")

    return {
        "tool_start_callback": tool_start,
        "tool_complete_callback": tool_complete,
        "tool_progress_callback": tool_progress,
        "tool_gen_callback": lambda name: progress(f"Preparing {name}"),
        "status_callback": lambda kind, text=None: progress(text if text is not None else kind),
    }


def _reset_session_agent(sid: str, session: dict) -> dict:
    tokens = _set_session_context(session["session_key"])
    try:
        new_agent = _make_agent(
            sid,
            session["session_key"],
            session_id=session["session_key"],
            # Preserve this session's chosen model across /new so a reset
            # doesn't silently revert to global config (or to a model another
            # session set). See the cross-session-contamination note in
            # _apply_model_switch.
            model_override=session.get("model_override"),
        )
    finally:
        _clear_session_context(tokens)
    session["agent"] = new_agent
    session["config_model_seen"] = _config_model_target()
    session["attached_images"] = []
    session["edit_snapshots"] = {}
    session["image_counter"] = 0
    session["running"] = False
    session["show_reasoning"] = _load_show_reasoning()
    session["tool_progress_mode"] = _load_tool_progress_mode()
    session["tool_started_at"] = {}
    with session["history_lock"]:
        session["history"] = []
        session["history_version"] = int(session.get("history_version", 0)) + 1
    info = _session_info(new_agent, session)
    _emit("session.info", sid, info)
    _restart_slash_worker(sid, session)
    return info


def _schedule_mcp_late_refresh(sid: str, agent) -> None:
    """Refresh a session's tool snapshot when MCP discovery lands late.

    The agent snapshots ``agent.tools`` once at build time and never re-reads
    the registry (run_agent/agent_init). ``_make_agent`` briefly joins the
    background MCP discovery thread (``wait_for_mcp_discovery``, bounded by the
    ``mcp_discovery_timeout`` config value, default 1.5s) so
    already-spawning servers land in that snapshot — but a server that takes
    longer than the bound to connect (common for an HTTP MCP server on first
    connect) lands *after* the agent is built. Its tools are then absent from
    both the agent and the banner for the whole session, even though the
    classic CLI shows them (the CLI re-derives ``get_tool_definitions`` at
    banner render time, which re-waits, so it picks them up).

    This schedules an off-critical-path daemon that waits for discovery to
    finish, then rebuilds the snapshot and re-emits ``session.info`` so both
    the agent's callable tools and the banner count catch up — the same
    rebuild ``/reload-mcp`` performs, but automatic.

    Cache safety: the rebuild only runs while the session is still pre-first-
    turn (no API call made yet → nothing cached to invalidate). If the user
    has already sent a message, we leave the snapshot frozen rather than
    invalidate the prompt cache mid-conversation — those late tools then
    require an explicit ``/reload-mcp`` (which gates on user consent), exactly
    as today. No-op when discovery already finished before the agent build.
    """
    try:
        from tui_gateway.entry import mcp_discovery_in_flight, join_mcp_discovery
    except Exception:
        return
    if not mcp_discovery_in_flight():
        return

    def _wait_then_refresh() -> None:
        # Bounded but generous — a server still not connected after this is
        # genuinely slow/dead; the user can /reload-mcp once it recovers.
        if not join_mcp_discovery(timeout=30.0):
            return
        with _sessions_lock:
            session = _sessions.get(sid)
            # Session may have been closed/reset while we waited.
            if session is None or session.get("agent") is not agent:
                return
            # Cache safety: never rebuild the tool list once the conversation
            # has started — that would invalidate the cached prompt prefix.
            if (
                int(getattr(agent, "_user_turn_count", 0) or 0) > 0
                or int(getattr(agent, "_api_call_count", 0) or 0) > 0
            ):
                return
            try:
                from tools.mcp_tool import refresh_agent_mcp_tools

                added = refresh_agent_mcp_tools(agent, quiet_mode=True)
            except Exception as exc:
                logger.warning(
                    "Late MCP refresh: tool snapshot rebuild failed for %s: %s",
                    sid,
                    exc,
                )
                return
            # No new tools landed (discovery added nothing) → don't churn the client.
            if not added:
                return
            info = _session_info(agent, session)
        # Emit outside the lock — write_json must not block under _sessions_lock.
        _emit("session.info", sid, info)
    threading.Thread(
        target=_wait_then_refresh,
        name=f"tui-mcp-late-refresh-{sid}",
        daemon=True,
    ).start()


def _resolve_runtime_with_fallback(
    resolve_kwargs: dict | None = None,
) -> dict:
    """Resolve runtime provider with init-time fallback on auth failure.

    Mirrors the fallback pattern in ``cron/scheduler.py`` and
    ``hermes_cli/cli_agent_setup_mixin.py``: when the primary provider
    raises ``AuthError``, walk the configured ``fallback_providers`` /
    ``fallback_model`` chain before giving up.
    """
    from hermes_cli.auth import AuthError
    from hermes_cli.runtime_provider import resolve_runtime_provider

    kwargs = resolve_kwargs or {}
    try:
        return resolve_runtime_provider(**kwargs)
    except AuthError as primary_exc:
        fb_chain = _load_fallback_model() or []
        for entry in fb_chain:
            if not isinstance(entry, dict):
                continue
            fb_provider = (entry.get("provider") or "").strip()
            if not fb_provider:
                continue
            try:
                fb_kwargs: dict = {"requested": fb_provider}
                if entry.get("base_url"):
                    fb_kwargs["explicit_base_url"] = entry["base_url"]
                if entry.get("api_key"):
                    fb_kwargs["explicit_api_key"] = entry["api_key"]
                runtime = resolve_runtime_provider(**fb_kwargs)
                import logging

                logging.getLogger(__name__).warning(
                    "Primary auth failed (%s), falling back to %s",
                    primary_exc,
                    fb_provider,
                )
                return runtime
            except Exception:
                continue
        raise


def _make_agent(
    sid: str,
    key: str,
    session_id: str | None = None,
    session_db=None,
    model_override: dict | str | None = None,
    provider_override: str | None = None,
    reasoning_config_override: dict | None = None,
    service_tier_override: str | None = None,
    skip_context_files: bool | None = None,
):
    from run_agent import AIAgent

    # MCP tool discovery runs in a background daemon thread at startup so a
    # dead server can't freeze the shell.  The agent snapshots its tool list
    # once here and never re-reads it, so briefly wait for in-flight discovery
    # to land before building — bounded, so a slow/dead server still can't
    # block. Dashboard /api/ws uses hermes_cli.mcp_startup; TUI stdio keeps
    # its existing tui_gateway.entry-owned thread.
    try:
        from hermes_cli.mcp_startup import wait_for_mcp_discovery

        wait_for_mcp_discovery()
    except Exception:
        pass
    try:
        from tui_gateway.entry import wait_for_mcp_discovery

        wait_for_mcp_discovery()
    except Exception:
        pass

    cfg = _load_cfg()
    agent_cfg = cfg.get("agent") or {}
    system_prompt = _prompt_text(agent_cfg.get("system_prompt", ""))
    startup_skills = _parse_tui_skills_env()
    loaded_skills: list[str] = []
    if startup_skills:
        from agent.skill_commands import build_preloaded_skills_prompt

        skills_prompt, loaded_skills, missing_skills = build_preloaded_skills_prompt(
            startup_skills,
            task_id=session_id or key,
        )
        if missing_skills:
            raise ValueError(f"Unknown skill(s): {', '.join(missing_skills)}")
        if skills_prompt:
            system_prompt = "\n\n".join(
                part for part in (system_prompt, skills_prompt) if part
            ).strip()
    # Prefer a per-session model override (set by a prior in-session /model
    # switch) over global config/env resolution. Resume-time stored sessions may
    # also pass scalar model/provider/runtime knobs from the persisted DB row.
    if isinstance(model_override, dict) and model_override.get("model"):
        model = str(model_override.get("model") or "")
        requested_provider = model_override.get("provider") or provider_override or None
        override_base_url = model_override.get("base_url")
        override_api_key = model_override.get("api_key")
        override_api_mode = model_override.get("api_mode")
        resolve_kwargs = {}
        if str(requested_provider or "").strip().lower() == "custom":
            # Session rows persisted before the custom-provider identity fix
            # (see _runtime_model_config) stored the resolved provider
            # "custom", which _get_named_custom_provider cannot match back to
            # a named ``providers:`` / ``custom_providers:`` entry — the
            # rebuild then either raised auth_unavailable, silently resolved
            # placeholder credentials against the patched-back base_url, or
            # (when no base_url was stored) routed to the OpenRouter default
            # with no key, surfacing as "No LLM provider configured". Recover
            # the entry identity from the persisted base_url, falling back to
            # the configured provider when the override carries no base_url
            # (the recurring Desktop/TUI regression vector).
            from hermes_cli.runtime_provider import canonical_custom_identity

            recovered = canonical_custom_identity(base_url=override_base_url or None)
            if recovered:
                requested_provider = recovered
            if override_base_url:
                # Failing identity recovery, still hand the base_url to the
                # direct-alias branch so pool/env credentials resolve for it.
                resolve_kwargs["explicit_base_url"] = override_base_url
        resolve_kwargs["requested"] = requested_provider
        resolve_kwargs["target_model"] = model or None
        runtime = _resolve_runtime_with_fallback(resolve_kwargs)
        # The switch already resolved concrete credentials/endpoint; honor them
        # so a custom/named endpoint survives the rebuild even if global
        # resolution would pick a different one.
        if override_base_url:
            runtime["base_url"] = override_base_url
        if override_api_key:
            runtime["api_key"] = override_api_key
        if override_api_mode:
            runtime["api_mode"] = override_api_mode
    else:
        model, requested_provider = _resolve_startup_runtime()
        if isinstance(model_override, str) and model_override:
            model = model_override
        if provider_override:
            requested_provider = provider_override
        runtime = _resolve_runtime_with_fallback({
            "requested": requested_provider,
            "target_model": model or None,
        })
    _pr = _load_provider_routing()
    # Computer use (desktop control via cua-driver) is an EXPLICIT opt-in
    # capability: it lives in its own `computer_use` toolset (removed from
    # _HERMES_CORE_TOOLS, off by default via _DEFAULT_OFF_TOOLSETS) and is
    # only enabled here when the computer-use skill is explicitly preloaded
    # (ARGUS_TUI_SKILLS=computer-use, or an identifier resolving to it).
    enabled_toolsets = _load_enabled_toolsets()
    disabled_toolsets = None
    _cu_skill_preloaded = any(
        str(s or "").strip().lower() in {"computer-use", "computer_use"}
        for s in (*startup_skills, *loaded_skills)
    )
    if enabled_toolsets is not None:
        if _cu_skill_preloaded:
            if "computer_use" not in enabled_toolsets:
                enabled_toolsets = [*enabled_toolsets, "computer_use"]
        else:
            filtered = [t for t in enabled_toolsets if t != "computer_use"]
            if len(filtered) != len(enabled_toolsets):
                enabled_toolsets = filtered
    elif not _cu_skill_preloaded:
        # enabled_toolsets=None expands to EVERY toolset; the only way to keep
        # the opt-in tool out of that expansion is an explicit disabled entry.
        disabled_toolsets = ["computer_use"]
    return AIAgent(
        model=model,
        max_iterations=_cfg_max_turns(cfg, 90),
        provider=runtime.get("provider"),
        base_url=runtime.get("base_url"),
        api_key=runtime.get("api_key"),
        api_mode=runtime.get("api_mode"),
        acp_command=runtime.get("command"),
        acp_args=runtime.get("args"),
        credential_pool=runtime.get("credential_pool"),
        quiet_mode=True,
        # verbose_logging controls DEBUG-level agent logging; it is intentionally
        # independent of tool_progress_mode (which only controls per-tool
        # display detail).  See cli.py PR (decoupling fix) for the matching
        # change on the classic CLI side.
        verbose_logging=False,
        reasoning_config=(
            reasoning_config_override
            if reasoning_config_override is not None
            else _load_reasoning_config()
        ),
        service_tier=(
            service_tier_override
            if service_tier_override is not None
            else _load_service_tier()
        ),
        enabled_toolsets=enabled_toolsets,
        disabled_toolsets=disabled_toolsets,
        # OpenRouter provider-routing prefs (config.yaml `provider_routing`).
        # Mirrors the messaging gateway + CLI so the desktop/TUI honors the same
        # routing instead of letting OpenRouter pick providers at random.
        providers_allowed=_pr.get("only"),
        providers_ignored=_pr.get("ignore"),
        providers_order=_pr.get("order"),
        provider_sort=_pr.get("sort"),
        provider_require_parameters=_pr.get("require_parameters", False),
        provider_data_collection=_pr.get("data_collection"),
        platform="tui",
        session_id=session_id or key,
        session_db=session_db if session_db is not None else _get_db(),
        ephemeral_system_prompt=system_prompt or None,
        checkpoints_enabled=is_truthy_value(os.environ.get("ARGUS_TUI_CHECKPOINTS")),
        pass_session_id=is_truthy_value(os.environ.get("ARGUS_TUI_PASS_SESSION_ID")),
        skip_context_files=(skip_context_files if skip_context_files is not None
                            else is_truthy_value(os.environ.get("ARGUS_IGNORE_RULES"))),
        skip_memory=is_truthy_value(os.environ.get("ARGUS_IGNORE_RULES")),
        fallback_model=_load_fallback_model(),
        **_agent_cbs(sid),
    )


def _init_session(
    sid: str,
    key: str,
    agent,
    history: list,
    cols: int = 80,
    cwd: str | None = None,
    session_db=None,
):
    now = time.time()
    with _sessions_lock:
        _sessions[sid] = {
            "agent": agent,
            "session_key": key,
            "history": history,
            "history_lock": threading.Lock(),
            "history_version": 0,
            "inflight_turn": None,
            "queued_prompt": None,
            "queued_prompts": [],
            "created_at": now,
            "last_active": now,
            "running": False,
            "attached_images": [],
            "image_counter": 0,
            "cwd": cwd or _completion_cwd(),
            "cols": cols,
            "slash_worker": None,
            "show_reasoning": _load_show_reasoning(),
            "tool_progress_mode": _load_tool_progress_mode(),
            "edit_snapshots": {},
            "tool_started_at": {},
            # Per-session model override set by an in-session /model switch.
            # Honored on rebuild (/new, resume) so a switch in THIS session
            # never leaks into siblings via process-global env vars.
            "model_override": None,
            # Pin async event emissions to whichever transport created the
            # session (stdio for Ink, JSON-RPC WS for the dashboard sidebar).
            "transport": current_transport() or _stdio_transport,
        }
    # ★ Resume 去重修复 (eager 路径): 恢复的历史已在 DB, 落盘游标必须钉到末尾 +
    #   预种身份集, 否则首个 turn 的 _flush_messages_to_session_db 会把整段历史当
    #   "新消息"再 append 一遍 → reopen 后消息翻倍。deferred 路径在 _start_agent_build
    #   里同样处理。全新 session(history 空)不进此分支, 行为不变。
    if agent is not None and isinstance(history, list) and history:
        try:
            agent._last_flushed_db_idx = len(history)
            agent._flushed_db_message_ids = {
                id(m) for m in history if isinstance(m, dict)
            }
            agent._flushed_db_message_session_id = getattr(agent, "session_id", None)
        except Exception:
            logger.debug("resume flush-cursor sync (init) failed", exc_info=True)
    db = session_db if session_db is not None else _get_db()
    if db is not None:
        row = db.get_session(key)
        if row and row.get("cwd"):
            with _sessions_lock:
                if sid in _sessions:
                    _sessions[sid]["cwd"] = row["cwd"]
        else:
            try:
                _cwd = _sessions[sid]["cwd"]
                db.update_session_cwd(key, _cwd)
                # git branch/root probes run off the hot path (see _set_session_cwd).
                _persist_session_git_meta(_sessions[sid], _cwd)
            except Exception:
                logger.debug("failed to persist resumed session cwd", exc_info=True)
    _register_session_cwd(_sessions[sid])
    try:
        _attach_worker(
            sid,
            _sessions[sid],
            _SlashWorker(key, getattr(agent, "model", _resolve_model())),
        )
    except Exception:
        # Defer hard-failure to slash.exec; chat still works without slash worker.
        _sessions[sid]["slash_worker"] = None
    try:
        from tools.approval import register_gateway_notify, load_permanent_allowlist

        register_gateway_notify(key, lambda data: _emit_approval_request(sid, data))
        load_permanent_allowlist()
    except Exception:
        pass
    # Surface the self-improvement background review's "💾 …" summary as a
    # review.summary event so Ink can render it as a persistent system line
    # in the transcript. In the CLI path this message is printed via
    # prompt_toolkit; the TUI has no equivalent print surface, so without
    # this callback the review would write the skill/memory change silently.
    try:
        agent.background_review_callback = lambda message, _sid=sid: _emit(
            "review.summary", _sid, {"text": str(message)}
        )
        # Honor display.memory_notifications (off | on | verbose) like the
        # messaging gateway and CLI do — otherwise the review always behaved as
        # "on" on the TUI/desktop and a user who set "off" was ignored.
        agent.memory_notifications = _load_memory_notifications()
    except Exception:
        # Bare AIAgents that don't expose the attribute (unlikely, but keep
        # session startup resilient).
        pass
    _wire_callbacks(sid)
    with _sessions_lock:
        if sid in _sessions:
            _sessions[sid]["_notif_stop"] = _start_notification_poller(sid, _sessions[sid])
            _maybe_start_monitor_engine(
                sid, _sessions[sid], getattr(agent, "frame_buffer", None))
    _notify_session_boundary("on_session_reset", key)
    _emit("session.info", sid, _session_info(agent, _sessions.get(sid, {})))
    _schedule_mcp_late_refresh(sid, agent)


def _new_session_key() -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def _with_checkpoints(session, fn):
    return fn(session["agent"]._checkpoint_mgr, _session_cwd(session))


def _resolve_checkpoint_hash(mgr, cwd: str, ref: str) -> str:
    try:
        checkpoints = mgr.list_checkpoints(cwd)
        idx = int(ref) - 1
    except ValueError:
        return ref
    if 0 <= idx < len(checkpoints):
        return checkpoints[idx].get("hash", ref)
    raise ValueError(f"Invalid checkpoint number. Use 1-{len(checkpoints)}.")


def _content_display_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (int, float)):
        return str(content)
    if isinstance(content, list):
        parts = []
        for part in content:
            text = _content_display_text(part).strip()
            if text:
                parts.append(text)
        return "\n".join(parts)
    if isinstance(content, dict):
        kind = content.get("type")
        if kind in {"text", "input_text", "output_text"}:
            return str(content.get("text") or content.get("content") or "")
        if kind in {"image_url", "input_image", "image"}:
            return "[image]"
        if kind in {"input_audio", "audio"}:
            return "[audio]"
        if kind:
            return f"[{kind}]"
        if "text" in content:
            return str(content.get("text") or "")
        return "[structured content]"
    return str(content)


def _coerce_message_text(content: Any) -> str:
    """Render ``message['content']`` as a plain string for transport.

    Provider-side, ``content`` may be a string (most common), a list of
    multimodal parts (e.g. ``[{"type": "text", "text": "..."},
    {"type": "image_url", "image_url": {...}}]``), or a single structured
    dict. Calling ``.strip()`` on a list raises ``'list' object has no
    attribute 'strip'`` and breaks session resume entirely.

    Image parts (``image_url``) are preserved by appending the underlying
    URL (data: or http:) into the text. The desktop renderer pulls these
    back out via ``extractEmbeddedImages`` so the user sees the image
    instead of the URL — and it stops the resume payload from disagreeing
    with the cached message (which would otherwise cause the inline image
    to flash, then disappear when the resume payload overwrites the cache).

    Other structured dict shapes (audio, unknown types) fall back to a
    bracketed placeholder so resume doesn't drop the message entirely.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (int, float)):
        return str(content)
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
                continue
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str):
                chunks.append(text)
                continue
            kind = part.get("type")
            if kind in {"text", "input_text", "output_text"}:
                t = part.get("text") or part.get("content") or ""
                if t:
                    chunks.append(str(t))
                continue
            if kind in {"image_url", "input_image", "image"}:
                image_url = part.get("image_url")
                url = ""
                if isinstance(image_url, dict):
                    candidate = image_url.get("url")
                    if isinstance(candidate, str):
                        url = candidate
                elif isinstance(image_url, str):
                    url = image_url
                if url:
                    chunks.append(f"\n{url}")
                else:
                    chunks.append("\n[image]")
                continue
            if kind in {"input_audio", "audio"}:
                chunks.append("\n[audio]")
                continue
            if kind:
                chunks.append(f"\n[{kind}]")
        return "".join(chunks)
    if isinstance(content, dict):
        kind = content.get("type")
        if kind in {"text", "input_text", "output_text"}:
            return str(content.get("text") or content.get("content") or "")
        if kind in {"image_url", "input_image", "image"}:
            image_url = content.get("image_url")
            url = ""
            if isinstance(image_url, dict):
                candidate = image_url.get("url")
                if isinstance(candidate, str):
                    url = candidate
            elif isinstance(image_url, str):
                url = image_url
            return url or "[image]"
        if kind in {"input_audio", "audio"}:
            return "[audio]"
        if kind:
            return f"[{kind}]"
        if "text" in content:
            return str(content.get("text") or "")
        return "[structured content]"
    return str(content)


def _history_to_messages(history: list[dict]) -> list[dict]:
    messages = []
    tool_call_args = {}

    for m in history:
        if not isinstance(m, dict):
            continue
        # QueryWorker answers are projected to the main model as normal Q/A
        # history, but the browser already rendered the original user bubble and
        # the worker-owned answer in their correlated slots. Suppress only this
        # internal duplicate representation from history/reopen rendering.
        if _mm_query_projection_fields(m) is not None:
            continue
        # ★ monitor/watcher/query-worker 通知元素 (type=mm_notice) →
        #   重建成前端气泡描述。
        #   reopen 时把它们还原为对应 worker subRole 的消息, 使全量
        #   context 在前端与实时收到的气泡长得一样。放在 role 白名单校验【之前】,
        #   因为持久态可能只剩 content dict、顶层 role 仍是 assistant。
        if _is_mm_notice(m):
            f = _mm_notice_fields(m)
            _mm_text = (f.get("text") or "").strip()
            if not _mm_text:
                continue
            _kind = f.get("kind") or "monitor"
            # Monitor + watcher notices are no longer replayed as chat bubbles —
            # they hydrate the right multimodal panel via list_mm_monitor_alerts
            # / list_mm_watcher_reports RPCs instead. Legacy rows from before
            # the sidechannel split are simply skipped here (they still live in
            # session["history"] as mm_notice, invisible to the LLM and to UI).
            if _kind in ("monitor", "watcher"):
                continue
            _sub = (
                "watcher_report" if _kind == "watcher"
                else "query_worker" if _kind == "query"
                else "monitor"
            )
            _msg = {
                "role": "user" if _kind == "query_user" else "assistant",
                "text": _mm_text,
                "monitorLabel": f.get("label") or "",
                "eventId": f.get("event_id") or "",
            }
            if _kind != "query_user":
                _msg["subRole"] = _sub
            if _kind == "watcher":
                _msg["deepReportRid"] = f.get("event_id") or ""
                if f.get("round") is not None:
                    _msg["deepRound"] = f.get("round")
            elif _kind == "query":
                _msg["requestId"] = f.get("event_id") or ""
            elif _kind == "query_user":
                _msg["requestId"] = f.get("event_id") or ""
            # ★ 带上源元素的 timestamp (秒), 前端据此还原气泡时间; 否则 reopen 后
            #   monitor/watcher 气泡时间会被前端兜底成"打开那一刻"。
            if m.get("timestamp"):
                _msg["timestamp"] = m.get("timestamp")
            messages.append(_msg)
            continue
        role = m.get("role")
        if role not in {"user", "assistant", "tool", "system"}:
            continue
        content_text = _coerce_message_text(m.get("content"))
        if role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                tc_id = tc.get("id", "")
                if tc_id and fn.get("name"):
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    tool_call_args[tc_id] = (fn["name"], args)
            if not content_text.strip():
                continue
        if role == "tool":
            tc_id = m.get("tool_call_id", "")
            tc_info = tool_call_args.get(tc_id) if tc_id else None
            name = (tc_info[0] if tc_info else None) or m.get("tool_name") or "tool"
            args = (tc_info[1] if tc_info else None) or {}
            # `context` is the CALL side (a preview of args — the command that
            # ran). The tool's own return value lives in m["content"] and used
            # to be dropped here, so a resumed session showed "✓ terminal" with
            # an empty disclosure whose only content was the 80-char command
            # echoed back. Ship a capped projection of the result as `content`
            # so history matches the live stream. Omitted when empty rather
            # than sent as "" — clients key "has detail" off its presence.
            body, result_summary = _history_tool_result(name, m.get("content"))
            row = {"role": "tool", "name": name, "context": _tool_ctx(name, args)}
            # Same classified args the live tool.start carries, so a reopened
            # session's tool rows expand to the same detail instead of being
            # strictly poorer than the stream that produced them.
            arg_fields = _tool_arg_fields(name, args)
            if arg_fields:
                row["args_fields"] = arg_fields
            if tc_id:
                row["tool_call_id"] = tc_id
            if body:
                row["content"] = body
            if result_summary:
                row["summary"] = result_summary
            messages.append(row)
            continue
        # An assistant turn may carry only reasoning/thinking content with no
        # visible text (extended-thinking turns, thinking-only recovery
        # responses). Such a turn is persisted with its reasoning fields and is
        # recallable from the transcript, but dropping it here as "empty" makes
        # it vanish from the resumed/reloaded session view while the desktop's
        # reasoning disclosure has nothing to render. Keep it when it carries
        # reasoning so the "Thinking…" block still shows. (#44022)
        reasoning_keys = (
            "reasoning",
            "reasoning_content",
            "reasoning_details",
            "codex_reasoning_items",
        )
        has_reasoning = role == "assistant" and any(
            m.get(key) for key in reasoning_keys
        )
        if not content_text.strip() and not has_reasoning:
            continue
        msg = {"role": role, "text": content_text}
        if role == "assistant":
            for key in reasoning_keys:
                if key in m and m.get(key) is not None:
                    msg[key] = m.get(key)
        messages.append(msg)

    return messages


def _coerce_seed_history(value: Any) -> list[dict]:
    if not isinstance(value, list):
        return []

    history = []
    for item in value:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        if role not in ("user", "assistant", "system"):
            continue

        content = item.get("content")
        if content is None:
            content = item.get("text")
        if not isinstance(content, str) or not content.strip():
            continue

        history.append({"role": role, "content": content})

    return history


def _inflight_text(value: Any) -> str:
    return _content_display_text(value).strip()


def _start_inflight_turn(session: dict, text: Any) -> None:
    now = time.time()
    session["inflight_turn"] = {
        "assistant": "",
        "started_at": now,
        "streaming": True,
        "updated_at": now,
        "user": _inflight_text(text),
    }


def _append_inflight_delta(session: dict, delta: Any) -> None:
    text = "" if delta is None else str(delta)
    if not text:
        return
    turn = session.get("inflight_turn")
    if not isinstance(turn, dict):
        turn = {"assistant": "", "streaming": True, "user": ""}
    turn["assistant"] = f"{turn.get('assistant') or ''}{text}"
    turn["streaming"] = True
    turn["updated_at"] = time.time()
    session["inflight_turn"] = turn


def _clear_inflight_turn(session: dict) -> None:
    session["inflight_turn"] = None


def _resolve_voice_task(session: dict, task_seq: Any, text: str) -> None:
    """Resolve a queued/active VoiceAgent Future without constructing a worker."""
    if task_seq is None:
        return
    voice = session.get("_mm_voice_agent")
    if voice is None:
        return
    try:
        voice.notify_main_reply(int(task_seq), text)
    except Exception as exc:
        logger.debug("voice task resolution failed seq=%s: %s", task_seq, exc)


def _claim_composer_attachments(session: dict, *, user_originated: bool) -> list:
    """Take the images staged in the composer, but only for their owner.

    Only a prompt the human actually sent owns those attachments. Handing them
    to anything else both exposes the images to an unrelated prompt and makes
    the user's next real submit silently lose them.

    ★ Stated as a positive requirement, not as a list of exemptions. This used
      to read ``if internal_origin: images = []``, which spared only the two
      hidden hooks that happened to set that flag — every other machine-injected
      prompt (async-delegation completions, background process notifications,
      watch-pattern matches, goal follow-ups) still swallowed the composer's
      attachments, and each new injected kind would have had to remember to opt
      out. Keying on ``user_originated`` makes the safe answer the default: that
      parameter already defaults to False, so anything injected is covered for
      free.

    Callers must hold ``session["history_lock"]``.
    """
    if not user_originated:
        return []
    images = list(session.get("attached_images", []))
    session["attached_images"] = []
    return images


def _enqueue_prompt(
    session: dict,
    text: Any,
    transport: Any,
    *,
    user_originated: bool = True,
    origin: str = "user",
    metadata: Optional[dict] = None,
) -> int:
    """Append one independent foreground turn to the per-session FIFO.

    A busy session used to have one ``queued_prompt`` slot. A second submit was
    concatenated into the first, which changed two user questions into one model
    turn and made it impossible to stress-test or inspect their trajectories
    independently. Keep every submit as its own item instead. ``transport`` is
    pinned so each drained turn streams back to the client that sent it.

    ``queued_prompt`` remains a compatibility alias for older status/debug code;
    the authoritative state is ``queued_prompts``. Returns the 1-based position.
    """
    queue = session.setdefault("queued_prompts", [])
    # Migrate an in-memory session created by an older server revision without
    # losing its already accepted prompt.
    legacy = session.get("queued_prompt")
    if legacy and not queue:
        queue.append(legacy)
    item = {
        "text": text,
        "transport": transport,
        "user_originated": bool(user_originated),
        "origin": str(origin or "user"),
        "queued_at": time.time(),
    }
    if metadata:
        item.update(dict(metadata))
    queue.append(item)
    # ★ 队列上限 50 (防无限堆积): 超限从队头丢最旧的。被丢的若是 VoiceAgent 委派,
    #   必须 resolve 它的 Future (否则 VoiceAgent 那边 await 到 300s 才超时)。
    _MAX_QUEUED = 50
    while len(queue) > _MAX_QUEUED:
        dropped = queue.pop(0)
        _dseq = dropped.get("voice_task_seq")
        if _dseq is not None:
            _resolve_voice_task(session, _dseq, "队列已满, 较早的语音指令被丢弃")
        logger.warning("[queue] queued_prompts over %d, dropped oldest (origin=%s)",
                       _MAX_QUEUED, dropped.get("origin", "user"))
    session["queued_prompt"] = queue[0] if queue else None
    return len(queue)


def _log_busy_diagnosis(sid: str, session: dict) -> None:
    """When a new prompt lands on a busy session, dump WHY the prior turn is
    still running. Screen-share stall investigation: users hit "queued 1min"
    and can't tell if the previous turn is genuinely still LLM-streaming or
    if session["running"] leaked. Emits one info line with:
      - the last per-stage milestone the previous turn reached (set by
        _run_prompt_submit's _trace helper — stashed on session)
      - seconds since that milestone (so `stalled_for_s=45` is a red flag)
      - inflight_turn's last delta timestamp if any
    """
    import time as _t
    _sid_short = (sid or "-")[-6:]
    stage = session.get("_mm_last_stage") or "<unknown>"
    stage_ts = session.get("_mm_last_stage_ts") or 0.0
    stalled = (_t.monotonic() - stage_ts) if stage_ts else -1.0
    turn = session.get("inflight_turn") or {}
    last_delta = turn.get("last_delta_at") or 0.0
    since_delta = (_t.time() - last_delta) if last_delta else -1.0
    logger.warning(
        "[mm-busy] sid=%s new prompt queued; prior turn last_stage=%r "
        "stalled_for_s=%.1f last_delta_ago_s=%.1f "
        "inflight_streaming=%s",
        _sid_short, stage, stalled, since_delta,
        turn.get("streaming"),
    )


def _handle_busy_submit(
    rid,
    sid: str,
    session: dict,
    text: Any,
    transport: Any,
    *,
    queue_only: bool = False,
    metadata: Optional[dict] = None,
) -> dict:
    """Apply the ``display.busy_input_mode`` policy to a prompt that lands while
    a turn is in flight, instead of rejecting it with ``session busy``.

    The old rejection forced clients into a deadline-bounded busy-retry that
    silently dropped the send when turn teardown outlived the deadline (e.g. a
    slow, non-interruptible tool like ``web_search`` running when the user hits
    stop). The message is instead queued to run as the next turn — and, for the
    default ``interrupt`` policy, the live turn is interrupted so it winds down
    promptly. Drained in ``run``'s tail (see ``_run_prompt_submit``).

    Modes: ``interrupt`` (default) → interrupt + queue; ``queue`` → queue
    without interrupting; ``steer`` → inject into the live turn if accepted,
    else queue.
    """
    mode = "queue" if queue_only else _load_busy_input_mode()
    agent = session.get("agent")
    # ★ Never interrupt a monitor-hook turn. A hook fire runs a full main-agent
    # turn in the background (_run_hook_turn → _run_prompt_submit); if the user
    # sends a prompt while it's mid-API-call, the default `interrupt` mode would
    # abort it with "Operation interrupted: waiting for model response". The user
    # prompt isn't trying to stop the hook — it just landed concurrently. So for
    # a hook turn we FORCE queue-without-interrupt: the hook finishes, then its
    # _run_prompt_submit tail drains this queued prompt. (Interrupting a normal
    # user turn — the user redirecting themselves — is unchanged.)
    if session.get("_monitor_hook_running"):
        position = _enqueue_prompt(session, text, transport, metadata=metadata)
        session["last_active"] = time.time()
        _emit("multimodal.trajectory", sid, {
            "worker": "MainScheduler", "phase": "prompt_queued",
            "queue_position": position, "origin": "user",
            "text": _inflight_text(text), "reason": "monitor_hook_running",
            "client_request_id": str(
                (metadata or {}).get("client_request_id") or ""),
        })
        return _ok(rid, {
            "status": "queued", "queue_position": position,
            **({"client_request_id": metadata.get("client_request_id")}
               if metadata and metadata.get("client_request_id") else {}),
        })
    if mode == "steer" and agent is not None and hasattr(agent, "steer"):
        try:
            if agent.steer(text):
                session["last_active"] = time.time()
                return _ok(rid, {"status": "steered"})
        except Exception:
            pass  # fall through to queue
    if mode != "queue" and agent is not None and hasattr(agent, "interrupt"):
        try:
            agent.interrupt()
        except Exception:
            pass
    position = _enqueue_prompt(session, text, transport, metadata=metadata)
    session["last_active"] = time.time()
    _emit("multimodal.trajectory", sid, {
        "worker": "MainScheduler", "phase": "prompt_queued",
        "queue_position": position, "origin": "user",
        "text": _inflight_text(text), "busy_mode": mode,
        "client_request_id": str(
            (metadata or {}).get("client_request_id") or ""),
    })
    return _ok(rid, {
        "status": "queued", "queue_position": position,
        **({"client_request_id": metadata.get("client_request_id")}
           if metadata and metadata.get("client_request_id") else {}),
    })


def _drain_queued_prompt(rid, sid: str, session: dict) -> bool:
    """Fire a queued next-turn prompt if one is waiting and the session is idle.

    Returns True if a queued prompt was dispatched (the caller should then skip
    lower-priority follow-ups this cycle — the user's message wins). Mirrors the
    claim-under-lock pattern used by the goal-continuation re-fire.
    """
    with session["history_lock"]:
        queue = session.setdefault("queued_prompts", [])
        legacy = session.get("queued_prompt")
        if legacy and not queue:
            queue.append(legacy)
        if not queue or session.get("running"):
            return False
        queued = queue.pop(0)
        queued_origin = str(queued.get("origin") or "user")
        internal_origin = (
            "monitor_hook" if queued_origin == "monitor_hook" else ""
        )
        session["queued_prompt"] = queue[0] if queue else None
        session["running"] = True
        # Monitor hooks are hidden internal turns. Claim their interrupt guard
        # under the same lock as ``running`` so a user prompt cannot land in the
        # gap between dequeue and _run_prompt_submit and abort the hook. The turn
        # releases this flag in its own finally block below.
        if internal_origin:
            session["_monitor_hook_running"] = True
        if queued.get("voice_task_seq") is not None:
            session["_voice_active_seq"] = queued.get("voice_task_seq")
        if queued.get("voice_input"):
            session["_mm_voice_turn"] = True
        if queued.get("transport") is not None:
            session["transport"] = queued["transport"]
    try:
        # A queued voice utterance deliberately withheld its user bubble at
        # submit time so the transcript never showed `user, user` (see
        # _submit_main). Its turn starts now, so paint it now — same event and
        # same request id, so the answer stream still correlates.
        if queued_origin == "voice_agent" and str(queued.get("text") or "").strip():
            _voice_turn_id = str(queued.get("voice_turn_id") or "")
            _emit("multimodal.asr_final",
                  str(queued.get("voice_live_sid") or sid), {
                      "text": queued.get("text"),
                      "request_id": str(queued.get("client_request_id") or ""),
                      **({"turn_id": _voice_turn_id} if _voice_turn_id else {}),
                  })
        _emit("multimodal.trajectory", sid, {
            "worker": "MainScheduler", "phase": "prompt_dequeued",
            "origin": queued_origin,
            "queued_for_sec": max(0.0, time.time() - float(
                queued.get("queued_at") or time.time())),
            "remaining": len(session.get("queued_prompts") or []),
            "text": _inflight_text(queued.get("text")),
            "client_request_id": str(queued.get("client_request_id") or ""),
        })
        run_kwargs = {
            "user_originated": bool(queued.get("user_originated", True)),
            "client_request_id": str(queued.get("client_request_id") or ""),
            "internal_origin": internal_origin,
        }
        if queued.get("anchor_frozen"):
            run_kwargs.update({
                "anchor_ts": _optional_finite_float(queued.get("anchor_ts")),
                "anchor_frozen": True,
            })
        _run_prompt_submit(
            rid, sid, session, queued["text"], **run_kwargs)
    except Exception as exc:
        print(
            f"[tui_gateway] queued prompt dispatch failed: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        voice_seq = queued.get("voice_task_seq")
        with session["history_lock"]:
            session["running"] = False
            if internal_origin:
                session["_monitor_hook_running"] = False
            if session.get("_voice_active_seq") == voice_seq:
                session.pop("_voice_active_seq", None)
        _resolve_voice_task(
            session, voice_seq,
            f"主 Agent 队列任务启动失败：{type(exc).__name__}: {exc}")
    return True


# ── Live-watcher completion hook: queued (never dropped) main-agent turns ──────
# A live_watcher that finishes with hook_main_agent must run its follow-up
# instruction reliably. Like monitor hooks and voice/user turns, it is retained
# until the foreground session becomes idle rather than being dropped.
def _enqueue_watcher_hook(session: dict, *, rid: str, label: str, task: str,
                          report: str = "") -> None:
    """Append one watcher-completion hook to the session's persistent queue.
    Drained when the session is idle — NEVER dropped. Deduped by rid for the
    whole live session, including after dispatch, so a resumed/re-fired
    completion cannot invoke the same hook twice."""
    lock = session.get("history_lock")
    if lock is None:
        return
    with lock:
        seen = session.setdefault("_watcher_hook_seen_ids", {})
        if not isinstance(seen, dict):
            seen = {}
            session["_watcher_hook_seen_ids"] = seen
        if rid in seen:
            return
        q = session.setdefault("_watcher_hook_queue", [])
        seen[rid] = True
        q.append({"rid": rid, "label": label or rid, "task": task or "",
                  "report": report or ""})


def _format_watcher_hook_message(hook: dict) -> str:
    """Build the user-message text handed to the main agent when a hooked
    live_watcher finishes. 固定格式 (与 monitor hook 一致):
        <instruction>. Reference (subagent execution result): <report>
    result = 本次调研报告全文。不再有 {{result}} 占位符、也不再加完成说明/前缀。"""
    task = str(hook.get("task") or "").strip()
    report = str(hook.get("report") or "").strip()
    return _append_hook_result(task, report)


def _format_watcher_hook_fallback(hook: dict) -> str:
    """User-visible fallback when the hidden synthesis turn cannot complete.

    The watcher report is already the durable, consolidated execution result.
    Returning it is strictly better than completing the hidden turn with an
    empty/error response and leaving the foreground UI waiting forever.
    """
    report = str(hook.get("report") or "").strip()
    if not report:
        return ""
    label = str(hook.get("label") or "Watcher").strip() or "Watcher"
    return (
        f"{label}已完成，但主 Agent 的二次整理暂时失败。"
        "以下是 Watcher 已生成的完整总结：\n\n"
        f"{report}"
    )


def _drain_watcher_hook(rid, sid: str, session: dict) -> bool:
    """Fire ONE queued watcher-completion hook if the session is idle. Returns
    True if a hook turn was dispatched (caller should skip lower-priority
    follow-ups this cycle). Claims the busy gate under history_lock exactly like
    _drain_queued_prompt / the monitor hook."""
    lock = session.get("history_lock")
    if lock is None:
        return False
    with lock:
        q = session.get("_watcher_hook_queue") or []
        if not q or session.get("running") or session.get("_monitor_hook_running"):
            return False
        hook = q.pop(0)
        session["_watcher_hook_queue"] = q
        # Claim both gates for the whole hidden turn. ``running`` routes a new
        # prompt through the busy handler; ``_monitor_hook_running`` makes that
        # handler queue without interrupting the internal Watcher hook.
        session["running"] = True
        session["_monitor_hook_running"] = True
    _hid = f"__watcher_hook__{rid}"
    try:
        # _run_prompt_submit owns both flags from here through asynchronous turn
        # teardown. Its finally block clears the hook guard before tail-draining
        # user prompts / the next Watcher hook, so FIFO chaining remains
        # deterministic without exposing the live hook to user interrupts.
        _run_prompt_submit(
            _hid,
            sid,
            session,
            _format_watcher_hook_message(hook),
            internal_origin="watcher_hook",
            internal_fallback_text=_format_watcher_hook_fallback(hook),
        )
    except Exception as exc:
        print(
            f"[tui_gateway] watcher hook dispatch failed: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        with session["history_lock"]:
            session["running"] = False
            session["_monitor_hook_running"] = False
    return True


def _inflight_snapshot(session: dict) -> dict | None:
    turn = session.get("inflight_turn")
    if not isinstance(turn, dict):
        return None
    user = str(turn.get("user") or "").strip()
    assistant = str(turn.get("assistant") or "")
    streaming = bool(turn.get("streaming"))
    if not user and not assistant and not streaming:
        return None
    return {
        "assistant": assistant,
        "streaming": streaming,
        "user": user,
    }


# ── Methods: session ─────────────────────────────────────────────────


@method("session.create")
def _(rid, params: dict) -> dict:
    sid = uuid.uuid4().hex[:8]
    key = _new_session_key()
    cols = int(params.get("cols", 80))
    history = _coerce_seed_history(params.get("messages"))
    title = str(params.get("title") or "").strip()
    # When set, this is a branch: the new chat copies an existing conversation's
    # history and links back to it so list_sessions_rich keeps it visible and the
    # sidebar can nest it under its parent. Mirrors the TUI /branch marker.
    parent_session_id = str(params.get("parent_session_id") or "").strip() or None
    # Did the client pick a workspace, or are we falling back to the gateway's
    # launch directory? Only an explicit choice is persisted as the session's
    # workspace (see _ensure_session_db_row); otherwise it lands in "No
    # workspace" instead of whatever folder the desktop launched in.
    raw_cwd = str(params.get("cwd") or "").strip()
    try:
        explicit_cwd = bool(raw_cwd) and os.path.isdir(os.path.abspath(os.path.expanduser(raw_cwd)))
    except Exception:
        explicit_cwd = False
    resolved_cwd = _completion_cwd(params)
    source = str(params.get("source") or "tui").strip() or "tui"
    _enable_gateway_prompts()

    # ``profile`` (app-global remote mode): a new chat started under a non-launch
    # profile must build its agent + persist against THAT profile's home/state.db,
    # not the dashboard's launch profile. Stored on the session so _start_agent_build
    # and each turn re-bind HERMES_HOME. None/own profile → launch (unchanged).
    profile = (params.get("profile") or "").strip() or None
    profile_home = _profile_home(profile)

    # The desktop composer owns its model/effort/fast as plain UI state and ships
    # it on every session.create. Honor each as a PER-SESSION override (built into
    # the agent below) — never a global config write, so picking a model/effort
    # for a new chat can't mutate the profile default. provider is optional
    # (resolved at build).
    create_model = str(params.get("model") or "").strip()
    session_model_override = (
        {"model": create_model, "provider": str(params.get("provider") or "").strip() or None}
        if create_model
        else None
    )
    create_reasoning_override = None
    if effort := str(params.get("reasoning_effort") or "").strip():
        try:
            from hermes_constants import parse_reasoning_effort

            create_reasoning_override = parse_reasoning_effort(effort)
        except Exception:
            create_reasoning_override = None
    # Only pin "fast" when explicitly requested; leaving it None lets the build
    # fall back to the profile default service tier rather than forcing normal.
    create_service_tier_override = "priority" if params.get("fast") else None

    ready = threading.Event()
    now = time.time()
    lease, limit_message = _claim_active_session_slot(key, live_session_id=sid)
    if limit_message is not None:
        return _err(rid, 4090, limit_message)

    with _sessions_lock:
        _sessions[sid] = {
            "agent": None,
            "agent_error": None,
            "agent_ready": ready,
            "attached_images": [],
            "close_on_disconnect": is_truthy_value(params.get("close_on_disconnect", False)),
            "active_session_lease": lease,
            "cols": cols,
            "created_at": now,
            "edit_snapshots": {},
            "explicit_cwd": explicit_cwd,
            "history": history,
            "history_lock": threading.Lock(),
            "history_version": 0,
            "image_counter": 0,
            "cwd": resolved_cwd,
            "inflight_turn": None,
            "queued_prompt": None,
            "queued_prompts": [],
            "last_active": now,
            "model_override": session_model_override,
            "create_reasoning_override": create_reasoning_override,
            "create_service_tier_override": create_service_tier_override,
            "parent_session_id": parent_session_id,
            "pending_title": title or None,
            "profile_home": str(profile_home) if profile_home is not None else None,
            "running": False,
            "session_key": key,
            "show_reasoning": _load_show_reasoning(),
            "source": source,
            "slash_worker": None,
            "tool_progress_mode": _load_tool_progress_mode(),
            "tool_started_at": {},
            "transport": current_transport() or _stdio_transport,
        }
        _register_session_cwd(_sessions[sid])

    # NOTE: we intentionally do NOT persist a DB row here. Every TUI/desktop
    # launch (and every "New agent" / draft) opens a session here just to paint
    # the composer, so eagerly creating a row left an "Untitled" empty session
    # behind for every launch the user never typed into. The row is now created
    # lazily on the first prompt (see _ensure_session_db_row + prompt.submit),
    # and the AIAgent's own INSERT-OR-IGNORE persists it on the first turn too.

    # Return the lightweight session immediately so Ink can paint the composer
    # + skeleton panel, then build the real AIAgent just after this response is
    # flushed.  This keeps startup responsive while still hydrating tools/skills
    # without requiring the user to submit a first prompt.
    _schedule_agent_build(sid)
    _schedule_session_cap_enforcement()  # trim detached idle sessions over the cap

    return _ok(
        rid,
        {
            "session_id": sid,
            "stored_session_id": key,
            "message_count": len(history),
            "messages": _history_to_messages(history),
            "info": {
                # Reflect the per-session model override (desktop composer pick)
                # in the immediate response so the client doesn't briefly clobber
                # its sticky pick with the global default before the deferred
                # build's session.info lands.
                "model": (
                    session_model_override.get("model")
                    if session_model_override
                    else _resolve_model()
                ),
                **(
                    {"provider": session_model_override["provider"]}
                    if session_model_override and session_model_override.get("provider")
                    else {}
                ),
                "tools": {},
                "skills": {},
                "cwd": _sessions[sid]["cwd"],
                "branch": _git_branch_for_cwd(_sessions[sid]["cwd"]),
                "lazy": True,
                "desktop_contract": DESKTOP_BACKEND_CONTRACT,
                "profile_name": _current_profile_name(),
            },
        },
    )


@method("session.list")
def _(rid, params: dict) -> dict:
    db = _get_db()
    if db is None:
        return _db_unavailable_error(rid, code=5006)
    try:
        # Resume picker should surface human conversation sessions from every
        # user-facing surface — CLI, TUI, all gateway platforms (including new
        # ones not enumerated here), ACP adapter clients, webhook sessions,
        # custom `ARGUS_SESSION_SOURCE` values, and older installs with
        # different source labels. We deny-list only the noisy internal
        # sources (``tool`` sub-agent runs) rather than allow-listing a
        # fixed set of platform names that goes stale whenever a new
        # platform is added or a user names their own source.
        deny = frozenset({"tool"})

        limit = int(params.get("limit", 200) or 200)
        # Over-fetch modestly so per-source filtering doesn't leave us
        # short; the compression-tip projection in ``list_sessions_rich``
        # can also merge rows.
        fetch_limit = max(limit * 2, 200)
        rows = [
            s
            for s in db.list_sessions_rich(source=None, limit=fetch_limit, order_by_last_active=True)
            if (s.get("source") or "").strip().lower() not in deny
        ][:limit]
        return _ok(
            rid,
            {
                "sessions": [
                    {
                        "id": s["id"],
                        "title": s.get("title") or "",
                        "preview": s.get("preview") or "",
                        "started_at": s.get("started_at") or 0,
                        "message_count": s.get("message_count") or 0,
                        "source": s.get("source") or "",
                    }
                    for s in rows
                ]
            },
        )
    except Exception as e:
        return _err(rid, 5006, str(e))


@method("session.most_recent")
def _(rid, params: dict) -> dict:
    """Return the most recent human-facing session id, or ``None``.

    Mirrors ``session.list``'s deny-list behaviour (drops ``tool``
    sub-agent rows).  Used by TUI auto-resume when
    ``display.tui_auto_resume_recent`` is on; the field is also handy
    for any CLI tooling that wants "latest session" without paginating
    the full list.

    Contract: a ``{"session_id": null}`` result means "no eligible
    session found right now".  Errors are also folded into that
    null-result shape (and logged) so callers don't have to special-
    case JSON-RPC error envelopes for what is a normal "no answer".
    """
    db = _get_db()
    if db is None:
        return _ok(rid, {"session_id": None})
    try:
        deny = frozenset({"tool"})
        # Over-fetch by a generous bounded amount so heavy sub-agent
        # users (lots of recent ``tool`` rows) don't get a false
        # "no eligible session" answer.  ``session.list`` uses a
        # similar over-fetch strategy.
        rows = db.list_sessions_rich(source=None, limit=200, order_by_last_active=True)
        for row in rows:
            src = (row.get("source") or "").strip().lower()
            if src in deny:
                continue
            return _ok(
                rid,
                {
                    "session_id": row.get("id"),
                    "title": row.get("title") or "",
                    "started_at": row.get("started_at") or 0,
                    "source": row.get("source") or "",
                },
            )
        return _ok(rid, {"session_id": None})
    except Exception:
        logger.exception("session.most_recent failed")
        return _ok(rid, {"session_id": None})


@method("project.facts")
def _(rid, params: dict) -> dict:
    """Structured project facts for a cwd — manifests, package manager, the
    exact verify commands, and context files.

    The same detection the coding-context posture (#43316) bakes into the system
    prompt, exposed so UIs (the desktop verify surface) consume it instead of
    re-sniffing. ``{"facts": null}`` means the cwd isn't a code workspace.
    """
    try:
        from agent.coding_context import project_facts_for

        return _ok(rid, {"facts": project_facts_for(params.get("cwd"))})
    except Exception:
        logger.exception("project.facts failed")
        return _ok(rid, {"facts": None})


@method("verification.status")
def _(rid, params: dict) -> dict:
    """Best known coding verification evidence for a cwd/session.

    Read-only consumer of the core ledger. It never runs checks and never
    upgrades targeted evidence into a repository-wide guarantee.
    """
    try:
        from agent.verification_evidence import verification_status

        return _ok(
            rid,
            {
                "verification": verification_status(
                    session_id=params.get("session_id") or params.get("session_key"),
                    cwd=params.get("cwd"),
                )
            },
        )
    except Exception:
        logger.exception("verification.status failed")
        return _ok(rid, {"verification": {"status": "unknown", "evidence": None}})


def _lazy_resume_info(cwd: str, *, model: str = "", provider: str = "") -> dict:
    """session.info for a not-yet-built session (the shape session.create
    returns). tools/skills land later when the deferred build emits session.info."""
    info = {
        "cwd": cwd,
        "branch": _git_branch_for_cwd(cwd),
        "model": model or _resolve_model(),
        "tools": {},
        "skills": {},
        "lazy": True,
        "desktop_contract": DESKTOP_BACKEND_CONTRACT,
        "profile_name": _current_profile_name(),
    }
    if provider:
        info["provider"] = provider
    return info


def _deferred_session_record(
    session_key: str,
    *,
    cols: int,
    cwd: str,
    history: list,
    lease,
    source: str = "tui",
    close_on_disconnect: bool = False,
    display_history_prefix: list | None = None,
    profile_home: Path | None = None,
    lazy: bool = False,
    model_override=None,
    resume_runtime_overrides: dict | None = None,
) -> dict:
    """A live-session record whose AIAgent is built later (lazy watch / cold
    resume) — _init_session's shape minus the agent."""
    now = time.time()
    return {
        "agent": None,
        "agent_error": None,
        "agent_ready": threading.Event(),
        "attached_images": [],
        "close_on_disconnect": close_on_disconnect,
        "active_session_lease": lease,
        "cols": cols,
        "created_at": now,
        "cwd": cwd,
        "display_history_prefix": display_history_prefix or [],
        "edit_snapshots": {},
        "explicit_cwd": False,
        "history": history,
        "history_lock": threading.Lock(),
        "history_version": 0,
        "image_counter": 0,
        "inflight_turn": None,
        "last_active": now,
        "lazy": lazy,
        "model_override": model_override,
        "pending_title": None,
        "profile_home": str(profile_home) if profile_home is not None else None,
        "resume_runtime_overrides": resume_runtime_overrides,
        "resume_session_id": session_key,
        "running": False,
        "session_key": session_key,
        "show_reasoning": _load_show_reasoning(),
        "slash_worker": None,
        "source": source,
        "tool_progress_mode": _load_tool_progress_mode(),
        "tool_started_at": {},
        "transport": current_transport() or _stdio_transport,
    }


def _claim_or_reuse_live(
    sid: str, session_key: str, record: dict, lease
) -> tuple[str, dict] | None:
    """Register ``record`` as the live session for ``session_key`` under the
    resume lock, or — if a concurrent resume already won — release ``lease`` and
    return the winner for the caller to reuse."""
    with _session_resume_lock:
        live = _find_live_session_by_key(session_key)
        if live is not None:
            if lease is not None:
                lease.release()
            return live
        with _sessions_lock:
            _sessions[sid] = record
            _register_session_cwd(_sessions[sid])
    return None


def _schedule_agent_build(sid: str, delay: float = 0.05) -> None:
    """Pre-warm a deferred session's agent off the response path (session.create
    and cold resume both build through here; _sess() also builds on demand)."""

    def _run():
        session = _sessions.get(sid)
        if session is not None:
            _start_agent_build(sid, session)

    timer = threading.Timer(delay, _run)
    timer.daemon = True
    timer.start()


@method("session.resume")
def _(rid, params: dict) -> dict:
    target = params.get("session_id", "")
    if not target:
        return _err(rid, 4006, "session_id required")
    try:
        cols = int(params.get("cols", 80))
    except (TypeError, ValueError):
        cols = 80
    # ``profile`` (app-global remote mode): resume a session that lives in another
    # local profile's state.db. None/own profile → the launch profile (unchanged).
    profile = (params.get("profile") or "").strip() or None
    profile_home = _profile_home(profile)

    # In a profile scope, the agent OWNS a long-lived db handle bound to that
    # profile (do NOT auto-close it here). Otherwise reuse the shared launch db.
    if profile_home is not None:
        from hermes_state import SessionDB

        db = SessionDB(db_path=profile_home / "state.db")
    else:
        db = _get_db()
    if db is None:
        return _db_unavailable_error(rid, code=5000)

    found = db.get_session(target)
    if not found:
        found = db.get_session_by_title(target)
        if found:
            target = found["id"]
        elif is_truthy_value(params.get("lazy", False)) and _child_run_active(target):
            # Race: a watch window opened on a freshly-spawned subagent. The
            # child relays `subagent.start` (which carries child_session_id and
            # triggers the window) BEFORE its first run_conversation() flushes
            # the DB row via _ensure_db_session, so db.get_session(target) is
            # momentarily empty. On slower hosts (notably WSL2, where SQLite +
            # process scheduling widen the gap) the window's resume consistently
            # lands inside this window and used to hard-fail "session not found"
            # — the frontend then 404'd on the REST messages fallback and the
            # window spun forever. The child is provably live (_child_run_active),
            # so proceed into the lazy branch with empty history; the live mirror
            # streams the whole turn anyway and the row exists by upgrade time.
            found = {}
        else:
            return _err(rid, 4007, "session not found")

    # Follow the compression-continuation chain to the live tip so a resume on
    # a rotated-out parent id binds to the descendant that actually holds the
    # post-compression turns. Auto-compression ends the session and forks a
    # continuation child; without this, resuming the original id (the desktop's
    # routed id when the chat was opened before it rotated) reloads the parent
    # transcript and the response generated after compression is missing — the
    # "I came back and the reply isn't there" bug on large sessions. Resolving
    # here also re-anchors the fast path below so a still-live rotated session
    # is reused (by its new key) instead of rebuilding a duplicate agent on the
    # stale parent. Skipped for lazy watch windows, which intentionally attach
    # to the exact child branch they were opened on.
    if found and not is_truthy_value(params.get("lazy", False)):
        try:
            tip = db.resolve_resume_session_id(target)
        except Exception:
            tip = target
        if tip and tip != target:
            target = tip
            found = db.get_session(target) or found

    profile_resume_cwd = str(found.get("cwd") or "").strip() or _profile_configured_cwd(
        profile_home
    )

    def _reuse_live_payload(sid: str, session: dict) -> dict:
        payload = _live_session_payload(
            sid,
            session,
            cols=cols,
            touch=True,
            transport=current_transport() or _stdio_transport,
        )
        payload["resumed"] = target
        # A lazy watch session never owns a run loop, so its payload's running
        # flag is always False — overlay the child-run registry so a reconnecting
        # watch window keeps its busy indicator while the child is still mid-run.
        if session.get("agent") is None and _child_run_active(target):
            payload["running"] = True
            payload["status"] = "streaming"
        return payload

    # Fast path: if the session is already live, reuse it under the lock.
    #
    # ★ The caller's ``source`` still matters here. A conversation first opened by
    #   the TUI/desktop (source=tui) or a delegated worker (source=tool) was built
    #   WITHOUT the multimodal runtime; reusing it verbatim for the dashboard's
    #   /multimodal page left that page with no frame_buffer and no MonitorEngine
    #   (set_monitor → "Monitor backend 未就绪"). Attach the runtime on demand
    #   instead. Promotion runs OUTSIDE _session_resume_lock: the engine startups
    #   it calls use bounded waits, and holding the resume lock across them would
    #   stall every other session.resume.
    _promote_live = None
    with _session_resume_lock:
        live = _find_live_session_by_key(target)
        if live is not None:
            _want_mm = (
                str(params.get("source") or "").strip().lower() == "multimodal")
            if _want_mm and not _is_multimodal_runtime_session(live[1]):
                _promote_live = live
            else:
                return _ok(rid, _reuse_live_payload(*live))
    if _promote_live is not None:
        _promote_session_to_multimodal(*_promote_live)
        return _ok(rid, _reuse_live_payload(*_promote_live))

    # Lazy/watch resume: register the live session WITHOUT building an agent.
    # Used by the desktop's subagent windows — the child runs inside the
    # parent's turn, so its window only needs the stored history plus a
    # transport for the child-mirror's live events. Skipping _make_agent here
    # is what keeps the window cheap while the backend is busy running the
    # delegation. A later prompt.submit upgrades it via _start_agent_build
    # (resume_session_id keeps the upgrade on the stored conversation).
    if is_truthy_value(params.get("lazy", False)):
        sid = uuid.uuid4().hex[:8]
        lease, limit_message = _claim_active_session_slot(target, live_session_id=sid)
        if limit_message is not None:
            return _err(rid, 4090, limit_message)
        try:
            db.reopen_session(target)
            # The child's OWN conversation only — include_ancestors would prepend
            # the parent's transcript onto the subagent's branch.
            history = db.get_messages_as_conversation(target)
        except Exception as e:
            if lease is not None:
                lease.release()
            return _err(rid, 5000, f"resume failed: {e}")
        cwd = profile_resume_cwd or os.getenv("TERMINAL_CWD", os.getcwd())
        record = _deferred_session_record(
            target,
            cols=cols,
            cwd=cwd,
            history=history,
            lease=lease,
            source=str(params.get("source") or "tui").strip() or "tui",
            close_on_disconnect=is_truthy_value(params.get("close_on_disconnect", False)),
            profile_home=profile_home,
            lazy=True,
        )
        if (live := _claim_or_reuse_live(sid, target, record, lease)) is not None:
            return _ok(rid, _reuse_live_payload(*live))
        # A delegated child mid-run emits no session events of its own — report
        # its liveness from the relay registry so the window shows a busy turn.
        child_running = _child_run_active(target)
        messages = _history_to_messages(history)
        return _ok(
            rid,
            {
                "session_id": sid,
                "resumed": target,
                "message_count": len(messages),
                "messages": messages,
                "info": _lazy_resume_info(cwd),
                "inflight": None,
                "running": child_running,
                "session_key": target,
                "started_at": record["created_at"],
                "status": "streaming" if child_running else "idle",
            },
        )

    # Cold resume default: register the live session and read its stored
    # transcript, but build the agent OFF the response path. _make_agent can
    # block for seconds (MCP discovery, prompt/skill build, AIAgent
    # construction), and every resume caller (desktop + Ink TUI) awaits this RPC
    # before it paints — so building eagerly is the bulk of the multi-second
    # "switching sessions is frozen" latency. Return the full display transcript
    # immediately and pre-warm the agent on a short timer (the same deferred-
    # build contract session.create uses); _sess() also builds on demand if the
    # first prompt beats the timer. A caller that needs the agent built
    # synchronously (e.g. tests of the build race) passes ``eager_build: true``
    # to fall through to the eager path below. Distinct from the lazy/watch
    # branch above: a normal resume restores the full ancestor history and the
    # session's persisted runtime identity, and is a real (upgradable) session.
    if not is_truthy_value(params.get("eager_build", False)):
        sid = uuid.uuid4().hex[:8]
        lease, limit_message = _claim_active_session_slot(target, live_session_id=sid)
        if limit_message is not None:
            return _err(rid, 4090, limit_message)
        # Interactive resume routes approvals/clarify through gateway prompts;
        # the deferred build wires the remaining per-session callbacks.
        _enable_gateway_prompts()
        try:
            db.reopen_session(target)
            raw_history = db.get_messages_as_conversation(target)
            display_history = db.get_messages_as_conversation(target, include_ancestors=True)
        except Exception as e:
            if lease is not None:
                lease.release()
            return _err(rid, 5000, f"resume failed: {e}")
        # Reopen reconciliation: a monitor / deep-research that was "running" when
        # the app/tab last closed (or crashed) is NOT running now — its engine job
        # and the video stream died with the process. Flip those receipts to
        # "interrupted" only in the display projection, and stash the jobs so the
        # agent build below re-registers them (disabled) for the panel's toggles.
        # The model-fed history remains byte-stable for prompt caching. This
        # is the backstop for the crash path where _finalize_session never ran;
        # the clean-close path already reconciled at finalize time. In-memory only
        # (no destructive DB rewrite) — the eventual finalize of THIS session
        # persists it, and a re-reopen re-reconciles idempotently.
        _mm_orphans: list = []
        try:
            # display_history 是回前端的 transcript → 在这条上收集孤儿 event id,
            # 孤儿气泡标 _mm_orphan (下面 _history_to_messages 会带出), 前端据此丢弃。
            _reconcile_stale_mm_jobs(display_history, session_id=target,
                                     orphans_out=_mm_orphans)
        except Exception:
            pass
        # Display keeps the full transcript; the model-fed history drops a
        # dangling/interrupted tool-call tail so a session killed mid-loop does
        # not replay the unanswered call forever (#29086).
        prefix = display_history[: max(0, len(display_history) - len(raw_history))]
        history = sanitize_replay_history(raw_history)
        # Restore identity-independent preferences only. The deferred build
        # resolves model/provider/credentials as one bundle from current config.
        overrides = _stored_session_runtime_overrides(found) or {}
        model_override = overrides.get("model_override") or {}
        cwd = profile_resume_cwd or os.getenv("TERMINAL_CWD", os.getcwd())
        record = _deferred_session_record(
            target,
            cols=cols,
            cwd=cwd,
            history=history,
            lease=lease,
            source=str(params.get("source") or "tui").strip() or "tui",
            close_on_disconnect=is_truthy_value(params.get("close_on_disconnect", False)),
            display_history_prefix=prefix,
            profile_home=profile_home,
            model_override=overrides.get("model_override"),
            resume_runtime_overrides=overrides or None,
        )
        if (live := _claim_or_reuse_live(sid, target, record, lease)) is not None:
            return _ok(rid, _reuse_live_payload(*live))

        _schedule_agent_build(sid)
        _schedule_session_cap_enforcement()  # trim detached idle sessions over the cap

        messages = _history_to_messages(display_history)
        return _ok(
            rid,
            {
                "session_id": sid,
                "resumed": target,
                "message_count": len(messages),
                "messages": messages,
                # ★ 孤儿 monitor/watcher event id (history 有、本 session 磁盘无) →
                #   前端据此丢弃对应气泡 + 顶部提示。
                "orphan_event_ids": _mm_orphans,
                "info": _lazy_resume_info(
                    cwd,
                    model=model_override.get("model") or "",
                    provider=overrides.get("provider_override") or "",
                ),
                "inflight": None,
                "running": False,
                "session_key": target,
                "started_at": record["created_at"],
                "status": "idle",
            },
        )

    # Build the agent OUTSIDE the lock — _make_agent can block for seconds
    # (MCP discovery, prompt/skill build, AIAgent construction). Holding
    # _session_resume_lock across it would stall session.close on the main
    # dispatch thread (it's not a _LONG_HANDLER), blocking fast-path RPCs.
    sid = uuid.uuid4().hex[:8]
    lease, limit_message = _claim_active_session_slot(target, live_session_id=sid)
    if limit_message is not None:
        return _err(rid, 4090, limit_message)
    _enable_gateway_prompts()
    home_token = (
        set_hermes_home_override(str(profile_home)) if profile_home is not None else None
    )
    try:
        db.reopen_session(target)
        raw_history = db.get_messages_as_conversation(target)
        display_history = db.get_messages_as_conversation(
            target, include_ancestors=True
        )
        # The display transcript keeps every row so the user still sees their
        # full history.  The model-fed history is sanitized: a session whose
        # last turn died mid-tool-loop persists a dangling assistant(tool_calls)
        # (or interrupted assistant→tool) tail; replaying it makes the model
        # re-issue the unanswered call forever — the permanent-"thinking" stuck
        # session in #29086.  The messaging gateway already strips this; this is
        # the WebUI/TUI resume path picking up the same cleanup.
        display_history_prefix = display_history[
            : max(0, len(display_history) - len(raw_history))
        ]
        history = sanitize_replay_history(raw_history)
        # 孤儿检测 (磁盘为权威): 与 deferred 路径一致, 收集 history 有、本 session 磁盘
        # 无的 monitor/watcher event id, 回前端丢弃 + 提示。
        _mm_orphans_eager: list = []
        try:
            _reconcile_stale_mm_jobs(display_history, session_id=target,
                                     orphans_out=_mm_orphans_eager)
        except Exception:
            pass
        messages = _history_to_messages(display_history)
        tokens = _set_session_context(target)
        try:
            # Pass the profile's db so the agent persists turns to the right
            # state.db; home override is active here so config/skills/model
            # resolve to the profile too. Runtime identity is restored from the
            # stored session row so switching chats does not inherit whatever
            # global model another chat last selected.
            stored_runtime_overrides = _stored_session_runtime_overrides(found)
            agent = _make_agent(
                sid,
                target,
                session_id=target,
                session_db=db,
                **stored_runtime_overrides,
            )
        finally:
            _clear_session_context(tokens)
    except Exception as e:
        if lease is not None:
            lease.release()
        return _err(rid, 5000, f"resume failed: {e}")
    finally:
        if home_token is not None:
            reset_hermes_home_override(home_token)

    # Double-checked locking: another concurrent resume may have created the
    # live session while we were building. Re-check under the lock; if it won,
    # discard our just-built agent and reuse theirs (no worker/poller wired yet).
    with _session_resume_lock:
        live = _find_live_session_by_key(target)
        if live is not None:
            try:
                if hasattr(agent, "close"):
                    agent.close()
            except Exception:
                pass
            if lease is not None:
                lease.release()
            other_sid, other_session = live
            payload = _live_session_payload(
                other_sid,
                other_session,
                cols=cols,
                touch=True,
                transport=current_transport() or _stdio_transport,
            )
            payload["resumed"] = target
            return _ok(rid, payload)
        try:
            init_home_token = (
                set_hermes_home_override(str(profile_home))
                if profile_home is not None
                else None
            )
            try:
                _init_session(
                    sid,
                    target,
                    agent,
                    history,
                    cols=cols,
                    cwd=profile_resume_cwd,
                    session_db=db,
                )
            finally:
                if init_home_token is not None:
                    reset_hermes_home_override(init_home_token)
            if sid in _sessions:
                if stored_runtime_overrides.get("model_override") is not None:
                    _sessions[sid]["model_override"] = stored_runtime_overrides[
                        "model_override"
                    ]
                _sessions[sid]["display_history_prefix"] = display_history_prefix
                # Remember the profile home so each turn re-binds HERMES_HOME (the
                # agent persists to its own db, but mid-turn home reads — memory,
                # skills — must resolve to the resumed profile too).
                if profile_home is not None:
                    _sessions[sid]["profile_home"] = str(profile_home)
                _sessions[sid]["active_session_lease"] = lease
        except Exception as e:
            if lease is not None:
                lease.release()
            return _err(rid, 5000, f"resume failed: {e}")
        session = _sessions.get(sid) or {}
    return _ok(
        rid,
        {
            "session_id": sid,
            "resumed": target,
            "message_count": len(messages),
            "messages": messages,
            "orphan_event_ids": _mm_orphans_eager,
            "info": _session_info(agent, session),
            "inflight": None,
            "running": False,
            "session_key": target,
            "started_at": float(session.get("created_at") or time.time()),
            "status": "idle",
        },
    )


@method("session.cwd.set")
def _(rid, params: dict) -> dict:
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    if session.get("running"):
        return _err(rid, 4009, "session busy")
    raw = str(params.get("cwd", "") or "").strip()
    if not raw:
        return _err(rid, 4016, "cwd required")
    try:
        cwd = _set_session_cwd(session, raw)
    except ValueError as e:
        return _err(rid, 4017, str(e))
    agent = session.get("agent")
    info = _session_info(agent, session) if agent is not None else {
        "cwd": cwd,
        "branch": _git_branch_for_cwd(cwd),
        "lazy": True,
    }
    _emit("session.info", params.get("session_id", ""), info)
    return _ok(rid, info)


def _session_pending_kind(sid: str) -> str:
    for rid, (owner_sid, _ev) in list(_pending.items()):
        if owner_sid != sid:
            continue
        event, _payload = _pending_prompt_payloads.get(rid, ("input.request", {}))
        return str(event).removesuffix(".request")
    return ""


def _session_live_status(sid: str, session: dict) -> str:
    if _session_pending_kind(sid):
        return "waiting"
    ready = session.get("agent_ready")
    # Unset + build never started = a lazy watch session sitting idle, not a
    # session stuck mid-construction.
    if ready is not None and not ready.is_set() and session.get("agent_build_started"):
        return "starting"
    if session.get("running"):
        return "working"
    return "idle"


def _message_preview(history: list) -> str:
    for msg in reversed(history or []):
        text = _content_display_text(msg.get("content", msg.get("text", ""))).strip()
        if text:
            return " ".join(text.split())[:160]
    return ""


def _session_live_title(session: dict, key: str) -> str:
    title = str(session.get("pending_title") or "").strip()
    db = _get_db()
    if db is not None:
        try:
            title = str(db.get_session_title(key) or title or "").strip()
        except Exception:
            pass
    return title


def _session_live_item(sid: str, session: dict, current_sid: str = "") -> dict:
    key = _session_lookup_key(session, fallback=sid)
    agent = session.get("agent")
    history = list(session.get("history") or [])
    status = _session_live_status(sid, session)
    inflight = _inflight_snapshot(session)
    preview = _message_preview(history)
    if inflight:
        preview = inflight.get("assistant") or inflight.get("user") or preview
        preview = " ".join(str(preview).split())[:160]
    now = time.time()
    return {
        "current": sid == current_sid,
        "id": sid,
        "last_active": float(session.get("last_active") or session.get("created_at") or now),
        "message_count": len(history),
        "model": str(getattr(agent, "model", "") or _resolve_model()),
        "preview": preview,
        "session_key": key,
        "started_at": float(session.get("created_at") or now),
        "status": status,
        "title": _session_live_title(session, key),
    }


def _session_lookup_key(session: dict, *, fallback: str = "") -> str:
    agent = session.get("agent")
    return str(
        getattr(agent, "session_id", None)
        or session.get("session_key")
        or fallback
        or ""
    )


def _find_live_session_by_key(session_key: str) -> tuple[str, dict] | None:
    for sid, session in list(_sessions.items()):
        if session.get("_finalized"):
            continue
        if _session_lookup_key(session, fallback=sid) == session_key:
            return sid, session
    return None


def _fallback_session_info(session: dict) -> dict:
    agent = session.get("agent")
    if agent is not None:
        return _session_info(agent)
    return {
        "cwd": os.getenv("TERMINAL_CWD", os.getcwd()),
        "lazy": True,
        "model": _resolve_model(),
        "skills": {},
        "tools": {},
    }


def _live_session_payload(
    sid: str,
    session: dict,
    *,
    cols: int | None = None,
    touch: bool = False,
    transport: Transport | None = None,
) -> dict:
    with session["history_lock"]:
        if cols is not None:
            session["cols"] = cols
        if transport is not None:
            session["transport"] = transport
        if touch:
            session["last_active"] = time.time()
        history = list(session.get("display_history_prefix") or []) + list(
            session.get("history") or []
        )
        inflight = _inflight_snapshot(session)
        running = bool(session.get("running"))
    payload = {
        "info": _fallback_session_info(session),
        "message_count": len(history),
        "messages": _history_to_messages(history),
        "running": running,
        "session_id": sid,
        "session_key": _session_lookup_key(session, fallback=sid),
        "started_at": float(session.get("created_at") or time.time()),
        "status": _session_live_status(sid, session),
    }
    if inflight:
        payload["inflight"] = inflight
    return payload


@method("session.active_list")
def _(rid, params: dict) -> dict:
    """Return live TUI sessions in this gateway process.

    Unlike ``session.list`` this is not a historical DB browser: it reports only
    sessions with in-memory agents/workers that the current TUI can switch to
    without closing siblings.
    """
    current = str(params.get("current_session_id") or "")
    try:
        with _sessions_lock:
            snapshot = list(_sessions.items())
    except Exception as e:
        return _err(rid, 5036, f"could not enumerate active sessions: {e}")

    # Liveness filter (#38950): a session whose teardown has begun (``_finalized``)
    # is dead — its agent/worker are being released and it is no longer
    # attachable — but it can briefly remain in ``_sessions`` until the reaper
    # pops it (the WS grace-reap and idle reaper both set ``_finalized`` inside
    # ``_teardown_session`` before the pop). Counting these inflated the footer's
    # "N sessions" count, which only ever went up until a gateway restart. Drop
    # them here so the count reflects genuinely attachable sessions. We do NOT
    # filter on ``transport is _detached_ws_transport`` (the WS-detached drop
    # sentinel): a detached session is still attachable via a quick reconnect /
    # session.resume until the grace-reap finalizes it, and a standalone
    # ``hermes --tui`` session legitimately rides the real stdio transport and
    # must stay visible.
    # Keep the natural creation/insertion order from ``_sessions``.  The
    # frontend marks the focused session with ``current``; it should not jump to
    # the top just because the user switched to it.
    rows = [
        _session_live_item(sid, session, current)
        for sid, session in snapshot
        if not session.get("_finalized")
    ]
    return _ok(rid, {"sessions": rows})


@method("session.activate")
def _(rid, params: dict) -> dict:
    """Attach the frontend to an already-live TUI session.

    This intentionally does not close the previously focused session; it merely
    returns enough state for Ink to redraw around another live session id.
    """
    sid = str(params.get("session_id") or "")
    session, err = _sess_nowait({"session_id": sid}, rid)
    if err:
        return err
    assert session is not None

    return _ok(
        rid,
        _live_session_payload(
            sid,
            session,
            touch=True,
            transport=current_transport() or _stdio_transport,
        ),
    )


@method("session.delete")
def _(rid, params: dict) -> dict:
    """Delete a stored session and its on-disk transcript files.

    Used by the TUI resume picker (``d`` key) so users can prune old
    sessions without dropping to the CLI.  Refuses to delete a session
    that is currently active in this gateway process — those rows are
    still being written to and removing them out from under the live
    agent corrupts message ordering and trips FK constraints when the
    next message append flushes.
    """
    target = params.get("session_id", "")
    if not target:
        return _err(rid, 4006, "session_id required")
    db = _get_db()
    if db is None:
        return _db_unavailable_error(rid, code=5036)
    # Block deletion of any session currently bound to a live TUI session
    # in this process.  The picker hides the active session anyway, but a
    # racing caller could still target it.  Snapshot via ``list(...)``
    # because ``_sessions`` is mutated by concurrent RPCs on the thread
    # pool — iterating the dict directly can raise ``RuntimeError:
    # dictionary changed size during iteration``.  If even the snapshot
    # raises, fail closed (refuse the delete) rather than fail open.
    try:
        with _sessions_lock:
            snapshot = list(_sessions.values())
    except Exception as e:
        return _err(rid, 5036, f"could not enumerate active sessions: {e}")
    active = {s.get("session_key") for s in snapshot if s.get("session_key")}
    if target in active:
        return _err(rid, 4023, "cannot delete an active session")
    sessions_dir = get_hermes_home() / "sessions"
    try:
        deleted = db.delete_session(target, sessions_dir=sessions_dir)
    except Exception as e:
        return _err(rid, 5036, f"delete failed: {e}")
    if not deleted:
        return _err(rid, 4007, "session not found")
    return _ok(rid, {"deleted": target})


@method("session.title")
def _(rid, params: dict) -> dict:
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    db = _get_db()
    if db is None:
        return _db_unavailable_error(rid, code=5007)
    key = session["session_key"]
    if "title" not in params:
        fallback = session.get("pending_title") or ""
        try:
            resolved_title = db.get_session_title(key) or ""
            if fallback:
                if db.set_session_title(key, fallback):
                    session["pending_title"] = None
                    resolved_title = fallback
                else:
                    existing_row = db.get_session(key)
                    existing_title = ((existing_row or {}).get("title") or "").strip()
                    if existing_title == fallback:
                        session["pending_title"] = None
                        resolved_title = fallback
                    elif not resolved_title:
                        resolved_title = fallback
            elif resolved_title:
                session["pending_title"] = None
        except Exception:
            resolved_title = fallback
        _emit_session_info_for_session(params.get("session_id", ""), session)
        return _ok(
            rid,
            {
                "title": resolved_title,
                "session_key": key,
            },
        )
    title = (params.get("title", "") or "").strip()
    if not title:
        return _err(rid, 4021, "title required")
    try:
        if db.set_session_title(key, title):
            session["pending_title"] = None
            _emit_session_info_for_session(params.get("session_id", ""), session)
            return _ok(rid, {"pending": False, "title": title})
        # rowcount == 0 can mean "same value" as well as "missing row".
        existing_row = db.get_session(key)
        if existing_row:
            session["pending_title"] = None
            _emit_session_info_for_session(params.get("session_id", ""), session)
            return _ok(
                rid,
                {
                    "pending": False,
                    "title": (existing_row.get("title") or title),
                },
            )
        # No row yet (the DB write is deferred to the first prompt so empty
        # drafts don't litter the sidebar). An explicit /title is clear user
        # intent, not an abandoned draft — so persist the row NOW and set the
        # title, mirroring the messaging gateway's _handle_title_command. The
        # old behavior only queued pending_title and relied on the post-turn
        # apply block; if that turn never landed under this session_key the
        # title was silently lost and the sidebar fell back to the message
        # preview. Creating the row up front removes that race entirely. The
        # min-messages sidebar filter keeps a titled 0-message row hidden, so
        # a /title'd-but-never-used draft still doesn't clutter the list.
        _ensure_session_db_row(session)
        with _session_db(session) as scoped_db:
            if scoped_db is not None and scoped_db.set_session_title(key, title):
                session["pending_title"] = None
                _emit_session_info_for_session(params.get("session_id", ""), session)
                return _ok(rid, {"pending": False, "title": title})
        # Row creation didn't take (DB unavailable, or a concurrent writer) —
        # fall back to queuing so the post-turn apply block can still recover.
        session["pending_title"] = title
        _emit_session_info_for_session(params.get("session_id", ""), session)
        return _ok(rid, {"pending": True, "title": title})
    except ValueError as e:
        return _err(rid, 4022, str(e))
    except Exception as e:
        return _err(rid, 5007, str(e))


def _main_runtime_from_agent(agent) -> dict | None:
    """Build an aux-client main_runtime override from a live agent.

    Lets a one-shot inherit the session's provider/model/credentials so its
    output matches the model the user is actually coding with, instead of
    falling back to the cheapest auto-detected backend.
    """
    if agent is None:
        return None
    runtime: dict = {}
    for field in ("provider", "model", "base_url", "api_key", "api_mode", "auth_mode"):
        value = getattr(agent, field, None)
        if isinstance(value, str) and value.strip():
            runtime[field] = value.strip()
        elif field == "api_key" and callable(value):
            runtime[field] = value
    return runtime or None


@method("llm.oneshot")
def _(rid, params: dict) -> dict:
    """Run a single stateless LLM request outside any conversation.

    Generic helper for small generative chores (e.g. a commit message from a
    diff). Accepts either a named ``template`` + ``variables`` or an explicit
    ``instructions`` / ``input`` pair. When ``session_id`` resolves to a live
    session the call inherits that agent's model; otherwise it uses the
    configured auxiliary ``task`` backend. Never mutates session history, so
    prompt caching is untouched.
    """
    template = (params.get("template") or "").strip() or None
    instructions = params.get("instructions") or ""
    user_input = params.get("input") or ""
    variables = params.get("variables") if isinstance(params.get("variables"), dict) else {}
    task = (params.get("task") or "title_generation").strip() or "title_generation"

    try:
        max_tokens = int(params.get("max_tokens") or 1024)
    except (TypeError, ValueError):
        max_tokens = 1024
    temperature = params.get("temperature")
    if temperature is not None:
        try:
            temperature = float(temperature)
        except (TypeError, ValueError):
            temperature = None

    if not template and not str(instructions).strip() and not str(user_input).strip():
        return _err(rid, 4030, "llm.oneshot requires a template or instructions/input")

    # Optional: inherit the live session's model (no error if absent).
    session = _sessions.get(params.get("session_id") or "")
    main_runtime = _main_runtime_from_agent(session.get("agent")) if session else None

    try:
        from agent.oneshot import run_oneshot

        text = run_oneshot(
            instructions=instructions,
            user_input=user_input,
            template=template,
            variables=variables,
            task=task,
            max_tokens=max_tokens,
            temperature=temperature if temperature is not None else 0.3,
            main_runtime=main_runtime,
        )
    except KeyError as e:
        return _err(rid, 4031, str(e))
    except ValueError as e:
        return _err(rid, 4032, str(e))
    except Exception as e:
        logger.warning("llm.oneshot failed: %s", e)
        return _err(rid, 5030, f"one-shot generation failed: {e}")

    return _ok(rid, {"text": text})


@method("handoff.request")
def _(rid, params: dict) -> dict:
    """Queue a handoff of this session to a messaging platform.

    Desktop parity with the CLI ``/handoff`` command: we only write
    ``handoff_state='pending'`` onto the persisted session row. The actual
    transfer is performed by the separate ``argus gateway`` process, whose
    ``_handoff_watcher`` claims the row, re-binds the session to the platform's
    home channel, and forges a synthetic turn. The desktop then polls
    ``handoff.state`` for the terminal result.
    """
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    if session.get("running"):
        return _err(
            rid,
            4009,
            "session busy — wait for the current turn to finish, then retry the handoff",
        )

    platform_name = (params.get("platform", "") or "").strip().lower()
    if not platform_name:
        return _err(rid, 4023, "platform required")

    # Validate against the live gateway config — an unconfigured platform or a
    # missing home channel would leave the handoff pending forever, so reject
    # up front with a clear, actionable message (mirrors cli.py).
    try:
        from gateway.config import Platform, load_gateway_config
    except Exception as e:  # pragma: no cover — gateway pkg always ships
        return _err(rid, 5021, f"could not load gateway config: {e}")
    try:
        platform = Platform(platform_name)
    except (ValueError, KeyError):
        return _err(rid, 4024, f"unknown platform '{platform_name}'")
    try:
        gw_config = load_gateway_config()
    except Exception as e:
        return _err(rid, 5021, f"could not load gateway config: {e}")
    pcfg = gw_config.platforms.get(platform)
    if not pcfg or not pcfg.enabled:
        return _err(
            rid,
            4025,
            f"platform '{platform_name}' is not configured/enabled in the gateway",
        )
    home = gw_config.get_home_channel(platform)
    if not home or not home.chat_id:
        return _err(
            rid,
            4026,
            f"no home channel configured for {platform_name} — set one with "
            "/sethome on the destination chat first",
        )

    # The watcher transfers a persisted DB row, so make sure one exists even
    # for a brand-new empty chat (mirrors the CLI's set_session_title stub).
    _ensure_session_db_row(session)

    with _session_db(session) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5007)
        key = session["session_key"]
        try:
            if not db.get_session(key):
                db.set_session_title(key, f"handoff-{key[:8]}")
            ok = db.request_handoff(key, platform_name)
        except Exception as e:
            return _err(rid, 5007, str(e))

    if not ok:
        return _err(
            rid,
            4027,
            "session is already in flight for handoff — wait for it to settle, then retry",
        )
    return _ok(
        rid,
        {
            "queued": True,
            "session_key": key,
            "platform": platform_name,
            "home_name": home.name,
        },
    )


@method("handoff.state")
def _(rid, params: dict) -> dict:
    """Poll the handoff state for a session.

    Returns ``{state, platform, error}`` where ``state`` is one of
    ``pending|running|completed|failed`` (or empty when no handoff record
    exists). Desktop polls this after ``handoff.request``.
    """
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    with _session_db(session) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5007)
        record = db.get_handoff_state(session["session_key"])

    record = record or {}
    return _ok(
        rid,
        {
            "state": record.get("state") or "",
            "platform": record.get("platform") or "",
            "error": record.get("error") or "",
        },
    )


@method("handoff.fail")
def _(rid, params: dict) -> dict:
    """Mark an in-flight handoff as failed so the user can retry.

    Desktop calls this when its bounded poll times out. Only pending/running
    rows are changed so a late success from the gateway watcher is not clobbered.
    """
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    reason = str(params.get("error") or "handoff failed").strip()[:500]
    with _session_db(session) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5007)
        key = session["session_key"]
        record = db.get_handoff_state(key) or {}
        state = record.get("state") or ""
        if state in {"pending", "running"}:
            db.fail_handoff(key, reason)
            return _ok(rid, {"failed": True, "state": "failed"})

    return _ok(rid, {"failed": False, "state": state})


@method("session.usage")
def _(rid, params: dict) -> dict:
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    agent = session.get("agent")
    usage: dict = (
        _get_usage(agent)
        if agent is not None
        else {"calls": 0, "input": 0, "output": 0, "total": 0}
    )
    # Nous credits block — agent-independent (a portal fetch), so it shows even
    # with zero API calls or on a resumed session. The TUI /usage panel renders
    # these lines regardless of `calls`. Fail-open: [] when not logged into Nous
    # or on any portal hiccup.
    try:
        from agent.account_usage import nous_credits_lines

        credits = nous_credits_lines()
        if credits:
            usage["credits_lines"] = credits
    except Exception:
        pass
    return _ok(rid, usage)


def _pet_frame_counts(spritesheet) -> dict:
    """Real (padding-trimmed) frame count per state, for the desktop canvas.

    Fail-open: a decode hiccup returns ``{}`` and the canvas falls back to its
    static ``framesPerState`` rather than breaking the (cosmetic) pet.
    """
    try:
        from agent.pet import render

        return render.state_frame_counts(str(spritesheet))
    except Exception:  # noqa: BLE001 - cosmetic, never break the surface
        return {}


_pet_payload_cache_lock = threading.Lock()
_pet_payload_cache: dict[tuple, dict] = {}


def _pet_sheet_revision(spritesheet) -> str:
    """Stable revision id for one spritesheet file."""
    try:
        stat = spritesheet.stat()
        return f"{stat.st_mtime_ns}:{stat.st_size}"
    except Exception:  # noqa: BLE001 - cosmetic, never break the surface
        return "0:0"


def _pet_payload_cache_key(pet, *, scale: float) -> tuple | None:
    """Cache key for the expensive sprite payload build."""
    try:
        stat = pet.spritesheet.stat()
    except Exception:  # noqa: BLE001
        return None
    return (
        str(pet.spritesheet),
        stat.st_mtime_ns,
        stat.st_size,
        pet.slug,
        pet.display_name,
        round(scale, 4),
    )


def _clone_pet_payload(payload: dict) -> dict:
    """Shallow-clone cached payloads so callers can't mutate shared state."""
    out = dict(payload)
    if isinstance(payload.get("framesByState"), dict):
        out["framesByState"] = dict(payload["framesByState"])
    if isinstance(payload.get("framesByRow"), dict):
        out["framesByRow"] = dict(payload["framesByRow"])
    if isinstance(payload.get("stateRows"), list):
        out["stateRows"] = list(payload["stateRows"])
    return out


def _pet_row_frame_counts(spritesheet) -> dict:
    """Real frame count per concrete spritesheet row name."""
    try:
        from PIL import Image

        from agent.pet import constants, render

        with Image.open(spritesheet) as opened:
            image = opened.convert("RGBA")
        cols = max(1, image.width // constants.FRAME_W)
        row_count = max(1, image.height // constants.FRAME_H)
        rows = constants.state_rows_for_grid(row_count)
        out: dict[str, int] = {}
        for row_idx, name in enumerate(rows[:row_count]):
            top = row_idx * constants.FRAME_H
            count = 0
            for col in range(cols):
                left = col * constants.FRAME_W
                frame = image.crop((left, top, left + constants.FRAME_W, top + constants.FRAME_H))
                if render._frame_is_blank(frame):
                    break
                count += 1
            out[name] = count
        return out
    except Exception:  # noqa: BLE001 - cosmetic, never break the surface
        return {}


def _pet_config_scale() -> float:
    """Configured ``display.pet.scale`` (or the engine default), never raises."""
    from agent.pet import constants

    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        display = cfg.get("display", {}) if isinstance(cfg.get("display"), dict) else {}
        pet_cfg = display.get("pet", {}) if isinstance(display.get("pet"), dict) else {}
        return float(pet_cfg.get("scale", constants.DEFAULT_SCALE) or constants.DEFAULT_SCALE)
    except Exception:  # noqa: BLE001
        return constants.DEFAULT_SCALE


def _pet_sprite_payload(pet, *, scale: float) -> dict:
    """Build the renderer payload (spritesheet bytes + geometry) for *pet*.

    Shared by ``pet.info`` (the active mascot) and ``pet.hatch`` (the unadopted
    preview) so both feed the desktop canvas / TUI from one shape.
    """
    import base64

    from agent.pet import constants

    cache_key = _pet_payload_cache_key(pet, scale=scale)
    if cache_key is not None:
        with _pet_payload_cache_lock:
            cached = _pet_payload_cache.get(cache_key)
        if cached is not None:
            return _clone_pet_payload(cached)

    raw = pet.spritesheet.read_bytes()
    suffix = pet.spritesheet.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/webp"
    payload = {
        "slug": pet.slug,
        "displayName": pet.display_name,
        "mime": mime,
        "spritesheetBase64": base64.standard_b64encode(raw).decode("ascii"),
        "spritesheetRevision": _pet_sheet_revision(pet.spritesheet),
        "frameW": constants.FRAME_W,
        "frameH": constants.FRAME_H,
        "framesPerState": constants.FRAMES_PER_STATE,
        "framesByState": _pet_frame_counts(pet.spritesheet),
        "framesByRow": _pet_row_frame_counts(pet.spritesheet),
        "loopMs": constants.LOOP_MS,
        "scale": scale,
        "stateRows": _pet_state_rows(pet.spritesheet),
    }
    if cache_key is not None:
        with _pet_payload_cache_lock:
            _pet_payload_cache[cache_key] = payload
            while len(_pet_payload_cache) > 8:
                _pet_payload_cache.pop(next(iter(_pet_payload_cache)))
    return _clone_pet_payload(payload)


def _pet_active_selection():
    """Resolve configured active pet + scale from config."""
    from agent.pet import constants, store

    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        display = cfg.get("display", {}) if isinstance(cfg.get("display"), dict) else {}
        pet_cfg = display.get("pet", {}) if isinstance(display.get("pet"), dict) else {}
    except Exception:
        pet_cfg = {}

    enabled = bool(pet_cfg.get("enabled"))
    configured_slug = str(pet_cfg.get("slug", "") or "")
    pet = store.resolve_active_pet(configured_slug) if enabled else None
    scale = float(pet_cfg.get("scale", constants.DEFAULT_SCALE) or constants.DEFAULT_SCALE)
    return enabled, pet, scale


def _pet_state_rows(spritesheet) -> list[str]:
    """Row taxonomy for the concrete active pet sheet.

    Hermes has to support both the legacy 8-row petdex atlas and the current
    Codex/petdex 9-row atlas. The desktop canvas gets this list and indexes it
    with the same `PetState` names the Python renderer uses.
    """
    try:
        from PIL import Image

        from agent.pet import constants

        with Image.open(spritesheet) as image:
            row_count = max(1, image.height // constants.FRAME_H)
        return list(constants.state_rows_for_grid(row_count))
    except Exception:  # noqa: BLE001 - cosmetic, never break the surface
        from agent.pet import constants

        return list(constants.STATE_ROWS)


@method("pet.info")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Return the active petdex pet for surfaces that render sprites.

    Shared by the desktop (canvas) and the TUI (half-block). Carries the
    spritesheet bytes (base64) plus the engine's frame geometry + state-row
    taxonomy so the renderer is a thin, framework-native consumer. The
    activity→state decision is mirrored from ``agent.pet.state`` client-side.

    Agent-independent (reads config + disk), so it works on any session and
    before the agent finishes building. Fail-open: returns ``enabled=False``
    on any error rather than erroring the surface.
    """
    try:
        enabled, pet, scale = _pet_active_selection()

        if not enabled or pet is None or not pet.exists:
            return _ok(rid, {"enabled": False})

        return _ok(rid, {"enabled": True, **_pet_sprite_payload(pet, scale=scale)})
    except Exception as exc:  # noqa: BLE001 - cosmetic, never break the surface
        logger.debug("pet.info failed: %s", exc)
        return _ok(rid, {"enabled": False})


@method("pet.info.meta")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Cheap active-pet metadata used to avoid full payload refreshes."""
    try:
        enabled, pet, scale = _pet_active_selection()
        if not enabled or pet is None or not pet.exists:
            return _ok(rid, {"enabled": False})
        return _ok(
            rid,
            {
                "enabled": True,
                "slug": pet.slug,
                "displayName": pet.display_name,
                "scale": scale,
                "spritesheetRevision": _pet_sheet_revision(pet.spritesheet),
            },
        )
    except Exception as exc:  # noqa: BLE001 - cosmetic, never break the surface
        logger.debug("pet.info.meta failed: %s", exc)
        return _ok(rid, {"enabled": False})


@method("pet.cells")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Return half-block cell frames for one pet state (TUI renderer).

    The TUI can't draw a canvas, so the engine downsamples the spritesheet to
    a grid of half-block cells and the Ink side paints them with native color
    props. Each cell is ``[tr,tg,tb,ta, br,bg,bb,ba]`` (top + bottom pixel).

    Params: ``state`` (idle/run/review/failed/wave/jump), ``cols`` (width).
    Fail-open: ``enabled=False`` on any problem.
    """
    try:
        from agent.pet import constants, render, store
        from agent.pet.render import PetRenderer

        try:
            from hermes_cli.config import load_config

            cfg = load_config()
            display = cfg.get("display", {}) if isinstance(cfg.get("display"), dict) else {}
            pet_cfg = display.get("pet", {}) if isinstance(display.get("pet"), dict) else {}
        except Exception:
            pet_cfg = {}

        if not bool(pet_cfg.get("enabled")):
            return _ok(rid, {"enabled": False})

        pet = store.resolve_active_pet(str(pet_cfg.get("slug", "") or ""))
        if pet is None or not pet.exists:
            return _ok(rid, {"enabled": False})

        state = str(params.get("state") or constants.PetState.IDLE.value)
        scale = float(pet_cfg.get("scale", constants.DEFAULT_SCALE) or constants.DEFAULT_SCALE)
        cols = int(params.get("cols") or 0) or constants.resolve_cols(scale, pet_cfg.get("unicode_cols", 0))

        # Graphics path: when the TUI is attached to a real TTY (``graphics``)
        # and the terminal speaks the kitty protocol, return a Unicode-
        # placeholder payload for a crisp image instead of half-blocks. Env
        # detection (KITTY_WINDOW_ID / TERM / TERM_PROGRAM) is shared with the
        # Ink process since it spawns us; the dashboard PTY (xterm.js) has no
        # such env, so it falls through to half-blocks automatically. Only
        # kitty is grid-safe in Ink — iTerm/sixel stay on the fallback.
        if params.get("graphics"):
            configured = str(pet_cfg.get("render_mode", "auto") or "auto").lower()
            gmode = render.detect_terminal_graphics() if configured in ("", "auto") else configured
            if gmode == "kitty":
                image_id = render.kitty_image_id(pet.slug)
                # kitty sizes from scaled pixels (_cell_box), so unicode_cols is moot here.
                payload = PetRenderer(
                    str(pet.spritesheet), mode="kitty", scale=scale
                ).kitty_payload(state, image_id=image_id)
                if payload:
                    kcount = len(payload["frames"]) or 1
                    return _ok(
                        rid,
                        {
                            "enabled": True,
                            "slug": pet.slug,
                            "displayName": pet.display_name,
                            "state": state,
                            "graphics": "kitty",
                            "imageId": image_id,
                            "color": render.kitty_color_hex(image_id),
                            "cols": payload["cols"],
                            "rows": payload["rows"],
                            "placeholder": payload["placeholder"],
                            "frames": payload["frames"],
                            "frameMs": constants.LOOP_MS / max(1, kcount),
                            "scale": scale,
                        },
                    )

        renderer = PetRenderer(
            str(pet.spritesheet),
            mode="unicode",
            scale=scale,
            unicode_cols=cols,
        )
        count = renderer.frame_count(state) or 1
        frames = []
        for i in range(count):
            grid = renderer.cells(state, i, cols=cols)
            frames.append(
                [[[*top, *bottom] for (top, bottom) in row] for row in grid]
            )

        return _ok(
            rid,
            {
                "enabled": True,
                "slug": pet.slug,
                "displayName": pet.display_name,
                "state": state,
                "cols": cols,
                "frameMs": constants.LOOP_MS / max(1, count),
                "frames": frames,
                "scale": scale,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.cells failed: %s", exc)
        return _ok(rid, {"enabled": False})


@method("pet.gallery")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """List adoptable pets for the desktop appearance picker.

    Returns the petdex gallery merged with local install state plus the
    current config (active slug + enabled). Agent-independent. Fail-open:
    returns whatever is installed locally if the gallery can't be reached, so
    the picker still works offline.

    Param ``localOnly`` (bool): skip the remote petdex manifest fetch and return
    only locally-installed pets. The desktop loads this first so the user's own
    pets render instantly instead of waiting on the (possibly slow) manifest.
    """
    local_only = bool(params.get("localOnly"))
    try:
        from agent.pet import store

        try:
            from hermes_cli.config import load_config

            cfg = load_config()
            display = cfg.get("display", {}) if isinstance(cfg.get("display"), dict) else {}
            pet_cfg = display.get("pet", {}) if isinstance(display.get("pet"), dict) else {}
        except Exception:
            pet_cfg = {}

        installed = {p.slug: p for p in store.installed_pets()}

        gallery: list[dict] = []
        seen: set[str] = set()
        try:
            from agent.pet.manifest import fetch_manifest, prefetch

            # Local-only: skip the network entirely, but kick off a background
            # warm so the follow-up full request usually hits a cached manifest.
            if local_only:
                prefetch()

            for entry in [] if local_only else fetch_manifest():
                seen.add(entry.slug)
                gallery.append(
                    {
                        "slug": entry.slug,
                        "displayName": entry.display_name,
                        "installed": entry.slug in installed,
                        "spritesheetUrl": entry.spritesheet_url,
                        # petdex exposes no popularity metric; "curated" (its
                        # hand-picked/official set, identified by the asset path)
                        # is the closest signal, so the picker can surface it first.
                        "curated": "/curated/" in entry.spritesheet_url,
                        "generated": entry.slug in installed and installed[entry.slug].generated,
                    }
                )
        except Exception as exc:  # noqa: BLE001 - offline: fall back to installed
            logger.debug("pet.gallery manifest fetch failed: %s", exc)

        # Always include locally-installed pets even if the gallery is unreachable.
        for slug, pet in installed.items():
            if slug not in seen:
                gallery.append(
                    {
                        "slug": slug,
                        "displayName": pet.display_name,
                        "installed": True,
                        "spritesheetUrl": "",
                        "generated": pet.generated,
                    }
                )

        return _ok(
            rid,
            {
                "enabled": bool(pet_cfg.get("enabled")),
                "active": str(pet_cfg.get("slug", "") or ""),
                "pets": gallery,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.gallery failed: %s", exc)
        return _ok(rid, {"enabled": False, "active": "", "pets": []})


@method("pet.select")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Adopt a pet from the desktop picker: install (if needed) + activate.

    Params: ``slug`` (required). Writes ``display.pet.*`` to config and returns
    ``{ok, slug, displayName}``. The surface re-pulls ``pet.info`` to render it.
    """
    slug = str(params.get("slug") or "").strip()
    if not slug:
        return _err(rid, 4004, "missing slug")
    try:
        from agent.pet import store
        from agent.pet.manifest import ManifestError
        from hermes_cli.pets import _set_active

        try:
            pet = store.install_pet(slug)
        except (store.PetStoreError, ManifestError) as exc:
            return _err(rid, 5031, f"could not adopt '{slug}': {exc}")
        _set_active(slug)
        return _ok(rid, {"ok": True, "slug": slug, "displayName": pet.display_name})
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.select failed: %s", exc)
        return _err(rid, 5031, f"pet.select failed: {exc}")


@method("pet.remove")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Uninstall a pet from the desktop picker (delete its on-disk directory).

    Params: ``slug`` (required). If the removed pet was the active one, the
    display is turned off so nothing tries to render a now-missing sprite.
    Returns ``{ok, slug}`` where ``ok`` reflects whether a directory was deleted.
    """
    slug = str(params.get("slug") or "").strip()
    if not slug:
        return _err(rid, 4004, "missing slug")
    try:
        from agent.pet import store
        from hermes_cli.pets import _clear_active_if

        removed = store.remove_pet(slug)

        # If that was the active pet, stop surfaces pointing at a deleted sprite.
        try:
            _clear_active_if(slug)
        except Exception as exc:  # noqa: BLE001 - removal already succeeded
            logger.debug("pet.remove config update failed: %s", exc)

        return _ok(rid, {"ok": removed, "slug": slug})
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.remove failed: %s", exc)
        return _err(rid, 5031, f"pet.remove failed: {exc}")


@method("pet.export")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Export an installed pet as a re-importable ``.zip`` (pet.json + sprite).

    Params: ``slug`` (required). Returns ``{ok, filename, zipBase64}`` — the
    client decodes the base64 and saves it. Heavy-ish (reads + zips files) but
    small; runs inline.
    """
    slug = str(params.get("slug") or "").strip()
    if not slug:
        return _err(rid, 4004, "missing slug")
    try:
        import base64

        from agent.pet import store

        filename, data = store.export_pet(slug)
        return _ok(
            rid,
            {"ok": True, "filename": filename, "zipBase64": base64.standard_b64encode(data).decode("ascii")},
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.export failed: %s", exc)
        return _err(rid, 5031, f"pet.export failed: {exc}")


@method("pet.rename")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Rename an installed pet's display name + realign its slug/dir.

    Params: ``slug`` + ``name`` (both required). Lets the generate flow hatch
    with a provisional name and apply the user's chosen name at adopt time.
    Returns ``{ok, slug, displayName}`` with the (possibly new) slug.
    """
    slug = str(params.get("slug") or "").strip()
    name = str(params.get("name") or "").strip()
    if not slug:
        return _err(rid, 4004, "missing slug")
    if not name:
        return _err(rid, 4004, "missing name")
    try:
        from agent.pet import store

        new_slug = store.rename_pet(slug, name)
        if not new_slug:
            return _err(rid, 5031, "pet.rename failed")

        # The dir may have moved; if the renamed pet was active, follow the slug
        # in config so surfaces don't point at the old (now-missing) directory.
        if new_slug != slug:
            try:
                from hermes_cli.pets import _rename_active_if

                _rename_active_if(slug, new_slug)
            except Exception as exc:  # noqa: BLE001 - rename already succeeded
                logger.debug("pet.rename config update failed: %s", exc)

        return _ok(rid, {"ok": True, "slug": new_slug, "displayName": name})
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.rename failed: %s", exc)
        return _err(rid, 5031, f"pet.rename failed: {exc}")


@method("pet.thumb")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Return a small idle-frame PNG (data URI) for one pet — the picker preview.

    Cropped + cached server-side so the renderer gets a same-origin data URL
    instead of a CDN ``<img>`` (which the desktop CSP / R2 hotlink rules break).
    Params: ``slug`` (required), ``url`` (optional petdex spritesheet URL used
    only for not-yet-installed pets). Fail-open: ``{ok: false}`` with no error.
    """
    slug = str(params.get("slug") or "").strip()
    if not slug:
        return _err(rid, 4004, "missing slug")
    try:
        import base64

        from agent.pet import store

        data = store.thumbnail_png(slug, source_url=str(params.get("url") or ""))
        if not data:
            return _ok(rid, {"ok": False, "slug": slug})

        return _ok(
            rid,
            {
                "ok": True,
                "slug": slug,
                "dataUri": "data:image/png;base64," + base64.standard_b64encode(data).decode("ascii"),
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.thumb failed: %s", exc)
        return _ok(rid, {"ok": False, "slug": slug})


@method("pet.disable")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Turn the pet off from the desktop picker (``display.pet.enabled=false``)."""
    try:
        from hermes_cli.pets import _set_enabled

        _set_enabled(False)
        return _ok(rid, {"ok": True})
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.disable failed: %s", exc)
        return _err(rid, 5031, f"pet.disable failed: {exc}")


@method("pet.scale")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Persist ``display.pet.scale`` from the desktop slider. Params: ``scale``.

    Clamped to the engine bounds. The renderer updates its own ``$petInfo`` for
    instant feedback; this just makes the change durable + visible to the other
    terminal surfaces on their next read.
    """
    try:
        from hermes_cli.pets import set_pet_scale

        scale, err = set_pet_scale(params.get("scale"))
        if err:
            return _err(rid, 4004, err)
        return _ok(rid, {"ok": True, "scale": scale})
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.scale failed: %s", exc)
        return _err(rid, 5031, f"pet.scale failed: {exc}")


def _pet_gen_root():
    """Profile-scoped staging dir for in-progress generation drafts."""
    from hermes_constants import get_hermes_home

    root = get_hermes_home() / "cache" / "pet-gen"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _pet_gen_sweep(root, *, max_age_s: float = 3600.0) -> None:
    """Drop stale draft staging dirs so cache never grows unbounded."""
    import shutil
    import time

    try:
        now = time.time()
        for child in root.iterdir():
            if child.is_dir() and now - child.stat().st_mtime > max_age_s:
                shutil.rmtree(child, ignore_errors=True)
    except Exception as exc:  # noqa: BLE001 - cleanup is best-effort
        logger.debug("pet-gen sweep failed: %s", exc)


def _pet_png_data_uri(path, *, max_px: int = 160) -> str:
    """Downscaled PNG data URI for a draft image (small preview payload)."""
    import base64
    import io

    from PIL import Image

    with Image.open(path) as opened:
        img = opened.convert("RGBA")
    img.thumbnail((max_px, max_px), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.standard_b64encode(buf.getvalue()).decode("ascii")


# Cooperative cancellation for the heavy pet generation paths. The client's Stop
# aborts its RPC immediately, but the worker-pool generation keeps running unless
# told to stop — pet.cancel flips a token's flag, which generate_base_drafts /
# hatch_pet poll between provider calls to skip work they haven't started.
_pet_cancel_lock = threading.Lock()
_pet_cancelled: set[str] = set()
_PET_REFERENCE_MIME_EXT = {
    "png": "png",
    "jpeg": "jpg",
    "jpg": "jpg",
    "webp": "webp",
    "gif": "gif",
}
try:
    _PET_REFERENCE_MAX_BYTES = max(
        1,
        int(os.environ.get("ARGUS_PET_REFERENCE_MAX_BYTES") or str(16 * 1024 * 1024)),
    )
except (TypeError, ValueError):
    _PET_REFERENCE_MAX_BYTES = 16 * 1024 * 1024


def _pet_reference_images_from_data_url(ref_raw: str, stage) -> list:
    """Decode + validate a reference-image data URL into the stage dir."""
    import base64
    import binascii
    import re as _re

    match = _re.match(r"^data:image/([a-zA-Z0-9.+-]+);base64,(.*)$", ref_raw, _re.DOTALL)
    if not match:
        raise ValueError("invalid reference image format")

    mime = match.group(1).lower()
    ext = _PET_REFERENCE_MIME_EXT.get(mime)
    if ext is None:
        raise ValueError("unsupported reference image type")

    payload = "".join(match.group(2).split())
    approx = (len(payload) * 3) // 4
    if approx > _PET_REFERENCE_MAX_BYTES:
        raise ValueError("reference image too large")

    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("invalid reference image data") from exc

    if len(raw) > _PET_REFERENCE_MAX_BYTES:
        raise ValueError("reference image too large")

    ref_path = stage / f"reference.{ext}"
    ref_path.write_bytes(raw)
    return [ref_path]


def _pet_cancel_arm(token: str) -> None:
    """Clear a stale cancel flag at the start of a generate/hatch run."""
    with _pet_cancel_lock:
        _pet_cancelled.discard(token)


def _pet_cancel_request(token: str) -> None:
    with _pet_cancel_lock:
        _pet_cancelled.add(token)


def _pet_is_cancelled(token: str) -> bool:
    with _pet_cancel_lock:
        return token in _pet_cancelled


def _pet_cancel_release(token: str) -> None:
    with _pet_cancel_lock:
        _pet_cancelled.discard(token)


@method("pet.cancel")
def _(rid, params: dict) -> dict:
    """Signal an in-flight ``pet.generate``/``pet.hatch`` (by token) to stop.

    Best-effort + idempotent: cancelling an unknown/finished token is a no-op.
    Stays off the worker pool so it lands while a heavy generation is occupying
    it. Returns ``{ok: True}``.
    """
    token = str(params.get("token") or "").strip()
    if token:
        _pet_cancel_request(token)
    return _ok(rid, {"ok": True})


@method("pet.generate.status")
def _(rid, params: dict) -> dict:
    """Whether pet generation is possible right now.

    True only when a reference-capable image backend (Nous Portal / OpenRouter /
    OpenAI gpt-image) is configured — the desktop checks this on open so it can
    offer setup instead of a dead prompt. Cheap (config + plugin discovery).
    """
    try:
        from agent.pet.generate.imagegen import (
            GenerationError,
            list_sprite_providers,
            resolve_provider,
        )

        try:
            resolve_provider(require_references=True)
            available = True
        except GenerationError:
            available = False
        try:
            providers = list_sprite_providers()
        except Exception as exc:  # noqa: BLE001 - picker is best-effort
            logger.debug("pet provider list failed: %s", exc)
            providers = []
        return _ok(rid, {"available": available, "providers": providers})
    except Exception as exc:  # noqa: BLE001 - never break the surface
        logger.debug("pet.generate.status failed: %s", exc)
        return _ok(rid, {"available": False, "providers": []})


@method("pet.generate")
def _(rid, params: dict) -> dict:
    """Generate candidate base looks for a new pet (the draft/variant step).

    Params: ``prompt`` (required unless ``referenceImage`` is given), ``count``
    (default 4), ``style`` (default ``auto``), ``referenceImage`` (optional data
    URL — a user photo/reference every draft is grounded on, e.g. to make *their*
    pet). Returns ``{ok, token, drafts:[{index, dataUri}]}`` — the token keys the
    staged base images for a later ``pet.hatch``. Heavy (network): worker pool.
    """
    prompt = str(params.get("prompt") or "").strip()
    ref_raw = str(params.get("referenceImage") or "").strip()
    if not prompt and not ref_raw:
        return _err(rid, 4004, "missing prompt")
    try:
        count = max(1, min(4, int(params.get("count") or 4)))
    except (TypeError, ValueError):
        count = 4
    style = str(params.get("style") or "auto").strip() or "auto"

    try:
        import shutil
        import uuid

        from agent.pet.generate import generate_base_drafts
        from agent.pet.generate.imagegen import GenerationError, resolve_provider

        root = _pet_gen_root()
        _pet_gen_sweep(root)

        # Token up front so each draft can be staged + streamed the moment it
        # lands, instead of the user staring at a blank grid until all N finish.
        token = uuid.uuid4().hex[:12]
        _pet_cancel_arm(token)
        stage = root / token
        stage.mkdir(parents=True, exist_ok=True)

        reference_images = None
        if ref_raw:
            try:
                reference_images = _pet_reference_images_from_data_url(ref_raw, stage)
            except ValueError as exc:
                _pet_cancel_release(token)
                return _err(rid, 4004, str(exc))

        # Optional desktop picker override: resolve the chosen provider up front so
        # a bad/uncredentialed pick fails fast instead of mid-fan-out.
        provider_name = str(params.get("provider") or "").strip()
        sprite = None
        if provider_name:
            try:
                sprite = resolve_provider(require_references=bool(reference_images), prefer=provider_name)
            except GenerationError as exc:
                _pet_cancel_release(token)
                return _err(rid, 5031, str(exc))

        concept = prompt or "a pet based on the reference image"
        out: list[dict] = []

        # Hand the token to the client up front (token-only init event) so a Stop
        # fired before the first draft lands can still target this run.
        try:
            _emit("pet.generate.progress", "", {"token": token, "count": count})
        except Exception as exc:  # noqa: BLE001 - streaming is best-effort
            logger.debug("pet.generate init emit failed: %s", exc)

        def _on_draft(index: int, src) -> None:
            dest = stage / f"draft-{index}.png"
            try:
                shutil.copyfile(src, dest)
                data_uri = _pet_png_data_uri(dest)
            except Exception as exc:  # noqa: BLE001 - skip a bad draft, keep the rest
                logger.debug("pet.generate draft %d failed: %s", index, exc)
                return
            out.append({"index": index, "dataUri": data_uri})
            # Stream this draft to the client so the grid fills in live. Best-
            # effort: a transport hiccup must not abort the generation itself.
            try:
                _emit(
                    "pet.generate.progress",
                    "",
                    {"token": token, "index": index, "dataUri": data_uri, "count": count},
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("pet.generate progress emit failed: %s", exc)

        try:
            generate_base_drafts(
                concept,
                n=count,
                style=style,
                reference_images=reference_images,
                provider=sprite,
                on_draft=_on_draft,
                is_cancelled=lambda: _pet_is_cancelled(token),
            )
        except GenerationError as exc:
            _pet_cancel_release(token)
            return _err(rid, 5031, str(exc))

        cancelled = _pet_is_cancelled(token)
        _pet_cancel_release(token)
        if cancelled:
            return _err(rid, 5031, "generation cancelled")
        if not out:
            return _err(rid, 5031, "generation produced no usable drafts")
        out.sort(key=lambda d: d["index"])
        return _ok(rid, {"ok": True, "token": token, "drafts": out})
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.generate failed: %s", exc)
        return _err(rid, 5031, f"pet.generate failed: {exc}")


@method("pet.hatch")
def _(rid, params: dict) -> dict:
    """Turn a chosen base draft into a full pet — installed but NOT yet active.

    Generation is expensive and the result varies, so hatch produces a *preview*
    the surface plays (all frames) before the user commits: the pet is written to
    the store (so it can be rendered + later activated) but the active pet is left
    untouched. Adopt with ``pet.select`` or throw it away with ``pet.remove``.

    Params: ``token`` + ``index`` (from ``pet.generate``), ``name`` (required),
    ``description`` (optional), ``prompt`` (optional concept for row prompts),
    ``style`` (optional). Returns ``{ok, slug, displayName, warnings, pet}`` where
    ``pet`` is the renderer payload. Heavy (network + raster): worker pool.
    """
    token = str(params.get("token") or "").strip()
    # Hatch cancellation rides its own key, not the generation token: hatching a
    # draft mid-generation means pet.generate is still releasing `token`, which
    # would otherwise wipe the arm we set here. Falls back to `token` for clients
    # that don't send one.
    cancel_token = str(params.get("cancelToken") or "").strip() or token
    index = params.get("index", 0)
    name = str(params.get("name") or "").strip()
    if not token:
        return _err(rid, 4004, "missing token")
    if not name:
        return _err(rid, 4004, "missing name")
    try:
        index = int(index)
    except (TypeError, ValueError):
        index = 0

    try:
        from agent.pet import store
        from agent.pet.generate import hatch_pet
        from agent.pet.generate.imagegen import GenerationError, resolve_provider

        base = _pet_gen_root() / token / f"draft-{index}.png"
        if not base.is_file():
            return _err(rid, 4004, "draft expired — generate again")

        # Optional desktop picker override (rows always need reference grounding).
        provider_name = str(params.get("provider") or "").strip()
        sprite = None
        if provider_name:
            try:
                sprite = resolve_provider(require_references=True, prefer=provider_name)
            except GenerationError as exc:
                return _err(rid, 5031, str(exc))

        _pet_cancel_arm(cancel_token)
        slug = store.unique_slug(name)

        def _on_progress(event: str, detail: str) -> None:
            # Row progress is encoded as "<state>:<done>:<total>" so the egg
            # screen can show "Drawing <state>… (n/total)"; other phases
            # (compose, save) pass through as-is. Best-effort streaming.
            payload: dict = {"event": event, "detail": detail}
            if event == "row" and detail.count(":") == 2:
                state, done, total = detail.split(":")
                payload = {"event": "row", "state": state, "done": done, "total": total}
            try:
                _emit("pet.hatch.progress", "", payload)
            except Exception as exc:  # noqa: BLE001
                logger.debug("pet.hatch progress emit failed: %s", exc)

        try:
            result = hatch_pet(
                base_image=base,
                slug=slug,
                display_name=name,
                description=str(params.get("description") or ""),
                concept=str(params.get("prompt") or name),
                style=str(params.get("style") or "auto").strip() or "auto",
                provider=sprite,
                on_progress=_on_progress,
                is_cancelled=lambda: _pet_is_cancelled(cancel_token),
            )
        except GenerationError as exc:
            return _err(rid, 5031, str(exc))
        finally:
            _pet_cancel_release(cancel_token)

        pet = store.load_pet(result.slug)
        payload = _pet_sprite_payload(pet, scale=_pet_config_scale()) if pet else {}
        return _ok(
            rid,
            {
                "ok": True,
                "slug": result.slug,
                "displayName": result.display_name,
                "warnings": result.validation.get("warnings", []),
                "pet": payload,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.hatch failed: %s", exc)
        return _err(rid, 5031, f"pet.hatch failed: {exc}")


@method("credits.view")
def _(rid, params: dict) -> dict:
    """Structured Nous credit view for the TUI /credits command.

    Account-independent (a portal fetch gated on "a Nous account is logged in"),
    so it works with no live agent / on a resumed session — same as the /usage
    credits block. Returns the surface-agnostic CreditsView fields so the TUI can
    render a clickable top-up <Link>. Fail-open: a portal hiccup or logged-out
    account yields {logged_in: false}, never an error the user has to parse.
    """
    try:
        from agent.account_usage import build_credits_view

        view = build_credits_view()
        return _ok(
            rid,
            {
                "logged_in": bool(view.logged_in),
                "balance_lines": [
                    line for line in view.balance_lines if not line.lstrip().startswith("📈")
                ],
                "identity_line": view.identity_line,
                "topup_url": view.topup_url,
                "depleted": bool(view.depleted),
            },
        )
    except Exception:
        # Fail-open: TUI treats this as "not logged in" and shows the prompt.
        return _ok(rid, {"logged_in": False, "balance_lines": [], "identity_line": None, "topup_url": None, "depleted": False})


# ===========================================================================
# Phase 2b terminal billing RPC methods
# ===========================================================================
#
# These return STRUCTURED success envelopes (result.ok / result.error) rather
# than JSON-RPC-level errors, so the TUI's rpc() promise always resolves and the
# Ink side can branch on the typed billing error code (insufficient_scope,
# rate_limited, no_payment_method, …) to render the right affordance instead of
# landing in a generic catch. The data-building lives in the shared core
# (agent/billing_view.py + hermes_cli/nous_billing.py) — same as /credits.


def _serialize_billing_error(exc) -> dict:
    """Map a BillingError into the result.error envelope the TUI branches on."""
    from hermes_cli.nous_billing import (
        BillingRateLimited,
        BillingScopeRequired,
    )

    kind = "error"
    if isinstance(exc, BillingScopeRequired):
        kind = "insufficient_scope"
    elif isinstance(exc, BillingRateLimited):
        kind = "rate_limited"
    elif getattr(exc, "error", None):
        kind = str(exc.error)
    return {
        "ok": False,
        "error": kind,
        "message": str(exc),
        "portal_url": getattr(exc, "portal_url", None),
        "retry_after": getattr(exc, "retry_after", None),
        "payload": getattr(exc, "payload", {}) or {},
    }


def _serialize_billing_state(state) -> dict:
    """Serialize a BillingState for the wire (Decimals → strings, money-safe)."""
    from agent.billing_view import format_money

    def _s(value):
        return None if value is None else str(value)

    card = None
    if state.card is not None:
        card = {"brand": state.card.brand, "last4": state.card.last4, "masked": state.card.masked}
    monthly_cap = None
    if state.monthly_cap is not None:
        mc = state.monthly_cap
        monthly_cap = {
            "limit_usd": _s(mc.limit_usd),
            "limit_display": format_money(mc.limit_usd),
            "spent_this_month_usd": _s(mc.spent_this_month_usd),
            "spent_display": format_money(mc.spent_this_month_usd),
            "is_default_ceiling": mc.is_default_ceiling,
        }
    auto_reload = None
    if state.auto_reload is not None:
        ar = state.auto_reload
        auto_reload = {
            "enabled": ar.enabled,
            "threshold_usd": _s(ar.threshold_usd),
            "threshold_display": format_money(ar.threshold_usd),
            "reload_to_usd": _s(ar.reload_to_usd),
            "reload_to_display": format_money(ar.reload_to_usd),
        }
    return {
        "ok": True,
        "logged_in": state.logged_in,
        "org_name": state.org_name,
        "org_slug": state.org_slug,
        "role": state.role,
        "is_admin": state.is_admin,
        "can_charge": state.can_charge,
        "balance_usd": _s(state.balance_usd),
        "balance_display": format_money(state.balance_usd),
        "cli_billing_enabled": state.cli_billing_enabled,
        "charge_presets": [_s(p) for p in state.charge_presets],
        "charge_presets_display": [format_money(p) for p in state.charge_presets],
        "min_usd": _s(state.min_usd),
        "max_usd": _s(state.max_usd),
        "card": card,
        "monthly_cap": monthly_cap,
        "auto_reload": auto_reload,
        "portal_url": state.portal_url,
        "error": state.error,
    }


@method("billing.state")
def _(rid, params: dict) -> dict:
    """GET /api/billing/state → serialized BillingState (Screen 1 + 5).

    Fail-open like credits.view: a logged-out / unreachable portal yields
    {ok:true, logged_in:false}. No scope required for this endpoint.
    """
    try:
        from agent.billing_view import build_billing_state

        state = build_billing_state()
        return _ok(rid, _serialize_billing_state(state))
    except Exception:
        return _ok(rid, {"ok": True, "logged_in": False, "error": "could not load billing state"})


@method("billing.charge")
def _(rid, params: dict) -> dict:
    """POST /api/billing/charge → {ok, chargeId} or a typed error envelope.

    params: {amount_usd: str|number, idempotency_key?: str}. If no key is
    supplied, the server-side core mints a fresh one and returns it so the TUI can
    reuse it on retry of the SAME purchase.
    """
    from hermes_cli.nous_billing import BillingError, post_charge
    from agent.billing_view import new_idempotency_key

    amount = params.get("amount_usd")
    if amount is None:
        return _ok(rid, {"ok": False, "error": "invalid_request", "message": "amount_usd is required"})
    key = params.get("idempotency_key") or new_idempotency_key()
    try:
        result = post_charge(amount_usd=amount, idempotency_key=key)
        return _ok(rid, {"ok": True, "charge_id": result.get("chargeId"), "idempotency_key": key})
    except BillingError as exc:
        env = _serialize_billing_error(exc)
        env["idempotency_key"] = key  # so the TUI can reuse on retry
        return _ok(rid, env)
    except Exception as exc:
        return _ok(rid, {"ok": False, "error": "error", "message": str(exc), "idempotency_key": key})


@method("billing.charge_status")
def _(rid, params: dict) -> dict:
    """GET /api/billing/charge/{id} → {ok, status, ...} or typed error.

    The poll. Caller drives the 2s/5-min cadence; this is a single status read.
    """
    from hermes_cli.nous_billing import BillingError, get_charge_status

    charge_id = params.get("charge_id")
    if not charge_id:
        return _ok(rid, {"ok": False, "error": "invalid_charge_id", "message": "charge_id is required"})
    try:
        result = get_charge_status(charge_id)
        return _ok(
            rid,
            {
                "ok": True,
                "status": result.get("status"),
                "amount_usd": result.get("amountUsd"),
                "settled_at": result.get("settledAt"),
                "reason": result.get("reason"),
            },
        )
    except BillingError as exc:
        return _ok(rid, _serialize_billing_error(exc))
    except Exception as exc:
        return _ok(rid, {"ok": False, "error": "error", "message": str(exc)})


@method("billing.auto_reload")
def _(rid, params: dict) -> dict:
    """PATCH /api/billing/auto-top-up → {ok:true} or typed error (Screen 2).

    params: {enabled: bool, threshold: number, top_up_amount: number}.
    """
    from hermes_cli.nous_billing import BillingError, patch_auto_top_up

    try:
        enabled = bool(params.get("enabled"))
        threshold = params.get("threshold")
        top_up_amount = params.get("top_up_amount")
        if threshold is None or top_up_amount is None:
            return _ok(rid, {"ok": False, "error": "invalid_request", "message": "threshold and top_up_amount are required"})
        patch_auto_top_up(enabled=enabled, threshold=threshold, top_up_amount=top_up_amount)
        return _ok(rid, {"ok": True})
    except BillingError as exc:
        return _ok(rid, _serialize_billing_error(exc))
    except Exception as exc:
        return _ok(rid, {"ok": False, "error": "error", "message": str(exc)})


@method("billing.step_up")
def _(rid, params: dict) -> dict:
    """Run the lazy billing:manage step-up device flow → {ok, granted}.

    Triggered by the TUI after a billing call returns error=insufficient_scope.
    Returns granted:false when the server silently downscopes (non-admin / unticked).

    Runs on the thread pool (in _LONG_HANDLERS): the device flow blocks for the
    whole device-code lifetime (minutes), so it must not stall the main stdin loop.
    The verification URL/code reach the TUI via an out-of-band ``billing.step_up.
    verification`` event (a plain print would be dropped by the JSON-RPC stdout
    pipe), and the browser is opened TUI-side via openExternalUrl — never with the
    gateway's headless webbrowser.open (hence open_browser=False).
    """
    sid = params.get("session_id") or ""
    try:
        from hermes_cli.auth import step_up_nous_billing_scope

        def _on_verification(url: str, code: str) -> None:
            _emit(
                "billing.step_up.verification",
                sid,
                {"verification_url": url, "user_code": code},
            )

        granted = step_up_nous_billing_scope(
            open_browser=False, on_verification=_on_verification
        )
        return _ok(rid, {"ok": True, "granted": bool(granted)})
    except Exception as exc:
        return _ok(rid, {"ok": False, "error": "error", "message": str(exc), "granted": False})


@method("session.status")
def _(rid, params: dict) -> dict:
    session, err = _sess_nowait(params, rid)
    if err:
        return err

    from hermes_constants import display_hermes_home

    key = session.get("session_key") or params.get("session_id") or ""
    agent = session.get("agent")
    meta = {}
    db = _get_db()
    if db and key:
        try:
            meta = db.get_session(key) or {}
        except Exception:
            meta = {}

    def _dt(value, fallback: datetime | None = None) -> datetime:
        if value:
            try:
                return datetime.fromtimestamp(float(value))
            except Exception:
                pass
        return fallback or datetime.now()

    created = _dt(meta.get("started_at"))
    updated = created
    for field in ("updated_at", "last_updated_at", "last_activity_at"):
        if meta.get(field):
            updated = _dt(meta.get(field), created)
            break

    usage = _get_usage(agent) if agent is not None else {}
    provider = getattr(agent, "provider", None) or "unknown"
    model = getattr(agent, "model", None) or "(unknown)"
    lines = [
        "Hermes TUI Status",
        "",
        f"Session ID: {key}",
        f"Path: {display_hermes_home()}",
    ]
    title = (meta.get("title") or "").strip()
    if title:
        lines.append(f"Title: {title}")
    lines.extend(
        [
            f"Model: {model} ({provider})",
            f"Created: {created.strftime('%Y-%m-%d %H:%M')}",
            f"Last Activity: {updated.strftime('%Y-%m-%d %H:%M')}",
            f"Tokens: {int(usage.get('total') or 0):,}",
            f"Agent Running: {'Yes' if session.get('running') else 'No'}",
        ]
    )
    return _ok(rid, {"output": "\n".join(lines)})


@method("session.history")
def _(rid, params: dict) -> dict:
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    history = list(session.get("history", []))
    db = _get_db()
    if db is not None and session.get("session_key"):
        try:
            history = db.get_messages_as_conversation(
                session["session_key"], include_ancestors=True
            )
        except Exception:
            pass
    return _ok(
        rid,
        {
            "count": len(history),
            "messages": _history_to_messages(history),
        },
    )


@method("session.undo")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    # Reject during an in-flight turn.  If we mutated history while
    # the agent thread is running, prompt.submit's post-run history
    # write would either clobber the undo (version matches) or
    # silently drop the agent's output (version mismatch, see below).
    # Neither is what the user wants — make them /interrupt first.
    if session.get("running"):
        return _err(
            rid, 4009, "session busy — /interrupt the current turn before /undo"
        )
    removed = 0
    with session["history_lock"]:
        history = session.get("history", [])
        while history and history[-1].get("role") in {"assistant", "tool"}:
            history.pop()
            removed += 1
        if history and history[-1].get("role") == "user":
            history.pop()
            removed += 1
        if removed:
            session["history_version"] = int(session.get("history_version", 0)) + 1
    return _ok(rid, {"removed": removed})


@method("session.compress")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    if session.get("running"):
        return _err(
            rid, 4009, "session busy — /interrupt the current turn before /compress"
        )
    sid = params.get("session_id", "")
    focus_topic = str(params.get("focus_topic", "") or "").strip()
    try:
        from agent.manual_compression_feedback import summarize_manual_compression
        from agent.model_metadata import estimate_request_tokens_rough

        with session["history_lock"]:
            before_messages = list(session.get("history", []))
            history_version = int(session.get("history_version", 0))
        before_count = len(before_messages)
        _agent = session["agent"]
        _sys_prompt = getattr(_agent, "_cached_system_prompt", "") or ""
        _tools = getattr(_agent, "tools", None) or None
        before_tokens = (
            estimate_request_tokens_rough(
                before_messages, system_prompt=_sys_prompt, tools=_tools
            )
            if before_count
            else 0
        )

        if before_count >= 4:
            focus_suffix = f', focus: "{focus_topic}"' if focus_topic else ""
            _status_update(
                sid,
                "compressing",
                f"⠋ compressing {before_count} messages "
                f"(~{before_tokens:,} tok){focus_suffix}…",
            )

        try:
            removed, usage = _compress_session_history(
                session,
                focus_topic,
                approx_tokens=before_tokens,
                before_messages=before_messages,
                history_version=history_version,
            )
            with session["history_lock"]:
                messages = list(session.get("history", []))
            after_count = len(messages)
            # Re-read system prompt + tools after compression — _compress_context
            # may have rebuilt the system prompt (_cached_system_prompt=None).
            _sys_prompt_after = (
                getattr(_agent, "_cached_system_prompt", "") or _sys_prompt
            )
            _tools_after = getattr(_agent, "tools", None) or _tools
            after_tokens = (
                estimate_request_tokens_rough(
                    messages,
                    system_prompt=_sys_prompt_after,
                    tools=_tools_after,
                )
                if after_count
                else 0
            )
            agent = session["agent"]
            _sync_session_key_after_compress(sid, session)
            summary = summarize_manual_compression(
                before_messages, messages, before_tokens, after_tokens
            )
            info = _session_info(agent, session)
            _emit("session.info", sid, info)
            return _ok(
                rid,
                {
                    "status": "compressed",
                    "removed": removed,
                    "before_messages": before_count,
                    "after_messages": after_count,
                    "before_tokens": before_tokens,
                    "after_tokens": after_tokens,
                    "summary": summary,
                    "usage": usage,
                    "info": info,
                    "messages": messages,
                },
            )
        finally:
            # Always clear the pinned compressing status so the bar
            # reverts to neutral whether compaction succeeded, was a
            # no-op, or raised.
            _status_update(sid, "ready")
    except Exception as e:
        return _err(rid, 5005, str(e))


@method("session.save")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err

    agent = session["agent"]
    # Mirror the classic CLI /save: snapshot under the Hermes profile home
    # (~/.argus/sessions/saved/) rather than the project/workspace CWD, and
    # include the system prompt so the export matches the dashboard save.
    saved_dir = get_hermes_home() / "sessions" / "saved"
    try:
        saved_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return _err(rid, 5011, f"failed to create save directory {saved_dir}: {e}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = saved_dir / f"hermes_conversation_{timestamp}.json"

    with session["history_lock"]:
        messages = list(session.get("history", []))

    session_id = getattr(agent, "session_id", None) or session.get("session_key") or ""
    # Prefer the agent's session_start datetime (matches the classic CLI export);
    # fall back to the gateway session's created_at timestamp.
    agent_start = getattr(agent, "session_start", None)
    if isinstance(agent_start, datetime):
        session_start = agent_start.isoformat()
    else:
        created_at = session.get("created_at")
        session_start = (
            datetime.fromtimestamp(created_at).isoformat()
            if isinstance(created_at, (int, float))
            else ""
        )

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": getattr(agent, "model", ""),
                    "session_id": session_id,
                    "session_start": session_start,
                    "system_prompt": getattr(agent, "_cached_system_prompt", "") or "",
                    "messages": messages,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        return _ok(rid, {"file": str(path)})
    except Exception as e:
        return _err(rid, 5011, str(e))


@method("session.close")
def _(rid, params: dict) -> dict:
    sid = params.get("session_id", "")
    # Serialize against the WS-orphan reaper (which also pops under
    # _session_resume_lock) so a disconnect-reap and an explicit close can't
    # both tear the same session down. _close_session_by_id is the single
    # idempotent teardown path (pop + _teardown_session) and returns False
    # when the session is already gone.
    with _session_resume_lock:
        return _ok(rid, {"closed": _close_session_by_id(sid, end_reason="tui_close")})


@method("session.branch")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    db = _get_db()
    if db is None:
        return _db_unavailable_error(rid, code=5008)
    old_key = session["session_key"]
    with session["history_lock"]:
        history = [dict(msg) for msg in session.get("history", [])]
    if not history:
        return _err(rid, 4008, "nothing to branch — send a message first")
    new_key = _new_session_key()
    new_sid = uuid.uuid4().hex[:8]
    lease, limit_message = _claim_active_session_slot(new_key, live_session_id=new_sid)
    if limit_message is not None:
        return _err(rid, 4090, limit_message)
    branch_name = params.get("name", "")
    try:
        if branch_name:
            title = branch_name
        else:
            current = db.get_session_title(old_key) or "branch"
            title = (
                db.get_next_title_in_lineage(current)
                if hasattr(db, "get_next_title_in_lineage")
                else f"{current} (branch)"
            )
        db.create_session(
            new_key,
            source=_session_source(session),
            model=_resolve_model(),
            # Stable _branched_from marker so list_sessions_rich() keeps the
            # branch visible in /resume and /sessions. The TUI branch leaves
            # the parent live (no end_reason='branched'), so the legacy
            # end_reason heuristic never matches it — the marker is the only
            # thing that surfaces TUI branches. See issue #20856.
            model_config={"_branched_from": old_key},
            parent_session_id=old_key,
            cwd=_session_cwd(session),
        )
        for msg in history:
            db.append_message(
                session_id=new_key,
                role=msg.get("role", "user"),
                content=msg.get("content"),
            )
        db.set_session_title(new_key, title)
    except Exception as e:
        if lease is not None:
            lease.release()
        return _err(rid, 5008, f"branch failed: {e}")
    try:
        tokens = _set_session_context(new_key)
        try:
            agent = _make_agent(new_sid, new_key, session_id=new_key)
        finally:
            _clear_session_context(tokens)
        _init_session(
            new_sid, new_key, agent, list(history), cols=session.get("cols", 80)
        )
        if new_sid in _sessions:
            _sessions[new_sid]["active_session_lease"] = lease
    except Exception as e:
        if lease is not None:
            lease.release()
        return _err(rid, 5000, f"agent init failed on branch: {e}")
    return _ok(rid, {"session_id": new_sid, "title": title, "parent": old_key})


@method("session.interrupt")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    # Safety net: if the turn's run thread is already gone but `running` stayed
    # stuck (a crash/desync that skipped the run loop's `finally`), force-clear it
    # so the session can't be permanently bricked at 4009 "session busy" — every
    # send/restore/resume would otherwise reject until a full backend restart.
    # Always tell the agent to interrupt when the session claims a run is active:
    # stale flags are cleared below, and fresh turns clear the interrupt flag at
    # entry. This keeps a stale/missing thread handle from making Stop a no-op.
    run_thread = session.get("_run_thread")
    run_thread_alive = run_thread is not None and run_thread.is_alive()
    should_interrupt = bool(session.get("running"))
    if should_interrupt and hasattr(session["agent"], "interrupt"):
        session["agent"].interrupt()
    with session["history_lock"]:
        cancelled_prompts = [
            q for q in (session.get("queued_prompts") or [])
            if isinstance(q, dict)
        ]
        cancelled_voice_seqs = [
            q.get("voice_task_seq")
            for q in cancelled_prompts
            if q.get("voice_task_seq") is not None
        ]
        session["_turn_cancel_requested"] = True
        session["queued_prompt"] = None
        session["queued_prompts"] = []
    for voice_seq in cancelled_voice_seqs:
        _resolve_voice_task(session, voice_seq, "任务已被用户取消")
    # Every queued browser query owns a preallocated answer slot. Explicitly
    # close those slots when Stop clears the FIFO; a cancelled queued turn never
    # reaches _run_prompt_submit and cannot otherwise emit its own completion.
    for queued in cancelled_prompts:
        request_id = str(queued.get("client_request_id") or "")
        if request_id:
            _emit("message.complete", params.get("session_id", ""), {
                "request_id": request_id,
                "text": "已取消",
                "status": "cancelled",
            })
    if not run_thread_alive:
        with session["history_lock"]:
            if session.get("running"):
                session["running"] = False
                _clear_inflight_turn(session)

    # Stop = stop the TURN (cooperative interrupt above also kills the in-flight
    # foreground subprocess). Background processes the agent started (dev servers,
    # watchers) are intentionally left running — kill those individually with the
    # "x" on the task row (process.kill). Don't reap them here.
    # Scope the pending-prompt release to THIS session.  A global
    # _clear_pending() would collaterally cancel clarify/sudo/secret
    # prompts on unrelated sessions sharing the same tui_gateway
    # process, silently resolving them to empty strings.
    _clear_pending(params.get("session_id", ""))
    try:
        from tools.approval import resolve_gateway_approval

        resolve_gateway_approval(session["session_key"], "deny", resolve_all=True)
    except Exception:
        pass
    return _ok(rid, {"status": "interrupted"})


# ── Delegation: subagent tree observability + controls ───────────────
# Powers the TUI's /agents overlay (see ui-tui/src/components/agentsOverlay).
# The registry lives in tools/delegate_tool — these handlers are thin
# translators between JSON-RPC and the Python API.


@method("delegation.status")
def _(rid, params: dict) -> dict:
    from tools.delegate_tool import (
        is_spawn_paused,
        list_active_subagents,
        _get_max_concurrent_children,
        _get_max_spawn_depth,
    )

    return _ok(
        rid,
        {
            "active": list_active_subagents(),
            "paused": is_spawn_paused(),
            "max_spawn_depth": _get_max_spawn_depth(),
            "max_concurrent_children": _get_max_concurrent_children(),
        },
    )


@method("delegation.pause")
def _(rid, params: dict) -> dict:
    from tools.delegate_tool import set_spawn_paused

    paused = bool(params.get("paused", True))
    return _ok(rid, {"paused": set_spawn_paused(paused)})


@method("subagent.interrupt")
def _(rid, params: dict) -> dict:
    from tools.delegate_tool import interrupt_subagent

    subagent_id = str(params.get("subagent_id") or "").strip()
    if not subagent_id:
        return _err(rid, 4000, "subagent_id required")
    ok = interrupt_subagent(subagent_id)
    return _ok(rid, {"found": ok, "subagent_id": subagent_id})


# ── Spawn-tree snapshots: TUI-written, disk-persisted ────────────────
# The TUI is the source of truth for subagent state (it assembles payloads
# from the event stream).  On turn-complete it posts the final tree here;
# /replay and /replay-diff fetch past snapshots by session_id + filename.
#
# Layout:  $HERMES_HOME/spawn-trees/<session_id>/<timestamp>.json
# Each file contains { session_id, started_at, finished_at, subagents: [...] }.


def _spawn_trees_root():
    from hermes_constants import get_hermes_home

    root = get_hermes_home() / "spawn-trees"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _spawn_tree_session_dir(session_id: str):
    safe = (
        "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id) or "unknown"
    )
    d = _spawn_trees_root() / safe
    d.mkdir(parents=True, exist_ok=True)
    return d


# Per-session append-only index of lightweight snapshot metadata.  Read by
# `spawn_tree.list` so scanning doesn't require reading every full snapshot
# file (Copilot review on #14045).  One JSON object per line.
_SPAWN_TREE_INDEX = "_index.jsonl"


def _append_spawn_tree_index(session_dir, entry: dict) -> None:
    try:
        with (session_dir / _SPAWN_TREE_INDEX).open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        # Index is a cache — losing a line just means list() falls back
        # to a directory scan for that entry.  Never block the save.
        logger.debug("spawn_tree index append failed: %s", exc)


def _read_spawn_tree_index(session_dir) -> list[dict]:
    index_path = session_dir / _SPAWN_TREE_INDEX
    if not index_path.exists():
        return []
    out: list[dict] = []
    try:
        with index_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return out


@method("spawn_tree.save")
def _(rid, params: dict) -> dict:
    session_id = str(params.get("session_id") or "").strip()
    subagents = params.get("subagents") or []
    if not isinstance(subagents, list) or not subagents:
        return _err(rid, 4000, "subagents list required")

    from datetime import datetime

    started_at = params.get("started_at")
    finished_at = params.get("finished_at") or time.time()
    label = str(params.get("label") or "")
    ts = datetime.utcfromtimestamp(float(finished_at)).strftime("%Y%m%dT%H%M%S")
    fname = f"{ts}.json"
    d = _spawn_tree_session_dir(session_id or "default")
    path = d / fname
    try:
        payload = {
            "session_id": session_id,
            "started_at": float(started_at) if started_at else None,
            "finished_at": float(finished_at),
            "label": label,
            "subagents": subagents,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        return _err(rid, 5000, f"spawn_tree.save failed: {exc}")

    _append_spawn_tree_index(
        d,
        {
            "path": str(path),
            "session_id": session_id,
            "started_at": payload["started_at"],
            "finished_at": payload["finished_at"],
            "label": label,
            "count": len(subagents),
        },
    )

    return _ok(rid, {"path": str(path), "session_id": session_id})


@method("spawn_tree.list")
def _(rid, params: dict) -> dict:
    session_id = str(params.get("session_id") or "").strip()
    limit = int(params.get("limit") or 50)
    cross_session = bool(params.get("cross_session"))

    if cross_session:
        root = _spawn_trees_root()
        roots = [p for p in root.iterdir() if p.is_dir()]
    else:
        roots = [_spawn_tree_session_dir(session_id or "default")]

    entries: list[dict] = []
    for d in roots:
        indexed = _read_spawn_tree_index(d)
        if indexed:
            # Skip index entries whose snapshot file was manually deleted.
            entries.extend(
                e for e in indexed if (p := e.get("path")) and Path(p).exists()
            )
            continue

        # Fallback for legacy (pre-index) sessions: full scan.  O(N) reads
        # but only runs once per session until the next save writes the index.
        for p in d.glob("*.json"):
            if p.name == _SPAWN_TREE_INDEX:
                continue
            try:
                stat = p.stat()
                try:
                    raw = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    raw = {}
                subagents = raw.get("subagents") or []
                entries.append(
                    {
                        "path": str(p),
                        "session_id": raw.get("session_id") or d.name,
                        "finished_at": raw.get("finished_at") or stat.st_mtime,
                        "started_at": raw.get("started_at"),
                        "label": raw.get("label") or "",
                        "count": len(subagents) if isinstance(subagents, list) else 0,
                    }
                )
            except OSError:
                continue

    entries.sort(key=lambda e: e.get("finished_at") or 0, reverse=True)
    return _ok(rid, {"entries": entries[:limit]})


@method("spawn_tree.load")
def _(rid, params: dict) -> dict:
    from pathlib import Path

    raw_path = str(params.get("path") or "").strip()
    if not raw_path:
        return _err(rid, 4000, "path required")

    # Reject paths escaping the spawn-trees root.
    root = _spawn_trees_root().resolve()
    try:
        resolved = Path(raw_path).resolve()
        resolved.relative_to(root)
    except (ValueError, OSError) as exc:
        return _err(rid, 4030, f"path outside spawn-trees root: {exc}")

    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _err(rid, 5000, f"spawn_tree.load failed: {exc}")

    return _ok(rid, payload)


@method("session.steer")
def _(rid, params: dict) -> dict:
    """Inject a user message into the next tool result without interrupting.

    Mirrors AIAgent.steer(). Safe to call while a turn is running — the text
    lands on the last tool result of the next tool batch and the model sees
    it on its next iteration. No interrupt, no new user turn, no role
    alternation violation.
    """
    text = (params.get("text") or "").strip()
    if not text:
        return _err(rid, 4002, "text is required")
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    agent = session.get("agent")
    if agent is None or not hasattr(agent, "steer"):
        return _err(rid, 4010, "agent does not support steer")
    try:
        accepted = agent.steer(text)
    except Exception as exc:
        return _err(rid, 5000, f"steer failed: {exc}")
    return _ok(rid, {"status": "queued" if accepted else "rejected", "text": text})


@method("terminal.resize")
def _(rid, params: dict) -> dict:
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    session["cols"] = int(params.get("cols", 80))
    return _ok(rid, {"cols": session["cols"]})


# ── Methods: prompt ──────────────────────────────────────────────────


@method("prompt.submit")
def _(rid, params: dict) -> dict:
    sid, text = params.get("session_id", ""), params.get("text", "")
    client_request_id = _ensure_client_request_id(
        params.get("client_request_id"))
    truncate_user_ordinal = params.get("truncate_before_user_ordinal")
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    # Thinking on/off is derived from ``agent.reasoning_effort`` in config.yaml
    # (parse_reasoning_effort → reasoning_config). No per-request flag anymore;
    # _apply_mm_deep_thinking reads reasoning_config directly.
    # Re-bind to the current client transport for this request. This keeps
    # streaming events on the active websocket even if an earlier disconnect
    # or fallback moved the session transport to stdio.
    if (t := current_transport()) is not None:
        session["transport"] = t
    with session["history_lock"]:
        # ★ `or queued_prompts` closes the ordering hole at the turn boundary.
        #   The turn tail sets running=False under this lock and only re-acquires
        #   it afterwards to drain (see _run_prompt_submit's finally/_drain
        #   pair), so a submit landing in between used to find running=False and
        #   start immediately — overtaking messages that had been waiting in the
        #   FIFO. Refusing the direct path while anything is queued makes
        #   "sent means it waits its turn" hold for the whole queue.
        #   queue_only is forced in that case: there is no live turn to
        #   interrupt, and calling agent.interrupt() here would arm an interrupt
        #   against the NEXT turn instead.
        _queue_backlog = bool(session.get("queued_prompts"))
        if session.get("running") or _queue_backlog:
            # Don't reject a mid-turn prompt — queue it (and, by default,
            # interrupt the live turn) so it runs as the next turn. See
            # _handle_busy_submit for why the old "session busy" rejection
            # dropped messages when teardown outlived the client's retry window.
            _log_busy_diagnosis(sid, session)
            return _handle_busy_submit(
                rid,
                sid,
                session,
                text,
                t or session.get("transport"),
                # Multimodal's composer intentionally accepts more questions
                # while the foreground turn is streaming. Those submits must
                # wait in FIFO without aborting a direct answer already in flight.
                queue_only=(
                    bool(params.get("queue_if_busy", False))
                    or not session.get("running")
                ),
                metadata={
                    "client_request_id": client_request_id,
                },
            )
        # A watch session's run lives in the PARENT turn, so its own running
        # flag is False — without this, typing mid-run builds a second agent
        # racing the in-flight child on the same stored session (interleaved
        # transcript, stale fork). After the run completes, submitting is fine:
        # the upgrade resumes the child's transcript as a normal conversation.
        if session.get("lazy") and _child_run_active(str(session.get("session_key") or "")):
            return _err(rid, 4009, "subagent still running — wait for it to finish")
        if truncate_user_ordinal is not None:
            try:
                ordinal = int(truncate_user_ordinal)
            except (TypeError, ValueError):
                return _err(rid, 4004, "truncate_before_user_ordinal must be an integer")
            history = session.get("history", [])
            user_indices = [i for i, m in enumerate(history) if m.get("role") == "user"]
            if ordinal >= len(user_indices):
                return _err(rid, 4018, "target user message is no longer in session history")
            truncated = history[: user_indices[ordinal]]
            session["history"] = truncated
            session["history_version"] = int(session.get("history_version", 0)) + 1
            if (db := _get_db()) is not None:
                try:
                    db.replace_messages(session["session_key"], truncated)
                except Exception as exc:
                    print(f"[tui_gateway] prompt.submit: replace_messages failed: {exc}", file=sys.stderr)
        session["running"] = True
        session["_turn_cancel_requested"] = False
        session["last_active"] = time.time()
        _start_inflight_turn(session, text)
    _emit("multimodal.trajectory", sid, {
        "worker": "MainScheduler", "phase": "prompt_started",
        "origin": "user", "text": _inflight_text(text),
        "client_request_id": client_request_id,
        "queued_count": len(session.get("queued_prompts") or []),
    })

    # Persist the DB row lazily, now that the user has actually sent a message.
    _ensure_session_db_row(session)
    # A branch becomes real here: copy its parent's transcript into the row so it
    # resumes with full context (the agent won't persist the seed itself).
    _persist_branch_seed(session)
    _start_agent_build(sid, session)

    def run_after_agent_ready() -> None:
        err = _wait_agent(session, rid)
        if err:
            _emit(
                "error",
                sid,
                {
                    "request_id": client_request_id,
                    "message": err.get("error", {}).get(
                        "message", "agent initialization failed"
                    )
                },
            )
            with session["history_lock"]:
                session["running"] = False
                _clear_inflight_turn(session)
            return
        with session["history_lock"]:
            if session.get("_turn_cancel_requested") or not session.get("running"):
                session["running"] = False
                _clear_inflight_turn(session)
                return
        # ★ 真·用户输入: 前端已本地加过 user 气泡 → 不回显 (user_originated=True)。
        _run_prompt_submit(
            rid, sid, session, text,
            user_originated=True,
            client_request_id=client_request_id,
        )

    run_thread = threading.Thread(target=run_after_agent_ready, daemon=True)
    # Keep a handle so session.interrupt can tell a live turn from a stuck
    # `running` flag (a turn that died without clearing it) and recover the latter.
    session["_run_thread"] = run_thread
    run_thread.start()
    return _ok(rid, {
        "status": "streaming",
        "client_request_id": client_request_id,
    })


def _notification_event_belongs_elsewhere(session: dict, evt: dict) -> bool:
    """True if ``evt`` is owned by a *different* live session.

    Background-process events carry the ``session_key`` of the session that
    started the process. Since all desktop sessions share one process-wide
    completion queue, each poller must skip events it doesn't own so a
    background job's completion surfaces in the session that launched it — not
    whichever poller happened to dequeue first. Orphaned events (owner gone)
    and global/system events (empty ``session_key``) return False so the
    current poller still handles them rather than losing them.
    """
    evt_key = str(evt.get("session_key") or "")
    if not evt_key:
        return False
    if evt_key == str(session.get("session_key") or ""):
        return False
    try:
        with _sessions_lock:
            snapshot = list(_sessions.values())
    except Exception:
        # If we can't safely enumerate live sessions, fail open so we don't
        # crash the poller thread or drop the event.
        return False

    return any(
        s is not session and str(s.get("session_key") or "") == evt_key
        for s in snapshot
    )


def _notification_event_dedup_key(evt: dict) -> tuple:
    """Return the UI-emission identity for a process notification event.

    Completion events are terminal notifications for a background process, so
    they remain one-shot per process session. Watch-match events are not
    terminal: a single background process can legitimately match the same or
    different patterns many times, so include event-specific content to avoid
    suppressing later distinct matches from the same process.
    """
    evt_type = evt.get("type", "completion")
    evt_sid = evt.get("session_id", "")
    if evt_type == "watch_match":
        return (
            evt_sid,
            evt_type,
            evt.get("command", ""),
            evt.get("pattern", ""),
            evt.get("output", ""),
            evt.get("suppressed", 0),
            evt.get("message_id", ""),
        )
    if evt_type.startswith("watch_overflow_") or evt_type == "watch_disabled":
        return (
            evt_sid,
            evt_type,
            evt.get("command", ""),
            evt.get("message", ""),
            evt.get("suppressed", 0),
        )
    if evt_type == "async_delegation":
        # Async-delegation completions have no process session_id; without
        # this the fallthrough keys every one as ("", "async_delegation")
        # and the second completion's status update is suppressed forever.
        return (evt.get("delegation_id", ""), evt_type)
    return (evt_sid, evt_type)


def _notification_poller_loop(
    stop_event: threading.Event, sid: str, session: dict
) -> None:
    """Poll completion_queue and dispatch notifications autonomously.

    Runs in a daemon thread started by _init_session(). Emits a
    status.update (kind=process) for user visibility, then chains an
    agent turn via _run_prompt_submit if the session is idle.

    NOTE: The completion_queue is global (one per process). If multiple
    TUI sessions coexist, whichever poller wakes first grabs the event,
    even if the process was started by a different session. This matches
    CLI/gateway behavior (single session per process).
    """
    from tools.process_registry import process_registry, format_process_notification

    _emitted = set()  # dedup re-queued events so same completion isn't emitted 50 times while session is busy
    while not stop_event.is_set() and not session.get("_finalized"):
        # Between-turn drain of queued live_watcher completion hooks. A hook that
        # was enqueued while the session was idle (no running turn to chain from)
        # would otherwise wait for the next user prompt; fire it here. Idle-gated
        # inside _drain_watcher_hook, so it's a cheap no-op while the agent is busy.
        try:
            _drain_hid = f"__watcher_hook_poll__{int(time.time() * 1000)}"
            if _drain_watcher_hook(_drain_hid, sid, session):
                continue  # a hook turn ran; re-loop to check for the next one
        except Exception as _wh_exc:
            logger.debug("[mm-research] poller hook drain failed: %s", _wh_exc)

        # Between-turn drain of the unified queued_prompts FIFO (user keyboard +
        # VoiceAgent delegations both live here). A prompt enqueued while idle
        # would otherwise wait for the next turn to chain from. Idle-gated inside
        # _drain_queued_prompt, so it's a cheap no-op while the agent is busy.
        try:
            _voice_hid = f"__queued_poll__{int(time.time() * 1000)}"
            if _drain_queued_prompt(_voice_hid, sid, session):
                continue
        except Exception as _vh_exc:
            logger.debug("[mm-research] poller queued-prompt drain failed: %s", _vh_exc)

        try:
            evt = process_registry.completion_queue.get(timeout=0.5)
        except Exception:
            continue

        # Multiple desktop sessions share this one process-wide queue. Only
        # consume events that belong to *this* session — otherwise a background
        # process started in session A would surface its completion in whichever
        # session's poller happened to wake first (Ben's "reported in a
        # different session" bug). Leave foreign events for their owner.
        if _notification_event_belongs_elsewhere(session, evt):
            process_registry.completion_queue.put(evt)
            time.sleep(0.1)
            continue

        _evt_sid = evt.get("session_id", "")
        if evt.get("type") == "completion" and process_registry.is_completion_consumed(_evt_sid):
            continue

        text = format_process_notification(evt)
        if not text:
            continue

        # Only emit the same notification identity to TUI once — re-queued
        # completions get re-emitted every 0.5s otherwise when session is busy,
        # while distinct watch_match events from the same process must remain
        # visible independently.
        _dedup_key = _notification_event_dedup_key(evt)
        if _dedup_key not in _emitted:
            _emit("status.update", sid, {"kind": "process", "text": text})
            _emitted.add(_dedup_key)

        with session["history_lock"]:
            if session.get("running"):
                process_registry.completion_queue.put(evt)
                continue
            session["running"] = True

        rid = f"__notif__{int(time.time() * 1000)}"
        try:
            _emit("message.start", sid)
            _run_prompt_submit(rid, sid, session, text)
        except Exception as exc:
            print(
                f"[tui_gateway] notification poller dispatch failed: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            with session["history_lock"]:
                session["running"] = False

    # Drain any remaining events after stop signal (process all pending
    # before exiting so nothing is lost on shutdown). Events owned by other
    # live sessions are set aside and re-queued so their poller still sees them.
    deferred: list = []
    while not process_registry.completion_queue.empty():
        try:
            evt = process_registry.completion_queue.get_nowait()
        except Exception:
            break
        if _notification_event_belongs_elsewhere(session, evt):
            deferred.append(evt)
            continue
        _evt_sid = evt.get("session_id", "")
        if evt.get("type") == "completion" and process_registry.is_completion_consumed(_evt_sid):
            continue
        text = format_process_notification(evt)
        if not text:
            continue

        _dedup_key = _notification_event_dedup_key(evt)
        if _dedup_key not in _emitted:
            _emit("status.update", sid, {"kind": "process", "text": text})
            _emitted.add(_dedup_key)

        with session["history_lock"]:
            if session.get("running"):
                process_registry.completion_queue.put(evt)
                break
            session["running"] = True

        rid = f"__notif__{int(time.time() * 1000)}"
        try:
            _emit("message.start", sid)
            _run_prompt_submit(rid, sid, session, text)
        except Exception as exc:
            print(
                f"[tui_gateway] notification poller dispatch failed: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            with session["history_lock"]:
                session["running"] = False

    # Hand any other sessions' events back to the shared queue.
    for evt in deferred:
        process_registry.completion_queue.put(evt)


def _start_notification_poller(sid: str, session: dict) -> threading.Event:
    """Start the background notification poller for a TUI session."""
    stop = threading.Event()
    t = threading.Thread(
        target=_notification_poller_loop,
        args=(stop, sid, session),
        daemon=True,
    )
    t.start()
    return stop


# --------------------------------------------------------------------------- #
# Multimodal phase 3 — per-session video monitors (MonitorEngine).
#
# The old single per-session sync daemon (_multimodal_monitor_loop) has been
# replaced by agent.multimodal.monitor_engine.MonitorEngine: a WatcherAgent-style
# background thread hosting a private asyncio loop, where EACH monitor is a
# long-lived async job. The gateway only supplies session-aware callbacks (speak
# / notify / gate hooks) and shares the agent.mm_monitors registry by reference.
# See _maybe_start_monitor_engine below.
#
# The monitor subsystem is always on (the old global monitor_enabled master
# switch was removed in v33). No monitor job does anything until the user opts in
# via set_monitor — each monitor gates on its own per-monitor enabled flag.
# --------------------------------------------------------------------------- #


def _maybe_start_monitor_engine(sid: str, session: dict, frame_buffer):
    """Start the per-session MonitorEngine (async job container) if multimodal
    monitoring is available. The constructed engine is published into ``session``
    before startup so finalize always owns a stoppable reference; the return value
    is the ready engine or None. Best-effort - never breaks the agent build.

    The engine hosts one long-lived async job per monitor. The gateway supplies
    session-aware callbacks (speak / notify / gate hooks) and shares
    agent.mm_monitors by reference so set_monitor create/delete drives job spawn/
    cancel.
    """
    if frame_buffer is None:
        return None
    engine = None
    try:
        agent = session.get("agent")
        if agent is None:
            return None
        if not hasattr(agent, "mm_monitors") or agent.mm_monitors is None:
            agent.mm_monitors = {}
        from agent.multimodal.monitor_engine import MonitorEngine

        def _run_hook_turn(mid, label, task, result_text=""):
            """Run a monitor hook as an ephemeral internal main-agent turn.

            Mirrors _notification_poller_loop: runs _run_prompt_submit in this
            (short-lived) thread so the monitor loop isn't blocked. The busy gate
            (_monitor_hook_running / session.running) was already claimed by the
            caller under history_lock. _run_prompt_submit owns the guard until
            the asynchronous turn actually finishes.

            ★ hook_main_agent 使能时, 把本次命中的判定结果 (result_text) 按固定格式
            拼进指令一起发给主 Agent。
            """
            rid = f"__mon_hook__{int(time.time() * 1000)}"
            # ★ 固定拼接: <instruction>. Reference (subagent execution result): <result>
            task = _append_hook_result(task, result_text)
            # Send the hooked task to the main agent AS A USER MESSAGE. (不再加
            # '[监控「label」触发]' 方括号前缀 —— 该前导标签既进 LLM 又显示在前端,
            # 已按用户要求去掉。)
            _msg = task
            try:
                _run_prompt_submit(
                    rid,
                    sid,
                    session,
                    _msg,
                    internal_origin="monitor_hook",
                )
            except Exception as exc:
                logger.warning("[mm-monitor] hook turn dispatch failed (%s): %s", mid, exc)
                # _run_prompt_submit normally clears session.running; on a throw
                # before it takes over, release it so the session isn't stuck.
                _lk = session.get("history_lock")
                if _lk is not None:
                    with _lk:
                        session["running"] = False
                        session["_monitor_hook_running"] = False

        def _speak_cb(mid, m, text):
            lock = session.get("history_lock")
            _evidence = m.get("_delivery_evidence")
            if not isinstance(_evidence, dict):
                _evidence = None
            # ★ 二选一: 命中后【要么】触发主 Agent (hook)【要么】只弹提醒气泡, 不并存。
            #   hook_main_agent 使能 → 只走 hook, 不发气泡 / 不写 context / 不 TTS
            #   (否则 reopen 会重建出一条本不该有的气泡)。
            _hook_task = (str(m.get("hook_instruction") or "").strip()
                          if m.get("hook_main_agent") else "")
            if _hook_task:
                # ── 只触发主 Agent。空闲就立即运行；忙时进入和用户输入相同的
                #   FIFO，但标记为内部来源：不回显 synthetic user 气泡，
                #   也不写入主 Agent canonical history。
                if lock is None:
                    return True
                _label = str(m.get("label", "") or m.get("brief", "") or mid)[:40]
                _go = False
                _position = 0
                _queued_text = ""
                with lock:
                    if not session.get("running") and not session.get("_monitor_hook_running"):
                        session["running"] = True
                        session["_monitor_hook_running"] = True
                        _go = True
                    else:
                        # Queue while holding the same lock used by prompt.submit
                        # and turn teardown. Otherwise a completion can observe an
                        # empty queue between our busy check and append, stranding
                        # this hook until another unrelated turn finishes.
                        _queued_text = _append_hook_result(_hook_task, text)
                        _position = _enqueue_prompt(
                            session,
                            _queued_text,
                            session.get("transport"),
                            user_originated=False,
                            origin="monitor_hook",
                        )
                if _go:
                    # 命中判定结果 (text = verdict reason) 由 _run_hook_turn 按固定
                    # 格式拼进指令一起发给主 Agent。
                    threading.Thread(
                        target=_run_hook_turn,
                        args=(mid, _label, _hook_task, text),
                        name=f"mon-hook-{mid}", daemon=True).start()
                else:
                    logger.info(
                        "[mm-monitor] hook fire queued (main agent busy) "
                        "mid=%s position=%d", mid, _position)
                    _emit("multimodal.trajectory", sid, {
                        "worker": "MainScheduler",
                        "phase": "monitor_hook_queued",
                        "monitor_id": mid,
                        "queue_position": _position,
                        "text": _queued_text,
                    })
                return True

            # ── 无 hook: 只弹提醒气泡 (原行为)。 ──────────────────────────────
            _lbl = str(m.get("label", "") or "").strip()
            if not _lbl:
                _b = str(m.get("brief", "") or "").strip()
                _lbl = (_b[:10] + "…") if len(_b) > 10 else _b
            tag = {"source": "monitor", "monitor_id": mid,
                   "monitor_label": _lbl}
            # This is a side-channel notification, not a foreground AIAgent
            # turn. It can stream next to the main answer without claiming
            # session.running or blocking/being dropped by user conversation.
            _emit("message.start", sid, tag)
            _emit("message.delta", sid, {**tag, "text": text})
            _emit("message.complete", sid, {
                **tag,
                "text": text,
                "evidence": copy.deepcopy(_evidence),
            })
            # Persist to the sidechannel DB (NOT session["history"]). The main
            # agent never sees these alerts (they go to the right multimodal
            # panel via list_mm_monitor_alerts on session.resume). The
            # emit above is the live push; this write is for reopen restore.
            try:
                _durable_sid = str(session.get("session_key") or "")
                if _durable_sid:
                    with _session_db(session) as _side_db:
                        if _side_db is not None:
                            _side_db.insert_mm_monitor_alert(
                                _durable_sid, monitor_id=str(mid),
                                text=text, label=_lbl or None,
                                evidence=copy.deepcopy(_evidence),
                            )
            except Exception as _exc:
                logger.debug("[mm-monitor] sidechannel write failed: %s", _exc)
            live = (getattr(agent, "mm_monitors", None) or {}).get(mid)
            if live is not None:
                live["last_speak_ts"] = time.time()
            # TTS 播报旁路: 喇叭开时把 monitor 气泡交给 announcer。
            try:
                if bool(session.get("_mm_tts_on")):
                    _ann = _get_voice_agent(session)
                    if _ann is not None:
                        _ann.submit("monitor", text, task_id=str(mid))
            except Exception as _tx:
                logger.debug("monitor TTS announce skipped: %s", _tx)
            return True

        def _notify_cb(kind, mid, m, text):
            # ★ 监控【过程失败/停用】通知: 只发一个 multimodal.toast 事件, 前端在右侧面板
            #   最下方弹一个动画小框 (3s 淡出)。不进 LLM history、不发主 Agent 气泡 (不再
            #   _emit "error")、不再顶部错误通知。工具返回结果 (create/enable) 才进主 Agent,
            #   那是 role=tool 消息, 不走这里。
            try:
                # ★ 五态统一: "interrupted"(含原熔断 disabled) 属"任务已停/暂停"类 →
                #   warning; 其余(过程失败等)→ error。
                _emit("multimodal.toast", sid, {
                    "level": "warning" if kind in ("disabled", "interrupted") else "error",
                    "kind": kind, "monitor_id": mid, "text": text,
                })
            except Exception:
                pass
            try:
                _recompute_mm_monitor_active(agent)
                from tools.monitor_tool import _push_monitors_event
                _push_monitors_event(sid, agent)
            except Exception:
                pass

        def _is_source_off():
            # Authoritative: skip new-frame evaluation only when the video source
            # is OFF (frontend UI switch stopped it, or nothing ever captured) —
            # NOT the 10s frame-freshness gate (that's only for the main agent's
            # single-frame Q&A; a background monitor must not go blind on a static
            # scene / backgrounded tab). Prefer the WatcherAgent's is_source_live()
            # (set by multimodal.source_stopped); fall back to "ever captured".
            try:
                re = session.get("_mm_live_watcher_agent")
                if re is not None and hasattr(re, "is_source_live"):
                    return not re.is_source_live()
                buf = getattr(agent, "frame_buffer", None) or frame_buffer
                if buf is None:
                    return True
                return bool(getattr(buf, "size", 0) == 0
                            and getattr(buf, "_last_push_wall", None) is None)
            except Exception:
                return False

        def _is_session_busy():
            return bool(session.get("running"))

        def _monitor_emit_cb(event: str, payload: dict) -> None:
            """Relay engine progress and publish terminal registry state."""
            _emit(event, sid, payload)
            if (event != "multimodal.trajectory"
                    or not isinstance(payload, dict)
                    or payload.get("phase") != "job_done"):
                return
            monitor_id = str(payload.get("monitor_id") or "").strip()
            if not monitor_id:
                return
            live = (getattr(agent, "mm_monitors", None) or {}).get(monitor_id)
            trigger_mode = (
                live.get("trigger_mode") if isinstance(live, dict)
                else payload.get("trigger_mode")
            )
            try:
                _mark_monitor_tool_result_done(
                    session, monitor_id, trigger_mode=trigger_mode)
            except Exception:
                logger.debug(
                    "[mm-monitor] failed to persist completion receipt (%s)",
                    monitor_id,
                    exc_info=True,
                )
            _recompute_mm_monitor_active(agent)
            _push_mm_registries(sid, agent)

        # Publish before start under the same lifecycle lock used by
        # _finalize_session.  This closes the constructed-but-invisible window:
        # a concurrent finalize either wins first (and construction is refused),
        # or captures this exact engine and stops it.
        with _MM_ACTIVE_MEMORY_BACKENDS_LOCK:
            if session.get("_finalized"):
                logger.info(
                    "[mm-monitor] session finalized before engine build "
                    "(sid=%s)", sid,
                )
                return None
            existing = session.get("_mm_monitor_engine")
            if existing is not None:
                try:
                    if existing.is_healthy():
                        return existing
                except Exception:
                    pass
                # Another caller may still be inside start().  It already owns
                # the session slot; never construct a competing runtime.
                logger.info(
                    "[mm-monitor] engine startup already in progress "
                    "(sid=%s)", sid,
                )
                return None
            engine = MonitorEngine(
                frame_buffer,
                monitors_ref=agent.mm_monitors,
                sid=sid,
                emit_cb=_monitor_emit_cb,
                speak_cb=_speak_cb,
                notify_cb=_notify_cb,
                is_source_off=_is_source_off,
                is_session_busy=_is_session_busy,
            )
            session["_mm_monitor_engine"] = engine

        engine.start()
        healthy = engine.is_healthy()
        with _MM_ACTIVE_MEMORY_BACKENDS_LOCK:
            accepted = bool(
                healthy
                and not session.get("_finalized")
                and session.get("_mm_monitor_engine") is engine
            )
            if not accepted and session.get("_mm_monitor_engine") is engine:
                session["_mm_monitor_engine"] = None
        if not accepted:
            logger.warning(
                "[mm-monitor] engine failed readiness check (sid=%s)", sid)
            engine.stop()
            return None
        logger.info("[mm-monitor] engine started (sid=%s)", sid)
        return engine
    except Exception as exc:
        if engine is not None:
            with _MM_ACTIVE_MEMORY_BACKENDS_LOCK:
                if session.get("_mm_monitor_engine") is engine:
                    session["_mm_monitor_engine"] = None
            try:
                engine.stop()
            except Exception:
                pass
        logger.warning("[mm-monitor] engine not started: %s", exc, exc_info=True)
        return None


def _maybe_start_live_watcher_agent(sid, frame_buffer, memory_backend, session=None):
    """Start the on-demand multimodal WatcherAgent if multimodal is enabled.

    WatcherAgent is a sibling of MemoryBackend — it shares the same
    FrameBuffer (and the MemoryStore/ConversationLog via the memory_backend
    ref when present) but runs Router/Search/Recall on its OWN asyncio loop
    so that the main agent's synchronous tool handler can submit work and
    proactive results stream back via _emit.

    ``session`` is the session dict. The watcher is fully decoupled from the main
    agent chat: per-round process + final report go ONLY to the watcher panel
    (_emit). The one exception is an explicit completion HOOK (hook_main_agent) —
    that enqueues a user message for the main agent (queued, never dropped).
    """
    if frame_buffer is None:
        logger.info("[watcher] no frame_buffer; engine not started (sid=%s)", sid)
        return None
    engine = None
    watcher_key = None
    try:
        from agent.multimodal.hermes_glue import flatten_mm_config
        mm = flatten_mm_config(_load_cfg() or {})
        if not mm.get("enabled", True):
            logger.info("[watcher] settings.enabled=False; engine not started")
            return None
        # Multimodal memory is always on.  If its sole owning backend did not
        # reach READY, fail closed instead of quietly constructing a standalone
        # Watcher with a second MemoryStore/Embedding/Recall bundle.  Apart from
        # recreating split-brain state, that orphan Watcher would prevent a
        # later healthy memory-backend retry from replacing it because the two
        # runtimes no longer share the same resources.
        if memory_backend is None:
            logger.warning(
                "[watcher] memory backend unavailable; "
                "Watcher not started (sid=%s)", sid,
            )
            return None
        from agent.multimodal.watcher_engine import WatcherAgent

        def _on_delegation_start(request_id, brief):
            # Intentionally a no-op. The watcher writes nothing to history; the
            # set_live_watcher tool result already carries {request_id,
            # status:"running"} and IS the receipt shown to the user. All progress
            # and the final report go to the watcher panel via _emit.
            return

        def _on_round_report(request_id, round_idx, report_text):
            # Fired per productive analysis round. The watcher is now FULLY decoupled
            # from the main agent: per-round process/results live ONLY in the watcher
            # panel (UI emit). We do NOT touch session["history"] here — no turn-2
            # append, no lock contention with the main agent / user turns. The final
            # consolidated summary is pushed once at completion (watcher.final).
            if session is None:
                return
            text = (report_text or "").strip()
            if not text:
                return
            # Live UI only: append this round's report to the rid-anchored watcher
            # panel card (frontend routes report_append into the DeepWindow, not the
            # center chat).
            try:
                _emit("watcher.report_append", sid, {
                    "request_id": str(request_id),
                    "round": int(round_idx),
                    "text": text,
                })
            except Exception as _exc:
                logger.debug("[mm-watcher] round report UI emit failed: %s", _exc)
            # Persist each round's report to the sidechannel DB (NOT to
            # session["history"]). Restored on session.resume into the right
            # DeepWindow via list_mm_watcher_reports so a reopened session's
            # watcher panel does not lose its per-round segments. The main
            # agent LLM never sees these — the watcher is fully decoupled.
            try:
                _agent_w = session.get("agent")
                _went = (getattr(_agent_w, "mm_watchers", None) or {}).get(str(request_id))
                _wlabel = ""
                if isinstance(_went, dict):
                    # Match the live-bubble label logic: label first, fall back
                    # to task_instruction.
                    _wlabel = str(_went.get("label")
                                  or _went.get("task_instruction") or "").strip()
                _durable_sid = str(session.get("session_key") or "")
                if _durable_sid:
                    with _session_db(session) as _side_db:
                        if _side_db is not None:
                            _side_db.insert_mm_watcher_report(
                                _durable_sid, watcher_id=str(request_id),
                                round_idx=int(round_idx or 0),
                                text=text, label=_wlabel or None)
            except Exception as _exc:
                logger.debug("[mm-watcher] sidechannel write failed: %s", _exc)
            # ★ TTS 播报旁路 (开关开启时): 每轮【段解读】也交 announcer 播报 —— 之前只有
            #   watcher.final(最终报告)播, 导致"边看边给的实时战术分析"从不出声。announcer
            #   自带拥塞控制 (watcher 优先级 2, 播不过来会丢/加速), 不会把队列冲爆。
            try:
                if bool(session.get("_mm_tts_on")):
                    _ann = _get_voice_agent(session)
                    if _ann is not None:
                        _ann.submit("watcher", text, task_id=str(request_id))
            except Exception as _exc:
                logger.debug("[mm-watcher] round report TTS submit failed: %s", _exc)

            # Per-round reports intentionally stop here. They belong only to the
            # watcher panel + sidechannel store; the main agent is invoked once at
            # successful completion with the consolidated full report below.

        def _on_delegation_complete(
            request_id, brief, full_text, stop_reason="normal"
        ):
            # Fired exactly once after the engine has consolidated every segment.
            # stop_reason ∈ {normal, disabled, deleted, source_end, task_complete}.
            # Only source_end/task_complete satisfy the explicit completion-hook
            # contract; pause/delete never synthesize a main-agent turn.
            if session is None:
                return
            final_text = str(full_text or "").strip()
            successful = stop_reason not in ("disabled", "deleted")
            if successful:
                try:
                    _mark_watcher_tool_result_complete(session, str(request_id))
                except Exception as _exc:
                    logger.debug("[mm-watcher] mark-complete failed: %s", _exc)
            if successful and final_text:
                try:
                    _emit("watcher.final", sid, {
                        "request_id": str(request_id),
                        "brief": str(brief or "").strip().replace("\n", " ")[:40],
                        "text": final_text,
                    })
                    durable_sid = str(session.get("session_key") or "")
                    if durable_sid:
                        with _session_db(session) as _side_db:
                            if _side_db is not None:
                                _side_db.upsert_mm_watcher_final(
                                    durable_sid,
                                    watcher_id=str(request_id),
                                    text=final_text,
                                )
                    if bool(session.get("_mm_tts_on")):
                        announcer = _get_voice_agent(session)
                        if announcer is not None:
                            announcer.submit(
                                "watcher", final_text, task_id=str(request_id))
                except Exception as _exc:
                    logger.debug(
                        "[mm-watcher] final report delivery failed: %s", _exc)
            try:
                _emit("watcher.complete", sid, {
                    "request_id": str(request_id),
                    "brief": str(brief or "").strip().replace("\n", " ")[:40],
                    "stop_reason": str(stop_reason or "normal"),
                })
            except Exception as _exc:
                logger.debug("[mm-watcher] deep complete UI emit failed: %s", _exc)

            # Registry state + the one completion hook. The queued hook receives
            # the consolidated report, never an individual segment report.
            try:
                _agent = session.get("agent")
                _researches = getattr(_agent, "mm_watchers", None) or {}
                _ent = _researches.get(str(request_id))
                if _ent is not None:
                    if stop_reason == "disabled":
                        _ent["status"] = "interrupted"
                    else:
                        _ent["status"] = "done"
                    fire_hook = bool(
                        stop_reason in {"source_end", "task_complete"}
                        and _ent.get("hook_main_agent")
                        and str(_ent.get("hook_instruction") or "").strip()
                        and final_text
                    )
                    if fire_hook:
                        label = (
                            _ent.get("label")
                            or str(brief or "")[:20]
                            or str(request_id)
                        )
                        _enqueue_watcher_hook(
                            session,
                            rid=str(request_id),
                            label=str(label),
                            task=str(_ent.get("hook_instruction") or "").strip(),
                            report=final_text,
                        )
                        _run_research_hook_turn(str(request_id))
                    if stop_reason == "deleted":
                        _researches.pop(str(request_id), None)
                    try:
                        from tools.live_watcher_tool import _push_watchers_event
                        _push_watchers_event(
                            _agent and getattr(_agent, "session_id", "") or sid, _agent)
                    except Exception:
                        pass
            except Exception as _hexc:
                logger.debug(
                    "[mm-research] completion hook/cleanup failed: %s", _hexc)

        def _run_research_hook_turn(rid_str):
            """Attempt to fire the NEXT queued watcher-completion hook immediately.
            If the main agent is busy, _drain_watcher_hook is a no-op and the hook
            stays in the per-session FIFO, to be drained by the notification
            poller / turn tail when idle. Monitor hooks have the same no-drop
            contract through the foreground prompt FIFO.
            Runs the drain in a short-lived thread so the engine loop isn't blocked
            for the whole main-agent turn."""
            _drid = f"__research_hook__{int(time.time() * 1000)}"
            try:
                threading.Thread(
                    target=lambda: _drain_watcher_hook(_drid, sid, session),
                    name=f"research-hook-{rid_str}", daemon=True).start()
            except Exception as _texc:
                logger.warning("[mm-research] hook thread start failed (%s): %s",
                               rid_str, _texc)

        def _research_registry_cb(request_id):
            """Return the tool-layer research entry for the engine's batch-boundary
            read (op=update/delete). Thread-safe: plain dict read."""
            try:
                _agent = session.get("agent")
                return (getattr(_agent, "mm_watchers", None) or {}).get(str(request_id))
            except Exception:
                return None

        def _on_query_complete(task_id, parent_id, query, text, status):
            """Finish a QueryWorker-owned answer without re-entering AIAgent.

            ``parent_id`` is the browser's preallocated answer-slot id.  Results
            may finish out of order; routing by this stable id keeps Q2's answer
            under Q2 even when Q3 finishes first.  Persist a UI-only notice and
            a bounded sidecar ledger, but never splice the result into old main
            conversation messages (prompt-cache prefix remains immutable).
            """
            raw_answer = (text or "").strip()
            answer = raw_answer or "QueryWorker 没有返回可用结果。"
            now = time.time()
            original_query = str(query or "")
            internal_query = False
            lock = session.get("history_lock")
            if lock is not None:
                with lock:
                    ledger = session.setdefault("_mm_worker_tasks", {})
                    row = ledger.setdefault(str(task_id), {
                        "task_id": str(task_id),
                        "parent_user_message_id": str(parent_id),
                        "reply_owner": "query_worker",
                        "query": str(query or ""),
                        "created_at": now,
                    })
                    # The worker may receive an expanded recall/search brief.
                    # The existing dispatch row retains the user's exact text;
                    # that is the Q the main agent must remember as its own turn.
                    original_query = str(row.get("query") or query or "")
                    row.update({
                        "status": str(status or "complete"),
                        "completed_at": now,
                        "result": answer,
                    })
                    internal_query = _settle_internal_query_worker_locked(
                        session,
                        parent_id=str(parent_id),
                        task_id=str(task_id),
                    )
                    # Only a genuinely completed, non-empty answer becomes a
                    # conversational Q/A pair. Error/cancel/no-answer outcomes
                    # remain visible in the worker UI but are absent from the
                    # main agent's history -- both Q and A stay out.
                    if (
                        not internal_query
                        and str(status or "complete") == "complete"
                        and raw_answer
                    ):
                        results = session.setdefault("_mm_query_results", [])
                        results.append({
                            "task_id": str(task_id),
                            "parent_user_message_id": str(parent_id),
                            "projection_key": str(parent_id or task_id),
                            "query": original_query,
                            "answer": raw_answer,
                            "status": "complete",
                            "originated_at": float(row.get("created_at") or now),
                            "completed_at": now,
                            "projection_state": "pending",
                        })
                        # Bound already-delivered debug history while never
                        # dropping pending/reserved answers under query pressure.
                        while len(results) > 100:
                            committed_idx = next((
                                idx for idx, item in enumerate(results)
                                if isinstance(item, dict)
                                and item.get("projection_state") == "committed"
                            ), None)
                            if committed_idx is None:
                                break
                            del results[committed_idx]

            tag = {
                "source": "query_worker",
                "request_id": str(parent_id),
                "task_id": str(task_id),
            }
            _emit("message.complete", sid, {
                **tag,
                "text": answer,
                "status": str(status or "complete"),
            })
            # UI/reopen sidecar. `_strip_mm_context` removes the notice itself
            # from model history; a successful result is separately synchronized
            # as a plain historical user/assistant pair on the next turn.
            if not internal_query:
                _append_mm_context(
                    session, kind="query", text=answer,
                    event_id=str(parent_id), label=original_query,
                    status=str(status or "complete"),
                )
            try:
                if bool(session.get("_mm_tts_on")):
                    announcer = _get_voice_agent(session)
                    if announcer is not None:
                        announcer.submit("assistant", answer, task_id=str(task_id))
            except Exception as exc:
                logger.debug("query-worker TTS announce skipped: %s", exc)

        durable_id = str((session or {}).get("session_key") or sid)
        watcher_key = _mm_memory_backend_registry_key(durable_id, sid)

        # At most one Watcher may own a durable conversation while its thread
        # is alive. Construction + registry/session publication are atomic with
        # finalize; start/waits remain outside the lifecycle lock.
        while True:
            created = False
            with _MM_ACTIVE_MEMORY_BACKENDS_LOCK:
                if session is not None and session.get("_finalized"):
                    logger.info(
                        "[watcher] session finalized before Watcher build "
                        "(sid=%s)", sid,
                    )
                    return None

                existing = _MM_ACTIVE_WATCHERS.get(watcher_key)
                if existing is not None and _mm_watcher_is_stopped(existing):
                    if _MM_ACTIVE_WATCHERS.get(watcher_key) is existing:
                        _MM_ACTIVE_WATCHERS.pop(watcher_key, None)
                    existing = None

                if existing is None:
                    engine = WatcherAgent(
                        frame_buffer,
                        memory_backend=memory_backend,
                        emit_cb=lambda ev, payload: _emit(ev, sid, payload),
                        sid=sid,
                        on_delegation_complete=_on_delegation_complete,
                        on_delegation_start=_on_delegation_start,
                        on_round_report=_on_round_report,
                        research_registry_cb=_research_registry_cb,
                        on_query_complete=_on_query_complete,
                    )
                    _MM_ACTIVE_WATCHERS[watcher_key] = engine
                    if session is not None:
                        session["_mm_live_watcher_agent"] = engine
                    add_stopped_callback = getattr(
                        engine, "add_stopped_callback", None)
                    if callable(add_stopped_callback):
                        add_stopped_callback(
                            lambda stopped, _key=watcher_key, _engine=engine,
                            _session=session: _clear_mm_watcher_guard(
                                _key,
                                _engine if stopped is None else stopped,
                                _session,
                            )
                        )
                    created = True
                else:
                    engine = existing
                    same_owner = bool(
                        session is None
                        or session.get("_mm_live_watcher_agent") is engine
                    )
                    same_resources = bool(
                        getattr(engine, "frame_buffer", None) is frame_buffer
                        and getattr(engine, "_memory_backend", None)
                        is memory_backend
                    )

            if created:
                break

            # Only the original session may reuse its Watcher: every callback
            # captures that session/sid. A reopened transport must wait for the
            # old owner to stop, never silently inherit its callback routing.
            if same_owner and same_resources:
                if _mm_watcher_is_ready(engine):
                    return engine
                wait_ready = getattr(engine, "wait_ready", None)
                if (callable(wait_ready)
                        and wait_ready(_MM_WATCHER_STARTUP_TIMEOUT_SEC) is True
                        and _mm_watcher_is_ready(engine)):
                    return engine

            if _mm_watcher_is_stopped(engine):
                continue
            wait_stopped = getattr(engine, "wait_stopped", None)
            if (callable(wait_stopped)
                    and wait_stopped(_MM_WATCHER_STOP_TIMEOUT_SEC) is True):
                continue
            logger.warning(
                "[watcher] active Watcher guard refused a second runtime "
                "(sid=%s durable=%s same_owner=%s same_resources=%s state=%s)",
                sid, durable_id, same_owner, same_resources,
                getattr(engine, "state", "unknown"),
            )
            return None

        if not engine.start(timeout=_MM_WATCHER_STARTUP_TIMEOUT_SEC):
            logger.warning(
                "[watcher] engine failed to reach ready/healthy state "
                "(sid=%s): %s", sid,
                getattr(engine, "startup_error", None)
                or f"startup timed out after {_MM_WATCHER_STARTUP_TIMEOUT_SEC:.0f}s",
            )
            try:
                engine.stop(timeout=_MM_WATCHER_STOP_TIMEOUT_SEC)
            except TypeError:
                engine.stop()
            except Exception as stop_exc:
                logger.debug(
                    "[watcher] failed engine cleanup raised: %s", stop_exc)
            return None

        # Startup may complete in the same instant session.close wins. Since the
        # engine was published first, finalize can stop it; this check prevents a
        # ready-but-rejected runtime from escaping to its caller.
        with _MM_ACTIVE_MEMORY_BACKENDS_LOCK:
            accepted = bool(
                _MM_ACTIVE_WATCHERS.get(watcher_key) is engine
                and (session is None or (
                    not session.get("_finalized")
                    and session.get("_mm_live_watcher_agent") is engine
                ))
            )
        if not accepted:
            try:
                engine.stop(timeout=_MM_WATCHER_STOP_TIMEOUT_SEC)
            except TypeError:
                engine.stop()
            except Exception:
                pass
            return None
        logger.info("[watcher] engine started (sid=%s)", sid)
        return engine
    except Exception as exc:
        if engine is not None:
            try:
                engine.stop(timeout=_MM_WATCHER_STOP_TIMEOUT_SEC)
            except TypeError:
                try:
                    engine.stop()
                except Exception:
                    pass
            except Exception:
                pass
        logger.warning("[watcher] engine not started: %s", exc, exc_info=True)
        return None


def _mm_memory_backend_registry_key(session_id: str, sid: str = "") -> tuple[str, str]:
    """Return the process-local ownership key for one durable conversation."""
    try:
        profile_home = os.path.realpath(os.fspath(get_hermes_home()))
    except Exception:
        profile_home = os.path.realpath(os.fspath(_hermes_home))
    durable_id = str(session_id or sid or "").strip()
    return profile_home, durable_id


def _mm_memory_backend_is_stopped(backend: Any) -> bool:
    """Read the real lifecycle flag without treating Mock-like values as true."""
    try:
        return getattr(backend, "is_stopped", False) is True
    except Exception:
        return False


def _mm_watcher_is_ready(watcher: Any) -> bool:
    """Read a real Watcher readiness marker without Mock truthiness."""
    try:
        marker = getattr(watcher, "is_ready", None)
        if marker is True:
            return True
        if callable(marker):
            return marker() is True
    except Exception:
        pass
    # Compatibility for older WatcherAgent versions that expose only health.
    try:
        return getattr(watcher, "_healthy", False) is True
    except Exception:
        return False


def _mm_watcher_is_stopped(watcher: Any) -> bool:
    """Read a real Watcher stopped marker without Mock truthiness."""
    try:
        marker = getattr(watcher, "is_stopped", False)
        return marker() is True if callable(marker) else marker is True
    except Exception:
        return False


def _clear_mm_watcher_guard(
    key: tuple[str, str], watcher: Any, session: dict | None = None,
) -> None:
    """Release Watcher ownership only after its real owner thread stopped."""
    with _MM_ACTIVE_MEMORY_BACKENDS_LOCK:
        if _MM_ACTIVE_WATCHERS.get(key) is watcher:
            _MM_ACTIVE_WATCHERS.pop(key, None)
        if (session is not None
                and session.get("_mm_live_watcher_agent") is watcher):
            session["_mm_live_watcher_agent"] = None


def _clear_mm_memory_backend_guard(
    key: tuple[str, str], backend: Any,
) -> None:
    """Remove a registry entry iff it still belongs to the stopped backend."""
    with _MM_ACTIVE_MEMORY_BACKENDS_LOCK:
        if _MM_ACTIVE_MEMORY_BACKENDS.get(key) is backend:
            _MM_ACTIVE_MEMORY_BACKENDS.pop(key, None)


def _maybe_start_memory_backend(
    sid, session_id, frame_buffer, *, session: dict | None = None,
):
    """Start the independent layered-memory backend if enabled + buffer exists.

    Returns a READY MemoryBackend for Watcher consumption, or None.  ``session``
    receives the constructed backend immediately, even when startup later times
    out, so finalize never loses the only stoppable reference.
    Gated on multimodal.memory_enabled; best-effort (never breaks agent build).
    ``sid`` is the live transport id used for frontend events. ``session_id``
    is the durable conversation key persisted into the memory database so the
    debug UI can deterministically bind a database to a reopened session.
    """
    if frame_buffer is None:
        logger.info("[mm-memory] no frame_buffer; backend not started (sid=%s)", sid)
        return None
    backend = None
    try:
        from agent.multimodal.hermes_glue import build_config
        hermes_cfg = _load_cfg() or {}
        # Multimodal memory is ALWAYS ON — there is no memory_enabled opt-out
        # (the config knob was removed on purpose). An idle backend is cheap
        # (no frames → no work); a genuinely-broken one must LOUD-FAIL below with
        # its own real cause, not silently stay off.
        from agent.multimodal.memory_backend import MemoryBackend
        key = _mm_memory_backend_registry_key(session_id, sid)

        # At most one backend object may own a (profile, durable session) while
        # its thread is alive.  A stopped entry is pruned and may be rebuilt;
        # every other state remains guarded until bounded lifecycle waits below
        # prove the old thread has fully exited.
        while True:
            created = False
            with _MM_ACTIVE_MEMORY_BACKENDS_LOCK:
                if session is not None and session.get("_finalized"):
                    logger.info(
                        "[mm-memory] session finalized before backend build "
                        "(sid=%s)", sid,
                    )
                    return None
                existing = _MM_ACTIVE_MEMORY_BACKENDS.get(key)
                if existing is not None and _mm_memory_backend_is_stopped(existing):
                    _MM_ACTIVE_MEMORY_BACKENDS.pop(key, None)
                    existing = None

                if existing is None:
                    # Build Config once while the per-profile home override is
                    # active.  Construction, guard registration, and session
                    # publication occur under the same lock used by finalize.
                    runtime_config = build_config(
                        hermes_cfg, session_id=session_id)
                    backend = MemoryBackend(
                        frame_buffer,
                        hermes_cfg=hermes_cfg,
                        emit_cb=lambda ev, payload: _emit(ev, sid, payload),
                        session_id=session_id,
                        runtime_config=runtime_config,
                    )
                    _MM_ACTIVE_MEMORY_BACKENDS[key] = backend
                    if session is not None:
                        session["_mm_memory_backend"] = backend
                    add_stopped_callback = getattr(
                        backend, "add_stopped_callback", None)
                    if callable(add_stopped_callback):
                        add_stopped_callback(
                            lambda stopped, _key=key, _backend=backend:
                            _clear_mm_memory_backend_guard(
                                _key, _backend if stopped is None else stopped))
                    created = True
                else:
                    backend = existing
                    if session is not None:
                        session["_mm_memory_backend"] = backend

            if created:
                break

            same_buffer = getattr(backend, "frame_buffer", None) is frame_buffer
            if same_buffer and getattr(backend, "is_ready", False) is True:
                logger.info(
                    "[mm-memory] reusing ready backend (sid=%s durable=%s)",
                    sid, session_id,
                )
                return backend

            # A duplicate call can meet the first one during STARTING. Wait only
            # for the configured startup bound; never construct alongside it.
            if same_buffer:
                wait_ready = getattr(backend, "wait_ready", None)
                if (callable(wait_ready)
                        and wait_ready(_MM_MEMORY_STARTUP_TIMEOUT_SEC)
                        and getattr(backend, "is_ready", False) is True):
                    return backend

            if _mm_memory_backend_is_stopped(backend):
                continue
            wait_stopped = getattr(backend, "wait_stopped", None)
            if (callable(wait_stopped)
                    and wait_stopped(_MM_MEMORY_STOP_TIMEOUT_SEC)):
                continue
            logger.warning(
                "[mm-memory] active backend guard refused a second runtime "
                "(sid=%s durable=%s same_frame_buffer=%s state=%s)",
                sid, session_id, same_buffer,
                getattr(backend, "state", "unknown"),
            )
            return None

        if not backend.start(timeout=_MM_MEMORY_STARTUP_TIMEOUT_SEC):
            reason = backend.startup_error or (
                f"startup timed out after {_MM_MEMORY_STARTUP_TIMEOUT_SEC:.0f}s")
            logger.error(
                "[mm-memory] backend failed before ready (sid=%s): %s",
                sid, reason,
            )
            # LOUD FAIL: memory is always-on, so a startup failure is a real
            # error the user must see — surface the backend's OWN cause (e.g. the
            # OCR/rapidocr RuntimeError) verbatim, not a downstream euphemism.
            try:
                _emit("error", sid, {
                    "message": f"多模态记忆后端启动失败: {reason}",
                    "source": "mm_memory_backend",
                })
            except Exception:
                pass
            backend.stop(timeout=_MM_MEMORY_STOP_TIMEOUT_SEC)
            return None
        logger.info(
            "[mm-memory] backend ready (sid=%s db=%s)",
            sid, getattr(getattr(backend, "mem", None), "db_path", ""),
        )
        return backend
    except Exception as exc:
        if backend is not None:
            try:
                backend.stop(timeout=_MM_MEMORY_STOP_TIMEOUT_SEC)
            except TypeError:
                try:
                    backend.stop()
                except Exception:
                    pass
            except Exception:
                pass
        logger.error("[mm-memory] backend not started: %s", exc, exc_info=True)
        # LOUD FAIL: always-on memory hit a hard error building/starting — tell
        # the user its real cause instead of silently leaving the panels empty.
        try:
            _emit("error", sid, {
                "message": f"多模态记忆后端启动失败: {exc}",
                "source": "mm_memory_backend",
            })
        except Exception:
            pass
        return None


def _is_subagent_msg_with_event(msg: Any, kind: str, event_id: str) -> bool:
    """Check whether ``msg`` is a stored sub-agent history entry matching
    (kind, event_id). Used for D3 dedup: only the most recent sub-agent
    message per (kind, event_id) is kept in history for the LLM.

    Marker layout (see agent.prompt_builder.format_subagent_marker):
        [SUB-AGENT <kind>:<label> #<event_id>]\\n<body>
    We match on the first line only — the prefix + " #<event_id>]" — so a
    body that happens to contain a lookalike string can't cause a false
    dedup hit."""
    if not isinstance(msg, dict) or msg.get("role") != "user":
        return False
    content = msg.get("content")
    if not isinstance(content, str):
        return False
    first_line = content.split("\n", 1)[0]
    return (first_line.startswith(f"[SUB-AGENT {kind}:")
            and first_line.endswith(f" #{event_id}]"))


def _append_subagent_message(
    session: dict, *, kind: str, event_id: str, label: str, text: str,
) -> None:
    """Append a background worker's output to session["history"] as a
    role=user message wrapped in the sub-agent marker (see
    ``agent.prompt_builder.format_subagent_marker``).

    D3 dedup: any previous entry with the same (kind, event_id) is removed
    first, so the LLM only sees the most recent message per event. The UI
    stream is unaffected — the front-end keeps every bubble it received.

    Bumps ``history_version`` so a concurrent prompt.submit sees a fresh
    history (same pattern as the personality-marker append at line ~4028).
    """
    text = (text or "").strip()
    if not text:
        return
    try:
        from agent.prompt_builder import format_subagent_marker
        content = format_subagent_marker(kind, event_id, label, text)
    except Exception as exc:
        logger.debug("[mm-subagent] format marker failed: %s", exc)
        return
    lock = session.get("history_lock")
    if lock is None:
        return
    with lock:
        history = session.get("history", [])
        if not isinstance(history, list):
            history = []
        filtered = [
            m for m in history
            if not _is_subagent_msg_with_event(m, kind, event_id)
        ]
        filtered.append({"role": "user", "content": content})
        session["history"] = filtered
        session["history_version"] = int(session.get("history_version", 0)) + 1


# ── 功能1: monitor/watcher 通知作为 type 标记元素写入 session context ──────────
#   monitor 命中 / watcher 每轮报告向前端注入通知的同时, 也往 session["history"]
#   写一份【type=mm_notice】的元素。前端展示用【全量】context (含这些); 但每次请求
#   LLM 前用 _strip_mm_context 过滤掉它们 (见 conversation_history 组装处)。
#
#   持久化设计 (关键): DB messages 表没有自由列, 所以把标记信息塞进 content 里 ——
#   content 存一个 dict {"type":"mm_notice","mm_kind":...,"text":...,...}。
#   SessionDB._encode_content 会 JSON 序列化 dict content (\x00json: 前缀),
#   _decode_content 原样还原; 且 get_messages_as_conversation 只对 *str* content
#   跑 sanitize, dict content 完整保留 —— 于是 reopen 后仍能从 content.type 识别并
#   (a) 继续过滤出 LLM, (b) 由 _history_to_messages 重建成 monitor/watcher 气泡。
#
#   内存态同时冗余顶层 mm_kind/mm_event_id/mm_label 便于快速判定; DB round-trip 后
#   顶层键会丢, 靠 content.type=="mm_notice" 兜底 (见 _is_mm_notice)。
#   role 用合法值 assistant 以兼容 DB / 其它消费者。
MM_NOTICE_TYPE = "mm_notice"


def _is_mm_notice(msg: Any) -> bool:
    """判定一条 history 元素是否为 monitor/watcher 通知 (功能1 的 type 标记元素)。

    同时覆盖两种形态:
      • 内存态: 顶层带 mm_kind (刚 append, 还没经过 DB round-trip)。
      • 持久态: content 是 dict 且 content["type"]=="mm_notice" (reopen 后, 顶层
        mm_kind 已在 DB 写读中丢失)。"""
    if not isinstance(msg, dict):
        return False
    if msg.get("mm_kind"):
        return True
    content = msg.get("content")
    return isinstance(content, dict) and content.get("type") == MM_NOTICE_TYPE


def _mm_notice_fields(msg: dict) -> dict:
    """从一条 mm_notice history 元素中抽出标准字段。
    兼容内存态 (顶层键) 与持久态 (content dict)。"""
    content = msg.get("content")
    if isinstance(content, dict) and content.get("type") == MM_NOTICE_TYPE:
        return {
            "kind": str(content.get("mm_kind") or ""),
            "event_id": str(content.get("mm_event_id") or ""),
            "label": str(content.get("mm_label") or ""),
            "round": content.get("mm_round"),
            "status": str(content.get("mm_status") or ""),
            "text": str(content.get("text") or ""),
        }
    return {
        "kind": str(msg.get("mm_kind") or ""),
        "event_id": str(msg.get("mm_event_id") or ""),
        "label": str(msg.get("mm_label") or ""),
        "round": msg.get("mm_round"),
        "status": str(msg.get("mm_status") or ""),
        "text": _content_display_text(content),
    }


def _flush_mm_notice_to_db(session: dict, entry: dict) -> None:
    """把一条 mm_notice 直接写进 session 的 state.db, 使其能在 reopen 后重建。

    必须直接写: turn 机制的增量 flush (_flush_messages_to_session_db) 用 id() 身份
    跳过所有已在 conversation_history 里的元素, 而 mm_notice 是带外 append 进
    session["history"] 的, 永远不会被增量 flush 写库。直接写是唯一的持久路径, 且
    因增量 flush 会 skip 它 (身份已在 history 里), 不会二次写。

    ★ 用 gateway 权威的落库入口 _session_db(session) (profile-aware) + session_key,
    与 reopen 读取 (db.get_messages_as_conversation(session_key)) 对齐 —— 之前误用
    agent._session_db / agent.session_id, 在 gateway 会话里往往是 None, 于是一条也
    没落库 (reopen 后 monitor/watcher 气泡全没了)。best-effort, 绝不抛异常。"""
    key = session.get("session_key")
    if not key:
        logger.info("[mm-context] db flush SKIP: no session_key")
        return
    try:
        # 首条 mm_notice 可能早于用户第一条消息 (monitor 后台命中), 行可能还没建 →
        # 先幂等建行 (INSERT OR IGNORE), 否则 append 的外键无处可挂。
        _ensure_session_db_row(session)
    except Exception:
        logger.warning("[mm-context] ensure session row failed", exc_info=True)
    try:
        with _session_db(session) as db:
            if db is None:
                logger.info("[mm-context] db flush SKIP: db is None (key=%s)", key)
                return
            db.append_message(
                session_id=key,
                role=entry.get("role", "assistant"),
                content=entry.get("content"),
            )
            logger.info("[mm-context] db flush OK: key=%s kind=%s",
                        key, entry.get("mm_kind"))
    except Exception as exc:
        logger.warning("[mm-context] db flush FAILED: %s", exc, exc_info=True)


def _append_mm_context(session: dict, *, kind: str, text: str,
                       event_id: str = "", label: str = "",
                       round_idx: Any = None, status: str = "") -> None:
    """把一条 monitor/watcher 通知写进 session context (type=mm_notice 元素)。

    kind: "monitor" | "watcher" | "query_user" | "query"。text: 通知正文。
    round_idx: watcher 段序号 (可空)。
    不入 LLM (发送前过滤), 仅供前端全量展示 / 基于 context 判断。写内存 history +
    直接落库 (reopen 可重建)。best-effort, 绝不抛异常。"""
    text = (text or "").strip()
    if not text:
        return
    lock = session.get("history_lock")
    if lock is None:
        return
    content = {
        "type": MM_NOTICE_TYPE,
        "mm_kind": str(kind),
        "mm_event_id": str(event_id or ""),
        "mm_label": str(label or ""),
        "text": text,
    }
    if status:
        content["mm_status"] = str(status)
    if round_idx is not None:
        try:
            content["mm_round"] = int(round_idx)
        except (TypeError, ValueError):
            pass
    entry = {
        "role": "assistant", "content": content,
        # 顶层冗余: 内存态快速判定 (DB round-trip 后会丢, 靠 content.type 兜底)。
        "mm_kind": str(kind), "mm_event_id": str(event_id or ""),
        "mm_label": str(label or ""),
    }
    if status:
        entry["mm_status"] = str(status)
    if "mm_round" in content:
        entry["mm_round"] = content["mm_round"]
    try:
        with lock:
            history = session.get("history", [])
            if not isinstance(history, list):
                history = []
            # Query UI records are keyed by the browser's stable answer-slot id.
            # Tool callbacks are best-effort and may repeat; never duplicate a
            # restored user/answer bubble for the same kind + request.
            if kind in {"query", "query_user"} and event_id:
                duplicate = any(
                    _is_mm_notice(existing)
                    and _mm_notice_fields(existing).get("kind") == kind
                    and _mm_notice_fields(existing).get("event_id") == str(event_id)
                    for existing in history
                )
                if duplicate:
                    return
            history.append(entry)
            session["history"] = history
            session["history_version"] = int(session.get("history_version", 0)) + 1
    except Exception as exc:
        logger.debug("[mm-context] append failed (%s): %s", kind, exc)
        return
    _flush_mm_notice_to_db(session, entry)


def _strip_mm_context(history: list) -> list:
    """发给 LLM 前丢掉所有 monitor/watcher 通知元素 (type=mm_notice)。返回新列表,
    不修改原 history。普通消息原样保留。"""
    if not isinstance(history, list):
        return history
    visible_to_model = [m for m in history if not _is_mm_notice(m)]
    return _normalize_mm_query_projection_history(visible_to_model)


# ── hook 指令 + 结果回传 ──────────────────────────────────────────────────────
#   hook_main_agent 使能时, monitor 命中 / watcher 完成会把 hook_instruction 连同
#   本次执行结果一起发给主 Agent。格式固定 (指令后换行, 结果另起一行):
#       <instruction>
#       monitor / watcher result for reference: <result>
#   result = monitor 判定正文 / watcher 调研报告。不再有 {{result}} 占位符机制。
def _append_hook_result(instruction: Any, result_text: Any) -> str:
    """把 hook_instruction 与本次执行结果按固定格式拼成发给主 Agent 的消息。

    instruction 为空 → 只回结果; result 为空 → 只回指令 (不拼空引用)。"""
    instr = str(instruction or "").strip()
    result = str(result_text or "").strip()
    if not result:
        return instr
    ref = f"monitor / watcher result for reference: {result}"
    if not instr:
        return ref
    return f"{instr}\n{ref}"


def _mark_monitor_tool_result_done(
    session: dict, monitor_id: str, *, trigger_mode: Any = "once",
) -> bool:
    """Compatibility no-op: completion must not rewrite canonical history."""
    # Kept as a compatibility shim for older imports. Completion is persisted
    # in the monitor event file and live registry; changing an earlier tool
    # message would invalidate the conversation's cached provider prefix.
    return False


def _mark_watcher_tool_result_complete(session: dict, rid: str) -> bool:
    """Flip the persisted set_live_watcher receipt's status → "done" (by rid).

    ★ 五态统一: 历史 receipt 的完成态统一用 "done" (不再用 "complete" —— 它是 done 的
    历史层同义词, 已废除)。The watcher stays DECOUPLED from the chat: this touches
    ONLY the tool receipt's `status` field, so the reopen reconcile leaves a
    FINISHED watcher alone instead of marking it "interrupted". Runs under
    history_lock. Returns True if a matching receipt was updated."""
    # Kept as a compatibility shim for older imports. Watch files, sidechannel
    # final rows, and the live registry own completion state; canonical chat
    # history is immutable after the turn that created the watcher.
    return False


def _reconcile_stale_mm_jobs(history: list, agent=None,
                             session_id: str = "", orphans_out: "list | None" = None) -> int:
    """Mark still-"running" multimodal jobs (set_monitor / set_live_watcher) as
    INTERRUPTED in the persisted history, because the process that ran them is
    gone (app/tab closed or crashed) — their engine jobs + the video stream did
    NOT survive. Without this, a reopened session's transcript keeps showing
    "监控已启动 / 深度研究进行中" for jobs that aren't actually running.

    A job's receipt is a role=tool message whose content is a JSON string:
      • set_live_watcher: {request_id: "req_...", status: "running", ...}
      • set_monitor:       {monitor_id: "mon_...", status: "running"/absent, op: ...}
    We flip status→"interrupted" (+ a reason) and drop the stale "note". A
    completed deep-research (status=="complete", carrying its report) is left
    untouched — that's a real result.

    When ``agent`` is given, each interrupted job is ALSO re-registered into
    ``agent.mm_monitors`` / ``agent.mm_watchers`` (enabled=False /
    status="interrupted", no engine job spawned) so the UI can list it and let
    the user flip it back on AFTER (re)starting the video stream. Mutates
    ``history`` in place. Returns the number of jobs reconciled.
    """
    import json as _json
    if not isinstance(history, list):
        return 0
    n = 0
    mons = getattr(agent, "mm_monitors", None) if agent is not None else None
    if agent is not None and mons is None:
        agent.mm_monitors = {}
        mons = agent.mm_monitors
    watchers = getattr(agent, "mm_watchers", None) if agent is not None else None
    if agent is not None and watchers is None:
        agent.mm_watchers = {}
        watchers = agent.mm_watchers

    # ★ 磁盘为权威真相源 (session-scoped): 先扫本 session 的磁盘事件文件, 得到该
    #   session 合法的 event id 全集。history 里的 monitor/watcher 气泡若其 event id
    #   不在这个磁盘集里 → 判为"孤儿"(orphan): 不 re-register、id 收集进 orphans_out,
    #   由 resume 传回前端丢弃 + 顶部提示。仅在给了 session_id 时启用 (旧调用不变)。
    _want_sid = str(session_id or "").strip()
    _disk_mon_ids: "set | None" = None
    _disk_monitors: "dict | None" = None
    _disk_watch_ids: "set | None" = None
    _disk_watchers: "dict | None" = None
    if _want_sid:
        try:
            from agent.multimodal import monitor_agent as _ma_scan
            _disk_monitors = _ma_scan.scan_all(session_id=_want_sid) or {}
            _disk_mon_ids = set(_disk_monitors.keys())
        except Exception:
            _disk_monitors = {}
            _disk_mon_ids = set()
        try:
            from agent.multimodal import watch_file as _wf_scan
            _disk_watchers = _wf_scan.scan_all(session_id=_want_sid) or {}
            _disk_watch_ids = set(_disk_watchers.keys())
        except Exception:
            _disk_watchers = {}
            _disk_watch_ids = set()

    # A deterministic local Monitor create deliberately leaves provider-facing
    # history as a plain user -> assistant receipt (no synthetic tool_call: it
    # would lack Gemini/Anthropic provider signatures). Recover those monitors
    # from the session-scoped event file, which already stores the complete
    # query/report/hook contract and is the authoritative source used above for
    # orphan checks. Reopened running jobs are disabled/interrupted until the
    # user explicitly re-enables them on a live stream; a one-shot job already
    # persisted as done remains done and disabled.
    if mons is not None and _disk_monitors:
        for mid, info in _disk_monitors.items():
            if mid in mons:
                continue
            disk_status = str(info.get("status") or "interrupted").strip().lower()
            if disk_status in ("running", "stopping"):
                try:
                    _ma_scan.set_status(mid, "interrupted")
                except Exception:
                    pass
                disk_status = "interrupted"
            elif disk_status == "complete":
                disk_status = "done"
            elif disk_status not in ("done", "interrupted"):
                disk_status = "interrupted"
            query = str(info.get("monitor_query") or "")
            restored = {
                "id": mid,
                "monitor_id": mid,
                "monitor_query": query,
                "brief": query,
                "label": str(info.get("label") or ""),
                "enabled": False,
                "trigger_mode": _normalize_mm_monitor_trigger_mode(
                    info.get("trigger_mode")),
                "silent": bool(info.get("silent", False)),
                "report_interval": info.get("report_interval"),
                "event_file": str(info.get("event_file") or ""),
                "hook_main_agent": bool(info.get("hook_main_agent", False)),
                "hook_instruction": str(info.get("hook_instruction") or ""),
                "status": disk_status,
                "created_at": 0.0,
                "last_speak_ts": 0.0,
            }
            if disk_status == "interrupted":
                restored["_interrupted"] = True
            mons[mid] = restored

    # Watcher state is also recoverable directly from its session-owned analyse
    # file.  This is the authoritative path for UI updates that never produced a
    # chat tool receipt and for process restarts where in-memory pacing/cursors
    # no longer exist.  Legacy files fall back to their query/status/round data.
    if watchers is not None and _disk_watchers:
        from agent.multimodal import watch_file as _wf_restore
        for wid, info in _disk_watchers.items():
            if wid in watchers:
                continue
            state = info.get("state") if isinstance(info.get("state"), dict) else {}
            disk_status = str(info.get("status") or "interrupted").strip().lower()
            if disk_status in ("running", "stopping"):
                disk_status = "interrupted"
                try:
                    _wf_restore.set_status(
                        wid, "interrupted",
                        round_idx=int(info.get("round_idx") or 0),
                        stop_reason="restart",
                    )
                except Exception:
                    pass
            elif disk_status == "complete":
                disk_status = "done"
            elif disk_status not in ("done", "interrupted"):
                disk_status = "interrupted"
            seg_base = max(
                int(info.get("round_idx") or 0),
                int(state.get("seg_base") or 0),
                int(_wf_restore.count_rounds(wid) or 0),
            )
            task_text = str(
                state.get("task_instruction") or info.get("query") or "")
            restored = {
                "id": wid,
                "watcher_id": wid,
                "task_instruction": task_text,
                "label": str(state.get("label") or ""),
                "hook_main_agent": bool(
                    state.get("hook_main_agent", False)),
                "hook_instruction": str(
                    state.get("hook_instruction") or ""),
                "ttl": str(state.get("ttl") or ""),
                "ttl_sec": state.get("ttl_sec"),
                "target_frames": state.get("target_frames"),
                "status": disk_status,
                "watch_file": str(_wf_restore.file_path(wid)),
                "created_at": 0.0,
                "_seg_base": seg_base,
            }
            if disk_status == "interrupted":
                restored["_interrupted"] = True
            watchers[wid] = restored

    for m in history:
        if not (isinstance(m, dict) and m.get("role") == "tool"):
            continue
        content = m.get("content")
        if not isinstance(content, str) or '"' not in content:
            continue
        try:
            data = _json.loads(content)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        rid = str(data.get("request_id") or "").strip()
        mid = str(data.get("monitor_id") or "").strip()
        status = str(data.get("status") or "").strip().lower()
        is_research = rid.startswith("req_")
        is_monitor = mid.startswith("mon_")
        if not (is_research or is_monitor):
            continue
        # A delete receipt/tombstone is terminal and intentionally absent from
        # scan_all(); never reinterpret that absence as an interrupted monitor.
        if is_monitor and (
                status == "deleted" or str(data.get("op") or "") == "delete"):
            continue
        # ★ 孤儿判定 (磁盘为权威): 该 event id 不在本 session 的磁盘集 → 不属于本
        #   session (或文件已删/旧无 session_id 文件)。标记为孤儿, 收集 id, 从
        #   history 中标注 _orphan 让前端丢弃, 且不 re-register 进内存注册表。
        if is_monitor and _disk_mon_ids is not None and mid not in _disk_mon_ids:
            if orphans_out is not None:
                orphans_out.append(mid)
            m["_mm_orphan"] = True
            continue
        if is_research and _disk_watch_ids is not None and rid not in _disk_watch_ids:
            if orphans_out is not None:
                orphans_out.append(rid)
            m["_mm_orphan"] = True
            continue
        # Completed research and Monitor receipts are monotonic terminal facts.
        # For Monitor, the disk/registry status may already be done while its
        # older create receipt still says running; synchronize the receipt to
        # done instead of allowing generic stale-job reconciliation to overwrite
        # the terminal state with interrupted.
        restored_monitor = (
            mons.get(mid) if is_monitor and mons is not None else None)
        restored_status = str(
            restored_monitor.get("status") if isinstance(restored_monitor, dict)
            else ""
        ).strip().lower()
        monitor_done = is_monitor and (
            status in ("done", "complete")
            or restored_status in ("done", "complete")
        )
        if is_research and status in ("done", "complete"):
            continue

        changed = False
        if is_monitor:
            use_receipt_mode = (
                not isinstance(restored_monitor, dict)
                or str(data.get("op") or "") == "update"
            )
            trigger_mode = _normalize_mm_monitor_trigger_mode(
                data.get("trigger_mode")
                if use_receipt_mode and "trigger_mode" in data
                else (
                    restored_monitor.get("trigger_mode")
                    if isinstance(restored_monitor, dict) else None
                )
            )
            desired_status = "done" if monitor_done else "interrupted"
            if data.get("status") != desired_status:
                data["status"] = desired_status
                changed = True
            if data.get("trigger_mode") != trigger_mode:
                data["trigger_mode"] = trigger_mode
                changed = True
            if monitor_done:
                if "interrupted_reason" in data:
                    data.pop("interrupted_reason", None)
                    changed = True
                if isinstance(restored_monitor, dict):
                    restored_monitor["enabled"] = False
                    restored_monitor["status"] = "done"
                    restored_monitor.pop("_interrupted", None)
                # A legacy receipt can be the only persisted completion marker.
                # Promote the event-file header so subsequent registry payloads,
                # which prefer disk status, remain consistent.
                if restored_status not in ("done", "complete") and _want_sid:
                    try:
                        from agent.multimodal import monitor_agent as _ma_done
                        _ma_done.set_status(mid, "done")
                    except Exception:
                        pass
            else:
                reason = (
                    "会话已关闭,后台作业未持续运行;如需继续,请重新开启视频流后在面板中开启此作业。"
                )
                if data.get("interrupted_reason") != reason:
                    data["interrupted_reason"] = reason
                    changed = True
                if "note" in data:
                    data.pop("note", None)
                    changed = True
        elif status != "interrupted":
            data["status"] = "interrupted"
            data["interrupted_reason"] = (
                "会话已关闭,后台作业未持续运行;如需继续,请重新开启视频流后在面板中开启此作业。"
            )
            data.pop("note", None)
            changed = True
        if changed:
            m["content"] = _json.dumps(data, ensure_ascii=False)
            n += 1
        # Backward compatibility for monitors updated before config headers
        # became atomically rewriteable. Disk remains the existence/session
        # authority, while chronological update receipts can overlay their
        # latest mutable contract onto an already restored disk entry. Create
        # receipts are intentionally not merged: UI-side updates may have no
        # chat receipt, and their newer disk header must win over old create data.
        if (is_monitor and mons is not None and mid in mons
                and str(data.get("op") or "") == "update"):
            restored = mons[mid]
            query = str(
                data.get("monitor_query") or data.get("brief") or "").strip()
            if query:
                restored["monitor_query"] = query
                restored["brief"] = query
            if "label" in data:
                restored["label"] = str(data.get("label") or "")
            if "silent" in data:
                restored["silent"] = bool(data.get("silent"))
            if "report_interval" in data:
                restored["report_interval"] = data.get("report_interval")
            if "trigger_mode" in data:
                restored["trigger_mode"] = _normalize_mm_monitor_trigger_mode(
                    data.get("trigger_mode"))
            if "hook_main_agent" in data:
                restored["hook_main_agent"] = bool(
                    data.get("hook_main_agent"))
            if "hook_instruction" in data:
                restored["hook_instruction"] = str(
                    data.get("hook_instruction") or "")
        # Re-register into the in-memory registry (disabled/interrupted, no job).
        if is_monitor and mons is not None and mid not in mons:
            # ★ event_file 路径是确定性的 (HERMES_HOME/monitor/monitor_<mid>.md)。
            #   优先用 receipt 里存的; 缺失就现算 —— 否则 reopen 后前端 registry 返回
            #   event_file=None, 主 Agent read_file(None) → "读不到监控文件" (文件其实
            #   在盘上, 只是重建时漏填了路径)。
            _evt_file = data.get("event_file")
            if not _evt_file:
                try:
                    from agent.multimodal.monitor_agent import event_file_path as _efp
                    _evt_file = str(_efp(mid))
                except Exception:
                    _evt_file = ""
            restored_status = (
                "done" if status in ("done", "complete") else "interrupted")
            restored_monitor = {
                "id": mid, "monitor_id": mid,
                "monitor_query": data.get("monitor_query", "") or data.get("brief", ""),
                "brief": data.get("monitor_query", "") or data.get("brief", ""),
                "label": data.get("label", ""),
                "enabled": False, "silent": bool(data.get("silent", False)),
                "status": restored_status,
                "trigger_mode": _normalize_mm_monitor_trigger_mode(
                    data.get("trigger_mode")),
                "report_interval": data.get("report_interval"),
                "event_file": _evt_file,
                # ★ 从 tool receipt 恢复 hook 能力 —— 否则 reopen 后重新开启监控, 命中时
                #   _speak_cb 读到的 hook_main_agent 为假, 不再触发主 Agent (与 watcher
                #   重建保持对齐, 后者一直带这两个字段)。
                "hook_main_agent": bool(data.get("hook_main_agent", False)),
                "hook_instruction": data.get("hook_instruction", ""),
                "created_at": 0.0, "last_speak_ts": 0.0,
            }
            if restored_status == "interrupted":
                restored_monitor["_interrupted"] = True
            mons[mid] = restored_monitor
        if is_research and watchers is not None and rid not in watchers:
            watchers[rid] = {
                "id": rid, "watcher_id": rid,
                # Prefer the new field; fall back to old persisted keys so a
                # session saved before the rename still reconciles.
                "task_instruction": (data.get("task_instruction", "")
                                     or data.get("watcher_text", "")
                                     or data.get("brief", "")),
                "label": data.get("label", ""),
                "hook_main_agent": bool(data.get("hook_main_agent", False)),
                "hook_instruction": data.get("hook_instruction", ""),
                "ttl": data.get("ttl", ""),
                "ttl_sec": data.get("ttl_sec"),
                "target_frames": data.get("target_frames"),
                "status": "interrupted", "_interrupted": True,
                "watch_file": data.get("watch_file", ""), "created_at": 0.0,
                "_seg_base": 0,
            }
            # req: on reopen, drop the last (possibly half-written) analysis
            # segment from the analyse file so a resumed job doesn't show a
            # truncated round. No-op if the run had already finished.
            try:
                from agent.multimodal import watch_file as _wf
                _wf.drop_last_incomplete_round(rid)
            except Exception:
                pass
        # Chronological update receipts are a backward-compatible overlay for
        # files created before watcher_state existed. New code persists the same
        # fields atomically in the analyse header, so either recovery source
        # reconstructs the latest mutable task contract.
        if (is_research and watchers is not None and rid in watchers
                and str(data.get("op") or "") == "update"):
            restored = watchers[rid]
            task_text = str(data.get("task_instruction") or "").strip()
            if task_text:
                restored["task_instruction"] = task_text
            if "label" in data:
                restored["label"] = str(data.get("label") or "")
            for key in ("ttl", "ttl_sec", "target_frames"):
                if key in data:
                    restored[key] = data.get(key)
            if "hook_main_agent" in data:
                restored["hook_main_agent"] = bool(
                    data.get("hook_main_agent"))
            if "hook_instruction" in data:
                restored["hook_instruction"] = str(
                    data.get("hook_instruction") or "")
    if agent is not None:
        _recompute_mm_monitor_active(agent)
    return n


def _interrupt_running_mm_jobs(sid: str, session: dict) -> int:
    """★ Video stream closed → TERMINATE all in-flight monitors/watchers so they
    do NOT survive into the next stream (no 复活/持续监控/持续深度研究).

    Semantics = 彻底终止不恢复:
      • Monitors: cancel each engine tick-job (remove_monitor) + flip
        agent.mm_monitors[mid] → enabled=False/status=interrupted; clear the
        active flag. The engine container itself stays (so the user can create a
        NEW monitor on the next stream), but no OLD monitor keeps ticking.
      • Watcher: mark_source_stopped (existing) + flip agent.mm_watchers[rid] →
        status=interrupted so the loop finishes and never re-arms; the source
        is not treated as live again for these jobs.
      • Reconcile a detached history snapshot to recover registry metadata.
        The live provider-visible history stays immutable mid-conversation;
        registry pushes carry the interrupted state to the desktop UI.

    Returns how many jobs were interrupted. Best-effort; never raises to caller.
    """
    n = 0
    agent = session.get("agent")
    if agent is None:
        return 0
    # ── Monitors: cancel engine jobs + flip registry ──────────────────────
    mon_engine = session.get("_mm_monitor_engine")
    mons = getattr(agent, "mm_monitors", None) or {}
    for mid, m in list(mons.items()):
        try:
            status = str(m.get("status") or "").strip().lower()
            if status in ("interrupted", "complete", "done", "deleted"):
                continue
            if mon_engine is not None:
                try:
                    mon_engine.remove_monitor(mid)
                except Exception:
                    pass
            m["enabled"] = False
            m["status"] = "interrupted"
            m["_interrupted"] = True
            n += 1
        except Exception:
            pass
    _recompute_mm_monitor_active(agent)
    # ── Watcher: signal source-stopped + flip registry so it can't re-arm ──
    # ★ 视频流结束 (界面停止共享): 只 mark_source_stopped —— 让运行中的 watcher 循环
    #   drain 完最后一批帧后以 stop_reason="source_end" 自然收尾, 从而【在主 agent 执行
    #   hook_main_agent 指令】(见 _on_delegation_complete)。绝不在这里设 _deleted:
    #   _deleted 会让循环以 "deleted" 收尾 → 抑制 hook, 与"流结束触发 hook"矛盾。
    #   只把【非运行中】(已 disabled/interrupted, 不会有 complete 回调的) 标成中断。
    watcher = session.get("_mm_live_watcher_agent")
    if watcher is not None:
        try:
            watcher.mark_source_stopped()
        except Exception:
            pass
    wrs = getattr(agent, "mm_watchers", None) or {}
    for rid, w in list(wrs.items()):
        try:
            st = str(w.get("status") or "").strip()
            if st == "running":
                # 运行中的交给 source_end 自然收尾 (可能触发 hook), 这里不动状态。
                n += 1
                continue
            # ★ 五态统一: 终态/收尾中都不动 (done/interrupted/deleted 已终; stopping 由
            #   引擎收尾后自己落终态; complete 是 done 历史遗留)。
            if st in ("interrupted", "complete", "done", "deleted", "stopping"):
                continue
            # 非运行态残留 → 标中断 (不会有 completion 回调来收尾它)。
            w["status"] = "interrupted"
            w["_interrupted"] = True
        except Exception:
            pass
    # ── Recover metadata without rewriting the cached chat prefix ──────────
    try:
        lock = session.get("history_lock")
        if lock is not None:
            with lock:
                history = session.get("history")
                if isinstance(history, list):
                    _reconcile_stale_mm_jobs(copy.deepcopy(history), agent)
        else:
            history = session.get("history")
            if isinstance(history, list):
                _reconcile_stale_mm_jobs(copy.deepcopy(history), agent)
    except Exception:
        pass
    if n:
        logger.info("[mm] video stream closed → interrupted %d monitor/watcher job(s) (sid=%s)",
                    n, sid)
    # Push the updated registries so the UI reflects the interrupted state.
    try:
        from tools.monitor_tool import _push_monitors_event
        _push_monitors_event(sid, agent)
    except Exception:
        pass
    try:
        from tools.live_watcher_tool import _push_watchers_event
        _push_watchers_event(sid, agent)
    except Exception:
        pass
    return n


def _run_prompt_submit(rid, sid: str, session: dict, text: Any,
                       *, user_originated: bool = False,
                       client_request_id: str = "",
                       internal_origin: str = "",
                       internal_fallback_text: str = "",
                       anchor_ts: Optional[float] = None,
                       anchor_frozen: bool = False) -> None:
    # Every turn must own a distinct stream key before emitting anything.  Keep
    # a caller-provided id byte-for-byte (within the protocol bound), otherwise
    # allocate once and reuse it for user echo/start/delta/complete, MM query
    # projection, and deferred persistence metadata throughout this invocation.
    client_request_id = _ensure_client_request_id(client_request_id)
    # ★ Per-turn tracing: dump [mm-trace] milestones to gateway stderr so we
    # can tell exactly where a screen-share query stalls. Compare timestamps
    # to see whether the latency lives in setup, ask-time anchoring/routing,
    # LLM first-byte, or elsewhere. All logs share a short sid-suffix + a
    # monotonic ms delta since t0 so grep on "[mm-trace] sid=..." gives you
    # one row per stage.
    import time as _t
    _trace_t0 = _t.monotonic()
    _sid_short = (sid or "-")[-6:]
    def _trace(stage: str, extra: str = "") -> None:
        dt = (_t.monotonic() - _trace_t0) * 1000
        # Stash so _log_busy_diagnosis can tell users where the prior turn
        # got stuck when their new prompt hits the busy guard.
        session["_mm_last_stage"] = stage
        session["_mm_last_stage_ts"] = _t.monotonic()
        logger.info("[mm-trace] sid=%s +%.0fms %s%s",
                    _sid_short, dt, stage,
                    (" " + extra) if extra else "")
    _trace("enter_run_prompt_submit")
    _event_tag = {"request_id": client_request_id}

    query_projection_reservation_id = ""
    query_projection_messages: list[dict] = []
    # A hidden Monitor/Watcher hook is not a foreground user turn and must not
    # commit pending QueryWorker Q/A projections into canonical history merely
    # by running.  Leave them pending for the next real user turn.
    if not internal_origin and _is_multimodal_runtime_session(session):
        query_projection_reservation_id = (
            f"qproj_{client_request_id or rid}_{uuid.uuid4().hex[:8]}"
        )
        query_projection_messages = _reserve_mm_query_turn_projection(
            session, query_projection_reservation_id)
        if query_projection_messages and not _commit_mm_query_turn_projection(
                session, query_projection_reservation_id,
                query_projection_messages):
            _finish_mm_query_turn_projection(
                session, query_projection_reservation_id, committed=False)
            query_projection_messages = []
    with session["history_lock"]:
        persisted_history = list(session["history"])
        history_version = int(session.get("history_version", 0))
        images = _claim_composer_attachments(
            session, user_originated=user_originated)
        if not isinstance(session.get("inflight_turn"), dict):
            # Internal completion hooks may produce a visible Assistant answer,
            # but their implementation prompt must never reappear as a recovered
            # user bubble after reconnect/resume.
            _start_inflight_turn(session, "" if internal_origin else text)
    # Projection commits are append-only, so this remains a cache-stable prefix.
    history = persisted_history
    agent = session["agent"]
    _trace(
        "session_locked",
        f"history_len={len(history)} images={len(images)} "
        f"query_qa_pairs={len(query_projection_messages) // 2}",
    )
    if hasattr(agent, "clear_interrupt"):
        try:
            agent.clear_interrupt()
        except Exception:
            pass
    # ★ 后端发起的 ordinary turn (通知轮询、goal 跟进等) 其 user
    #   指令不是用户在前端输入的 → 前端没本地加过 user 气泡。回显一条 message.user_echo,
    #   让主 agent 对话页把这条触发指令作为正式 UserMessage 显示。普通 prompt.submit
    #   (user_originated=True) 跳过, 避免跟前端本地已加的气泡重复。
    # Internal monitor/watcher hooks are hidden control context: only their
    # Assistant result is user-visible, never a synthetic `You` bubble.
    if not user_originated and not internal_origin:
        _echo = text if isinstance(text, str) else ""
        if _echo.strip():
            _emit("message.user_echo", sid, {**_event_tag, "text": _echo})
    _emit("message.start", sid, _event_tag)
    _trace("emit_message_start")
    session["_mm_trace"] = _trace  # exposed so the inner run() can share it

    def run():
        approval_token = None
        session_tokens = []
        home_token = None  # per-turn HERMES_HOME override for a resumed remote profile
        goal_followup = None  # set by the post-turn goal hook below
        try:
            from tools.approval import (
                reset_current_session_key,
                set_current_session_key,
            )

            approval_token = set_current_session_key(session["session_key"])
            session_tokens = _set_session_context(session["session_key"])
            _profile_home_str = session.get("profile_home")
            if _profile_home_str:
                home_token = set_hermes_home_override(_profile_home_str)
            # The sudo password callback is thread-local (tools.terminal_tool
            # _callback_tls), so wiring it on the build thread doesn't reach this
            # turn thread — terminal sudo prompts would fall through to /dev/tty
            # and hang the headless gateway. Re-wire here so the prompt routes to
            # the sudo.request overlay. (secret capture is a module global, so
            # re-running is a harmless no-op.)
            _wire_callbacks(sid)
            # ── 模型生命周期: 启动时定死, 不每轮热更新 ──────────────────────
            # 用户要求: model 在会话 build 时由 config 加载一次, 生命周期内固定;
            # 不需要 config.yaml 热更新 (改 config 后已开会话不变, 新开会话/重启
            # 才用新值)。故不再每轮调 _sync_agent_model_with_config —— 它原本每轮
            # 读 config 比对并可能中途切换模型 + 写 "active model changed" 系统
            # 消息, 正是"历史/config 漂移"问题的来源之一。函数本体保留 (未来若需
            # 手动同步可复用), 仅摘除每轮自动调用。
            # _sync_agent_model_with_config(sid, session)  # 已停用: 见上
            # ★ Deep-thinking toggle re-read EVERY turn (agent built once per
            # session — reading it only at build froze the 🧠 toggle at its
            # first-turn value). Multimodal sessions only.
            if _is_multimodal_runtime_session(session):
                _apply_mm_deep_thinking(agent, session)
            cwd = _session_cwd(session)
            _register_session_cwd(session)
            cols = session.get("cols", 80)
            streamer = make_stream_renderer(cols)
            prompt = text

            if isinstance(prompt, str) and "@" in prompt:
                from agent.context_references import preprocess_context_references
                from agent.model_metadata import get_model_context_length

                ctx_len = get_model_context_length(
                    getattr(agent, "model", "") or _resolve_model(),
                    base_url=getattr(agent, "base_url", "") or "",
                    api_key=getattr(agent, "api_key", "") or "",
                    provider=getattr(agent, "provider", "") or "",
                    config_context_length=getattr(
                        agent, "_config_context_length", None
                    ),
                )
                ctx = preprocess_context_references(
                    prompt,
                    cwd=cwd,
                    allowed_root=cwd,
                    context_length=ctx_len,
                )
                if ctx.blocked:
                    _emit(
                        "error",
                        sid,
                        {
                            **_event_tag,
                            "message": "\n".join(ctx.warnings)
                            or "Context injection refused."
                        },
                    )
                    return
                prompt = ctx.message

            # Images are attached natively as OpenAI-style image_url content
            # parts. Provider adapters translate for Anthropic/Gemini/Bedrock/
            # etc. If the active model can't accept vision, the provider will
            # error — configure a vision-capable model.
            run_message: Any = prompt
            if images:
                from agent.image_routing import build_native_content_parts
                _parts, _skipped = build_native_content_parts(prompt, images)
                if _skipped:
                    print(
                        f"[tui_gateway] image attachment skipped "
                        f"{len(_skipped)} unreadable path(s)",
                        file=sys.stderr,
                    )
                if any(p.get("type") == "image_url" for p in _parts):
                    run_message = _parts

            # ── Multimodal phase 2: stamp the ask-time frame anchor for a
            # vision model for THIS turn ──────────────────────────────────────
            # The browser streams camera/screen frames into the session's
            # FrameBuffer via multimodal.frame. The main agent receives no
            # passive frame parts; QueryWorker and explicit raw-frame tools use
            # this anchor to resolve the frames that existed when the user asked.
            _trace_fn = session.get("_mm_trace")
            if _trace_fn:
                _trace_fn("before_vision_anchor")
            # ★ 记下"本条 UserMessage 发送时刻"的原始帧锚点。一次性
            #   QueryWorker 会据此取服务器实际收到的、未经 dHash 去重的
            #   ask-time 前 3 帧。不在工具真正执行时取 latest，避免混入提问
            #   后才出现的画面。get_current_frame 仍可以用该时间边界取稀疏原图。
            #   ★ v33: 主 Agent 【永不】被动注入图像帧。一次性当前/历史视觉问答
            #   交给 query_multimodal；用户明确要求取回/查看原始当前帧时才调
            #   get_current_frame。这里只 stamp 锚点, 不注入任何帧。
            try:
                _fb = getattr(agent, "frame_buffer", None)
                # Push-to-talk manual ASR snapshots this value at the user's
                # stop click, before waiting for the upstream final transcript.
                # Prefer that explicit per-turn boundary so QueryWorker cannot
                # accidentally see frames that arrived during ASR flush.
                _raw_anchor = _optional_finite_float(anchor_ts)
                if not anchor_frozen:
                    _raw_anchor = (
                        getattr(_fb, "monitor_latest_ts", None)
                        if _fb is not None else None
                    )
                # Older/custom FrameBuffer implementations expose only the
                # sparse timestamp. Preserve compatibility while production
                # uses the raw server-received ring.
                if (not anchor_frozen and _raw_anchor is None
                        and _fb is not None):
                    _raw_anchor = getattr(_fb, "latest_ts", None)
                session["_mm_send_anchor_ts"] = _raw_anchor
            except Exception:
                session["_mm_send_anchor_ts"] = None
            # ★ 同时记下"本条用户消息文本", 供 get_current_frame 在 supports_vision=false
            #   时作为 VQA fallback 的兜底 query (主 agent 未显式传 query 才用)。
            try:
                if isinstance(run_message, str):
                    session["_mm_last_user_text"] = run_message
                elif isinstance(run_message, list):
                    session["_mm_last_user_text"] = " ".join(
                        str(p.get("text", "")) for p in run_message
                        if isinstance(p, dict) and p.get("type") == "text").strip()
            except Exception:
                session["_mm_last_user_text"] = ""

            # First-delta latch: only fires once per turn. Distinguishes
            # "LLM never sent a token" (nothing after agent_run_start) from
            # "LLM sent quickly but rendering stalls" (first_delta close to
            # agent_run_start, then long gap before message.end).
            _first_delta_seen = {"v": False}
            def _stream(delta):
                if not _first_delta_seen["v"]:
                    _first_delta_seen["v"] = True
                    if _trace_fn:
                        _trace_fn("first_delta", f"len={len(delta)}")
                with session["history_lock"]:
                    _append_inflight_delta(session, delta)
                payload = {**_event_tag, "text": delta}
                if streamer and (r := streamer.feed(delta)) is not None:
                    payload["rendered"] = r
                _emit("message.delta", sid, payload)

            model_history = _strip_history_image_parts(
                _strip_mm_context(history)
            )
            if internal_origin:
                # The agent loop repairs/sanitizes messages in place.  The
                # strip helpers intentionally preserve ordinary message dicts
                # for normal turns, so a shallow list here would let a hidden
                # Monitor hook mutate the user's canonical cached prefix even
                # though its final result is never assigned back.  Give only
                # internal turns a detached working copy; ordinary turns keep
                # the existing zero-copy/cache-friendly path.
                model_history = copy.deepcopy(model_history)

            run_kwargs = {
                # Strip image parts from PAST user turns. Otherwise explicit
                # attachments and legacy persisted frames replay every turn →
                # linear prefill blow-up (the "越用越卡" bug). Any images
                # attached to the current request live in run_message, not in
                # history, so they are unaffected.
                # ★ 功能1: 先过滤 monitor/watcher 通知元素 (mm_kind), 再剥历史图片。
                #   这些通知只给前端全量展示, 绝不进 LLM。
                "conversation_history": model_history,
                "stream_callback": _stream,
            }
            try:
                if "task_id" in inspect.signature(agent.run_conversation).parameters:
                    run_kwargs["task_id"] = session["session_key"]
            except (TypeError, ValueError):
                pass

            # ── Voice-turn per-segment TTS ────────────────────────────────
            # In a VOICE turn, speak each COMPLETE text segment as it finishes
            # (the model may emit: thinking → tool → text → thinking → tool →
            # text; every text block should be spoken, not just the final one).
            # agent.interim_assistant_callback fires once per intermediate text
            # block (the ones preceding a tool call) with the clean visible text.
            # The final block (no trailing tool call) is spoken by the
            # message.complete hook below. All segments share ONE response_id so
            # the engine's serial TTS queue plays them back-to-back and the
            # frontend appends their PCM to a single timeline. Text turns leave
            # interim_assistant_callback untouched (no auto-speak).
            # ★ 旧 auto-TTS (语音输入 turn → 逐段自动播) 已停用: 现在 TTS 播报是独立的
            #   VoiceAgent 旁路子系统, 由独立开关 (_mm_tts_on) 驱动, 主 Agent turn
            #   完成后【整段】经改写层播报, 不再逐段。这段代码保留但门恒 False。
            _mm_is_voice = False   # was: bool(session.get("_mm_voice_turn"))
            _mm_tts_engine = session.get("_mm_live_watcher_agent")
            _mm_tts_rid = "tts_" + uuid.uuid4().hex[:8]
            _mm_tts_spoken: list = []  # texts already enqueued (dedup vs final)
            _prev_interim_cb = getattr(agent, "interim_assistant_callback", None)
            if _mm_is_voice and _mm_tts_engine is not None:
                def _mm_interim_tts(text, *, already_streamed=False):
                    # Speak every intermediate text segment immediately. TTS is
                    # a separate audio channel, so already_streamed (about the
                    # VISUAL stream) is irrelevant here. Best-effort — a TTS
                    # hiccup must never break the turn.
                    try:
                        t = (text or "").strip()
                        if not t:
                            return
                        _mm_tts_spoken.append(t)
                        _mm_tts_engine.enqueue_tts(t, _mm_tts_rid)
                    except Exception as _e:
                        logger.debug("interim TTS enqueue skipped: %s", _e)
                try:
                    agent.interim_assistant_callback = _mm_interim_tts
                except Exception:
                    pass

            if _trace_fn:
                _trace_fn("agent_run_start")
            _prev_parent_message_id = getattr(
                agent, "_active_parent_user_message_id", None)
            _prev_active_user_text = getattr(
                agent, "_active_user_message_text", None)
            _prev_defer_turn_persistence = getattr(
                agent, "_defer_current_turn_persistence", None)
            _prev_ephemeral_internal_turn = getattr(
                agent, "_ephemeral_internal_turn", None)
            _had_session_messages = hasattr(agent, "_session_messages")
            _prev_session_messages = (
                getattr(agent, "_session_messages", None)
                if internal_origin else None
            )
            _had_last_flushed_db_idx = hasattr(
                agent, "_last_flushed_db_idx")
            _prev_last_flushed_db_idx = (
                getattr(agent, "_last_flushed_db_idx", None)
                if internal_origin else None
            )
            _had_compression_enabled = hasattr(agent, "compression_enabled")
            _prev_compression_enabled = getattr(
                agent, "compression_enabled", True)
            agent._active_parent_user_message_id = str(client_request_id or "")
            agent._active_user_message_text = _mm_message_text(prompt)
            # A QueryWorker deferred handoff must leave no half-turn in main
            # history. Delay normal crash-resilience writes until finalization,
            # where only a deferred reply is reduced back to the prior prefix.
            agent._defer_current_turn_persistence = (
                _is_multimodal_runtime_session(session)
                and bool(str(client_request_id or ""))
            )
            agent._ephemeral_internal_turn = bool(internal_origin)
            # Automatic context compaction rotates/mutates the durable session.
            # Hidden control turns are ephemeral even when the provider reports
            # overflow, so surface a normal failed internal turn instead of ever
            # compressing the user's canonical conversation on its behalf.
            if internal_origin:
                agent.compression_enabled = False
            try:
                result = agent.run_conversation(run_message, **run_kwargs)
                if _trace_fn:
                    _trace_fn("agent_run_returned",
                              f"first_delta_seen={_first_delta_seen['v']}")
            finally:
                # Restore the previous interim callback so per-segment TTS never
                # leaks into a later (possibly text) turn on the same agent.
                if _mm_is_voice and _mm_tts_engine is not None:
                    try:
                        agent.interim_assistant_callback = _prev_interim_cb
                    except Exception:
                        pass
                if _prev_parent_message_id is None:
                    try:
                        delattr(agent, "_active_parent_user_message_id")
                    except AttributeError:
                        pass
                else:
                    agent._active_parent_user_message_id = _prev_parent_message_id
                if _prev_active_user_text is None:
                    try:
                        delattr(agent, "_active_user_message_text")
                    except AttributeError:
                        pass
                else:
                    agent._active_user_message_text = _prev_active_user_text
                if _prev_defer_turn_persistence is None:
                    try:
                        delattr(agent, "_defer_current_turn_persistence")
                    except AttributeError:
                        pass
                else:
                    agent._defer_current_turn_persistence = (
                        _prev_defer_turn_persistence)
                if _prev_ephemeral_internal_turn is None:
                    try:
                        delattr(agent, "_ephemeral_internal_turn")
                    except AttributeError:
                        pass
                else:
                    agent._ephemeral_internal_turn = (
                        _prev_ephemeral_internal_turn)
                if internal_origin:
                    if _had_session_messages:
                        agent._session_messages = _prev_session_messages
                    else:
                        try:
                            delattr(agent, "_session_messages")
                        except AttributeError:
                            pass
                    if _had_last_flushed_db_idx:
                        agent._last_flushed_db_idx = _prev_last_flushed_db_idx
                    else:
                        try:
                            delattr(agent, "_last_flushed_db_idx")
                        except AttributeError:
                            pass
                if _had_compression_enabled:
                    agent.compression_enabled = _prev_compression_enabled
                else:
                    try:
                        delattr(agent, "compression_enabled")
                    except AttributeError:
                        pass
                if _trace_fn:
                    _trace_fn("turn_finally_done")
            if "moa_one_shot_restore" in session:
                _restore = session.pop("moa_one_shot_restore", None)
                # Restore the model the user was on before the /moa one-shot.
                # The one-shot did a real in-place agent.switch_model() to MoA
                # (#53444), so undoing it must go back through the switch path —
                # resetting session["model_override"] alone would leave the live
                # agent's client pinned to MoA for the next turn.
                if isinstance(_restore, dict):
                    _prev_override = _restore.get("override")
                    _prev_model = _restore.get("model")
                    _prev_provider = _restore.get("provider")
                    if _prev_override is None:
                        session.pop("model_override", None)
                    else:
                        session["model_override"] = _prev_override
                    if _prev_model:
                        _raw = (
                            f"{_prev_model} --provider {_prev_provider}"
                            if _prev_provider
                            else _prev_model
                        )
                        try:
                            _apply_model_switch(
                                sid,
                                session,
                                _raw,
                                confirm_expensive_model=False,
                                pin_session_override=bool(_prev_override),
                            )
                        except Exception as _moa_restore_exc:
                            logger.warning(
                                "MoA one-shot model restore failed: %s",
                                _moa_restore_exc,
                            )
                elif _restore is None:
                    session.pop("model_override", None)
                else:
                    session["model_override"] = _restore

            last_reasoning = None
            status_note = None
            handoff_meta = None
            if isinstance(result, dict):
                if (not internal_origin
                        and isinstance(result.get("messages"), list)):
                    with session["history_lock"]:
                        current_version = int(session.get("history_version", 0))
                        if current_version == history_version:
                            # NOTE: the live-watcher is fully decoupled from the
                            # main agent — it never writes into session history
                            # (process + final report go to the watcher panel only).
                            # The legacy _mm_deep_rid preservation loop below is now
                            # a permanent no-op (nothing sets that key anymore) but
                            # kept as a harmless guard against any external writer.
                            _new_msgs = list(result["messages"])
                            try:
                                _live = session.get("history") or []
                                _have_rids = {
                                    m.get("_mm_deep_rid")
                                    for m in _new_msgs
                                    if isinstance(m, dict) and m.get("_mm_deep_rid")
                                }
                                for _m in _live:
                                    if (isinstance(_m, dict)
                                            and _m.get("_mm_deep_rid")
                                            and _m.get("_mm_deep_rid") not in _have_rids):
                                        _new_msgs.append(_m)
                            except Exception:
                                pass
                            session["history"] = _new_msgs
                            session["history_version"] = history_version + 1
                            # ★ Live-watcher is now FULLY decoupled from the main
                            # agent: its per-round process and final consolidated
                            # report live ONLY in the watcher panel (UI emits
                            # watcher.report_append / watcher.final). Nothing is
                            # written into session["history"], so there is no report
                            # region to re-merge here and no lock re-entry risk.
                        else:
                            # Monitor/watcher notices are a UI-only side channel:
                            # they may legitimately arrive while this turn is in
                            # flight and bump history_version. Preserve those
                            # append-only notices after the agent's result instead
                            # of treating them as a destructive edit and dropping
                            # the entire foreground answer from session history.
                            # Any other concurrent mutation still takes the strict
                            # mismatch path below.
                            _live_history = session.get("history") or []
                            _notice_tail = None
                            if (
                                isinstance(_live_history, list)
                                and len(_live_history) >= len(persisted_history)
                                and _live_history[:len(persisted_history)] == persisted_history
                            ):
                                _candidate_tail = _live_history[len(persisted_history):]
                                if _candidate_tail and all(
                                    _is_mm_notice(_m) for _m in _candidate_tail
                                ):
                                    _notice_tail = list(_candidate_tail)
                            if _notice_tail is not None:
                                _new_msgs = list(result["messages"]) + _notice_tail
                                session["history"] = _new_msgs
                                session["history_version"] = current_version + 1
                                logger.info(
                                    "[mm-context] merged %d concurrent notice(s) "
                                    "into completed foreground turn",
                                    len(_notice_tail),
                                )
                            else:
                                # History mutated externally during the turn
                                # (undo/compress/retry/rollback now guard on
                                # session.running, but this is the defensive
                                # backstop for any path that slips past).
                                # Surface the desync rather than silently
                                # dropping the agent's output — the UI can
                                # show the response and warn that it was
                                # not persisted.
                                print(
                                    f"[tui_gateway] prompt.submit: history_version mismatch "
                                    f"(expected={history_version} current={current_version}) — "
                                    f"agent output NOT written to session history",
                                    file=sys.stderr,
                                )
                                status_note = (
                                    "History changed during this turn — the response above is visible "
                                    "but was not saved to session history."
                                )

                # If auto-compression fired inside run_conversation(), agent.session_id
                # may have rotated. Sync session_key before downstream title/goal/finalize
                # handling uses it. Preserve pending_title (user intent) so it can be
                # applied to the continuation. Restart slash worker so subsequent
                # worker-backed commands (/title etc.) target the live session.
                # Fix for #20001.
                _sync_session_key_after_compress(
                    sid, session, clear_pending_title=False, restart_slash_worker=True,
                )

                raw = result.get("final_response", "")
                if isinstance(result.get("handoff"), dict):
                    handoff_meta = dict(result["handoff"])
                status = (
                    "interrupted"
                    if result.get("interrupted")
                    else "error"
                    if result.get("error") or result.get("failed")
                    else "complete"
                )
                # When the backend produced no visible response AND reported a
                # real error (e.g. invalid model slug → provider 4xx), surface
                # that error as the visible text instead of shipping an empty
                # turn to Ink. Mirrors classic CLI behavior at cli.py where
                # (failed|partial) + no final_response → "Error: <detail>".
                # Leaves the None-with-no-error path untouched: an empty
                # successful turn still renders as empty, and the existing
                # "(empty)" sentinel handling stays in its own lane.
                if (not raw) and result.get("error") and (
                    result.get("failed") or result.get("partial")
                ):
                    raw = f"Error: {result.get('error')}"
                lr = result.get("last_reasoning")
                if isinstance(lr, str) and lr.strip():
                    last_reasoning = lr.strip()
            else:
                raw = str(result)
                status = "complete"

            _watcher_fallback_used = False
            if (
                internal_origin == "watcher_hook"
                and str(internal_fallback_text or "").strip()
                and (
                    status != "complete"
                    or not isinstance(raw, str)
                    or not raw.strip()
                )
            ):
                logger.warning(
                    "[mm-watcher] completion synthesis failed or returned "
                    "empty; delivering consolidated watcher report directly"
                )
                raw = str(internal_fallback_text).strip()
                status = "complete"
                _watcher_fallback_used = True

            _ephemeral_control = bool(
                status == "complete"
                and isinstance(handoff_meta, dict)
                and handoff_meta.get("control") == "handoff"
                and handoff_meta.get("history_policy") == "ephemeral_control"
            )
            payload = {
                **_event_tag,
                "text": raw,
                "usage": _get_usage(agent),
                "status": status,
            }
            if _ephemeral_control:
                payload.update({
                    "ephemeral_control": True,
                    "history_policy": "ephemeral_control",
                    "request_id": str(client_request_id or ""),
                })
            if last_reasoning:
                payload["reasoning"] = last_reasoning
            if status_note:
                payload["warning"] = status_note
            if _watcher_fallback_used:
                payload.update({
                    "watcher_fallback": True,
                    "warning": (
                        "Main-agent watcher synthesis failed; the consolidated "
                        "watcher report was delivered directly."
                    ),
                })
            rendered = render_message(raw, cols)
            if rendered:
                payload["rendered"] = rendered
            with session["history_lock"]:
                _clear_inflight_turn(session)
            _deferred_worker_reply = bool(
                isinstance(handoff_meta, dict)
                and handoff_meta.get("control") == "handoff"
                and handoff_meta.get("handoff_mode") == "deferred_reply"
            )
            if _deferred_worker_reply:
                # Keep the browser's preallocated answer slot open. QueryWorker
                # streams/completes it later with the same request_id, while the
                # main session is released in this turn's finally block. The
                # dispatch tool card already shows the deterministic ack.
                _emit("multimodal.trajectory", sid, {
                    "worker": "MainScheduler",
                    "phase": "reply_ownership_transferred",
                    "request_id": str(client_request_id or ""),
                    "parent_user_message_id": str(client_request_id or ""),
                    "reply_owner": handoff_meta.get("reply_owner"),
                    "task_ids": list(handoff_meta.get("task_ids") or []),
                    "status": "background",
                })
            else:
                _emit("message.complete", sid, payload)

            # ── Multimodal TTS hook (phase 8; per-segment voice mode) ──
            # Only auto-speak when the turn came from VOICE input (session flag
            # set by the asr_final handler); text turns wait for the user to
            # click ▶ → multimodal.tts_speak. Intermediate text segments were
            # already enqueued via interim_assistant_callback above; here we
            # enqueue the FINAL segment (skip if it was already spoken as an
            # interim block) and then close the turn's TTS response so its
            # queued audio plays back-to-back under one response_id.
            # Best-effort: TTS failure must NEVER break the chat turn.
            try:
                session.pop("_mm_voice_turn", False)   # 清残留标记 (旧逻辑已停用)
                # 独立 TTS 播报旁路。开关开启时，主 Agent turn 完成的
                # 整段答案交给口播子系统改写后播报，不入 history、不逐段。
                _tts_effective = bool(session.get("_mm_tts_on"))
                _final = raw.strip() if isinstance(raw, str) else ""
                # Correlation with a VoiceAgent-delegated task is independent
                # from whether the user currently has the speaker/TTS switch on.
                # Tying Future resolution to TTS made a switch-off mid-turn turn
                # every otherwise successful task into a five-minute timeout.
                # ★ pop 在 history_lock 内, 与写点 (抢门时设 _voice_active_seq) 对称,
                #   消除并发委派下"读写锁不对称"的隐式错配依赖 (review A)。
                with session["history_lock"]:
                    _seq = session.pop("_voice_active_seq", None)
                if _seq is not None:
                    _ann = _get_voice_agent(session)
                    if _ann is not None:
                        _reply = (
                            str(handoff_meta.get("ack")
                                or "Monitor operation handled.")
                            if _ephemeral_control
                            else _final or (
                                "主 Agent 未返回有效结果"
                                if status == "complete"
                                else f"主 Agent 任务已{status}"
                            )
                        )
                        try:
                            _ann.notify_main_reply(
                                _seq,
                                _reply,
                                speak=not _ephemeral_control,
                            )
                        except Exception as _voice_exc:
                            logger.debug("voice notify_main_reply err: %s", _voice_exc)
                elif (not internal_origin
                      and not _deferred_worker_reply
                      and not _ephemeral_control
                      and _tts_effective and status == "complete" and _final):
                    _ann = _get_voice_agent(session)
                    if _ann is not None:
                        _ann.submit("assistant", _final)
            except Exception as _tts_exc:
                logger.debug("multimodal TTS announce skipped: %s", _tts_exc)

            # ── /goal continuation (Ralph-style loop) ─────────────────
            # After every TUI turn, if a /goal is active, ask the judge
            # whether the goal is done and — if not and we're still under
            # budget — queue a continuation prompt to run after this
            # thread releases session["running"]. The verdict message
            # ("✓ Goal achieved" / "⏸ budget exhausted") is surfaced as
            # a system line so the user sees progress regardless of
            # outcome. Mirrors gateway/run._post_turn_goal_continuation.
            if (not internal_origin
                    and not _deferred_worker_reply
                    and not _ephemeral_control
                    and status == "complete"
                    and isinstance(raw, str) and raw.strip()):
                try:
                    from hermes_cli.goals import GoalManager

                    sid_key = session.get("session_key") or ""
                    if sid_key:
                        try:
                            goals_cfg = _load_cfg().get("goals") or {}
                            goal_max_turns = int(goals_cfg.get("max_turns", 20) or 20)
                        except Exception:
                            goal_max_turns = 20
                        goal_mgr = GoalManager(
                            session_id=sid_key,
                            default_max_turns=goal_max_turns,
                        )
                        if goal_mgr.is_active():
                            try:
                                from hermes_cli.goals import gather_background_processes as _gather_bg
                                _bg_procs = _gather_bg()
                            except Exception:
                                _bg_procs = None
                            decision = goal_mgr.evaluate_after_turn(
                                raw,
                                user_initiated=True,
                                background_processes=_bg_procs,
                            )
                            verdict_msg = decision.get("message") or ""
                            if verdict_msg:
                                _emit(
                                    "status.update",
                                    sid,
                                    {"kind": "goal", "text": verdict_msg},
                                )
                            if decision.get("should_continue"):
                                cont_prompt = decision.get("continuation_prompt") or ""
                                if cont_prompt:
                                    goal_followup = cont_prompt
                except Exception as _goal_exc:
                    print(
                        f"[tui_gateway] goal continuation hook failed: "
                        f"{type(_goal_exc).__name__}: {_goal_exc}",
                        file=sys.stderr,
                    )

            # Apply pending_title now that the DB row exists.
            _pending = session.get("pending_title")
            if _pending and status == "complete":
                _pdb = _get_db()
                if _pdb:
                    _session_key = session.get("session_key") or sid
                    try:
                        if _pdb.set_session_title(_session_key, _pending):
                            session["pending_title"] = None
                    except ValueError as exc:
                        # Invalid/duplicate title — non-retryable, drop it.
                        # Auto-title will take over. Fix for #19029.
                        session["pending_title"] = None
                        logger.info(
                            "Dropping pending title for session %s: %s",
                            _session_key, exc,
                        )
                    except Exception:
                        # Transient DB failure — keep pending_title for retry.
                        pass

            if (
                not internal_origin
                and not _deferred_worker_reply
                and
                not _ephemeral_control
                and
                status == "complete"
                and isinstance(raw, str)
                and raw.strip()
                and isinstance(text, str)
                and text.strip()
            ):
                try:
                    from agent.title_generator import maybe_auto_title

                    _title_key = session.get("session_key") or sid
                    maybe_auto_title(
                        _get_db(),
                        _title_key,
                        text,
                        raw,
                        session.get("history", []),
                        # Push the generated title live so the sidebar renames
                        # without waiting for the next list refresh (the titler
                        # runs async, after this turn's refresh already fired).
                        title_callback=lambda t, _k=_title_key: _emit(
                            "session.title", sid, {"session_id": _k, "title": t}
                        ),
                    )
                except Exception:
                    pass

            # CLI parity: when voice-mode TTS is on, speak the agent reply
            # (cli.py:_voice_speak_response).  Only the final text — tool
            # calls / reasoning already stream separately and would be
            # noisy to read aloud.
            if (
                not internal_origin
                and not _deferred_worker_reply
                and
                not _ephemeral_control
                and
                status == "complete"
                and isinstance(raw, str)
                and raw.strip()
                and _voice_tts_enabled()
            ):
                try:
                    from hermes_cli.voice import speak_text

                    spoken = raw
                    threading.Thread(
                        target=speak_text, args=(spoken,), daemon=True
                    ).start()
                except ImportError:
                    logger.warning("voice TTS skipped: hermes_cli.voice unavailable")
                except Exception as e:
                    logger.warning("voice TTS dispatch failed: %s", e)
        except Exception as e:
            import traceback

            trace = traceback.format_exc()
            try:
                os.makedirs(os.path.dirname(_CRASH_LOG), exist_ok=True)
                with open(_CRASH_LOG, "a", encoding="utf-8") as f:
                    f.write(
                        f"\n=== turn-dispatcher exception · "
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')} · sid={sid} ===\n"
                    )
                    f.write(trace)
            except Exception:
                pass
            print(
                f"[gateway-turn] {type(e).__name__}: {e}", file=sys.stderr, flush=True
            )
            _watcher_fallback = (
                str(internal_fallback_text or "").strip()
                if internal_origin == "watcher_hook"
                else ""
            )
            if _watcher_fallback:
                # The hidden hook already owns a message.start slot. Complete
                # that slot with the durable report so the Web client cannot be
                # left spinning merely because the optional synthesis failed.
                fallback_payload = {
                    **_event_tag,
                    "text": _watcher_fallback,
                    "usage": _get_usage(agent),
                    "status": "complete",
                    "watcher_fallback": True,
                    "warning": (
                        "Main-agent watcher synthesis raised an exception; the "
                        "consolidated watcher report was delivered directly."
                    ),
                }
                try:
                    rendered = render_message(_watcher_fallback, cols)
                    if rendered:
                        fallback_payload["rendered"] = rendered
                except Exception:
                    pass
                _emit("message.complete", sid, fallback_payload)
            else:
                _emit("error", sid, {**_event_tag, "message": str(e)})
            # A VoiceAgent task may be awaiting this exact turn. Resolve it on
            # the error path as well; otherwise the foreground gate is released
            # below but its Future remains stuck until the 300s timeout.
            # ★ pop 在锁内, 与写点对称 (review A)。
            with session["history_lock"]:
                _failed_voice_seq = session.pop("_voice_active_seq", None)
            if _failed_voice_seq is not None:
                try:
                    _voice = _get_voice_agent(session)
                    if _voice is not None:
                        _voice.notify_main_reply(
                            _failed_voice_seq,
                            f"主 Agent 任务处理出错：{type(e).__name__}: {e}",
                        )
                except Exception as _voice_exc:
                    logger.debug("voice error reply failed: %s", _voice_exc)
        finally:
            try:
                if approval_token is not None:
                    reset_current_session_key(approval_token)
            except Exception:
                pass
            if home_token is not None:
                reset_hermes_home_override(home_token)
            _clear_session_context(session_tokens)
            with session["history_lock"]:
                session["running"] = False
                session["last_active"] = time.time()
                _clear_inflight_turn(session)
                # This guard belongs to the whole asynchronous Monitor/Watcher
                # turn, not merely its dispatch thread. Releasing it here keeps
                # concurrent user submits queue-only until the hidden hook has
                # fully returned and its ephemeral history has been discarded.
                if internal_origin in {"monitor_hook", "watcher_hook"}:
                    session["_monitor_hook_running"] = False
            _emit("session.info", sid, _session_info(agent, session))

        # A user prompt that arrived mid-turn (interrupt + queue) wins over
        # every auto follow-up below — drain it first and skip them this cycle;
        # the goal judge / notifications re-evaluate at the end of that turn.
        if _drain_queued_prompt(rid, sid, session):
            return

        # A live_watcher that finished while this turn was running left its
        # completion hook queued (never dropped). Drain ONE now that we're idle —
        # it chains its own tail, which drains the next queued hook, and so on.
        if _drain_watcher_hook(rid, sid, session):
            return

        # (VoiceAgent 委派现在统一进 queued_prompts, 由上面的 _drain_queued_prompt
        #  一并 drain — 不再单独一套 _voice_hook_queue。)

        # Chain a goal-continuation turn if the judge said so. We do
        # this AFTER the finally releases session["running"], so the
        # nested _run_prompt_submit doesn't deadlock on the busy
        # guard. A real user prompt that races us wins because
        # prompt.submit sets running=True under the history_lock and
        # we check that guard before re-firing.
        if goal_followup:
            with session["history_lock"]:
                if session.get("running"):
                    # User already sent something — their turn wins,
                    # the judge will re-run on the next turn anyway.
                    return
                session["running"] = True
            try:
                _emit("message.start", sid)
                _run_prompt_submit(rid, sid, session, goal_followup)
            except Exception as _cont_exc:
                print(
                    f"[tui_gateway] goal continuation dispatch failed: "
                    f"{type(_cont_exc).__name__}: {_cont_exc}",
                    file=sys.stderr,
                )
                with session["history_lock"]:
                    session["running"] = False

        # Drain completion notifications that arrived during this turn.
        # The background poller handles between-turn delivery; this is
        # the safety net for events that arrived mid-turn.
        try:
            from tools.process_registry import process_registry

            for _evt, synth in process_registry.drain_notifications():
                with session["history_lock"]:
                    if session.get("running"):
                        process_registry.completion_queue.put(_evt)
                        break
                    session["running"] = True
                try:
                    _emit("message.start", sid)
                    _run_prompt_submit(rid, sid, session, synth)
                except Exception as _n_exc:
                    print(
                        f"[tui_gateway] completion notification dispatch failed: "
                        f"{type(_n_exc).__name__}: {_n_exc}",
                        file=sys.stderr,
                    )
                    with session["history_lock"]:
                        session["running"] = False
        except Exception as _drain_exc:
            print(
                f"[tui_gateway] completion queue drain failed: "
                f"{type(_drain_exc).__name__}: {_drain_exc}",
                file=sys.stderr,
            )

    run_thread = threading.Thread(target=run, daemon=True)
    session["_run_thread"] = run_thread
    run_thread.start()


@method("clipboard.paste")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    try:
        from hermes_cli.clipboard import has_clipboard_image, save_clipboard_image
    except Exception as e:
        return _err(rid, 5027, f"clipboard unavailable: {e}")

    session["image_counter"] = session.get("image_counter", 0) + 1
    img_dir = _hermes_home / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_path = (
        img_dir
        / f"clip_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{session['image_counter']}.png"
    )

    # Save-first: mirrors CLI keybinding path; more robust than has_image() precheck
    if not save_clipboard_image(img_path):
        session["image_counter"] = max(0, session["image_counter"] - 1)
        msg = (
            "Clipboard has image but extraction failed"
            if has_clipboard_image()
            else "No image found in clipboard"
        )
        return _ok(rid, {"attached": False, "message": msg})

    session.setdefault("attached_images", []).append(str(img_path))
    return _ok(
        rid,
        {
            "attached": True,
            "path": str(img_path),
            "count": len(session["attached_images"]),
            **_image_meta(img_path),
        },
    )


@method("image.attach")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    raw = str(params.get("path", "") or "").strip()
    if not raw:
        return _err(rid, 4015, "path required")
    try:
        from cli import (
            _IMAGE_EXTENSIONS,
            _detect_file_drop,
            _resolve_attachment_path,
            _split_path_input,
        )

        dropped = _detect_file_drop(raw)
        if dropped:
            image_path = dropped["path"]
            remainder = dropped["remainder"]
        else:
            path_token, remainder = _split_path_input(raw)
            image_path = _resolve_attachment_path(path_token)
            if image_path is None:
                return _err(rid, 4016, f"image not found: {path_token}")
        if image_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            return _err(rid, 4016, f"unsupported image: {image_path.name}")
        session.setdefault("attached_images", []).append(str(image_path))
        return _ok(
            rid,
            {
                "attached": True,
                "path": str(image_path),
                "count": len(session["attached_images"]),
                "remainder": remainder,
                "text": remainder or f"[User attached image: {image_path.name}]",
                **_image_meta(image_path),
            },
        )
    except Exception as e:
        return _err(rid, 5027, str(e))


# Byte-upload attach caps. 25 MB matches Anthropic's per-image limit; 50 MB / 25
# pages bounds a single PDF drop so it can't blow the context budget.
_ATTACH_BYTES_MAX_BYTES = 25 * 1024 * 1024
_PDF_ATTACH_MAX_BYTES = 50 * 1024 * 1024
_PDF_ATTACH_MAX_PAGES = 25

# Leading magic bytes → file extension, for filename-less uploads.
_IMAGE_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"\xff\xd8\xff", ".jpg"),
    (b"GIF87a", ".gif"),
    (b"GIF89a", ".gif"),
    (b"BM", ".bmp"),
)


def _decode_attach_base64(raw: str, *, mime_prefix: str) -> bytes | None:
    """Decode a base64 (optionally data-URL-wrapped) payload.

    Accepts ``data:<mime_prefix>...;base64,<b64>`` plus embedded whitespace.
    Returns the decoded bytes, or ``None`` when the input isn't valid base64.
    """
    import base64 as _base64
    import re as _re

    cleaned = raw.strip()
    m = _re.match(
        rf"^data:{_re.escape(mime_prefix)}[a-zA-Z0-9.+-]*;base64,(.*)$",
        cleaned,
        _re.DOTALL,
    )
    if m:
        cleaned = m.group(1)
    cleaned = _re.sub(r"\s+", "", cleaned)
    try:
        return _base64.b64decode(cleaned, validate=True)
    except Exception:
        return None


def _sniff_image_ext(img_bytes: bytes, filename: str = "") -> str:
    """Resolve an image extension from a filename hint, else magic bytes.

    Falls back to ``.png``. WebP needs the RIFF/WEBP container check, handled
    before the generic table.
    """
    if filename:
        suffix = Path(filename).suffix.lower()
        if suffix:
            return suffix
    head = img_bytes[:16]
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return ".webp"
    for sig, ext in _IMAGE_MAGIC:
        if head.startswith(sig):
            return ext
    return ".png"


def _allowed_image_extensions() -> frozenset[str]:
    try:
        from cli import _IMAGE_EXTENSIONS

        return frozenset(_IMAGE_EXTENSIONS)
    except Exception:
        return frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"})


def _queue_attached_image(session: dict, img_bytes: bytes, ext: str, *, prefix: str) -> Path:
    """Write image bytes into the gateway's images dir and queue them.

    Mirrors what ``image.attach`` does for a local path: appends to
    ``session["attached_images"]`` so the next ``prompt.submit`` picks it up via
    the existing native-image-attach pipeline. Returns the written path.
    """
    session["image_counter"] = session.get("image_counter", 0) + 1
    img_dir = _hermes_home / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = img_dir / f"{prefix}_{ts}_{session['image_counter']}{ext}"
    try:
        img_path.write_bytes(img_bytes)
    except Exception:
        session["image_counter"] = max(0, session["image_counter"] - 1)
        raise
    session.setdefault("attached_images", []).append(str(img_path))
    return img_path


@method("image.attach_bytes")
def _(rid, params: dict) -> dict:
    """Attach an image to the session from base64 bytes (remote-client path).

    A desktop app or web dashboard running on a DIFFERENT machine than the
    gateway can't hand us a local path — that file only exists on the client's
    disk. So it uploads the raw image bytes (base64) and we write them into the
    gateway's own images dir. The response shape mirrors ``image.attach`` so the
    client treats both identically.

    Params:
      content_base64 / data (str, required): base64 image bytes. Accepts a
        ``data:image/...;base64,`` prefix and embedded whitespace. ``data`` is
        an accepted alias for older desktop builds.
      filename / ext (str, optional): extension hint. Without it, magic bytes
        identify PNG/JPEG/GIF/WebP/BMP, falling back to ``.png``.
    """
    session, err = _sess(params, rid)
    if err:
        return err

    raw_b64 = str(params.get("content_base64") or params.get("data") or "").strip()
    if not raw_b64:
        return _err(rid, 4015, "content_base64 required")

    img_bytes = _decode_attach_base64(raw_b64, mime_prefix="image/")
    if img_bytes is None:
        return _err(rid, 4017, "data is not valid base64")
    if not img_bytes:
        return _err(rid, 4017, "image is empty")
    if len(img_bytes) > _ATTACH_BYTES_MAX_BYTES:
        mb = _ATTACH_BYTES_MAX_BYTES // (1024 * 1024)
        return _err(rid, 4018, f"image too large ({len(img_bytes)} bytes; cap is {mb} MB)")

    filename = str(params.get("filename", "") or "")
    ext_hint = str(params.get("ext", "") or "").strip().lower()
    if ext_hint and not ext_hint.startswith("."):
        ext_hint = "." + ext_hint
    ext = _sniff_image_ext(img_bytes, filename or (f"x{ext_hint}" if ext_hint else ""))
    if ext not in _allowed_image_extensions():
        return _err(rid, 4016, f"unsupported image extension: {ext}")

    try:
        img_path = _queue_attached_image(session, img_bytes, ext, prefix="upload")
    except Exception as e:
        return _err(rid, 5027, f"write failed: {e}")

    return _ok(
        rid,
        {
            "attached": True,
            "path": str(img_path),
            "count": len(session["attached_images"]),
            "remainder": "",
            "text": f"[User attached image: {img_path.name}]",
            "bytes": len(img_bytes),
            **_image_meta(img_path),
        },
    )


@method("multimodal.frame")
def _(rid, params: dict) -> dict:
    """Push a live video frame (camera/screen) into the session agent's
    FrameBuffer (multimodal phase 2).

    High-frequency (~2fps) best-effort channel: uses the non-blocking session
    lookup and silently no-ops if the agent isn't built yet or the multimodal
    FrameBuffer is unavailable — a dropped frame must never error the client or
    block the event loop. The next chat turn reads recent frames from this
    buffer to answer with vision.

    params: {session_id, ts: float, jpeg_b64: str}
    """
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    agent = session.get("agent")
    buf = getattr(agent, "frame_buffer", None) if agent else None
    if buf is None:
        # Agent not built yet / multimodal unavailable — drop silently.
        return _ok(rid, {"buffered": False, "reason": "no_buffer"})
    jpeg_b64 = str(params.get("jpeg_b64") or "")
    if not jpeg_b64:
        return _err(rid, 4015, "jpeg_b64 required")
    if jpeg_b64.startswith("data:"):
        comma = jpeg_b64.find(",")
        if comma >= 0:
            jpeg_b64 = jpeg_b64[comma + 1:]
    # Cheap sanity check only. A prior revision did full base64 decode +
    # validate=True here — measured 5-10 ms per ~200 KB frame, and this
    # handler used to run on the dispatcher thread (see _LONG_HANDLERS
    # above) so that decode blocked every chat SSE token behind it. The
    # frame is now routed to the RPC pool, but even there full decode is
    # wasted work: frame_to_image_content downstream decodes into a data
    # URL, and a malformed payload throws there with a clear traceback.
    # Just reject the obvious garbage (empty / not remotely base64).
    if len(jpeg_b64) < 100 or any(c.isspace() for c in jpeg_b64[:64]):
        return _err(rid, 4017, "jpeg_b64 is not valid base64")
    try:
        source_type = str(
            params.get("source_type") or params.get("source")
            or params.get("sourceType") or params.get("kind")
            or params.get("type") or "").strip().lower()
        # Desktop screen-share only: the specific Electron picker source the user
        # chose (e.g. 'window:12345:0' + '访达'). Empty on camera/web/legacy paths.
        # Part 3 (window AX/UIA capture) will look these up on the Frame to find
        # the target window; storing them per-frame is required so a mid-share
        # window switch keeps the historical frames auditable.
        source_id = str(params.get("source_id") or params.get("sourceId") or "").strip()
        source_name = str(
            params.get("source_name") or params.get("sourceName") or "").strip()
        generation_raw = params.get("capture_generation")
        capture_client_id = str(params.get("capture_client_id") or "").strip()
        capture_attempt_id = str(params.get("capture_attempt_id") or "").strip()
        try:
            generation = int(generation_raw) if generation_raw is not None else None
        except (TypeError, ValueError):
            generation = None
        # ★ SERVER-AUTHORITATIVE monotonic frame ts + epoch/wall anchors are now
        #   owned by FrameBuffer.push_live (maintained under its lock, so the
        #   epoch check-then-set can't race across concurrent pushes, and any
        #   non-gateway feed path gets the same anchors instead of None). The
        #   client-supplied `ts` is intentionally ignored (a 0 / non-monotonic
        #   client clock used to strand the monitor/watcher cursor).
        capture_lock = session.setdefault("_mm_capture_lock", threading.RLock())
        with capture_lock:
            if generation is not None:
                current_client_id = str(
                    session.get("_mm_capture_client_id") or "").strip()
                current_generation = session.get("_mm_capture_generation")
                current_attempt_id = str(
                    session.get("_mm_capture_attempt_id") or "").strip()
                if (capture_client_id and current_client_id
                        and capture_client_id != current_client_id):
                    return _ok(rid, {"buffered": False, "reason": "stale_capture"})
                if (capture_attempt_id and current_attempt_id
                        and capture_attempt_id != current_attempt_id):
                    return _ok(rid, {"buffered": False, "reason": "stale_capture"})
                if (current_generation is None
                        or generation != int(current_generation)
                        or not bool(session.get("_mm_capture_active", False))):
                    return _ok(rid, {"buffered": False, "reason": "stale_capture"})
            res = buf.push_live(
                jpeg_b64, source_type=source_type,
                source_id=source_id, source_name=source_name,
            )
            _record_mm_capture_anchor_pair(
                session,
                capture_attempt_id=capture_attempt_id,
                client_ts=params.get("ts"),
                server_ts=res.get("ts"),
            )
            # Monitor owns a short raw 2fps queue and must wake even when the same
            # frame was dropped from the long-term dHash-deduped buffer. Fall back to
            # ``stored`` for older/custom FrameBuffer implementations.
            if res.get("monitor_stored", res.get("stored")):
                monitor_engine = session.get("_mm_monitor_engine")
                if monitor_engine is not None and hasattr(monitor_engine, "notify_frame"):
                    try:
                        monitor_engine.notify_frame()
                    except Exception:
                        pass
    except Exception as e:
        return _err(rid, 5027, f"frame push failed: {e}")
    return _ok(rid, {"buffered": True, "size": res.get("size", buf.size)})


@method("multimodal.monitor_toggle")
def _(rid, params: dict) -> dict:
    """Enable/disable a single monitor from the UI toggle switch.

    params: {session_id, monitor_id, enabled: bool}

    A disabled monitor stays in the registry (so its label/brief persist) but
    the monitor daemon skips it each tick — proactive SPEAK is paused until it
    is re-enabled. Pushes an updated ``multimodal.monitors`` snapshot so the UI
    reflects the new state.
    """
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    agent = session.get("agent")
    mons = getattr(agent, "mm_monitors", None) if agent else None
    if not mons:
        return _err(rid, 4020, "no monitors registered")
    mid = str(params.get("monitor_id") or "").strip()
    if mid not in mons:
        return _err(rid, 4021, f"monitor_id {mid!r} not found")
    enabled = bool(params.get("enabled", True))
    # Route through the guarded set_monitor op so enabling re-applies the LIVE
    # STREAM guard (+ respawns the engine job). Without the stream the enable
    # FAILS → the UI toggle's optimistic update rolls back to off (the "点 on
    # 没流就自动弹回" behavior). A direct `enabled = True` here would flip the
    # flag over a dead stream. disable pauses + stops the job.
    from tools.monitor_tool import set_monitor as _set_monitor
    _sid = params.get("session_id") or session.get("session_key") or ""
    raw = _set_monitor(op=("enable" if enabled else "disable"),
                       monitor_id=mid, session_id=_sid)
    try:
        import json as _json
        _res = _json.loads(raw) if isinstance(raw, str) else (raw or {})
    except Exception:
        _res = {}
    if not _res.get("success", True):
        # Enable rejected (e.g. no live stream). Report failure so the UI rolls
        # the switch back to off. Registry flag is unchanged (still off).
        return _err(rid, 4022, str(_res.get("error") or "toggle failed"))
    # set_monitor already pushed the refreshed registry snapshot.
    return _ok(rid, {"monitor_id": mid, "enabled": bool(mons[mid].get("enabled", False))})


@method("multimodal.watcher_toggle")
def _(rid, params: dict) -> dict:
    """Enable/disable a single deep-research from the UI toggle switch.

    params: {session_id, watcher_id, enabled: bool}

    Mirrors monitor_toggle: routes through the guarded set_live_watcher op so
    enabling re-applies the live-stream guard (+ respawns the delegation). No
    stream → enable fails → the UI toggle rolls back to off.
    """
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    agent = session.get("agent")
    watchers = getattr(agent, "mm_watchers", None) if agent else None
    if not watchers:
        return _err(rid, 4020, "no watchers registered")
    rid_r = str(params.get("watcher_id") or "").strip()
    if rid_r not in watchers:
        return _err(rid, 4021, f"watcher_id {rid_r!r} not found")
    enabled = bool(params.get("enabled", True))
    from tools.live_watcher_tool import set_live_watcher as _set_lr
    _sid = params.get("session_id") or session.get("session_key") or ""
    raw = _set_lr(op=("enable" if enabled else "disable"),
                  watcher_id=rid_r, session_id=_sid)
    try:
        import json as _json
        _res = _json.loads(raw) if isinstance(raw, str) else (raw or {})
    except Exception:
        _res = {}
    if not _res.get("success", True):
        return _err(rid, 4022, str(_res.get("error") or "toggle failed"))
    _status = watchers.get(rid_r, {}).get("status", "")
    return _ok(rid, {"watcher_id": rid_r, "enabled": _status == "running"})


@method("multimodal.list_registries")
def _(rid, params: dict) -> dict:
    """前端进会话时主动拉取当前 monitor + watcher 注册表快照 (不依赖后端 push 时机)。

    params: {session_id}
    returns: {ready:bool, monitors:[…], watchers:[…]} —— 形状与 multimodal.monitors /
    multimodal.watchers 推送事件一致。没有 agent / 无注册表 → 空数组。用于启动时
    判断是否有未完成任务从而自动打开右侧深度面板。

    ★ 用 _sess_nowait (不阻塞): 之前用 _sess 会阻塞等整个 agent build 完成 (最多 30s),
    导致启动后多模态面板"好久才出现"。改回立即返回快照 (build 未完时可能为空); build
    完成时后台会【无条件 _push_mm_registries】(见 _build), 前端订阅到推送即填充 → 面板
    自动出现。这样既不阻塞, 又能拿到 reconcile 后的完整注册表。
    """
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    agent = session.get("agent")
    ready_event = session.get("agent_ready")
    ready = bool(
        agent is not None
        and ready_event is not None
        and callable(getattr(ready_event, "is_set", None))
        and ready_event.is_set()
    )
    monitors: list = []
    watchers: list = []
    try:
        mons = list((getattr(agent, "mm_monitors", {}) or {}).values()) if agent else []
        from tools.monitor_tool import _monitor_label as _mlabel
        from agent.multimodal import monitor_agent as _ma_reg
        monitors = []
        for m in mons:
            # ★ 五态 status: 文件头执行态优先, 兜底 enabled → running/interrupted。
            _fst = (_ma_reg.read_status(m.get("id")) or {}).get("status")
            _mstatus = _fst or ("running" if m.get("enabled", True) else "interrupted")
            monitors.append({
                "monitor_id": m.get("id"),
                "brief": m.get("monitor_query", "") or m.get("brief", ""),
                "monitor_query": m.get("monitor_query", "") or m.get("brief", ""),
                "label": _mlabel(m),
                "enabled": bool(m.get("enabled", True)),
                "status": _mstatus,
                "trigger_mode": _normalize_mm_monitor_trigger_mode(
                    m.get("trigger_mode")),
                "silent": bool(m.get("silent", False)),
                "report_interval": m.get("report_interval"),
                "created_at": m.get("created_at", 0.0),
            })
    except Exception:
        monitors = []
    try:
        rs = list((getattr(agent, "mm_watchers", {}) or {}).values()) if agent else []
        from tools.live_watcher_tool import _watcher_label as _wlabel
        # ★ 五态: 不再派生 deleting; 直接用 registry status。deleted 的不列。
        watchers = [{
            "watcher_id": r.get("id"),
            "label": _wlabel(r),
            "task_instruction": r.get("task_instruction", ""),
            "status": r.get("status", "running"),
            "hook_main_agent": bool(r.get("hook_main_agent", False)),
            "created_at": r.get("created_at", 0.0),
        } for r in rs if r.get("status") != "deleted"]
    except Exception:
        watchers = []
    return _ok(rid, {
        "ready": ready,
        "monitors": monitors,
        "watchers": watchers,
    })


@method("multimodal.list_monitor_alerts")
def _(rid, params: dict) -> dict:
    """Return all persisted monitor alerts for a session (right-panel hydrate).

    Called by the frontend on session.resume so that per-monitor alert history
    is rebuilt in the right multimodal panel. Bypasses the main-agent LLM path
    entirely — these live in the mm_monitor_alerts sidechannel table.

    params: {session_id}
    returns: {alerts: [{monitor_id, text, label, wall_ts, evidence}, ...]}
    """
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    durable_sid = str(session.get("session_key") or "")
    if not durable_sid:
        return _ok(rid, {"alerts": []})
    try:
        with _session_db(session) as _side_db:
            alerts = (
                _side_db.list_mm_monitor_alerts(durable_sid)
                if _side_db is not None else []
            )
    except Exception as exc:
        logger.warning("[mm-monitor] list_monitor_alerts failed: %s", exc)
        alerts = []
    return _ok(rid, {"alerts": alerts})


@method("multimodal.list_watcher_content")
def _(rid, params: dict) -> dict:
    """Return persisted watcher per-round reports (and finals) for a session.

    Called by the frontend on session.resume so DeepWindow segments and any
    final reports are rebuilt in the right multimodal panel. Bypasses the
    main-agent LLM path entirely — sourced from mm_watcher_reports +
    mm_watcher_finals sidechannel tables.

    params: {session_id}
    returns: {reports:[{watcher_id, round_idx, text, label, wall_ts}, ...],
              finals:[{watcher_id, text, wall_ts}, ...]}
    """
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    durable_sid = str(session.get("session_key") or "")
    if not durable_sid:
        return _ok(rid, {"reports": [], "finals": []})
    try:
        with _session_db(session) as _side_db:
            reports = (
                _side_db.list_mm_watcher_reports(durable_sid)
                if _side_db is not None else []
            )
    except Exception as exc:
        logger.warning("[mm-watcher] list_mm_watcher_reports failed: %s", exc)
        reports = []
    try:
        with _session_db(session) as _side_db:
            finals = (
                _side_db.list_mm_watcher_finals(durable_sid)
                if _side_db is not None else []
            )
    except Exception as exc:
        logger.warning("[mm-watcher] list_mm_watcher_finals failed: %s", exc)
        finals = []
    return _ok(rid, {"reports": reports, "finals": finals})


@method("multimodal.trajectory.list")
def _(rid, params: dict) -> dict:
    """Return the bounded unified worker trajectory for the live session."""
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    try:
        limit = max(1, min(
            int(params.get("limit") or _MM_TRAJECTORY_MAX_ENTRIES),
            _MM_TRAJECTORY_MAX_ENTRIES,
        ))
    except (TypeError, ValueError):
        limit = _MM_TRAJECTORY_MAX_ENTRIES
    with _sessions_lock:
        stored_rows = session.setdefault("_mm_trajectory", [])
        _bound_mm_trajectory(stored_rows)
        rows = copy.deepcopy(stored_rows[-limit:])
    return _ok(rid, {"entries": rows, "count": len(rows)})


@method("multimodal.get_watcher_report")
def _(rid, params: dict) -> dict:
    """读回某个深度研究任务的历史分析 (从 analyse/watch_<rid>.md 解析), 供前端重开
    历史任务时渲染只读窗口 (分段 + 最终报告)。

    params: {session_id, request_id | watcher_id}
    returns: read_structured() 的结构 (found/status/round_idx/query/rounds/final_report)。
    纯读文件, 不调模型; 找不到文件 → {found:false}。
    """
    # session 校验 (拿到会话即可; 报告文件按 rid 全局存, 不强依赖 agent)。
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    req = str(params.get("request_id") or params.get("watcher_id") or "").strip()
    if not req:
        return _err(rid, 4023, "missing request_id")
    try:
        from agent.multimodal import watch_file as _wf
        data = _wf.read_structured(req) or {"request_id": req, "found": False}
    except Exception as e:
        return _err(rid, 4024, f"read report failed: {e}")
    return _ok(rid, data)


@method("multimodal.user_audio")
def _(rid, params: dict) -> dict:
    """User speech → text → ask the agent (phase 4).

    The mic recording is transcribed by the configured ASR (Qwen3-ASR-style
    STTClient) and then submitted as if the user had typed it, so the same
    semantic multimodal routing and ask-time anchoring apply.

    params: {session_id, data_b64, mime}
    """
    session, err = _sess(params, rid)
    if err:
        return err
    import base64
    data_b64 = str(params.get("data_b64") or "")
    if not data_b64:
        return _err(rid, 4015, "data_b64 required")
    try:
        audio_bytes = base64.b64decode(data_b64)
    except Exception as exc:
        return _err(rid, 4017, f"data is not valid base64: {exc}")
    if len(audio_bytes) < 200:
        return _ok(rid, {"transcript": "", "reason": "too_short"})
    mime = str(params.get("mime") or "audio/webm")
    _sid = str(params.get("session_id") or "")
    try:
        import asyncio
        from agent.multimodal._dual_agent import STTClient
        from agent.multimodal.hermes_glue import build_config as _mm_cfg
        stt = STTClient(_mm_cfg())
        # Run the async transcribe on a dedicated private loop. asyncio.run()
        # would raise RuntimeError if this thread already had a running loop;
        # a fresh loop is always safe here (this RPC runs on the gateway thread).
        _stt_loop = asyncio.new_event_loop()
        try:
            text = _stt_loop.run_until_complete(
                stt.transcribe(audio_bytes, mime=mime))
        finally:
            _stt_loop.close()
    except Exception as exc:
        logger.debug("multimodal user_audio ASR failed: %s", exc)
        return _err(rid, 5027, f"ASR failed: {exc}")
    text = (text or "").strip()
    if not text:
        return _ok(rid, {"transcript": "", "reason": "empty"})
    # Submit as a normal turn (guard against colliding with a live turn).
    with session["history_lock"]:
        if session.get("running"):
            return _ok(rid, {"transcript": text, "queued": False, "reason": "busy"})
        session["running"] = True
    rid2 = f"__voice__{int(time.time() * 1000)}"
    client_request_id = _ensure_client_request_id(
        params.get("client_request_id"))
    session["_mm_voice_turn"] = True  # → auto-TTS in message.complete hook
    _emit("multimodal.asr_final", _sid, {
        "text": text,
        "request_id": client_request_id,
    })
    # Run the turn on a dedicated thread so this RPC returns the transcript
    # immediately (the caller wants the transcript ACK, not to block for the
    # whole LLM turn). Mirrors prompt.submit + the streaming asr_final path.
    def _voice_turn() -> None:
        try:
            # _run_prompt_submit emits message.start itself — no extra emit here
            # (a second one spawns a duplicate empty Assistant bubble).
            _run_prompt_submit(
                rid2,
                _sid,
                session,
                text,
                user_originated=True,
                client_request_id=client_request_id,
            )
        except Exception as exc:
            logger.debug("user_audio submit failed: %s", exc)
            with session["history_lock"]:
                session["running"] = False
    threading.Thread(target=_voice_turn, daemon=True,
                     name="mm-voice-turn").start()
    return _ok(rid, {
        "transcript": text,
        "submitted": True,
        "client_request_id": client_request_id,
    })


@method("multimodal.env_audio")
def _(rid, params: dict) -> dict:
    """Environment audio (people speaking in the video) → audio_observation in
    the memory backend (phase 4). No-op if the memory backend isn't running.

    params: {session_id, data_b64, mime, chunk_seq,
             client_start_ts, client_end_ts, client_duration_sec}
    """
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    backend = session.get("_mm_memory_backend")
    if backend is None:
        # Env-audio ASR rides on the memory backend. If it's absent here, the
        # backend failed to START (already loud-failed with its OWN cause via an
        # "error" event at startup) — so report an ASR-honest reason, not the
        # misleading bare "no_memory_backend" that made users chase the wrong
        # thing. The real cause is in that earlier error toast + the logs.
        return _ok(rid, {
            "ingested": False,
            "reason": "asr_backend_unavailable",
            "detail": "环境音 ASR 依赖记忆后端,但它未成功启动(见启动错误提示/日志)。",
        })
    import base64
    data_b64 = str(params.get("data_b64") or "")
    if not data_b64:
        return _err(rid, 4015, "data_b64 required")
    try:
        audio_bytes = base64.b64decode(data_b64)
    except Exception as exc:
        return _err(rid, 4017, f"data is not valid base64: {exc}")
    if len(audio_bytes) < 500:
        return _ok(rid, {"ingested": False, "reason": "too_short"})
    mime = str(params.get("mime") or "audio/webm")
    audio_sha256 = hashlib.sha256(audio_bytes).hexdigest()
    container, standalone_header = _audio_container_signature(audio_bytes, mime)
    client_start_ts = _optional_finite_float(params.get("client_start_ts"))
    client_end_ts = _optional_finite_float(
        params.get("client_end_ts", params.get("window_ts")))
    client_duration = _optional_finite_float(params.get("client_duration_sec"))
    try:
        default_duration = float(
            getattr(backend.cfg, "env_audio_window_sec", 5.0))
    except Exception:
        default_duration = 5.0

    # ★ Stamp env audio on the SAME server-authoritative timeline as video frames
    #   (FrameBuffer.now_ts = monotonic - frame epoch), NOT the client-supplied
    #   window_ts (performance.now-based, anchored at recorder start → a DIFFERENT
    #   epoch than the frame clock). Otherwise the memory writer's audio window
    #   [ask_ts - N, ask_ts] (ask_ts is a frame ts) selects the wrong / empty audio
    #   span and omni quotes / ASR blocks don't line up with the picture. Fall back
    #   to the client window_ts only if no frame has anchored the epoch yet.
    window_end_ts = None
    try:
        agent = session.get("agent")
        buf = getattr(agent, "frame_buffer", None) if agent else None
        if buf is not None:
            window_end_ts = buf.now_ts()
    except Exception:
        window_end_ts = None
    window_start_ts, window_end_ts, client_duration = _resolve_env_audio_window(
        server_end_ts=window_end_ts,
        client_start_ts=client_start_ts,
        client_end_ts=client_end_ts,
        client_duration_sec=client_duration,
        default_duration_sec=default_duration,
    )
    chunk_seq_raw = params.get("chunk_seq")
    try:
        chunk_seq = int(chunk_seq_raw) if chunk_seq_raw is not None else None
    except (TypeError, ValueError):
        chunk_seq = None
    chunk_id = str(params.get("chunk_id") or "").strip()
    if not chunk_id:
        seq_label = str(chunk_seq) if chunk_seq is not None else "legacy"
        chunk_id = f"env_{seq_label}_{audio_sha256[:10]}"
    metadata = {
        "capture_id": str(params.get("capture_id") or "").strip(),
        "chunk_id": chunk_id,
        "chunk_seq": chunk_seq,
        "sha256": audio_sha256,
        "sha256_short": audio_sha256[:12],
        "container": container,
        "standalone_header": standalone_header,
        "header_hex": audio_bytes[:16].hex(),
        "client_start_ts": client_start_ts,
        "client_end_ts": client_end_ts,
        "client_duration_sec": client_duration,
        "server_start_ts": window_start_ts,
        "server_end_ts": window_end_ts,
        "blob_timecode": _optional_finite_float(params.get("blob_timecode")),
    }
    try:
        accepted = backend.submit_env_audio(
            audio_bytes, mime=mime, window_ts=window_start_ts,
            metadata=metadata)
    except Exception as exc:
        return _err(rid, 5027, f"env_audio ingest failed: {exc}")
    if not accepted:
        logger.warning(
            "[mm-env-asr] audio slice rejected sid=%s chunk=%s seq=%s "
            "sha=%s bytes=%d mime=%s container=%s standalone=%s",
            params.get("session_id"), chunk_id, chunk_seq,
            audio_sha256[:12], len(audio_bytes), mime, container,
            standalone_header,
        )
        return _ok(rid, {
            "ingested": False,
            "reason": "memory_backend_not_ready",
        })
    logger.info(
        "[mm-env-asr] audio slice queued sid=%s chunk=%s seq=%s sha=%s "
        "bytes=%d mime=%s container=%s standalone=%s "
        "client=%.3f..%.3f/%.3fs server=%s..%s",
        params.get("session_id"), chunk_id, chunk_seq, audio_sha256[:12],
        len(audio_bytes), mime, container, standalone_header,
        client_start_ts if client_start_ts is not None else -1.0,
        client_end_ts if client_end_ts is not None else -1.0,
        client_duration,
        window_start_ts, window_end_ts,
    )
    return _ok(rid, {
        "ingested": True,
        "bytes": len(audio_bytes),
        "chunk_id": chunk_id,
        "sha256_short": audio_sha256[:12],
        "container": container,
        "standalone_header": standalone_header,
        "server_start_ts": window_start_ts,
        "server_end_ts": window_end_ts,
    })


# --------------------------------------------------------------------------- #
# Streaming realtime ASR (DashScope Qwen) — user speaks, server-side VAD
# segments speech and partial transcripts stream back as
# `multimodal.asr_partial`. Continuous Voice Dialog remains VAD-driven; a
# manual_turn buffers every final segment and commits exactly once on explicit
# finish. Backed by the per-session WatcherAgent's loop.
# Gracefully returns enabled:false when no dashscope_api_key is configured.
# --------------------------------------------------------------------------- #
_MM_ASR_RETIRED_TURNS_MAX = 32
_MM_CAPTURE_ANCHOR_EPOCHS_MAX = 8
_MM_CAPTURE_ANCHOR_PAIRS_MAX = 64


def _join_asr_segments(segments: list) -> str:
    """Join final VAD segments while preserving natural CJK/ASCII spacing."""
    joined = ""
    for raw in list(segments or []):
        text = str(raw or "").strip()
        if not text:
            continue
        if (joined and joined[-1].isascii() and joined[-1].isalnum()
                and text[0].isascii() and text[0].isalnum()):
            joined += " "
        joined += text
    return joined.strip()


def _snapshot_mm_frame_anchor(session: dict) -> Optional[float]:
    """Sample the server-authoritative latest frame timestamp for one turn."""
    try:
        agent = session.get("agent")
        frame_buffer = getattr(agent, "frame_buffer", None) if agent else None
        raw = (
            getattr(frame_buffer, "monitor_latest_ts", None)
            if frame_buffer is not None else None
        )
        if raw is None and frame_buffer is not None:
            raw = getattr(frame_buffer, "latest_ts", None)
        return _optional_finite_float(raw)
    except Exception:
        return None


def _mm_capture_anchor_epoch(
    session: dict, capture_attempt_id: str = "",
) -> str:
    """Stable key for one client/server capture-clock mapping epoch."""
    attempt_id = str(
        capture_attempt_id or session.get("_mm_capture_attempt_id") or ""
    ).strip()
    if attempt_id:
        return f"attempt:{attempt_id}"
    # Legacy Web captures may not yet send an attempt id.  Their client id,
    # generation, and start marker together still identify one clock epoch.
    return "legacy:{client}:{generation}:{started}".format(
        client=str(session.get("_mm_capture_client_id") or ""),
        generation=str(session.get("_mm_capture_generation") or ""),
        started=str(session.get("_mm_capture_client_started_at_ms") or ""),
    )


def _record_mm_capture_anchor_pair(
    session: dict,
    *,
    capture_attempt_id: str,
    client_ts: Any,
    server_ts: Any,
) -> None:
    """Record a bounded client-frame-ts -> FrameBuffer-ts correspondence."""
    client_value = _optional_finite_float(client_ts)
    server_value = _optional_finite_float(server_ts)
    if client_value is None or server_value is None:
        return
    epoch = _mm_capture_anchor_epoch(session, capture_attempt_id)
    epochs = session.setdefault("_mm_capture_anchor_pairs", {})
    rows = epochs.setdefault(epoch, [])
    rows.append((client_value, server_value))
    if len(rows) > _MM_CAPTURE_ANCHOR_PAIRS_MAX:
        del rows[:-_MM_CAPTURE_ANCHOR_PAIRS_MAX]
    # Dict insertion order is our epoch LRU. Refresh the active epoch and bound
    # old source/attempt mappings so long-lived desktop sessions stay constant.
    epochs.pop(epoch, None)
    epochs[epoch] = rows
    while len(epochs) > _MM_CAPTURE_ANCHOR_EPOCHS_MAX:
        epochs.pop(next(iter(epochs)), None)


def _resolve_mm_capture_anchor(
    session: dict,
    *,
    capture_attempt_id: str,
    client_anchor_ts: Any,
) -> Optional[float]:
    """Map a click-time client anchor onto the server FrameBuffer timeline.

    Select the latest frame whose client timestamp was at/before the click and
    return that frame's actual server timestamp.  Never compare or copy raw
    client-relative seconds into the server-monotonic FrameBuffer epoch.
    """
    client_anchor = _optional_finite_float(client_anchor_ts)
    if client_anchor is None:
        return _snapshot_mm_frame_anchor(session)
    epoch = _mm_capture_anchor_epoch(session, capture_attempt_id)
    rows = list((session.get("_mm_capture_anchor_pairs") or {}).get(epoch) or [])
    eligible = [
        (client_ts, server_ts)
        for client_ts, server_ts in rows
        if client_ts <= client_anchor
    ]
    if not eligible:
        if str(capture_attempt_id or "").strip():
            # Exact owner + finite click time, but every accepted frame belongs
            # to the future of that click.  Freeze "no frame"; falling back to
            # stop-time latest would admit the first post-click frame while ASR
            # was flushing.
            return None
        return _snapshot_mm_frame_anchor(session)
    # Concurrent frame RPCs can arrive out of client order. max() by client
    # time (and server time as a stable tie-break) still picks the click-bound
    # frame without admitting a post-click capture.
    return max(eligible, key=lambda pair: (pair[0], pair[1]))[1]


def _retire_asr_turn(session: dict, turn_id: str, result: dict) -> None:
    """Keep a small idempotency/tombstone cache for stopped ASR turns."""
    turn_id = str(turn_id or "").strip()
    if not turn_id:
        return
    retired = session.setdefault("_mm_asr_retired_turns", {})
    retired.pop(turn_id, None)
    retired[turn_id] = dict(result)
    while len(retired) > _MM_ASR_RETIRED_TURNS_MAX:
        retired.pop(next(iter(retired)), None)


def _abort_active_asr_turn(
    session: dict,
    *,
    reason: str,
    reopenable: bool,
) -> Optional[dict]:
    """Abort the session-owned ASR without emitting/submitting a user turn.

    Used by transport detach and session finalization, where relying on a
    renderer cancellation RPC is unsafe because that socket may already be
    closed.  A detach tombstone is explicitly reopenable by the *new current*
    transport using the same logical turn id; ordinary stop/cancel tombstones
    remain terminal.
    """
    capture_lock = session.setdefault("_mm_capture_lock", threading.RLock())
    with capture_lock:
        turn = session.get("_mm_asr_turn")
        if not isinstance(turn, dict):
            return None
        with turn["lock"]:
            state = str(turn.get("state") or "")
            if state == "stopped":
                return dict(turn.get("result") or {})
            if state == "stopping":
                # An explicit graceful finish may already own the Watcher key.
                # Flip its effective disposition before its trailing callbacks
                # or commit phase; that leader will publish the terminal result.
                abort_won = not bool(turn.get("stop_committed", False))
                if abort_won:
                    turn["disposition"] = "cancel"
                    turn["abort_reason"] = str(reason or "cancelled")
                session["_mm_asr_on"] = False
                turn_id = str(turn.get("turn_id") or "")
                reopenable_turns = session.setdefault(
                    "_mm_asr_reopenable_turns", {})
                if reopenable and abort_won:
                    reopenable_turns[turn_id] = turn.get("owner_transport")
                else:
                    reopenable_turns.pop(turn_id, None)
                return None
            turn["state"] = "stopping"
            turn["disposition"] = "cancel"
            turn["abort_reason"] = str(reason or "cancelled")
            turn_id = str(turn.get("turn_id") or "")
            mode = str(turn.get("mode") or "continuous")
            key = str(turn.get("key") or "")
            old_owner = turn.get("owner_transport")

    session["_mm_asr_on"] = False
    engine = session.get("_mm_live_watcher_agent")
    if engine is not None:
        try:
            try:
                engine.asr_stop(key, graceful=False)
            except TypeError:
                engine.asr_stop(key)
        except Exception:
            logger.debug("abortive ASR close failed: %s", reason, exc_info=True)

    result = {
        "ok": True,
        "turn_id": turn_id,
        "mode": mode,
        "disposition": "cancel",
        "transcript": "",
        "submitted": False,
        "reason": str(reason or "cancelled"),
        "anchor_ts": None,
        "graceful": False,
    }
    with turn["lock"]:
        turn["state"] = "stopped"
        turn["result"] = dict(result)
    with capture_lock:
        _retire_asr_turn(session, turn_id, result)
        reopenable_turns = session.setdefault(
            "_mm_asr_reopenable_turns", {})
        if reopenable:
            reopenable_turns[turn_id] = old_owner
            while len(reopenable_turns) > _MM_ASR_RETIRED_TURNS_MAX:
                reopenable_turns.pop(next(iter(reopenable_turns)), None)
        else:
            reopenable_turns.pop(turn_id, None)
    turn["stop_event"].set()
    return result


@method("multimodal.asr_start")
def _(rid, params: dict) -> dict:
    """Open a streaming ASR session.

    Modern callers pass ``{session_id, turn_id, mode}``, where ``manual_turn``
    buffers until an explicit finish and ``continuous`` submits each VAD final.
    ``turn_id`` is the ownership token for every subsequent audio/stop RPC.
    """
    # Voice-only Desktop sessions start life as ordinary ``source=tui``
    # runtimes. Promote on demand before looking up the watcher/ASR engine;
    # unlike source_started this does not mark video capture active or couple
    # mic stop to the camera/screen lifecycle.
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    _sid = str(params.get("session_id") or "")
    requested_mode = str(params.get("mode") or "").strip().lower()
    if requested_mode not in {"", "manual_turn", "continuous"}:
        return _err(rid, 4004, "mode must be manual_turn or continuous")
    raw_turn_id = str(params.get("turn_id") or "").strip()
    modern = bool(requested_mode or raw_turn_id)
    mode = requested_mode or "continuous"
    explicit_manual = requested_mode == "manual_turn"
    if len(raw_turn_id) > 128:
        return _err(rid, 4004, "turn_id exceeds 128 characters")
    supplied_turn_id = raw_turn_id
    if mode == "manual_turn" and not supplied_turn_id:
        return _err(rid, 4002, "turn_id is required for manual_turn")
    turn_id = supplied_turn_id or f"legacy-asr-{uuid.uuid4().hex}"
    capture_lock = session.setdefault("_mm_capture_lock", threading.RLock())
    request_transport = current_transport()
    key = f"asr:{_sid}"
    turn = {
        "turn_id": turn_id,
        "mode": mode,
        "modern": modern,
        "key": key,
        "segments": [],
        "partial": "",
        "state": "starting",
        "stop_event": threading.Event(),
        "lock": threading.RLock(),
        "owner_transport": request_transport,
        "started_at": time.monotonic(),
        "audio_delivery_failed": False,
        "result": None,
    }
    with capture_lock:
        owner = session.get("transport")
        if (request_transport is not None and owner is not None
                and owner is not request_transport):
            return _ok(rid, {"enabled": False, "reason": "stale_transport"})
        retired = session.get("_mm_asr_retired_turns") or {}
        if modern and turn_id in retired:
            reopenable_turns = session.get("_mm_asr_reopenable_turns") or {}
            old_owner = reopenable_turns.get(turn_id)
            can_reopen = bool(
                turn_id in reopenable_turns
                and request_transport is not None
                and session.get("transport") is request_transport
                and request_transport is not old_owner
            )
            if can_reopen:
                retired.pop(turn_id, None)
                reopenable_turns.pop(turn_id, None)
            else:
                return _ok(rid, {
                    "enabled": False,
                    "reason": "retired_turn",
                    "turn_id": turn_id,
                    "mode": mode,
                })
        active_turn = session.get("_mm_asr_turn")
        if (isinstance(active_turn, dict)
                and active_turn.get("state") in {
                    "starting", "connecting", "recording", "stopping",
                }):
            if (modern and active_turn.get("turn_id") == turn_id
                    and active_turn.get("state") == "recording"):
                active_owner = active_turn.get("owner_transport")
                if (request_transport is not None and active_owner is not None
                        and request_transport is not active_owner):
                    if session.get("transport") is request_transport:
                        # A resumed live session has explicitly rebound its
                        # transport.  The session-owned Qwen connection remains
                        # alive across the renderer WS disconnect (and
                        # asr_audio self-heals it if upstream died), so transfer
                        # the logical turn owner instead of opening a duplicate
                        # ASR socket or returning a false idempotent success to
                        # an unauthorized stale transport.
                        active_turn["owner_transport"] = request_transport
                    else:
                        return _ok(rid, {
                            "enabled": False,
                            "reason": "stale_transport",
                            "turn_id": turn_id,
                            "mode": str(active_turn.get("mode") or mode),
                        })
                active_mode = str(active_turn.get("mode") or mode)
                return _ok(rid, {
                    "enabled": True,
                    "turn_id": turn_id,
                    "mode": active_mode,
                    "idempotent": True,
                    **({"resumed": True}
                       if active_owner is not request_transport else {}),
                })
            return _ok(rid, {
                "enabled": False,
                "reason": "active_turn",
                "turn_id": str(active_turn.get("turn_id") or ""),
                "mode": str(active_turn.get("mode") or "continuous"),
            })
        # Publish a provisional owner before any slow build/promotion. Exact-id
        # stop/cancel and transport teardown can now retire it immediately
        # instead of waiting behind a minutes-long capture lock.
        session["_mm_asr_turn"] = turn
    # ★ 记下前端当前订阅的【live runtime sid】(前端 isMine 就是拿它比对)。VoiceAgent v2
    #   委派主 Agent 时 (_submit_main → _run_prompt_submit → _emit) 必须用这个 sid, 否则
    #   emit 用了 session_key (持久 id) → 前端 isMine 过滤掉 → 用户气泡 + Assistant 答案
    #   全部不显示。见 _get_voice_agent._submit_main。
    def _on_partial(text: str) -> None:
        # Live transcript preview — forward to frontend as-is.
        with turn["lock"]:
            if turn.get("state") not in {
                    "starting", "connecting", "recording", "stopping"}:
                return
            if (turn.get("state") == "stopping"
                    and turn.get("disposition") == "cancel"):
                return
            if mode == "manual_turn" and turn.get("audio_delivery_failed"):
                return
            if mode == "manual_turn":
                turn["partial"] = str(text or "")
        _emit("multimodal.asr_partial", _sid, {
            "text": text,
            "turn_id": turn_id,
        })

    def _on_eou_buffer_updated(buffer: list) -> None:
        """EOU 监听中状态: buffer 有更新 (追加或清空) → 通知前端显示已拼接段。"""
        with turn["lock"]:
            if (turn.get("state") == "stopped"
                    or turn.get("disposition") == "cancel"):
                return
        _emit("multimodal.asr_buffer", _sid, {
            "segments": list(buffer),
            "turn_id": turn_id,
        })

    def _dispatch_user_turn(
        text: str,
        *,
        anchor_ts: Optional[float] = None,
        anchor_frozen: bool = False,
    ) -> dict:
        """Commit exactly one voice-tagged foreground turn (or FIFO item)."""
        text = (text or "").strip()
        if not text:
            return {"submitted": False, "reason": "empty"}
        client_request_id = _ensure_client_request_id()
        lock = session.get("history_lock")
        queued_position = 0
        cancelled = False
        if lock is not None:
            with lock:
                with turn["lock"]:
                    cancelled = bool(
                        turn.get("disposition") == "cancel"
                        or turn.get("state") == "stopped"
                        or (turn.get("mode") == "manual_turn"
                            and turn.get("audio_delivery_failed"))
                    )
                    if not cancelled:
                        # Voice never interrupts and never overtakes: enqueue
                        # while a turn runs OR while anything is still waiting.
                        # Same invariant as _submit_main — see the comment there
                        # for the turn-boundary window this closes.
                        if (session.get("running")
                                or session.get("queued_prompts")):
                            queued_position = _enqueue_prompt(
                                session,
                                text,
                                session.get("transport"),
                                user_originated=True,
                                origin="voice_asr",
                                metadata={
                                    "voice_input": True,
                                    "client_request_id": client_request_id,
                                    "anchor_ts": anchor_ts,
                                    "anchor_frozen": anchor_frozen,
                                },
                            )
                        else:
                            session["running"] = True
                        if (turn.get("mode") == "manual_turn"
                                and turn.get("state") == "stopping"):
                            # Exact scheduler-admission linearization point:
                            # queue append/running claim and finish ownership
                            # become visible atomically to a racing cancel.
                            turn["stop_committed"] = True
        else:
            with turn["lock"]:
                cancelled = bool(
                    turn.get("disposition") == "cancel"
                    or turn.get("state") == "stopped"
                    or (turn.get("mode") == "manual_turn"
                        and turn.get("audio_delivery_failed"))
                )
                if (not cancelled and turn.get("mode") == "manual_turn"
                        and turn.get("state") == "stopping"):
                    turn["stop_committed"] = True
        if cancelled:
            return {
                "submitted": False,
                "cancelled": True,
                "reason": str(
                    turn.get("abort_reason")
                    or ("audio_delivery_failed"
                        if turn.get("audio_delivery_failed") else "cancelled")
                ),
            }
        _emit("multimodal.asr_final", _sid, {
            "text": text,
            "request_id": client_request_id,
            "turn_id": turn_id,
        })
        if queued_position:
            _emit("multimodal.trajectory", _sid, {
                "worker": "MainScheduler",
                "phase": "voice_prompt_queued",
                "origin": "voice_asr",
                "queue_position": queued_position,
                "text": text,
                "client_request_id": client_request_id,
                "anchor_ts": anchor_ts,
            })
            return {
                "submitted": True,
                "queued": True,
                "queue_position": queued_position,
                "client_request_id": client_request_id,
            }

        try:
            rid2 = f"__voice__{int(time.time() * 1000)}"
            session["_mm_voice_turn"] = True

            def _voice_turn() -> None:
                try:
                    kwargs = {
                        "user_originated": True,
                        "client_request_id": client_request_id,
                    }
                    if anchor_frozen:
                        kwargs.update({
                            "anchor_ts": anchor_ts,
                            "anchor_frozen": True,
                        })
                    _run_prompt_submit(
                        rid2, _sid, session, text, **kwargs)
                except Exception as exc:
                    logger.debug("asr turn submit failed: %s", exc)
                    if lock is not None:
                        with lock:
                            session["running"] = False

            threading.Thread(
                target=_voice_turn, daemon=True, name="mm-voice-turn").start()
            return {
                "submitted": True,
                "queued": False,
                "client_request_id": client_request_id,
            }
        except Exception as exc:
            logger.debug("asr turn dispatch failed: %s", exc)
            if lock is not None:
                with lock:
                    session["running"] = False
            return {"submitted": False, "reason": "dispatch_failed"}

    turn["submit_cb"] = _dispatch_user_turn

    def _on_final(text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        with turn["lock"]:
            if (turn.get("state") == "stopping"
                    and turn.get("disposition") == "cancel"):
                return
            if mode == "manual_turn" and turn.get("audio_delivery_failed"):
                return
        if mode == "manual_turn":
            # A VAD completion is a preview boundary, not a chat boundary.  A
            # manual recording may contain pauses and therefore many completed
            # segments; accumulate all of them until the explicit finish RPC.
            with turn["lock"]:
                if turn.get("state") not in {
                        "starting", "connecting", "recording", "stopping"}:
                    return
                segments = turn["segments"]
                # Qwen's adapter reconciles the one session.finished full
                # transcript before invoking us.  Every callback here is thus
                # a distinct VAD item and must be preserved even when adjacent
                # segments have identical text ("好" ... "好").  Text-based
                # dedupe at this layer silently deletes legitimate speech.
                segments.append(text)
                snapshot = list(segments)
                turn["partial"] = ""
            _emit("multimodal.asr_partial", _sid, {
                "text": "",
                "turn_id": turn_id,
            })
            _emit("multimodal.asr_buffer", _sid, {
                "segments": snapshot,
                "turn_id": turn_id,
            })
            return
        # Legacy continuous microphone semantics remain VAD-driven.  Modern
        # manual_turn returned above after buffering and reaches this helper
        # only from its explicit finish RPC.
        _dispatch_user_turn(text)

    def _on_speech_started() -> None:
        return

    # Runtime build/promotion is deliberately outside capture_lock.  The
    # provisional turn above is the ownership token a concurrent exact-id stop
    # can cancel while this potentially slow work is still running.
    activation_response = None
    runtime_tokens: list = []
    runtime_home_token = None
    try:
        runtime_tokens = _set_session_context(
            str(session.get("session_key") or _sid),
            str(session.get("cwd") or ""),
        )
        if profile_home := session.get("profile_home"):
            runtime_home_token = set_hermes_home_override(profile_home)
        _start_agent_build(_sid, session)
        wait_error = _wait_agent(
            session, rid, timeout=_MM_CAPTURE_ACTIVATION_TIMEOUT_SEC)
        if wait_error:
            activation_response = wait_error
        elif (session.get("_mm_live_watcher_agent") is None
              and not _promote_session_to_multimodal(_sid, session)):
            activation_response = _err(
                rid, 5027,
                "could not initialize the multimodal runtime for voice",
            )
    finally:
        if runtime_home_token is not None:
            reset_hermes_home_override(runtime_home_token)
        _clear_session_context(runtime_tokens)

    with capture_lock:
        retired_now = turn_id in (
            session.get("_mm_asr_retired_turns") or {})
        with turn["lock"]:
            turn_state = str(turn.get("state") or "")
        if (session.get("_mm_asr_turn") is not turn
                or retired_now or turn_state != "starting"):
            return _ok(rid, {
                "enabled": False,
                "reason": "retired_turn",
                "turn_id": turn_id,
                "mode": mode,
            })
        if activation_response is not None:
            session.pop("_mm_asr_turn", None)
            return activation_response
        # Revalidate immediately before the irreversible upstream ASR open so
        # a disconnected/replaced session cannot publish a late connection.
        owner = session.get("transport")
        if (_sessions.get(_sid) is not session or session.get("_finalized")
                or (request_transport is not None and owner is not None
                    and owner is not request_transport)):
            if session.get("_mm_asr_turn") is turn:
                session.pop("_mm_asr_turn", None)
            return _ok(rid, {"enabled": False, "reason": "stale_transport"})
        engine = session.get("_mm_live_watcher_agent")
        if engine is None:
            if session.get("_mm_asr_turn") is turn:
                session.pop("_mm_asr_turn", None)
            return _ok(rid, {"enabled": False, "reason": "no_router_engine"})
        session["_mm_live_sid"] = _sid
        with turn["lock"]:
            turn["state"] = "connecting"

    try:
        ok = engine.asr_start(
            key, _on_partial, _on_final, _on_speech_started)
        start_error = None
    except Exception as exc:
        ok = False
        start_error = exc

    cleanup_late_start = False
    with capture_lock:
        retired_now = turn_id in (
            session.get("_mm_asr_retired_turns") or {})
        with turn["lock"]:
            turn_state = str(turn.get("state") or "")
        owner = session.get("transport")
        stale_owner = bool(
            _sessions.get(_sid) is not session or session.get("_finalized")
            or (request_transport is not None and owner is not None
                and owner is not request_transport)
        )
        cancelled = bool(
            session.get("_mm_asr_turn") is not turn
            or retired_now or turn_state != "connecting" or stale_owner
        )
        if cancelled:
            cleanup_late_start = bool(ok)
        elif start_error is not None:
            session.pop("_mm_asr_turn", None)
        elif not ok:
            session.pop("_mm_asr_turn", None)
        else:
            with turn["lock"]:
                turn["state"] = "recording"
            session["_mm_asr_on"] = True

    if cleanup_late_start:
        try:
            try:
                engine.asr_stop(key, graceful=False)
            except TypeError:
                engine.asr_stop(key)
        except Exception:
            logger.debug("late cancelled asr_start cleanup failed", exc_info=True)
    if cancelled:
        return _ok(rid, {
            "enabled": False,
            "reason": "retired_turn" if retired_now else "stale_transport",
            "turn_id": turn_id,
            "mode": mode,
        })
    if start_error is not None:
        return _err(rid, 5027, f"asr_start failed: {start_error}")
    if not ok:
        return _ok(rid, {
            "enabled": False,
            "reason": "no_dashscope_key_or_disabled",
        })
    if ok:
        if not modern:
            return _ok(rid, {"enabled": True})
        return _ok(rid, {
            "enabled": True,
            "turn_id": turn_id,
            "mode": mode,
        })
    return _ok(rid, {"enabled": False})


@method("multimodal.asr_audio")
def _(rid, params: dict) -> dict:
    """Feed PCM16 owned by one exact ASR turn.

    Modern callers must include the ``turn_id`` returned by asr_start.  This
    prevents delayed audio from a stopped renderer turn entering a newer ASR
    connection that happens to share the same live session id.
    """
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    engine = session.get("_mm_live_watcher_agent")
    if engine is None:
        return _ok(rid, {"ok": False, "reason": "no_router_engine"})
    turn = session.get("_mm_asr_turn")
    if not isinstance(turn, dict):
        return _ok(rid, {"ok": False, "reason": "no_active_turn"})
    supplied_turn_id = str(params.get("turn_id") or "").strip()
    active_turn_id = str(turn.get("turn_id") or "")
    if turn.get("modern") and not supplied_turn_id:
        return _ok(rid, {
            "ok": False,
            "reason": "turn_id_required",
            "turn_id": active_turn_id,
        })
    if supplied_turn_id and supplied_turn_id != active_turn_id:
        return _ok(rid, {
            "ok": False,
            "reason": "stale_turn",
            "turn_id": supplied_turn_id,
            "active_turn_id": active_turn_id,
        })
    owner_transport = turn.get("owner_transport")
    request_transport = current_transport()
    if (request_transport is not None and owner_transport is not None
            and request_transport is not owner_transport):
        if session.get("transport") is request_transport:
            turn["owner_transport"] = request_transport
        else:
            return _ok(rid, {"ok": False, "reason": "stale_transport"})
    with turn["lock"]:
        if turn.get("state") != "recording":
            return _ok(rid, {
                "ok": False,
                "reason": "turn_not_recording",
                "turn_id": active_turn_id,
            })
        if turn.get("mode") == "manual_turn" and turn.get(
                "audio_delivery_failed", False):
            return _ok(rid, {
                "ok": False,
                "reason": "audio_delivery_failed",
                "turn_id": active_turn_id,
            })
    import base64 as _b64
    b64 = str(params.get("pcm_b64") or "")
    if not b64:
        return _ok(rid, {"ok": False, "reason": "empty"})
    try:
        pcm = _b64.b64decode(b64)
    except Exception:
        return _err(rid, 4017, "pcm_b64 not valid base64")
    try:
        delivered = bool(engine.asr_audio(str(turn.get("key") or ""), pcm))
    except Exception as exc:
        logger.debug("asr_audio failed: %s", exc)
        delivered = False
    if not delivered:
        if turn.get("mode") == "manual_turn":
            with turn["lock"]:
                turn["audio_delivery_failed"] = True
                turn["partial"] = ""
                turn["segments"] = []
        return _ok(rid, {
            "ok": False,
            "reason": "audio_delivery_failed",
            "turn_id": active_turn_id,
        })
    return _ok(rid, {"ok": True, "turn_id": active_turn_id})


@method("multimodal.asr_stop")
def _(rid, params: dict) -> dict:
    """Finish/cancel one exact ASR turn with idempotent ownership.

    ``disposition=finish`` commits one manual_turn only after the upstream
    ``session.finish`` handshake has flushed all final segments.
    ``disposition=cancel`` closes the stream but never submits.  Continuous
    Voice Dialog keeps its existing callback-driven semantics in both cases.
    """
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    _sid = str(params.get("session_id") or "")
    supplied_turn_id = str(params.get("turn_id") or "").strip()
    disposition = str(params.get("disposition") or "finish").strip().lower()
    if disposition not in {"finish", "cancel"}:
        return _err(rid, 4004, "disposition must be finish or cancel")
    modern_request = bool(supplied_turn_id or "disposition" in params)
    request_transport = current_transport()
    session_transport = session.get("transport")
    if (request_transport is not None and session_transport is not None
            and request_transport is not session_transport):
        return _ok(rid, {
            "ok": False,
            "reason": "stale_transport",
            "turn_id": supplied_turn_id,
        })
    capture_lock = session.setdefault("_mm_capture_lock", threading.RLock())

    leader = False
    wait_event = None
    turn = None
    with capture_lock:
        retired = session.get("_mm_asr_retired_turns") or {}
        if supplied_turn_id and supplied_turn_id in retired:
            return _ok(rid, dict(retired[supplied_turn_id]))

        active = session.get("_mm_asr_turn")
        if not isinstance(active, dict):
            if not supplied_turn_id:
                # Preserve the pre-turn-id no-op stop contract for legacy
                # continuous clients.
                return _ok(rid, {"ok": True})
            result = {
                "ok": True,
                "turn_id": supplied_turn_id,
                "transcript": "",
                "submitted": False,
                "disposition": disposition,
                "reason": "no_active_turn",
                "anchor_ts": None,
            }
            # Stop-before-start tombstone: a delayed cold-start RPC with this
            # token must not resurrect recording after the user stopped it.
            _retire_asr_turn(session, supplied_turn_id, result)
            return _ok(rid, result)

        active_turn_id = str(active.get("turn_id") or "")
        if active.get("modern") and not supplied_turn_id:
            return _ok(rid, {
                "ok": False,
                "reason": "turn_id_required",
                "turn_id": active_turn_id,
            })
        if supplied_turn_id and supplied_turn_id != active_turn_id:
            # Tombstone the unknown caller token without touching the actual
            # active turn.  This also closes the late-start race for that token.
            result = {
                "ok": True,
                "turn_id": supplied_turn_id,
                "transcript": "",
                "submitted": False,
                "disposition": disposition,
                "reason": "stale_turn",
                "active_turn_id": active_turn_id,
                "anchor_ts": None,
            }
            _retire_asr_turn(session, supplied_turn_id, result)
            return _ok(rid, result)
        owner_transport = active.get("owner_transport")
        if (request_transport is not None and owner_transport is not None
                and request_transport is not owner_transport):
            if session.get("transport") is request_transport:
                active["owner_transport"] = request_transport
            else:
                return _ok(rid, {
                    "ok": False,
                    "reason": "stale_transport",
                    "turn_id": active_turn_id,
                })

        turn = active
        with turn["lock"]:
            state = str(turn.get("state") or "")
            if state == "stopped" and isinstance(turn.get("result"), dict):
                return _ok(rid, dict(turn["result"]))
            wait_event = turn["stop_event"]
            if state == "stopping":
                # Cancellation is the dominant boundary operation until the
                # leader reaches its explicit commit point.  This covers a
                # profile/session switch racing a user-click finish: the
                # graceful Qwen flush may continue, but its callbacks and
                # buffered transcript can no longer become a main-agent turn.
                if (disposition == "cancel"
                        and not bool(turn.get("stop_committed", False))):
                    turn["disposition"] = "cancel"
                    turn["abort_reason"] = "cancelled"
                leader = False
            else:
                turn["state"] = "stopping"
                turn["disposition"] = disposition
                # Freeze ask-time before the potentially multi-second ASR flush.
                # ``None`` is meaningful (there was no frame yet), so the
                # downstream anchor_frozen flag must suppress latest-frame
                # fallback even in that case.
                requested_capture_id = str(
                    params.get("capture_attempt_id")
                    or params.get("capture_id") or "").strip()
                current_capture_id = str(
                    session.get("_mm_capture_attempt_id") or "").strip()
                capture_active = bool(
                    session.get("_mm_capture_active", False))
                capture_owner_matches = bool(
                    capture_active
                    and (
                        (requested_capture_id and current_capture_id
                         and requested_capture_id == current_capture_id)
                        # Legacy Web capture epoch: same authenticated session
                        # transport, but no attempt id in the stop payload.
                        or not requested_capture_id
                    )
                )
                if capture_owner_matches:
                    turn["anchor_ts"] = _resolve_mm_capture_anchor(
                        session,
                        capture_attempt_id=(
                            requested_capture_id or current_capture_id),
                        client_anchor_ts=params.get("anchor_ts"),
                    )
                    turn["anchor_capture_id"] = (
                        requested_capture_id or current_capture_id)
                else:
                    turn["anchor_ts"] = _snapshot_mm_frame_anchor(session)
                    turn["anchor_capture_id"] = current_capture_id
                leader = True

    if not leader:
        if wait_event is not None and wait_event.wait(timeout=7.0):
            with turn["lock"]:
                if isinstance(turn.get("result"), dict):
                    return _ok(rid, dict(turn["result"]))
        return _ok(rid, {
            "ok": False,
            "turn_id": str(turn.get("turn_id") or supplied_turn_id),
            "submitted": False,
            "reason": "stop_in_progress",
        })

    engine = session.get("_mm_live_watcher_agent")
    session["_mm_asr_on"] = False
    with turn["lock"]:
        audio_delivery_failed = bool(
            turn.get("mode") == "manual_turn"
            and turn.get("audio_delivery_failed", False)
        )
    close_result: dict = (
        {} if engine is not None else {
            "ok": False,
            "reason": "no_engine",
            "session_finished": False,
        }
    )
    if engine is not None:
        try:
            try:
                raw_close_result = engine.asr_stop(
                    str(turn.get("key") or ""),
                    graceful=(
                        disposition == "finish"
                        and not audio_delivery_failed
                    ),
                )
            except TypeError:
                raw_close_result = engine.asr_stop(
                    str(turn.get("key") or ""))
            if isinstance(raw_close_result, dict):
                close_result = dict(raw_close_result)
        except Exception as exc:
            logger.debug("asr_stop failed: %s", exc)
            close_result = {"ok": False, "reason": "finish_failed"}

    with turn["lock"]:
        disposition = str(
            turn.get("disposition") or disposition).strip().lower()
        forced_abort_reason = str(turn.get("abort_reason") or "").strip()
        audio_delivery_failed = bool(
            turn.get("mode") == "manual_turn"
            and turn.get("audio_delivery_failed", False)
        )
        segments = list(turn.get("segments") or [])
        pending_partial = str(turn.get("partial") or "").strip()
        incomplete_audio = bool(
            disposition == "finish"
            and turn.get("mode") == "manual_turn"
            and not audio_delivery_failed
            and pending_partial
        )
        canonical = str(close_result.get("transcript") or "").strip()
        if canonical:
            # Qwen session.finished is the only source that can authoritatively
            # refine spacing inside a completed item (cat -> cats). Prefer its
            # exact text over reconstructing character suffix callbacks.
            transcript = canonical
        else:
            transcript = _join_asr_segments(segments)
        anchor_ts = _optional_finite_float(turn.get("anchor_ts"))
        mode = str(turn.get("mode") or "continuous")

    # Clear preview state before materializing the one final user bubble.
    _emit("multimodal.asr_partial", _sid, {
        "text": "",
        "turn_id": str(turn.get("turn_id") or ""),
    })
    _emit("multimodal.asr_buffer", _sid, {
        "segments": [],
        "turn_id": str(turn.get("turn_id") or ""),
    })

    dispatch_result: dict = {"submitted": False}
    reason = ""
    if disposition == "cancel":
        transcript = ""
        reason = forced_abort_reason or "cancelled"
    elif mode == "manual_turn" and audio_delivery_failed:
        transcript = ""
        reason = "audio_delivery_failed"
    elif mode == "manual_turn" and incomplete_audio:
        # A visible live partial has not crossed a completed/session.finished
        # boundary.  Submitting the confirmed prefix would make an agent act on
        # a truncated command; submitting the partial would pretend an
        # unacknowledged provider guess is final.  Fail the whole manual turn.
        transcript = ""
        with turn["lock"]:
            disposition = str(
                turn.get("disposition") or disposition).strip().lower()
            forced_abort_reason = str(
                turn.get("abort_reason") or "").strip()
            if disposition != "cancel":
                turn["stop_committed"] = True
        reason = (
            forced_abort_reason or "cancelled"
            if disposition == "cancel"
            else ("finish_timeout" if close_result.get("timed_out")
                  else "incomplete_audio")
        )
    elif mode == "manual_turn":
        if transcript:
            dispatch_result = turn["submit_cb"](
                transcript,
                anchor_ts=anchor_ts,
                anchor_frozen=True,
            )
            if dispatch_result.get("cancelled"):
                with turn["lock"]:
                    disposition = str(
                        turn.get("disposition") or "cancel").strip().lower()
                    forced_abort_reason = str(
                        turn.get("abort_reason") or "cancelled").strip()
                transcript = ""
                reason = forced_abort_reason
            else:
                reason = str(dispatch_result.get("reason") or "")
        else:
            # There is no scheduler claim for silence.  Still finalize the stop
            # outcome under the turn lock so a pre-result cancel wins and a
            # later follower observes one immutable terminal result.
            with turn["lock"]:
                disposition = str(
                    turn.get("disposition") or disposition).strip().lower()
                forced_abort_reason = str(
                    turn.get("abort_reason") or "").strip()
                turn["stop_committed"] = True
            reason = (
                forced_abort_reason or "cancelled"
                if disposition == "cancel" else "empty"
            )
    else:
        # Continuous/VAD mode has already routed each completed segment through
        # VoiceAgent or the legacy foreground path.  Stop must not replay them.
        transcript = ""
        reason = "continuous_stopped"

    close_reason = str(close_result.get("reason") or "").strip()
    if (disposition == "finish" and not audio_delivery_failed
            and not incomplete_audio
            and close_result.get("timed_out")):
        reason = "finish_timeout"
    elif (disposition == "finish" and not audio_delivery_failed
          and not incomplete_audio
          and close_reason):
        # A transport/provider failure is more actionable than the derived
        # empty-transcript label.  If best-effort partial text was recovered it
        # is still submitted once, while this reason remains as a warning.
        reason = close_reason

    result_ok = True
    if disposition == "finish" and mode == "manual_turn":
        close_failed = bool(
            close_result and not bool(close_result.get("ok", True)))
        dispatch_failed = bool(transcript and not dispatch_result.get("submitted"))
        if (audio_delivery_failed or incomplete_audio or dispatch_failed
                or (close_failed and not transcript)):
            result_ok = False

    result = {
        "ok": result_ok,
        "turn_id": str(turn.get("turn_id") or ""),
        "mode": mode,
        "disposition": disposition,
        "transcript": transcript,
        "submitted": bool(dispatch_result.get("submitted", False)),
        "anchor_ts": anchor_ts,
        "capture_id": str(turn.get("anchor_capture_id") or ""),
    }
    for key in ("queued", "queue_position", "client_request_id"):
        if key in dispatch_result:
            result[key] = dispatch_result[key]
    if reason:
        result["reason"] = reason
    if close_result.get("error"):
        result["error"] = str(close_result.get("error"))[:300]
    if close_result:
        result["graceful"] = bool(close_result.get("session_finished", False))

    with turn["lock"]:
        turn["state"] = "stopped"
        turn["result"] = dict(result)
    with capture_lock:
        _retire_asr_turn(session, str(turn.get("turn_id") or ""), result)
    turn["stop_event"].set()
    if not modern_request and not turn.get("modern"):
        # Preserve the old compact response only for genuinely legacy callers.
        return _ok(rid, {"ok": True})
    return _ok(rid, result)


@method("multimodal.tts_speak")
def _(rid, params: dict) -> dict:
    """Speak an assistant reply on demand (the ▶ play button on text turns).

    params: {session_id, text}. Text turns skip the auto-speak path in the
    message.complete hook — the frontend fires this when the user clicks ▶.
    """
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    engine = session.get("_mm_live_watcher_agent")
    text = str(params.get("text") or "").strip()
    if engine is None or not text:
        return _ok(rid, {"ok": False,
                         "reason": "no_engine" if engine is None else "empty"})
    try:
        engine.submit_tts(text)
    except Exception as exc:
        return _err(rid, 5027, f"tts_speak failed: {exc}")
    return _ok(rid, {"ok": True})


def _get_voice_agent(session: dict):
    """Lazily build + cache the session's VoiceAgent (常驻语音交互 Agent)。

    ★ v2 已转正为唯一实现 (v1 VoiceAgent 已删)。绑定到 session 的 WatcherAgent (提供
      TTS 播放通道 + asyncio loop) 与 auxiliary.text 远端客户端。意图判断、
      self/main_agent 分诊、是否播报和口播拟词统一走这一个低延迟模型。v2 内部通过 is_interactive()
      / is_speaker_on() 决定实际是否生效: 即使 asr/tts 没开, 实例仍构建、hook 仍进来,
      只是静默不入回播队列。引擎未就绪时返回 None。"""
    ann = session.get("_mm_voice_agent")
    if ann is not None:
        return ann
    engine = session.get("_mm_live_watcher_agent")
    if engine is None:
        return None
    try:
        _cfg = _load_cfg() or {}
        voice_client, voice_model = None, ""
        try:
            # 统一走 auxiliary.text.remote_backend，用于意图判断和口播改写。
            from agent.auxiliary_client import get_async_text_auxiliary_client
            voice_client, voice_model = get_async_text_auxiliary_client(
                task="text")
            logger.info(
                "voice remote(text) model=%s (intent/route/speak/phrase fallback)",
                voice_model or "unavailable",
            )
        except Exception as _voice_exc:
            logger.debug("voice aux(text) client unavailable: %s", _voice_exc)
        # ★ merge: 磁盘真实文件是 voice_agent.py / class VoiceAgent (v2 已转正改名)。
        from agent.multimodal.voice_agent import VoiceAgent
        _sid = str(session.get("session_key") or session.get("_sid") or "unknown")

        # 主 Agent 提交回调: 用户作业进主 Agent 的入口. 用 threading 起
        # _run_prompt_submit. task_seq 记入 session 供主 Agent turn 结束后回调
        # va.notify_main_reply.
        def _submit_main(text: str, task_seq: int) -> None:
            rid2 = f"__voice_{int(time.time()*1000)}"
            # ★ 用 live runtime sid (前端 isMine 比对的那个), 不是 session_key(持久 id)。
            #   asr_start 时把它存进 session["_mm_live_sid"]。用错 sid → 前端过滤掉
            #   message.user_echo + message.complete → 用户气泡和 Assistant 答案都不显。
            _live_sid = str(session.get("_mm_live_sid") or _sid)
            lock = session.get("history_lock")
            if lock is None:
                return
            client_request_id = _ensure_client_request_id()
            active_asr_turn = session.get("_mm_asr_turn")
            active_turn_id = (
                str(active_asr_turn.get("turn_id") or "")
                if isinstance(active_asr_turn, dict) else ""
            )
            if (isinstance(active_asr_turn, dict)
                    and active_asr_turn.get("state") == "stopped"
                    and active_asr_turn.get("disposition") == "cancel"):
                return
            # VoiceAgent only calls this callback after routing the utterance to
            # the main agent.  Materialize the voice-tagged user bubble exactly
            # once, then carry its id through queued or immediate submit.
            #
            # ★ Emitted AFTER the busy gate below, never before it. Painting the
            #   bubble here unconditionally meant a queued utterance appeared in
            #   the transcript the moment it was recognized — while the previous
            #   turn was still running — so two quick utterances rendered as
            #   `user, user` with no assistant between them. History was always
            #   correct (the FIFO is claimed under history_lock, exactly like a
            #   typed prompt), but the visible transcript was not, and a
            #   user/user pair is the shape providers reject. The typed path
            #   shows a queue indicator instead of a bubble and only renders the
            #   message when its turn starts; voice now matches. The queued
            #   branch re-emits this event from _drain_queued_prompt.
            def _emit_voice_bubble() -> None:
                _emit("multimodal.asr_final", _live_sid, {
                    "text": text,
                    "request_id": client_request_id,
                    **({"turn_id": active_turn_id} if active_turn_id else {}),
                })
            with lock:
                # ★ `or queued_prompts` keeps speech in the order it was spoken.
                #   The turn tail releases ``running`` under the lock and only
                #   then re-acquires it to drain (see the finally/_drain pair at
                #   the end of _run_prompt_submit), so a fresh utterance landing
                #   in that window would find running=False and take the direct
                #   path — jumping ahead of an older utterance still sitting in
                #   the FIFO. Refusing the direct path whenever anything is
                #   queued makes "second one waits its turn" hold for the whole
                #   queue, not just the turn in flight.
                if session.get("running") or session.get("queued_prompts"):
                    # 主 Agent 忙 → 委派进统一 queued_prompts FIFO, 逐个执行 (不丢)。
                    position = _enqueue_prompt(
                        session,
                        text,
                        session.get("transport"),
                        user_originated=True,
                        origin="voice_agent",
                        metadata={
                            "voice_task_seq": task_seq,
                            "client_request_id": client_request_id,
                            # Carried so the drain can paint this utterance's
                            # bubble when its turn actually begins.
                            "voice_live_sid": _live_sid,
                            "voice_turn_id": active_turn_id,
                        },
                    )
                    _emit("multimodal.trajectory", _live_sid, {
                        "worker": "VoiceAgent", "phase": "main_agent_fifo",
                        "task_seq": task_seq, "queue_position": position,
                        "text": text,
                        "client_request_id": client_request_id,
                    })
                    _vtrace("route.main", sid=_live_sid, mode="fifo",
                            seq=task_seq, rid=client_request_id,
                            queue_pos=position, text=text)
                    return
                session["running"] = True
                session["_voice_active_seq"] = task_seq
            # Gate won → this turn starts now, so the bubble is in order.
            _emit_voice_bubble()
            # ★ 环节3 终点: 交给主 Agent 之前的最后一刻。text 从这里进 LLM。
            _vtrace("route.main", sid=_live_sid, mode="direct",
                    seq=task_seq, rid=client_request_id, text=text)
            def _turn():
                try:
                    # asr_final 已经渲染了唯一的语音用户气泡;
                    # 这里跳过 message.user_echo，并把同一 id 交给答案流。
                    _run_prompt_submit(
                        rid2,
                        _live_sid,
                        session,
                        text,
                        user_originated=True,
                        client_request_id=client_request_id,
                    )
                except Exception as exc:
                    logger.debug("voice submit_main err: %s", exc)
                    with lock:
                        session["running"] = False
                        if session.get("_voice_active_seq") == task_seq:
                            session.pop("_voice_active_seq", None)
                    _resolve_voice_task(
                        session, task_seq,
                        f"主 Agent 任务启动失败：{type(exc).__name__}: {exc}")
            threading.Thread(target=_turn, daemon=True,
                             name="voice-v2-main").start()

        def _is_busy() -> bool:
            return bool(session.get("running"))

        voice = VoiceAgent(
            engine=engine, session=session, sid=_sid,
            submit_main_agent_cb=_submit_main,
            is_session_busy=_is_busy,
            aux_client=voice_client, aux_model=voice_model or "",
            intent_client=voice_client, intent_model=voice_model or "",
            cfg=_cfg,
            emit_cb=lambda ev, payload: _emit(
                ev, str(session.get("_mm_live_sid") or _sid), payload),
        )
        voice.start()
        session["_mm_voice_agent"] = voice
        return voice
    except Exception as exc:
        logger.warning("voice agent build failed: %s", exc, exc_info=True)
        return None


@method("multimodal.tts_toggle")
def _(rid, params: dict) -> dict:
    """独立 TTS 语音播报开关 (与麦克风开关解耦)。

    params: {session_id, enabled}. 开启 → 主 Agent/monitor/watcher 气泡完成后自动经
    改写层口语化播报; 关闭 → 停止并清空播报子系统内存。
    """
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    request_transport = current_transport()
    session_transport = session.get("transport")
    if (request_transport is not None and session_transport is not None
            and request_transport is not session_transport):
        return _ok(rid, {
            "ok": False,
            "enabled": bool(session.get("_mm_tts_on", False)),
            "reason": "stale_transport",
        })
    on = bool(params.get("enabled"))
    session["_mm_tts_on"] = on
    # v2 无 set_enabled: 通过 is_speaker_on() 读 _mm_tts_on 自动判断是否播报。
    # 这里只需确保实例已懒建 (hook 进来时才有播报旁路)。
    _get_voice_agent(session)
    return _ok(rid, {"ok": True, "enabled": on})


@method("multimodal.tts_played")
def _(rid, params: dict) -> dict:
    """★ #2 播放 ack: 前端打断时回传"某条 TTS 实际播了多少"。

    params: {session_id, response_id, played_ms, total_ms}. VoiceAgent 据此把
    _self_recent 里对应那句按已播比例截断到字符, 让"我说过什么"对齐用户真听到的部分。
    仅对话模式的 voice_* rid 有意义; 其它 rid 静默忽略 (voice 内部�� rid 匹配, 匹配不到���改)。
    """
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    voice = session.get("_mm_voice_agent")
    resp_id = str(params.get("response_id") or "").strip()
    if voice is None or not resp_id:
        return _ok(rid, {"ok": False})
    try:
        played_ms = float(params.get("played_ms") or 0.0)
        total_ms = float(params.get("total_ms") or 0.0)
        voice.record_tts_played(resp_id, played_ms, total_ms)
    except Exception as exc:
        logger.debug("multimodal.tts_played failed: %s", exc)
        return _ok(rid, {"ok": False})
    return _ok(rid, {"ok": True})


@method("multimodal.stop_analysis")
def _(rid, params: dict) -> dict:
    """Ask a continuous deep-analysis (analysis/research) run to stop early.

    params: {session_id, request_id}. The WatcherAgent's continuous
    _run_delegation loop checks a per-request_id stop event each round; this sets
    it so the user / main agent can end the run before the video stops. The run
    still writes its final summary from whatever it has gathered so far.
    Returns {ok: bool} — ok=False when no active run matches the request_id.
    """
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    engine = session.get("_mm_live_watcher_agent")
    req = str(params.get("request_id") or "").strip()
    if engine is None or not req:
        return _ok(rid, {"ok": False,
                         "reason": "no_engine" if engine is None else "no_request_id"})
    try:
        ok = engine.stop_delegation(req)
    except Exception as exc:
        return _err(rid, 5027, f"stop_analysis failed: {exc}")
    return _ok(rid, {"ok": bool(ok)})


@method("multimodal.source_stopped")
def _(rid, params: dict) -> dict:
    """The video source (screen share / camera / video call) was closed.

    params: {session_id, started?: bool, source_type?: camera|screen}.
    `started=True` means a NEW source just began (clear the flag); otherwise the
    source stopped (set the flag). A
    continuous deep-analysis run uses this to decide whether to keep waiting for
    new frames — far more reliable than the old frame-idle heuristic that
    false-stopped on a static scene / lull. Best-effort; never errors the client.
    """
    started = bool(params.get("started", False))
    # Resolve without building first: the per-session capture lock below is the
    # linearization point for start/build/promote and stop. A stop that arrives
    # while activation is slow must run *after* that activation and tear down
    # what it created, rather than racing ahead and leaving resident jobs alive.
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    live_sid = str(params.get("session_id") or "").strip()
    generation_raw = params.get("capture_generation")
    capture_client_id = str(params.get("capture_client_id") or "").strip()
    capture_attempt_id = str(params.get("capture_attempt_id") or "").strip()
    client_started_raw = params.get("capture_client_started_at_ms")
    try:
        generation = int(generation_raw) if generation_raw is not None else None
    except (TypeError, ValueError):
        generation = None
    try:
        client_started_at = int(client_started_raw) if client_started_raw is not None else None
    except (TypeError, ValueError):
        client_started_at = None
    capture_lock = session.setdefault("_mm_capture_lock", threading.RLock())
    request_transport = current_transport()

    def _wrong_transport() -> bool:
        owner = session.get("transport")
        return bool(request_transport is not None and owner is not None
                    and owner is not request_transport)

    runtime_tokens: list = []
    runtime_home_token = None
    try:
        runtime_tokens = _set_session_context(
            str(session.get("session_key") or live_sid),
            str(session.get("cwd") or ""),
        )
        if profile_home := session.get("profile_home"):
            runtime_home_token = set_hermes_home_override(profile_home)
        with capture_lock:
            if started:
                if _wrong_transport():
                    return _ok(rid, {"ok": True, "stale": True})
                current_client_id = str(
                    session.get("_mm_capture_client_id") or "").strip()
                current_client_started_at = session.get(
                    "_mm_capture_client_started_at_ms")
                current_generation = session.get("_mm_capture_generation")
                current_attempt_id = str(
                    session.get("_mm_capture_attempt_id") or "").strip()
                capture_active = bool(session.get("_mm_capture_active", False))
                same_client = bool(
                    not capture_client_id or not current_client_id
                    or capture_client_id == current_client_id)
                stopped_owner_matches = bool(
                    current_generation is not None
                    and _capture_owner_is_retired(
                        session,
                        current_client_id,
                        int(current_generation),
                        capture_attempt_id,
                    )
                )
                if (not same_client and client_started_at is not None
                        and current_client_started_at is not None
                        and client_started_at < int(current_client_started_at)):
                    return _ok(rid, {"ok": True, "stale": True})
                if (same_client and generation is not None and current_generation is not None
                        and (generation < int(current_generation)
                             or (generation == int(current_generation)
                                 and stopped_owner_matches))):
                    return _ok(rid, {"ok": True, "stale": True})
                source_activation_changed = (
                    not capture_active
                    or not same_client
                    or (
                        generation is not None
                        and current_generation is not None
                        and generation != int(current_generation)
                    )
                )
                replacing_active_capture = bool(
                    capture_active and source_activation_changed)
                if source_activation_changed:
                    # Client-relative frame clocks restart with a new capture
                    # owner/source epoch.  Old pairs must never participate in
                    # a later manual-turn anchor conversion.
                    session.pop("_mm_capture_anchor_pairs", None)
                if generation is not None:
                    session["_mm_capture_generation"] = generation
                if capture_client_id:
                    session["_mm_capture_client_id"] = capture_client_id
                if capture_attempt_id:
                    session["_mm_capture_attempt_id"] = capture_attempt_id
                if client_started_at is not None:
                    session["_mm_capture_client_started_at_ms"] = client_started_at
                session["_mm_capture_active"] = True
                # Retired attempts remain monotonic. A reconnect may reuse this
                # generation with a fresh attempt id, but the explicitly
                # stopped predecessor must never be able to reclaim ownership.

                # Start is a readiness barrier: wait for the normal desktop
                # agent, then attach MM services without rebuilding its cached
                # system prompt. Holding the capture lock makes a concurrent
                # stop wait and deterministically tear these services down.
                _start_agent_build(live_sid, session)
                wait_error = _wait_agent(
                    session, rid, timeout=_MM_CAPTURE_ACTIVATION_TIMEOUT_SEC)
                if wait_error:
                    session["_mm_capture_active"] = False
                    return wait_error
                if _wrong_transport():
                    session["_mm_capture_active"] = False
                    return _ok(rid, {"ok": True, "stale": True})
                if (not _multimodal_runtime_ready(session)
                        and not _promote_session_to_multimodal(live_sid, session)):
                    session["_mm_capture_active"] = False
                    return _err(
                        rid, 5027,
                        "could not initialize the multimodal runtime for this session",
                    )
                if _wrong_transport():
                    session["_mm_capture_active"] = False
                    return _ok(rid, {"ok": True, "stale": True})

                if replacing_active_capture:
                    # A newer start can overtake the renderer's fire-and-forget
                    # stop RPC. Terminate jobs owned by the previous active
                    # generation here as part of the same capture transaction;
                    # the eventual older stop will then be safely stale.
                    _interrupt_running_mm_jobs(
                        session.get("session_key") or "", session)

                engine = session.get("_mm_live_watcher_agent")
                agent = session.get("agent")
                buf = getattr(agent, "frame_buffer", None) if agent else None
                source_type = str(
                    params.get("source_type") or params.get("source")
                    or params.get("sourceType") or params.get("kind")
                    or params.get("type") or "").strip().lower()
                if buf is not None and hasattr(buf, "set_source_type"):
                    buf.set_source_type(source_type)
                # Source lifecycle is shared by monitor and watcher. A
                # monitor-only session may legitimately have no WatcherEngine;
                # source tagging and raw-buffer reset must still happen.
                if engine is not None and source_activation_changed:
                    engine.mark_source_started()
            else:
                agent = session.get("agent")
                buf = getattr(agent, "frame_buffer", None) if agent else None
                current_client_id = str(
                    session.get("_mm_capture_client_id") or "").strip()
                current_generation = session.get("_mm_capture_generation")
                current_attempt_id = str(
                    session.get("_mm_capture_attempt_id") or "").strip()
                if (capture_client_id and current_client_id
                        and capture_client_id != current_client_id):
                    return _ok(rid, {"ok": True, "stale": True})
                if (capture_attempt_id and current_attempt_id
                        and capture_attempt_id != current_attempt_id):
                    return _ok(rid, {"ok": True, "stale": True})
                if (generation is not None and current_generation is not None
                        and generation < int(current_generation)):
                    return _ok(rid, {"ok": True, "stale": True})
                if generation is not None:
                    session["_mm_capture_generation"] = generation
                if capture_client_id:
                    session["_mm_capture_client_id"] = capture_client_id
                if capture_attempt_id:
                    session["_mm_capture_attempt_id"] = capture_attempt_id
                if client_started_at is not None:
                    session["_mm_capture_client_started_at_ms"] = client_started_at
                session["_mm_capture_active"] = False
                if generation is not None:
                    _retire_capture_owner(
                        session,
                        current_client_id or capture_client_id,
                        int(generation),
                        capture_attempt_id or current_attempt_id,
                    )
                if buf is not None and hasattr(buf, "set_source_type"):
                    buf.set_source_type("")
                # ★ Stream closed: terminate ALL in-flight monitors/watchers so
                #   they do not survive into the next stream (彻底终止不恢复). Keep
                #   teardown in the same capture transaction: otherwise a newer
                #   source-start can interleave here and the old stop would kill
                #   jobs belonging to the new generation.
                _interrupt_running_mm_jobs(
                    session.get("session_key") or "", session)
    except Exception as exc:
        return _err(rid, 5027, f"source_stopped failed: {exc}")
    finally:
        if runtime_home_token is not None:
            reset_hermes_home_override(runtime_home_token)
        _clear_session_context(runtime_tokens)
    return _ok(rid, {"ok": True, "capture_generation": generation})


@method("pdf.attach")
def _(rid, params: dict) -> dict:
    """Attach a PDF by rendering each page to PNG and queuing the pages.

    Anthropic's vision pipeline accepts images, not PDFs, so this runs
    ``pdftoppm`` (poppler-utils) at 150 DPI per page and queues each rendered
    page as an attached image. Accepts either a host ``path`` (local mode) or
    base64 ``content_base64`` (remote upload). Caps at 50 MB / 25 pages per call.

    Requires ``pdftoppm`` on $PATH (``apt install poppler-utils``); returns 5028
    if missing.
    """
    import shutil
    import subprocess
    import tempfile

    session, err = _sess(params, rid)
    if err:
        return err

    if shutil.which("pdftoppm") is None:
        return _err(rid, 5028, "pdftoppm not installed (poppler-utils package required)")

    raw_path = str(params.get("path", "") or "").strip()
    raw_b64 = str(params.get("content_base64") or params.get("data") or "").strip()
    if not raw_path and not raw_b64:
        return _err(rid, 4015, "path or content_base64 required")

    with tempfile.TemporaryDirectory(prefix="pdf_attach_") as td:
        td_path = Path(td)
        if raw_b64:
            pdf_bytes = _decode_attach_base64(raw_b64, mime_prefix="application/pdf")
            if pdf_bytes is None:
                return _err(rid, 4017, "data is not valid base64")
            if not pdf_bytes:
                return _err(rid, 4017, "decoded PDF is empty")
            if len(pdf_bytes) > _PDF_ATTACH_MAX_BYTES:
                mb = _PDF_ATTACH_MAX_BYTES // (1024 * 1024)
                return _err(rid, 4018, f"PDF too large ({len(pdf_bytes)} bytes; cap is {mb} MB)")
            if pdf_bytes[:5] != b"%PDF-":
                return _err(rid, 4017, "payload is not a PDF (missing %PDF- magic bytes)")
            pdf_path = td_path / "input.pdf"
            pdf_path.write_bytes(pdf_bytes)
            display_name = str(params.get("filename", "") or "uploaded.pdf")
        else:
            try:
                from cli import _resolve_attachment_path

                resolved = _resolve_attachment_path(raw_path)
            except Exception:
                resolved = None
            if resolved is None or not Path(resolved).is_file():
                return _err(rid, 4016, f"PDF not found: {raw_path}")
            if Path(resolved).suffix.lower() != ".pdf":
                return _err(rid, 4016, f"not a PDF: {Path(resolved).name}")
            if Path(resolved).stat().st_size > _PDF_ATTACH_MAX_BYTES:
                mb = _PDF_ATTACH_MAX_BYTES // (1024 * 1024)
                return _err(rid, 4018, f"PDF too large; cap is {mb} MB")
            pdf_path = Path(resolved)
            display_name = pdf_path.name

        try:
            first_page = int(params.get("first_page") or 1)
            last_page_param = params.get("last_page")
            last_page = int(last_page_param) if last_page_param is not None else None
        except (TypeError, ValueError):
            return _err(rid, 4015, "first_page/last_page must be integers")

        if first_page < 1:
            return _err(rid, 4015, "first_page must be >= 1")
        if last_page is None:
            last_page = first_page + _PDF_ATTACH_MAX_PAGES - 1
        if last_page < first_page:
            return _err(rid, 4015, "last_page must be >= first_page")
        if last_page - first_page + 1 > _PDF_ATTACH_MAX_PAGES:
            return _err(rid, 4019, f"page range exceeds cap of {_PDF_ATTACH_MAX_PAGES} pages per attach call")

        out_prefix = td_path / "page"
        argv = [
            "pdftoppm", "-png", "-r", "150",
            "-f", str(first_page), "-l", str(last_page),
            str(pdf_path), str(out_prefix),
        ]
        try:
            res = subprocess.run(argv, capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return _err(rid, 5028, "pdftoppm timed out (>120s)")
        if res.returncode != 0:
            tail = (res.stderr or res.stdout or "").strip().splitlines()[-3:]
            return _err(rid, 5028, "pdftoppm failed: " + " | ".join(tail))

        rendered = sorted(td_path.glob("page-*.png"))
        if not rendered:
            return _err(rid, 5028, "pdftoppm produced no pages (corrupt PDF?)")

        attached_pages = []
        for src in rendered:
            page_num = src.stem.split("-", 1)[-1]
            try:
                page_int = int(page_num)
            except ValueError:
                page_int = first_page + len(attached_pages)
            dst = _queue_attached_image(session, src.read_bytes(), ".png", prefix=f"pdf_p{page_num}")
            attached_pages.append({"path": str(dst), "page": page_int, **_image_meta(dst)})

        return _ok(
            rid,
            {
                "attached": True,
                "filename": display_name,
                "pages_attached": len(attached_pages),
                "pages": attached_pages,
                "count": len(session["attached_images"]),
                "text": f"[User attached PDF: {display_name} ({len(attached_pages)} page(s))]",
            },
        )


_ATTACHMENT_REF_NEEDS_QUOTING_RE = None


def _format_ref_value(value: str) -> str:
    """Quote a context-ref value when it contains whitespace or bracket chars.

    Mirrors the desktop ``formatRefValue`` so the staged ``@file:`` ref round-trips
    through ``agent.context_references`` cleanly.
    """
    import re as _re

    global _ATTACHMENT_REF_NEEDS_QUOTING_RE
    if _ATTACHMENT_REF_NEEDS_QUOTING_RE is None:
        _ATTACHMENT_REF_NEEDS_QUOTING_RE = _re.compile(r"""[\s()\[\]{}<>"'`]""")
    if not value or not _ATTACHMENT_REF_NEEDS_QUOTING_RE.search(value):
        return value
    if "`" not in value:
        return f"`{value}`"
    if '"' not in value:
        return f'"{value}"'
    if "'" not in value:
        return f"'{value}'"
    return value


def _attachment_ref_path(session: dict, target: Path) -> str:
    """Workspace-relative path for an attachment, or the absolute path if outside."""
    workspace = Path(_session_cwd(session)).resolve()
    try:
        rel = target.resolve().relative_to(workspace)
        return str(rel).replace(os.sep, "/")
    except ValueError:
        return str(target.resolve())


def _desktop_attachment_dir(session: dict) -> Path:
    root = Path(_session_cwd(session)).resolve() / ".argus" / "desktop-attachments"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _sanitize_attachment_name(name: str) -> str:
    import re as _re

    candidate = Path(str(name or "").strip()).name
    candidate = _re.sub(r"[\x00-\x1f]+", "_", candidate)
    candidate = candidate.strip().strip(".")
    return candidate or "attachment"


def _unique_attachment_path(root: Path, filename: str) -> Path:
    candidate = root / filename
    if not candidate.exists():
        return candidate
    stem = Path(filename).stem or "attachment"
    suffix = Path(filename).suffix
    counter = 2
    while True:
        next_candidate = root / f"{stem}-{counter}{suffix}"
        if not next_candidate.exists():
            return next_candidate
        counter += 1


def _resolve_gateway_attachment_path(raw: str) -> Path | None:
    """Resolve a raw path token to a gateway-visible file, or None."""
    if not raw:
        return None
    try:
        from cli import _detect_file_drop, _resolve_attachment_path, _split_path_input
    except Exception:
        return None

    dropped = _detect_file_drop(raw)
    if dropped:
        return Path(dropped["path"]).resolve()
    path_token, _remainder = _split_path_input(raw)
    resolved = _resolve_attachment_path(path_token)
    return Path(resolved).resolve() if resolved is not None else None


def _decode_attachment_data_url(data_url: str) -> bytes:
    """Decode a ``data:<any-mime>;base64,<b64>`` payload to bytes.

    Unlike ``_decode_attach_base64`` (image-mime-specific), this accepts any
    media type — text/csv, application/pdf, etc. — so non-image file uploads
    round-trip. Also tolerates a bare base64 string with no data-URL prefix.
    """
    import base64 as _base64
    import binascii as _binascii
    import re as _re

    cleaned = (data_url or "").strip()
    m = _re.match(r"^data:[^;,]*(?:;[^;,=]+=[^;,]+)*;base64,(.*)$", cleaned, _re.DOTALL | _re.I)
    if m:
        cleaned = m.group(1)
    cleaned = _re.sub(r"\s+", "", cleaned)
    try:
        return _base64.b64decode(cleaned, validate=True)
    except (ValueError, _binascii.Error) as exc:
        raise ValueError("invalid data_url payload") from exc


def _stage_session_file_attachment(
    session: dict,
    *,
    raw_path: str,
    data_url: str,
    name: str,
) -> tuple[Path, bool]:
    """Make a desktop file attachment available to the remote gateway agent.

    Three cases:
      1. The path resolves to a file already INSIDE the session workspace — use
         it as-is (no copy, ``uploaded=False``).
      2. The path resolves to a gateway-visible file OUTSIDE the workspace — copy
         it into ``.argus/desktop-attachments/`` so the ``@file:`` ref resolves.
      3. The path doesn't exist on the gateway (the common remote case: it's a
         path on the CLIENT's disk) — decode the uploaded ``data_url`` bytes and
         write them into ``.argus/desktop-attachments/``.

    Returns ``(stored_path, uploaded)``.
    """
    workspace = Path(_session_cwd(session)).resolve()
    resolved = _resolve_gateway_attachment_path(raw_path)
    if resolved is not None:
        try:
            resolved.relative_to(workspace)
            return resolved, False
        except ValueError:
            payload = resolved.read_bytes()
            filename = resolved.name
    else:
        if not data_url:
            raise ValueError("file not found on gateway and no data_url provided")
        payload = _decode_attachment_data_url(data_url)
        filename = _sanitize_attachment_name(name or Path(str(raw_path or "")).name)

    upload_dir = _desktop_attachment_dir(session)
    target = _unique_attachment_path(upload_dir, _sanitize_attachment_name(filename))
    target.write_bytes(payload)
    return target.resolve(), True


@method("file.attach")
def _(rid, params: dict) -> dict:
    """Stage a non-image file attachment into the session workspace.

    The image/PDF path renders to vision tiles; this one keeps the file as a
    readable artifact and returns a workspace-relative ``@file:`` ref so the
    agent's file tools (and ``agent.context_references``) can read it. Solves the
    remote-gateway case where the desktop passes a path that only exists on the
    CLIENT's disk: the client uploads ``data_url`` bytes and we materialize the
    file on the gateway.

    Params:
      session_id (str, required)
      path (str): client/host path of the file (used for naming + local-mode
        gateway-visible resolution).
      data_url (str): ``data:<mime>;base64,<b64>`` upload of the file bytes,
        required when the path isn't visible to the gateway.
      name (str, optional): preferred filename.
    """
    session, err = _sess(params, rid)
    if err:
        return err
    raw = str(params.get("path", "") or "").strip()
    data_url = str(params.get("data_url", "") or "").strip()
    name = str(params.get("name", "") or "").strip()
    if not raw and not data_url:
        return _err(rid, 4015, "path or data_url required")
    try:
        stored_path, uploaded = _stage_session_file_attachment(
            session, raw_path=raw, data_url=data_url, name=name
        )
        ref_path = _attachment_ref_path(session, stored_path)
        return _ok(
            rid,
            {
                "attached": True,
                "name": stored_path.name,
                "path": str(stored_path),
                "ref_path": ref_path,
                "ref_text": f"@file:{_format_ref_value(ref_path)}",
                "uploaded": uploaded,
            },
        )
    except Exception as e:
        return _err(rid, 5028, str(e))


@method("image.detach")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    raw = str(params.get("path", "") or "").strip()
    if not raw:
        return _err(rid, 4015, "path required")
    images = session.setdefault("attached_images", [])
    before = len(images)
    session["attached_images"] = [path for path in images if path != raw]
    return _ok(
        rid,
        {
            "detached": len(session["attached_images"]) != before,
            "count": len(session["attached_images"]),
        },
    )


@method("input.detect_drop")
def _(rid, params: dict) -> dict:
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    try:
        from cli import _detect_file_drop

        raw = str(params.get("text", "") or "")
        dropped = _detect_file_drop(raw)
        if not dropped:
            return _ok(rid, {"matched": False})

        drop_path = dropped["path"]
        remainder = dropped["remainder"]
        if dropped["is_image"]:
            session.setdefault("attached_images", []).append(str(drop_path))
            text = remainder or f"[User attached image: {drop_path.name}]"
            return _ok(
                rid,
                {
                    "matched": True,
                    "is_image": True,
                    "path": str(drop_path),
                    "count": len(session["attached_images"]),
                    "text": text,
                    **_image_meta(drop_path),
                },
            )

        text = f"[User attached file: {drop_path}]" + (
            f"\n{remainder}" if remainder else ""
        )
        return _ok(
            rid,
            {
                "matched": True,
                "is_image": False,
                "path": str(drop_path),
                "name": drop_path.name,
                "text": text,
            },
        )
    except Exception as e:
        return _err(rid, 5027, str(e))


@method("prompt.background")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    text, parent = params.get("text", ""), params.get("session_id", "")
    if not text:
        return _err(rid, 4012, "text required")
    task_id = f"bg_{uuid.uuid4().hex[:6]}"

    def run():
        session_tokens = _set_session_context(task_id, cwd=_session_cwd(session))
        try:
            from run_agent import AIAgent

            result = AIAgent(
                **_background_agent_kwargs(session["agent"], task_id)
            ).run_conversation(
                user_message=text,
                task_id=task_id,
            )
            _emit(
                "background.complete",
                parent,
                {
                    "task_id": task_id,
                    "text": (
                        result.get("final_response", str(result))
                        if isinstance(result, dict)
                        else str(result)
                    ),
                },
            )
        except Exception as e:
            _emit(
                "background.complete",
                parent,
                {"task_id": task_id, "text": f"error: {e}"},
            )
        finally:
            _clear_session_context(session_tokens)

    threading.Thread(target=run, daemon=True).start()
    return _ok(rid, {"task_id": task_id})


@method("preview.restart")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err

    url = str(params.get("url") or "").strip()
    cwd = str(params.get("cwd") or "").strip()
    context = str(params.get("context") or "").strip()

    if not url:
        return _err(rid, 4012, "url required")

    task_id = f"preview_{uuid.uuid4().hex[:6]}"
    parent = params.get("session_id", "")
    parent_history = _preview_restart_history(session)
    has_history = bool(parent_history)
    prompt = "\n".join(
        line
        for line in [
            "The desktop preview pane cannot load a local server URL.",
            "",
            f"Preview URL: {url}",
            f"Current working directory: {cwd or '(unknown)'}",
            "",
            f"Preview console:\n{context}" if context else "",
            "" if context else "",
            (
                "The conversation history above is from the user's main session — including the commands you (the assistant) previously ran to start servers, edit files, or check ports. Use it to figure out exactly which server should be running at this Preview URL. The user did not start a brand new task; recover what they had working."
                if has_history
                else None
            ),
            "Restart exactly the app intended for the Preview URL, not Argus Desktop itself.",
            "The Preview URL and port are the target. Preserve that target unless you conclude it is impossible.",
            "If the prior conversation shows a specific command that bound this URL/port, prefer re-running THAT exact command (in the same cwd) over guessing a new one.",
            "First inspect what process, if any, owns the Preview URL port. If a stale server exists, inspect its cwd and prefer that cwd over the Hermes/Desktop process cwd.",
            "The Current working directory is only a hint. Do not assume it is the preview app root when the port owner or files indicate another root.",
            "If the console shows a module-script MIME error for src/main.tsx or similar, a static server is serving source files. Do not restart python -m http.server or any dumb static server for that app.",
            "For module-script MIME failures, inspect package.json/vite config in the candidate app root and start the real dev server/bundler (for example npm/pnpm/yarn dev) so module transforms happen.",
            "Before declaring success, verify the Preview URL responds with the intended app, not Argus Desktop. If it serves Hermes/Desktop UI or another unrelated app, stop that process and report failure.",
            "Do not modify files. Do not ask the user unless blocked.",
            "Prefer existing project scripts or commands when they are clear.",
            "If a stale process owns the needed port, handle it safely.",
            "Start long-running servers detached/in the background, then return immediately.",
            "Do not run a foreground dev server command that blocks this background task.",
            "Keep the final response short: what command/server was started, or why it could not be restarted.",
        ]
        if line
    )

    # Normalize defensively: a malformed client path (embedded NUL, etc.) must
    # not blow up the whole restart — treat it as "no validated cwd".
    try:
        preview_cwd = os.path.abspath(os.path.expanduser(cwd)) if cwd else ""
        if preview_cwd and not os.path.isdir(preview_cwd):
            preview_cwd = ""
    except Exception:
        preview_cwd = ""

    def run():
        # Pin the validated preview cwd, else the parent workspace — never an
        # invalid client path, which would silently fall back to the launch dir.
        session_tokens = _set_session_context(task_id, cwd=(preview_cwd or _session_cwd(session)))
        try:
            from run_agent import AIAgent
            from tools.terminal_tool import register_task_env_overrides

            if preview_cwd:
                register_task_env_overrides(task_id, {"cwd": preview_cwd})

            history_note = (
                f" (with {len(parent_history)} parent-session messages of context)"
                if parent_history
                else ""
            )
            _emit(
                "preview.restart.progress",
                parent,
                {"task_id": task_id, "text": f"Starting hidden restart agent{history_note}"},
            )
            result = AIAgent(
                **_ephemeral_preview_agent_kwargs(session["agent"], task_id),
                **_preview_restart_callbacks(parent, task_id),
            ).run_conversation(
                user_message=prompt,
                task_id=task_id,
                conversation_history=parent_history or None,
            )
            text = (
                result.get("final_response", str(result))
                if isinstance(result, dict)
                else str(result)
            )
            _emit("preview.restart.complete", parent, {"task_id": task_id, "text": text})
        except Exception as e:
            _emit(
                "preview.restart.complete",
                parent,
                {"task_id": task_id, "text": f"error: {e}"},
            )
        finally:
            try:
                from tools.terminal_tool import clear_task_env_overrides

                clear_task_env_overrides(task_id)
            except Exception:
                pass
            _clear_session_context(session_tokens)

    threading.Thread(target=run, daemon=True).start()
    return _ok(rid, {"task_id": task_id})


# ── Methods: respond ─────────────────────────────────────────────────


def _respond(rid, params, key):
    r = params.get("request_id", "")
    with _prompt_lock:
        entry = _pending.get(r)
        if not entry:
            return _err(rid, 4009, f"no pending {key} request")
        _, ev = entry
        _answers[r] = params.get(key, "")
        ev.set()
    return _ok(rid, {"status": "ok"})


@method("clarify.respond")
def _(rid, params: dict) -> dict:
    return _respond(rid, params, "answer")


@method("terminal.read.respond")
def _(rid, params: dict) -> dict:
    # `text` is a JSON string of the serialized terminal buffer + line metadata.
    return _respond(rid, params, "text")


@method("sudo.respond")
def _(rid, params: dict) -> dict:
    return _respond(rid, params, "password")


@method("secret.respond")
def _(rid, params: dict) -> dict:
    return _respond(rid, params, "value")


@method("approval.respond")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    try:
        from tools.approval import resolve_gateway_approval

        return _ok(
            rid,
            {
                "resolved": resolve_gateway_approval(
                    session["session_key"],
                    params.get("choice", "deny"),
                    resolve_all=params.get("all", False),
                )
            },
        )
    except Exception as e:
        return _err(rid, 5004, str(e))


# ── Methods: config ──────────────────────────────────────────────────


@method("config.set")
def _(rid, params: dict) -> dict:
    key, value = params.get("key", ""), params.get("value", "")
    session = _sessions.get(params.get("session_id", ""))

    if key == "model":
        try:
            if not value:
                return _err(rid, 4002, "model value required")
            if session:
                # Reject during an in-flight turn.  agent.switch_model()
                # mutates self.model / self.provider / self.base_url /
                # self.client in place; the worker thread running
                # agent.run_conversation is reading those on every
                # iteration.  A mid-turn swap can send an HTTP request
                # with the new base_url but old model (or vice versa),
                # producing 400/404s the user never asked for.  Parity
                # with the gateway's running-agent /model guard.
                if session.get("running"):
                    return _err(
                        rid,
                        4009,
                        "session busy — /interrupt the current turn before switching models",
                    )
                from hermes_cli.model_switch import parse_model_flags

                parsed_flags = parse_model_flags(value)
                _model_input, explicit_provider, _persist_global, _force_refresh, _is_session = parsed_flags
                if session.get("agent") is None and not explicit_provider.strip():
                    session_id = params.get("session_id", "")
                    _start_agent_build(session_id, session)
                    init_err = _wait_agent(session, rid)
                    if init_err:
                        return init_err
                    if session.get("agent") is None:
                        return _err(rid, 5032, "agent initialization failed")
                result = _apply_model_switch(
                    params.get("session_id", ""),
                    session,
                    value,
                    confirm_expensive_model=bool(
                        params.get("confirm_expensive_model", False)
                    ),
                    parsed_flags=parsed_flags,
                )
            else:
                result = _apply_model_switch(
                    "",
                    {"agent": None},
                    value,
                    confirm_expensive_model=bool(
                        params.get("confirm_expensive_model", False)
                    ),
                )
            return _ok(
                rid,
                {
                    "key": key,
                    "value": result["value"],
                    "warning": result["warning"],
                    "confirm_required": result.get("confirm_required", False),
                    "confirm_message": result.get("confirm_message", ""),
                },
            )
        except Exception as e:
            return _err(rid, 5001, str(e))

    if key == "fast":
        raw = str(value or "").strip().lower()
        agent = session.get("agent") if session else None
        if agent is not None:
            current_fast = getattr(agent, "service_tier", None) == "priority"
        else:
            current_fast = _load_service_tier() == "priority"

        if raw in {"status"}:
            return _ok(
                rid,
                {"key": key, "value": "fast" if current_fast else "normal"},
            )

        if raw in {"", "toggle"}:
            nv = "normal" if current_fast else "fast"
        elif raw in {"fast", "on"}:
            nv = "fast"
        elif raw in {"normal", "off"}:
            nv = "normal"
        else:
            return _err(rid, 4002, f"unknown fast mode: {value}")

        overrides = None
        if nv == "fast":
            from hermes_cli.models import resolve_fast_mode_overrides

            target_model = (
                getattr(agent, "model", None) if agent is not None else _resolve_model()
            )
            if not target_model:
                return _err(
                    rid,
                    4002,
                    "fast mode is not available without a selected model",
                )
            overrides = resolve_fast_mode_overrides(target_model)
            if overrides is None:
                return _err(
                    rid,
                    4002,
                    "fast mode is not available for this model",
                )

        _write_config_key("agent.service_tier", nv)
        if agent is not None:
            agent.service_tier = "priority" if nv == "fast" else None
            current_overrides = dict(getattr(agent, "request_overrides", {}) or {})
            current_overrides.pop("service_tier", None)
            current_overrides.pop("speed", None)
            if nv == "fast":
                current_overrides.update(overrides)
            agent.request_overrides = current_overrides
            _persist_live_session_runtime(session)
            _emit(
                "session.info",
                params.get("session_id", ""),
                _session_info(agent, session),
            )
        return _ok(rid, {"key": key, "value": nv})

    if key == "busy":
        raw = str(value or "").strip().lower()
        if raw in {"", "status"}:
            return _ok(rid, {"key": key, "value": _load_busy_input_mode()})
        if raw not in {"queue", "steer", "interrupt"}:
            return _err(rid, 4002, f"unknown busy mode: {value}")
        _write_config_key("display.busy_input_mode", raw)
        return _ok(rid, {"key": key, "value": raw})

    if key == "verbose":
        cycle = ["off", "new", "all", "verbose"]
        cur = (
            session.get("tool_progress_mode", _load_tool_progress_mode())
            if session
            else _load_tool_progress_mode()
        )
        if value and value != "cycle":
            nv = str(value).strip().lower()
            if nv not in cycle:
                return _err(rid, 4002, f"unknown verbose mode: {value}")
        else:
            try:
                idx = cycle.index(cur)
            except ValueError:
                idx = 2
            nv = cycle[(idx + 1) % len(cycle)]
        _write_config_key("display.tool_progress", nv)
        if session:
            session["tool_progress_mode"] = nv
            agent = session.get("agent")
            if agent is not None:
                agent.verbose_logging = nv == "verbose"
        return _ok(rid, {"key": key, "value": nv})

    if key == "yolo":
        # Approval bypass. Two scopes:
        #   scope="session" (default) — same as the TUI's Shift+Tab. Toggles
        #     ONLY this session's _session_yolo flag; never touches global
        #     config, so CLI / TUI / cron behavior is unaffected.
        #   scope="global" (Shift+click the zap) — flips the persistent global
        #     approvals.mode in config.yaml between "off" (bypass on) and
        #     "manual" (bypass off). This DOES affect every session, the CLI,
        #     the TUI, and cron, and survives restarts.
        scope = str(params.get("scope") or "session").strip().lower()
        try:
            from tools.approval import (
                disable_session_yolo,
                enable_session_yolo,
                is_session_yolo_enabled,
            )

            raw = str(value or "").strip().lower()

            def _resolve_toggle(current: bool) -> bool:
                if raw in {"1", "on", "true", "yes"}:
                    return True
                if raw in {"0", "off", "false", "no"}:
                    return False
                return not current

            if scope == "global":
                from tools.approval import _normalize_approval_mode

                cfg = _load_cfg()
                appr = cfg.get("approvals") if isinstance(cfg, dict) else None
                if not isinstance(appr, dict):
                    appr = {}
                current = _normalize_approval_mode(appr.get("mode", "manual")) == "off"
                enable = _resolve_toggle(current)
                # Toggle between full bypass and the default manual gate. We do
                # not try to restore a prior "smart"/custom mode — the zap is a
                # binary on/off affordance; users with bespoke modes set them in
                # config.yaml.
                _write_config_key("approvals.mode", "off" if enable else "manual")
                nv = "1" if enable else "0"
                # Reflect the global flip in every live session's indicator.
                for sid, sess in list(_sessions.items()):
                    agent = sess.get("agent")
                    if agent is not None:
                        _emit("session.info", sid, _session_info(agent, sess))
                return _ok(rid, {"key": key, "value": nv, "scope": "global"})

            if session:
                current = is_session_yolo_enabled(session["session_key"])
                enable = _resolve_toggle(current)
                if enable:
                    enable_session_yolo(session["session_key"])
                    nv = "1"
                else:
                    disable_session_yolo(session["session_key"])
                    nv = "0"
                agent = session.get("agent")
                if agent is not None:
                    _emit(
                        "session.info",
                        params.get("session_id", ""),
                        _session_info(agent, session),
                    )
            else:
                current = is_truthy_value(os.environ.get("ARGUS_YOLO_MODE"))
                enable = _resolve_toggle(current)
                if enable:
                    os.environ["ARGUS_YOLO_MODE"] = "1"
                    nv = "1"
                else:
                    os.environ.pop("ARGUS_YOLO_MODE", None)
                    nv = "0"
            return _ok(rid, {"key": key, "value": nv, "scope": "session"})
        except Exception as e:
            return _err(rid, 5001, str(e))

    if key == "reasoning":
        try:
            from hermes_constants import parse_reasoning_effort

            arg = str(value or "").strip().lower()
            if arg in {"show", "on"}:
                cfg = _load_cfg()
                display = (
                    cfg.get("display") if isinstance(cfg.get("display"), dict) else {}
                )
                sections = (
                    display.get("sections")
                    if isinstance(display.get("sections"), dict)
                    else {}
                )
                display["show_reasoning"] = True
                sections["thinking"] = "expanded"
                display["sections"] = sections
                cfg["display"] = display
                _save_cfg(cfg)
                if session:
                    session["show_reasoning"] = True
                return _ok(rid, {"key": key, "value": "show"})
            if arg in {"hide", "off"}:
                cfg = _load_cfg()
                display = (
                    cfg.get("display") if isinstance(cfg.get("display"), dict) else {}
                )
                sections = (
                    display.get("sections")
                    if isinstance(display.get("sections"), dict)
                    else {}
                )
                display["show_reasoning"] = False
                sections["thinking"] = "hidden"
                display["sections"] = sections
                cfg["display"] = display
                _save_cfg(cfg)
                if session:
                    session["show_reasoning"] = False
                return _ok(rid, {"key": key, "value": "hide"})

            # /reasoning full | clamp — parity with the classic CLI's
            # reasoning_full toggle. The TUI renders thinking as an
            # expand/collapse section rather than a fixed 10-line recap, so
            # full maps to sections.thinking=expanded and clamp to collapsed.
            # display.reasoning_full is persisted too so the config key stays
            # consistent across the CLI and TUI surfaces.
            if arg in {"full", "all"}:
                cfg = _load_cfg()
                display = (
                    cfg.get("display") if isinstance(cfg.get("display"), dict) else {}
                )
                sections = (
                    display.get("sections")
                    if isinstance(display.get("sections"), dict)
                    else {}
                )
                display["reasoning_full"] = True
                sections["thinking"] = "expanded"
                display["sections"] = sections
                cfg["display"] = display
                _save_cfg(cfg)
                return _ok(rid, {"key": key, "value": "full"})
            if arg in {"clamp", "collapse", "short"}:
                cfg = _load_cfg()
                display = (
                    cfg.get("display") if isinstance(cfg.get("display"), dict) else {}
                )
                sections = (
                    display.get("sections")
                    if isinstance(display.get("sections"), dict)
                    else {}
                )
                display["reasoning_full"] = False
                sections["thinking"] = "collapsed"
                display["sections"] = sections
                cfg["display"] = display
                _save_cfg(cfg)
                return _ok(rid, {"key": key, "value": "clamp"})

            parsed = parse_reasoning_effort(arg)
            if parsed is None:
                return _err(rid, 4002, f"unknown reasoning value: {value}")
            # ── scope ────────────────────────────────────────────────────
            # "session": thinking effort is THIS session's own runtime
            #   preference — set it on the live agent (read fresh on every API
            #   call, so it lands on the next turn) and persist it into the
            #   session row's model_config, which _stored_session_runtime_
            #   overrides restores on resume. config.yaml is NOT touched: it is
            #   the git-tracked project baseline that sync_project_config()
            #   copies over HERMES_HOME on every start, so a live value stored
            #   there is erased at the next restart. New sessions still start
            #   from that baseline via _load_reasoning_config().
            # "global" (default): legacy /reasoning behaviour — write the
            #   config key so the choice becomes the baseline for new sessions.
            scope = str(params.get("scope") or "global").strip().lower()
            if scope == "session":
                if not session:
                    return _err(rid, 4001, "session not found")
                agent = session.get("agent")
                if agent is not None:
                    agent.reasoning_config = parsed
                    _persist_live_session_runtime(session)
                    _emit(
                        "session.info",
                        params.get("session_id", ""),
                        _session_info(agent, session),
                    )
                else:
                    # Agent not built yet (lazy build defers to the first
                    # prompt). Stash it where the deferred build already looks
                    # for a per-session reasoning override, so the first turn
                    # is built with this effort instead of the config default.
                    session["create_reasoning_override"] = parsed
                return _ok(rid, {"key": key, "value": arg, "scope": "session"})
            _write_config_key("agent.reasoning_effort", arg)
            if session and session.get("agent") is not None:
                session["agent"].reasoning_config = parsed
                _persist_live_session_runtime(session)
                _emit(
                    "session.info",
                    params.get("session_id", ""),
                    _session_info(session["agent"], session),
                )
            return _ok(rid, {"key": key, "value": arg, "scope": "global"})
        except Exception as e:
            return _err(rid, 5001, str(e))

    if key == "details_mode":
        nv = str(value or "").strip().lower()
        if nv not in _DETAIL_MODES:
            return _err(rid, 4002, f"unknown details_mode: {value}")
        cfg = _load_cfg()
        display = cfg.get("display") if isinstance(cfg.get("display"), dict) else {}
        sections = (
            display.get("sections") if isinstance(display.get("sections"), dict) else {}
        )
        display["details_mode"] = nv
        for section in _DETAIL_SECTION_NAMES:
            sections[section] = nv
        display["sections"] = sections
        cfg["display"] = display
        _save_cfg(cfg)
        return _ok(rid, {"key": key, "value": nv})

    if key.startswith("details_mode."):
        # Per-section override: `details_mode.<section>` writes to
        # `display.sections.<section>`. Empty value clears the explicit
        # override and lets frontend resolution apply built-in section defaults
        # before the global details_mode.
        section = key.split(".", 1)[1]
        if section not in _DETAIL_SECTION_NAMES:
            return _err(rid, 4002, f"unknown section: {section}")

        cfg = _load_cfg()
        display = cfg.get("display") if isinstance(cfg.get("display"), dict) else {}
        sections_cfg = (
            display.get("sections") if isinstance(display.get("sections"), dict) else {}
        )

        nv = str(value or "").strip().lower()
        if not nv:
            sections_cfg.pop(section, None)
            display["sections"] = sections_cfg
            cfg["display"] = display
            _save_cfg(cfg)
            return _ok(rid, {"key": key, "value": ""})

        if nv not in _DETAIL_MODES:
            return _err(rid, 4002, f"unknown details_mode: {value}")

        sections_cfg[section] = nv
        display["sections"] = sections_cfg
        cfg["display"] = display
        _save_cfg(cfg)
        return _ok(rid, {"key": key, "value": nv})

    if key == "thinking_mode":
        nv = str(value or "").strip().lower()
        allowed_tm = frozenset({"collapsed", "truncated", "full"})
        if nv not in allowed_tm:
            return _err(rid, 4002, f"unknown thinking_mode: {value}")
        _write_config_key("display.thinking_mode", nv)
        # Backward compatibility bridge: keep details_mode aligned.
        _write_config_key(
            "display.details_mode", "expanded" if nv == "full" else "collapsed"
        )
        return _ok(rid, {"key": key, "value": nv})

    if key == "compact":
        raw = str(value or "").strip().lower()
        cfg0 = _load_cfg()
        d0 = cfg0.get("display") if isinstance(cfg0.get("display"), dict) else {}
        cur_b = bool(d0.get("tui_compact", False))
        if raw in {"", "toggle"}:
            nv_b = not cur_b
        elif raw == "on":
            nv_b = True
        elif raw == "off":
            nv_b = False
        else:
            return _err(rid, 4002, f"unknown compact value: {value}")
        _write_config_key("display.tui_compact", nv_b)
        return _ok(rid, {"key": key, "value": "on" if nv_b else "off"})

    if key == "statusbar":
        raw = str(value or "").strip().lower()
        display = _load_cfg().get("display")
        d0 = display if isinstance(display, dict) else {}
        current = _coerce_statusbar(d0.get("tui_statusbar", "top"))

        if raw in {"", "toggle"}:
            nv = "top" if current == "off" else "off"
        elif raw == "on":
            nv = "top"
        elif raw in _STATUSBAR_MODES:
            nv = raw
        else:
            return _err(rid, 4002, f"unknown statusbar value: {value}")

        _write_config_key("display.tui_statusbar", nv)
        return _ok(rid, {"key": key, "value": nv})

    if key == "mouse":
        # Explicit None check rather than `value or ""` so falsy non-string
        # inputs (0, False) reach the alias map as themselves — both map to
        # 'off' via _MOUSE_TRACKING_ALIASES — instead of being collapsed to
        # '' and triggering the toggle path. The slash command always passes
        # a string, but programmatic JSON-RPC callers may send booleans.
        raw = ("" if value is None else str(value)).strip().lower()
        cfg = _load_cfg()
        display = cfg.get("display") if isinstance(cfg.get("display"), dict) else {}
        current = _display_mouse_tracking(display)

        if raw in {"", "toggle"}:
            nv = "all" if current == "off" else "off"
        elif raw in _MOUSE_TRACKING_ALIASES:
            nv = _MOUSE_TRACKING_ALIASES[raw]
        else:
            return _err(rid, 4002, f"unknown mouse value: {value}")

        _write_config_key("display.mouse_tracking", nv)
        return _ok(rid, {"key": key, "value": nv})

    if key == "indicator":
        # Use an explicit None check rather than `value or ""` so falsy
        # non-string inputs (0, False, []) still surface as themselves
        # in the error message instead of looking like a blank value.
        raw = ("" if value is None else str(value)).strip().lower()
        if raw not in _INDICATOR_STYLES:
            return _err(
                rid,
                4002,
                f"unknown indicator: {raw!r}; pick one of {'|'.join(_INDICATOR_STYLES)}",
            )
        _write_config_key("display.tui_status_indicator", raw)
        return _ok(rid, {"key": key, "value": raw})

    if key in {"cwd", "terminal.cwd", "workdir"}:
        raw = str(value or "").strip()
        if not raw:
            return _err(rid, 4002, "cwd required")
        cwd = os.path.abspath(os.path.expanduser(raw))
        if not os.path.isdir(cwd):
            return _err(rid, 4002, f"working directory does not exist: {raw}")
        _write_config_key("terminal.cwd", cwd)
        os.environ["TERMINAL_CWD"] = cwd
        return _ok(
            rid,
            {"key": "terminal.cwd", "value": cwd, "cwd": cwd, "branch": _git_branch_for_cwd(cwd)},
        )

    if key in {"prompt", "personality", "skin"}:
        try:
            cfg = _load_cfg()
            if key == "prompt":
                if value == "clear":
                    cfg.pop("custom_prompt", None)
                    nv = ""
                else:
                    cfg["custom_prompt"] = value
                    nv = value
                _save_cfg(cfg)
            elif key == "personality":
                sid_key = params.get("session_id", "")
                pname, new_prompt = _validate_personality(str(value or ""), cfg)
                _write_config_key("display.personality", pname)
                _write_config_key("agent.system_prompt", new_prompt)
                nv = str(value or "none")
                history_reset, info = _apply_personality_to_session(
                    sid_key, session, new_prompt, pname
                )
            else:
                _write_config_key(f"display.{key}", value)
                nv = value
                if key == "skin":
                    _emit("skin.changed", "", resolve_skin())
            resp = {"key": key, "value": nv}
            if key == "personality":
                resp["history_reset"] = history_reset
                if info is not None:
                    resp["info"] = info
            return _ok(rid, resp)
        except Exception as e:
            return _err(rid, 5001, str(e))

    return _err(rid, 4002, f"unknown config key: {key}")


# ---------------------------------------------------------------------------
# Projects — first-class, per-profile, multi-folder workspaces
# ---------------------------------------------------------------------------


# JSON-RPC error codes for the projects surface.
_E_PROJECTS = 5061  # generic failure
_E_NO_PROJECT = 5062  # id resolved to nothing
_E_PROJECT_ARG = 5063  # invalid argument (e.g. bad name/slug)


class _NoProject(Exception):
    """Raised inside a projects handler when ``params['id']`` resolves to None."""


def _projects_payload(conn) -> dict:
    from hermes_cli import projects_db as pdb

    return {
        "projects": [p.to_dict() for p in pdb.list_projects(conn, include_archived=True)],
        "active_id": pdb.get_active_id(conn),
    }


def _projects_method(name: str):
    """Register a projects RPC, injecting (pdb, conn) and unifying error mapping.

    Every project CRUD handler opened the per-profile DB, mapped a missing id to
    5062, bad args to 5063, and everything else to 5061. This collapses that
    boilerplate so each handler is just its one meaningful operation.
    """

    def decorator(fn):
        @method(name)
        def handler(rid, params: dict) -> dict:
            try:
                from hermes_cli import projects_db as pdb

                with pdb.connect_closing() as conn:
                    return fn(rid, params, pdb, conn)
            except _NoProject:
                return _err(rid, _E_NO_PROJECT, "no such project")
            except ValueError as e:
                return _err(rid, _E_PROJECT_ARG, str(e))
            except Exception as e:
                return _err(rid, _E_PROJECTS, str(e))

        return handler

    return decorator


def _require_project(pdb, conn, params: dict):
    """The project named by ``params['id']`` (or raise ``_NoProject``)."""
    proj = pdb.get_project(conn, str(params.get("id") or ""))
    if proj is None:
        raise _NoProject
    return proj


@_projects_method("projects.list")
def _(rid, params, pdb, conn) -> dict:
    return _ok(rid, _projects_payload(conn))


@_projects_method("projects.get")
def _(rid, params, pdb, conn) -> dict:
    return _ok(rid, {"project": _require_project(pdb, conn, params).to_dict()})


@_projects_method("projects.create")
def _(rid, params, pdb, conn) -> dict:
    pid = pdb.create_project(
        conn,
        name=str(params.get("name") or ""),
        slug=params.get("slug"),
        folders=params.get("folders") or [],
        primary_path=params.get("primary_path"),
        description=params.get("description"),
        icon=params.get("icon"),
        color=params.get("color"),
        board_slug=params.get("board_slug"),
    )
    if params.get("use"):
        pdb.set_active(conn, pid)
    proj = pdb.get_project(conn, pid)
    return _ok(rid, {"project": proj.to_dict() if proj else None})


@_projects_method("projects.update")
def _(rid, params, pdb, conn) -> dict:
    proj = _require_project(pdb, conn, params)
    pdb.update_project(
        conn,
        proj.id,
        name=params.get("name"),
        description=params.get("description"),
        icon=params.get("icon"),
        color=params.get("color"),
        board_slug=params.get("board_slug"),
    )
    return _ok(rid, {"project": pdb.get_project(conn, proj.id).to_dict()})


@_projects_method("projects.add_folder")
def _(rid, params, pdb, conn) -> dict:
    proj = _require_project(pdb, conn, params)
    pdb.add_folder(
        conn,
        proj.id,
        str(params.get("path") or ""),
        label=params.get("label"),
        is_primary=bool(params.get("is_primary")),
    )
    return _ok(rid, {"project": pdb.get_project(conn, proj.id).to_dict()})


@_projects_method("projects.remove_folder")
def _(rid, params, pdb, conn) -> dict:
    proj = _require_project(pdb, conn, params)
    pdb.remove_folder(conn, proj.id, str(params.get("path") or ""))
    return _ok(rid, {"project": pdb.get_project(conn, proj.id).to_dict()})


@_projects_method("projects.set_primary")
def _(rid, params, pdb, conn) -> dict:
    proj = _require_project(pdb, conn, params)
    pdb.set_primary(conn, proj.id, str(params.get("path") or ""))
    return _ok(rid, {"project": pdb.get_project(conn, proj.id).to_dict()})


@_projects_method("projects.archive")
def _(rid, params, pdb, conn) -> dict:
    proj = _require_project(pdb, conn, params)
    (pdb.restore_project if params.get("restore") else pdb.archive_project)(conn, proj.id)
    return _ok(rid, _projects_payload(conn))


@_projects_method("projects.delete")
def _(rid, params, pdb, conn) -> dict:
    proj = _require_project(pdb, conn, params)
    pdb.delete_project(conn, proj.id)
    return _ok(rid, _projects_payload(conn))


@_projects_method("projects.set_active")
def _(rid, params, pdb, conn) -> dict:
    pdb.set_active(conn, _require_project(pdb, conn, params).id if params.get("id") else None)
    return _ok(rid, {"active_id": pdb.get_active_id(conn)})


@_projects_method("projects.for_cwd")
def _(rid, params, pdb, conn) -> dict:
    cwd = _completion_cwd({"cwd": str(params.get("cwd") or "").strip()} if params.get("cwd") else {})
    proj = pdb.project_for_path(conn, cwd)
    return _ok(rid, {"project": proj.to_dict() if proj else None, "cwd": cwd, "branch": _git_branch_for_cwd(cwd)})


def _is_repo_junk(root: str) -> bool:
    """A git root we never auto-surface as a project: the bare home dir or
    anything under HERMES_HOME (~/.argus by default) — config/sessions/skills,
    not a workspace. User-created projects pointing there are still honored."""
    if not root:
        return True

    from hermes_constants import get_hermes_home

    real = os.path.realpath(root)
    home = os.path.realpath(os.path.expanduser("~"))
    hermes_home = os.path.realpath(str(get_hermes_home()))

    return real == home or real == hermes_home or real.startswith(hermes_home + os.sep)


def _discover_repos_payload(db, *, conn=None, backfill: bool = True) -> list[dict]:
    """Merge filesystem-scanned repos (cached) with session-derived repo roots.

    Repo-first: the disk scan (persisted by `projects.record_repos`) surfaces
    repos even with zero hermes sessions. Session-derived roots cover repos
    outside the scan roots. Both are junk-filtered (hermes home subtree + bare
    home) and carry their session totals for the overview.

    ``conn`` reuses an already-open projects.db connection (the tree path holds
    one); ``backfill`` persists resolved roots back onto session rows — kept off
    the per-turn tree path (grouping uses the live git resolver regardless) and
    done only on the explicit discover/record refresh.
    """
    _is_junk = _is_repo_junk
    repos: dict[str, dict] = {}

    def _agg(root: str) -> dict:
        return repos.setdefault(root, {"root": root, "label": "", "sessions": 0, "last_active": 0.0})

    # Session-derived roots (common repo root, folding worktrees; cached) +
    # backfill the column so persisted git_repo_root matches the tree grouping.
    cwd_rows = list(db.distinct_session_cwds())
    # Warm the per-cwd git probes in parallel so a cold first paint doesn't
    # serialize one subprocess per distinct cwd before this loop reads the cache.
    git_probe.warm_roots(str(r.get("cwd") or "") for r in cwd_rows)
    cwd_to_root: dict[str, str] = {}
    for row in cwd_rows:
        cwd = str(row.get("cwd") or "")
        root = _git_common_repo_root_for_cwd(cwd)
        if not root:
            continue
        cwd_to_root[cwd] = root
        if _is_junk(root):
            continue
        agg = _agg(root)
        agg["sessions"] += int(row.get("sessions") or 0)
        agg["last_active"] = max(agg["last_active"], float(row.get("last_active") or 0))

    if backfill:
        try:
            db.backfill_repo_roots(cwd_to_root)
        except Exception:
            logger.debug("failed to backfill repo roots", exc_info=True)

    # Filesystem-scanned roots from the cache (may have zero sessions). Reuse the
    # caller's projects.db connection when given, else open a short-lived one.
    try:
        from hermes_cli import projects_db as pdb

        def _read(c) -> None:
            for entry in pdb.list_discovered_repos(c):
                root = str(entry.get("root") or "")
                if not root or _is_junk(root):
                    continue
                agg = _agg(root)
                if entry.get("label"):
                    agg["label"] = entry["label"]
                agg["last_active"] = max(agg["last_active"], float(entry.get("last_seen") or 0))

        if conn is not None:
            _read(conn)
        else:
            with pdb.connect_closing() as own:
                _read(own)
    except Exception:
        logger.debug("failed to read discovered repo cache", exc_info=True)

    out = sorted(repos.values(), key=lambda r: r["last_active"], reverse=True)
    for r in out:
        r["label"] = r["label"] or os.path.basename(r["root"].rstrip("/\\")) or r["root"]
    return out


@method("projects.discover_repos")
def _(rid, params: dict) -> dict:
    """Repos for the desktop overview: scanned-from-disk (cached) ∪ session-derived."""
    try:
        db = _get_db()
        if db is None:
            return _ok(rid, {"repos": []})
        return _ok(rid, {"repos": _discover_repos_payload(db)})
    except Exception as e:
        return _err(rid, 5061, str(e))


@method("projects.record_repos")
def _(rid, params: dict) -> dict:
    """Persist git repo roots found by the client's filesystem scan, then return
    the merged repo list. The native crawl runs on the desktop (local fs); this
    caches the result so later reads are instant instead of re-walking disk."""
    try:
        from hermes_cli import projects_db as pdb

        pairs: list[tuple[str, str | None]] = []
        for item in params.get("repos") or []:
            if isinstance(item, str):
                pairs.append((item, None))
            elif isinstance(item, dict) and item.get("root"):
                pairs.append((str(item["root"]), item.get("label")))

        with pdb.connect_closing() as conn:
            pdb.record_discovered_repos(conn, pairs, replace=True)

        db = _get_db()
        return _ok(rid, {"repos": _discover_repos_payload(db) if db is not None else []})
    except Exception as e:
        return _err(rid, 5061, str(e))


# Sources excluded from the project tree: cron runs and tool/subagent children
# are not user conversations. Subagent/compression children are already dropped
# by list_sessions_rich(include_children=False); cron has its own section.
_PROJECT_TREE_EXCLUDED_SOURCES = ["cron"]


def _project_tree_row(r: dict) -> dict:
    """Project a SessionDB row to the minimal shape the sidebar renders.

    Keeps the fields the grouping needs (cwd / git_branch / git_repo_root) plus
    everything ``SidebarSessionRow`` reads, and drops the heavy columns
    (system_prompt, model_config, ...) so the tree payload stays lean.
    """
    return {
        "id": r.get("id"),
        "_lineage_root_id": r.get("_lineage_root_id"),
        # The sidebar nests branch/fork sessions under their parent
        # (flattenSessionsWithBranches keys on this); without it, lane rows can't
        # draw the └─ connector the flat Recents list shows.
        "parent_session_id": r.get("parent_session_id"),
        "title": r.get("title"),
        "preview": r.get("preview"),
        "started_at": r.get("started_at") or 0,
        "ended_at": r.get("ended_at"),
        "last_active": r.get("last_active") or r.get("started_at") or 0,
        "source": r.get("source"),
        "archived": bool(r.get("archived")),
        "message_count": r.get("message_count") or 0,
        "tool_call_count": r.get("tool_call_count") or 0,
        "input_tokens": r.get("input_tokens") or 0,
        "output_tokens": r.get("output_tokens") or 0,
        "model": r.get("model"),
        "is_active": False,
        "cwd": r.get("cwd"),
        "git_branch": r.get("git_branch"),
        "git_repo_root": r.get("git_repo_root"),
    }


def _project_tree_inputs(
    db, session_limit: int, *, include_discovered: bool
) -> tuple[list[dict], list[dict], list[dict], str | None]:
    """Gather (sessions, projects, discovered_repos, active_id) for build_tree.

    ``include_discovered`` is the zero-session-repo overview tier; the entered
    view (drill-in) skips it entirely — it only needs the project it's showing,
    which already has sessions — avoiding the distinct-cwd scan + git probes on
    that per-turn path. One projects.db connection serves both reads.
    """
    rows = db.list_sessions_rich(
        limit=session_limit,
        offset=0,
        order_by_last_active=True,
        min_message_count=1,
        include_children=False,
        exclude_sources=_PROJECT_TREE_EXCLUDED_SOURCES,
        include_archived=False,
    )
    sessions = [_project_tree_row(r) for r in rows]
    # Parallel-warm the git cache so build_tree's resolver reads it instead of
    # cold-probing each cwd in sequence (matters on the drill-in path, which
    # skips the discovery warm-up below).
    git_probe.warm_roots(s["cwd"] for s in sessions if s.get("cwd"))

    from hermes_cli import projects_db as pdb

    with pdb.connect_closing() as conn:
        projects = [p.to_dict() for p in pdb.list_projects(conn)]
        active_id = pdb.get_active_id(conn)
        # backfill stays off the hot tree path — grouping uses the live resolver.
        discovered = _discover_repos_payload(db, conn=conn, backfill=False) if include_discovered else []

    return sessions, projects, discovered, active_id


def _build_project_tree(
    db, *, preview_limit: int, hydrate: bool, session_limit: int, include_discovered: bool
) -> tuple[dict, str | None]:
    """Gather inputs and run the one authoritative builder. Returns (tree, active_id)."""
    from tui_gateway import project_tree

    sessions, projects, discovered, active_id = _project_tree_inputs(
        db, session_limit, include_discovered=include_discovered
    )
    tree = project_tree.build_tree(
        projects,
        sessions,
        discovered,
        _resolve_cwd_git,
        preview_limit=preview_limit,
        hydrate=hydrate,
        is_junk_root=_is_repo_junk,
    )
    return tree, active_id


@method("projects.tree")
def _(rid, params: dict) -> dict:
    """Authoritative project overview: project -> repo -> lane structure with
    counts + a few preview sessions per project, plus the flat set of session
    ids claimed by any project (so the desktop excludes them from flat Recents).
    Lanes carry no session rows here; drill-in uses ``projects.project_sessions``.
    """
    try:
        db = _get_db()
        if db is None:
            return _ok(rid, {"projects": [], "active_id": None, "scoped_session_ids": []})

        tree, active_id = _build_project_tree(
            db,
            preview_limit=int(params.get("preview_limit") or 3),
            hydrate=False,
            session_limit=int(params.get("session_limit") or 2000),
            include_discovered=True,
        )
        return _ok(
            rid,
            {"projects": tree["projects"], "active_id": active_id, "scoped_session_ids": tree["scoped_session_ids"]},
        )
    except Exception as e:
        return _err(rid, 5061, str(e))


@method("projects.project_sessions")
def _(rid, params: dict) -> dict:
    """Fully hydrated lanes (repo -> lane -> session rows) for one project,
    built from the same authoritative grouping as ``projects.tree`` so ids and
    membership match exactly. Used when the user enters a project."""
    try:
        project_id = str(params.get("project_id") or "")
        if not project_id:
            return _err(rid, 5063, "project_id required")

        db = _get_db()
        if db is None:
            return _ok(rid, {"project": None})

        # Drill-in only needs the entered project (which has sessions), so skip
        # the zero-session discovery tier entirely.
        tree, _active = _build_project_tree(
            db, preview_limit=0, hydrate=True, session_limit=int(params.get("session_limit") or 5000),
            include_discovered=False,
        )
        proj = next((p for p in tree["projects"] if p["id"] == project_id), None)
        return _ok(rid, {"project": proj})
    except Exception as e:
        return _err(rid, 5061, str(e))


@method("config.get")
def _(rid, params: dict) -> dict:
    key = params.get("key", "")
    if key == "provider":
        try:
            from hermes_cli.models import list_available_providers, normalize_provider

            model = _resolve_model()
            parts = model.split("/", 1)
            return _ok(
                rid,
                {
                    "model": model,
                    "provider": (
                        normalize_provider(parts[0]) if len(parts) > 1 else "unknown"
                    ),
                    "providers": list_available_providers(),
                },
            )
        except Exception as e:
            return _err(rid, 5013, str(e))
    if key == "profile":
        from hermes_constants import display_hermes_home

        return _ok(rid, {"home": str(_hermes_home), "display": display_hermes_home()})
    if key == "project":
        cfg_terminal = _load_cfg().get("terminal") or {}
        raw = str(params.get("cwd", "") or cfg_terminal.get("cwd", "") or "").strip()
        cwd = _completion_cwd({"cwd": raw} if raw else {})
        return _ok(rid, {"cwd": cwd, "branch": _git_branch_for_cwd(cwd)})
    if key == "full":
        return _ok(rid, {"config": _load_cfg()})
    if key == "prompt":
        return _ok(rid, {"prompt": _load_cfg().get("custom_prompt", "")})
    if key == "skin":
        return _ok(
            rid, {"value": (_load_cfg().get("display") or {}).get("skin", "default")}
        )
    if key == "indicator":
        # Normalize so a hand-edited config.yaml with stray casing or
        # an unknown value reads back the SAME value the TUI actually
        # rendered (frontend's `normalizeIndicatorStyle` falls back to
        # `_INDICATOR_DEFAULT` for the same inputs).  Otherwise
        # `/indicator` would print one thing while the UI shows another.
        raw = (_load_cfg().get("display") or {}).get("tui_status_indicator", "")
        norm = str(raw).strip().lower()
        return _ok(
            rid,
            {"value": norm if norm in _INDICATOR_STYLES else _INDICATOR_DEFAULT},
        )
    if key == "personality":
        return _ok(
            rid,
            {"value": (_load_cfg().get("display") or {}).get("personality") or "none"},
        )
    if key == "reasoning":
        cfg = _load_cfg()
        effort = str(
            (cfg.get("agent") or {}).get("reasoning_effort", "medium") or "medium"
        )
        display = (
            "show"
            if bool((cfg.get("display") or {}).get("show_reasoning", False))
            else "hide"
        )
        return _ok(rid, {"value": effort, "display": display})
    if key == "fast":
        return _ok(
            rid,
            {
                "value": (
                    "fast"
                    if (session := _sessions.get(params.get("session_id", "")))
                    and getattr(session.get("agent"), "service_tier", None)
                    == "priority"
                    else ("fast" if _load_service_tier() == "priority" else "normal")
                ),
            },
        )
    if key == "busy":
        return _ok(rid, {"value": _load_busy_input_mode()})
    if key == "details_mode":
        allowed_dm = frozenset({"hidden", "collapsed", "expanded"})
        raw = (
            str(
                (_load_cfg().get("display") or {}).get("details_mode", "collapsed")
                or "collapsed"
            )
            .strip()
            .lower()
        )
        nv = raw if raw in allowed_dm else "collapsed"
        return _ok(rid, {"value": nv})
    if key == "thinking_mode":
        allowed_tm = frozenset({"collapsed", "truncated", "full"})
        cfg = _load_cfg()
        raw = (
            str((cfg.get("display") or {}).get("thinking_mode", "") or "")
            .strip()
            .lower()
        )
        if raw in allowed_tm:
            nv = raw
        else:
            dm = (
                str(
                    (cfg.get("display") or {}).get("details_mode", "collapsed")
                    or "collapsed"
                )
                .strip()
                .lower()
            )
            nv = "full" if dm == "expanded" else "collapsed"
        return _ok(rid, {"value": nv})
    if key == "compact":
        on = bool((_load_cfg().get("display") or {}).get("tui_compact", False))
        return _ok(rid, {"value": "on" if on else "off"})
    if key == "statusbar":
        display = _load_cfg().get("display")
        raw = (
            display.get("tui_statusbar", "top") if isinstance(display, dict) else "top"
        )
        return _ok(rid, {"value": _coerce_statusbar(raw)})
    if key == "mouse":
        display = _load_cfg().get("display")
        return _ok(rid, {"value": _display_mouse_tracking(display)})
    if key == "mtime":
        cfg_path = _hermes_home / "config.yaml"
        try:
            return _ok(
                rid, {"mtime": cfg_path.stat().st_mtime if cfg_path.exists() else 0}
            )
        except Exception:
            return _ok(rid, {"mtime": 0})
    return _err(rid, 4002, f"unknown config key: {key}")


@method("setup.status")
def _(rid, params: dict) -> dict:
    try:
        from hermes_cli.main import _has_any_provider_configured

        return _ok(rid, {"provider_configured": bool(_has_any_provider_configured())})
    except Exception as e:
        return _err(rid, 5016, str(e))


@method("setup.runtime_check")
def _(rid, params: dict) -> dict:
    """Strict provider check: does the configured/default model actually resolve to a usable runtime?

    Unlike setup.status (which returns True if ANY provider auth state is
    discoverable, including indirect fallbacks like ``gh auth token`` for
    Copilot), this runs the same resolve_runtime_provider() call the agent
    uses on session creation. It returns ok=False with the auth error message
    when the user's configured model cannot actually be served, so UIs can
    surface onboarding before the user submits a doomed prompt.
    """
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider
        from hermes_cli.auth import has_usable_secret
        from hermes_cli.main import _has_any_provider_configured

        requested = str(params.get("provider") or "").strip() or None
        runtime = resolve_runtime_provider(requested=requested)
        provider_configured = bool(_has_any_provider_configured())
        provider = runtime.get("provider") or "provider"
        source = str(runtime.get("source") or "")
        if not provider_configured and provider == "bedrock" and source in {
            "iam-role",
            "aws-sdk-default-chain",
        }:
            return _ok(
                rid,
                {
                    "ok": False,
                    "provider": provider,
                    "model": runtime.get("model"),
                    "source": source,
                    "error": "No Hermes provider is configured.",
                },
            )

        api_key = runtime.get("api_key")
        api_key_text = "" if callable(api_key) else str(api_key or "").strip()
        credential_ok = (
            callable(api_key)
            or api_key_text in {"aws-sdk", "no-key-required"}
            or has_usable_secret(api_key_text)
            or bool(runtime.get("command"))
        )

        if not credential_ok:
            return _ok(
                rid,
                {
                    "ok": False,
                    "provider": provider,
                    "model": runtime.get("model"),
                    "source": runtime.get("source"),
                    "error": f"No usable credentials found for {provider}.",
                },
            )

        return _ok(
            rid,
            {
                "ok": True,
                "provider": runtime.get("provider"),
                "model": runtime.get("model"),
                "source": runtime.get("source"),
            },
        )
    except Exception as e:
        return _ok(rid, {"ok": False, "error": str(e)})


# ── Methods: tools & system ──────────────────────────────────────────


@method("process.stop")
def _(rid, params: dict) -> dict:
    try:
        from tools.process_registry import process_registry

        return _ok(rid, {"killed": process_registry.kill_all()})
    except Exception as e:
        return _err(rid, 5010, str(e))


def _session_processes(session: dict) -> list:
    """Background processes owned by this session (registry session_key match)."""
    from tools.process_registry import process_registry

    key = str(session.get("session_key") or "")
    owned = []
    for entry in process_registry.list_sessions():
        proc = process_registry.get(entry["session_id"])
        if proc is None or str(getattr(proc, "session_key", "") or "") != key:
            continue
        # The 200-char list preview is too thin for the desktop's inline
        # terminal viewer — ship a real tail alongside it.
        entry["output_tail"] = (proc.output_buffer or "")[-4000:]
        owned.append(entry)
    return owned


@method("process.list")
def _(rid, params: dict) -> dict:
    """Session-scoped view of the background process registry (desktop status stack)."""
    session, err = _sess(params, rid)
    if err:
        return err
    try:
        return _ok(rid, {"processes": _session_processes(session)})
    except Exception as e:
        return _err(rid, 5010, str(e))


@method("process.kill")
def _(rid, params: dict) -> dict:
    """Kill ONE background process — scoped to the caller's session so one
    window can't reap another session's work (unlike process.stop's kill_all)."""
    session, err = _sess(params, rid)
    if err:
        return err
    proc_id = str(params.get("process_id") or "")
    if not proc_id:
        return _err(rid, 4012, "process_id required")
    try:
        from tools.process_registry import process_registry

        proc = process_registry.get(proc_id)
        if proc is None or str(getattr(proc, "session_key", "") or "") != str(
            session.get("session_key") or ""
        ):
            return _err(rid, 4044, f"no such process: {proc_id}")
        return _ok(rid, process_registry.kill_process(proc_id))
    except Exception as e:
        return _err(rid, 5010, str(e))


@method("reload.mcp")
def _(rid, params: dict) -> dict:
    session = _sessions.get(params.get("session_id", ""))
    try:
        # Gate: /reload-mcp invalidates the prompt cache for this session.
        # Respect the ``approvals.mcp_reload_confirm`` config toggle — if
        # set (default true) AND the caller did not pass ``confirm=true``
        # in params, surface a warning to the transcript instead of just
        # reloading silently.  Users pass confirm=true either by
        # re-invoking after reading the warning, or by setting the
        # config key to false permanently.
        user_confirm = bool(params.get("confirm", False))
        if not user_confirm:
            try:
                from hermes_cli.config import load_config as _load_config

                _cfg = _load_config()
                _approvals = _cfg.get("approvals") if isinstance(_cfg, dict) else None
                _confirm_required = True
                if isinstance(_approvals, dict):
                    _confirm_required = bool(_approvals.get("mcp_reload_confirm", True))
            except Exception:
                _confirm_required = True
            if _confirm_required:
                # Return a structured response the Ink client can surface
                # as a warning/confirmation without actually reloading yet.
                # Ink's ops.ts reads ``status`` and prints ``message`` to
                # the transcript; a follow-up invocation with confirm=true
                # (or an `always` choice that flips the config) proceeds.
                return _ok(
                    rid,
                    {
                        "status": "confirm_required",
                        "message": (
                            "⚠️  /reload-mcp invalidates the prompt cache (next "
                            "message re-sends full input tokens). Reply `/reload-mcp "
                            "now` to proceed, or `/reload-mcp always` to proceed and "
                            "silence this prompt permanently."
                        ),
                    },
                )

        from tools.mcp_tool import shutdown_mcp_servers, discover_mcp_tools

        shutdown_mcp_servers()
        discover_mcp_tools()
        if session:
            agent = session["agent"]
            # Rebuild the cached agent's tool snapshot so the current session
            # picks up added/removed MCP tools without `/new` (which discards
            # history).  The agent snapshots tools once at build and never
            # re-reads the registry, so an explicit rebuild is required here.
            # The user already consented to the prompt-cache invalidation via
            # the confirm gate above.  Mirrors gateway/run.py::_execute_mcp_reload.
            try:
                from tools.mcp_tool import refresh_agent_mcp_tools

                # Explicit reload: re-resolve enabled toolsets so a server the
                # user just enabled in config this session is picked up.
                refresh_agent_mcp_tools(
                    agent,
                    enabled_override=_load_enabled_toolsets(),
                    quiet_mode=True,
                )
            except Exception as _exc:
                logger.warning(
                    "Failed to refresh cached agent tools after /reload-mcp: %s",
                    _exc,
                )
            _emit(
                "session.info",
                params.get("session_id", ""),
                _session_info(agent, session),
            )

        # Honor `always=true` by persisting the opt-out to config.
        if bool(params.get("always", False)):
            try:
                from cli import save_config_value as _save_cfg

                _save_cfg("approvals.mcp_reload_confirm", False)
            except Exception as _exc:
                logger.warning("Failed to persist mcp_reload_confirm=false: %s", _exc)

        return _ok(rid, {"status": "reloaded"})
    except Exception as e:
        return _err(rid, 5015, str(e))


@method("reload.env")
def _(rid, params: dict) -> dict:
    """Re-read ``~/.argus/.env`` into the gateway process via
    ``hermes_cli.config.reload_env``, matching classic CLI's ``/reload``
    handler.  Newly added API keys take effect on the next agent call
    without restarting the TUI.

    The credential pool / provider routing for any *already-constructed*
    agent does not auto-rebuild — that's the same behaviour as classic
    CLI's ``/reload``.  Users who want a brand-new credential resolution
    should follow with ``/new``.
    """
    try:
        from hermes_cli.config import reload_env

        count = reload_env()
        return _ok(rid, {"updated": int(count)})
    except Exception as e:
        return _err(rid, 5015, str(e))


@method("mm.readiness")
def _(rid, params: dict) -> dict:
    """Report multimodal readiness so the web/desktop clients can gate entry
    into the multimodal experience (or show a guided "not ready" page) instead
    of dropping the user into a chat where voice/memory/etc silently don't work.

    Returns the ``probe_mm_readiness`` report. ``deep=true`` (default in the
    gateway, opt-out via ``probe_endpoints=false``) additionally runs LLM
    endpoint TCP probes (including auxiliary.text.remote_backend) and checks
    the local OCR package — everything lives in readiness.py so there is one
    mental model for MM startup state.

    Shape:
      {"ready": bool, "capabilities": [{key,label,status,required,reason,fix, ...}]}
    """
    try:
        from agent.multimodal.readiness import probe_mm_readiness
        raw_cfg = None
        try:
            from hermes_cli.config import load_config
            raw_cfg = load_config() or None
        except Exception:
            raw_cfg = None
        try:
            from agent.multimodal.hermes_glue import build_config
            cfg = build_config()
        except Exception:
            cfg = None  # fall back to field defaults; probe still reports gaps
        deep = is_truthy_value(params.get("probe_endpoints", True))
        return _ok(rid, probe_mm_readiness(cfg, raw_cfg, deep=deep))
    except Exception as e:
        return _err(rid, 5016, str(e))


_TUI_HIDDEN: frozenset[str] = frozenset(
    {
        "sethome",
        "set-home",
        "commands",
        "approve",
        "deny",
    }
)

_TUI_EXTRA: list[tuple[str, str, str]] = [
    ("/compact", "Toggle compact display mode", "TUI"),
    ("/logs", "Show recent gateway log lines", "TUI"),
    (
        "/mouse",
        "Set mouse tracking preset [on|off|toggle|wheel|buttons|all]",
        "TUI",
    ),
    ("/sessions", "Switch between live TUI sessions", "TUI"),
]

# Commands that queue messages onto _pending_input in the CLI.
# In the TUI the slash worker subprocess has no reader for that queue,
# so slash.exec routes them to command.dispatch internally (which handles
# them and returns a structured payload) instead of erroring out and
# relying on a client-side fallback. See #48848.
_PENDING_INPUT_COMMANDS: frozenset[str] = frozenset(
    {
        "retry",
        "queue",
        "q",
        "steer",
        "plan",
        "goal",
        "moa",
        "undo",
        "learn",
    }
)

_WORKER_BLOCKED_COMMANDS: frozenset[str] = frozenset({"snapshot", "snap"})


@method("commands.catalog")
def _(rid, params: dict) -> dict:
    """Registry-backed slash metadata for the TUI — categorized, no aliases."""
    try:
        from hermes_cli.commands import (
            COMMAND_REGISTRY,
            SUBCOMMANDS,
            _build_description,
        )

        all_pairs: list[list[str]] = []
        canon: dict[str, str] = {}
        categories: list[dict] = []
        cat_map: dict[str, list[list[str]]] = {}
        cat_order: list[str] = []

        for cmd in COMMAND_REGISTRY:
            if cmd.name in _TUI_HIDDEN or cmd.gateway_only:
                continue

            c = f"/{cmd.name}"
            canon[c.lower()] = c
            for a in cmd.aliases:
                canon[f"/{a}".lower()] = c

            desc = _build_description(cmd)
            all_pairs.append([c, desc])

            cat = cmd.category
            if cat not in cat_map:
                cat_map[cat] = []
                cat_order.append(cat)
            cat_map[cat].append([c, desc])

        for name, desc, cat in _TUI_EXTRA:
            all_pairs.append([name, desc])
            if cat not in cat_map:
                cat_map[cat] = []
                cat_order.append(cat)
            cat_map[cat].append([name, desc])

        warning = ""
        try:
            qcmds = _load_cfg().get("quick_commands", {}) or {}
            if isinstance(qcmds, dict) and qcmds:
                bucket = "User commands"
                if bucket not in cat_map:
                    cat_map[bucket] = []
                    cat_order.append(bucket)
                for qname, qc in sorted(qcmds.items()):
                    if not isinstance(qc, dict):
                        continue
                    key = f"/{qname}"
                    canon[key.lower()] = key
                    qtype = qc.get("type", "")
                    if qtype == "exec":
                        default_desc = f"exec: {qc.get('command', '')}"
                    elif qtype == "alias":
                        default_desc = f"alias → {qc.get('target', '')}"
                    else:
                        default_desc = qtype or "quick command"
                    qdesc = str(qc.get("description") or default_desc)
                    qdesc = qdesc[:120] + ("…" if len(qdesc) > 120 else "")
                    all_pairs.append([key, qdesc])
                    cat_map[bucket].append([key, qdesc])
        except Exception as e:
            if not warning:
                warning = f"quick_commands discovery unavailable: {e}"

        skill_count = 0
        try:
            from agent.skill_commands import scan_skill_commands

            for k, info in sorted(scan_skill_commands().items()):
                d = str(info.get("description", "Skill"))
                all_pairs.append([k, d[:120] + ("…" if len(d) > 120 else "")])
                skill_count += 1
        except Exception as e:
            warning = f"skill discovery unavailable: {e}"

        for cat in cat_order:
            categories.append({"name": cat, "pairs": cat_map[cat]})

        sub = {k: v[:] for k, v in SUBCOMMANDS.items()}
        return _ok(
            rid,
            {
                "pairs": all_pairs,
                "sub": sub,
                "canon": canon,
                "categories": categories,
                "skill_count": skill_count,
                "warning": warning,
            },
        )
    except Exception as e:
        return _err(rid, 5020, str(e))


def _cli_exec_blocked(argv: list[str]) -> str | None:
    """Return user hint if this argv must not run headless in the gateway process."""
    if not argv:
        return "bare `hermes` is interactive — use `/hermes chat -q …` or run `hermes` in another terminal"
    a0 = argv[0].lower()
    if a0 == "setup":
        return "`argus setup` needs a full terminal — run it outside the TUI"
    if a0 == "gateway":
        return "`argus gateway` is long-running — run it in another terminal"
    if a0 == "sessions" and len(argv) > 1 and argv[1].lower() == "browse":
        return "`argus sessions browse` is interactive — use /resume here, or run browse in another terminal"
    if a0 == "config" and len(argv) > 1 and argv[1].lower() == "edit":
        return "`argus config edit` needs $EDITOR in a real terminal"
    return None


@method("cli.exec")
def _(rid, params: dict) -> dict:
    """Run `python -m hermes_cli.main` with argv; capture stdout/stderr (non-interactive only)."""
    argv = params.get("argv", [])
    if not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
        return _err(rid, 4003, "argv must be list[str]")
    hint = _cli_exec_blocked(argv)
    if hint:
        return _ok(rid, {"blocked": True, "hint": hint, "code": -1, "output": ""})
    try:
        r = subprocess.run(
            [sys.executable, "-m", "hermes_cli.main", *argv],
            capture_output=True,
            text=True,
            timeout=min(int(params.get("timeout", 240)), 600),
            cwd=os.getcwd(),
            # cli.exec runs `python -m hermes_cli.main` (can drive the agent) →
            # needs provider credentials. Tier-1 secrets still stripped (#29157).
            env=hermes_subprocess_env(inherit_credentials=True),
            stdin=subprocess.DEVNULL,
        )
        parts = [r.stdout or "", r.stderr or ""]
        out = "\n".join(p for p in parts if p).strip() or "(no output)"
        return _ok(
            rid, {"blocked": False, "code": r.returncode, "output": out[:48_000]}
        )
    except subprocess.TimeoutExpired:
        return _err(rid, 5016, "cli.exec: timeout")
    except Exception as e:
        return _err(rid, 5017, str(e))


@method("command.resolve")
def _(rid, params: dict) -> dict:
    try:
        from hermes_cli.commands import resolve_command

        r = resolve_command(params.get("name", ""))
        if r:
            return _ok(
                rid,
                {
                    "canonical": r.name,
                    "description": r.description,
                    "category": r.category,
                },
            )
        return _err(rid, 4011, f"unknown command: {params.get('name')}")
    except Exception as e:
        return _err(rid, 5012, str(e))


def _resolve_name(name: str) -> str:
    try:
        from hermes_cli.commands import resolve_command

        r = resolve_command(name)
        return r.name if r else name
    except Exception:
        return name


@method("command.dispatch")
def _(rid, params: dict) -> dict:
    name, arg = params.get("name", "").lstrip("/"), params.get("arg", "")
    resolved = _resolve_name(name)
    if resolved != name:
        name = resolved
    session = _sessions.get(params.get("session_id", ""))

    qcmds = _load_cfg().get("quick_commands", {})
    if name in qcmds:
        qc = qcmds[name]
        if qc.get("type") == "exec":
            r = subprocess.run(
                qc.get("command", ""),
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                stdin=subprocess.DEVNULL,
            )
            output = (
                (r.stdout or "")
                + ("\n" if r.stdout and r.stderr else "")
                + (r.stderr or "")
            ).strip()[:4000]
            if r.returncode != 0:
                return _err(
                    rid,
                    4018,
                    output or f"quick command failed with exit code {r.returncode}",
                )
            return _ok(rid, {"type": "exec", "output": output})
        if qc.get("type") == "alias":
            return _ok(rid, {"type": "alias", "target": qc.get("target", "")})

    try:
        from hermes_cli.plugins import (
            get_plugin_command_handler,
            resolve_plugin_command_result,
        )

        handler = get_plugin_command_handler(name)
        if handler:
            result = resolve_plugin_command_result(handler(arg))
            return _ok(rid, {"type": "plugin", "output": str(result or "")})
    except Exception:
        pass

    try:
        from agent.skill_commands import (
            scan_skill_commands,
            build_skill_invocation_message,
        )

        cmds = scan_skill_commands()
        key = f"/{name}"
        if key in cmds:
            msg = build_skill_invocation_message(
                key, arg, task_id=session.get("session_key", "") if session else ""
            )
            if msg:
                return _ok(
                    rid,
                    {
                        "type": "skill",
                        "message": msg,
                        "name": cmds[key].get("name", name),
                    },
                )
    except Exception:
        pass

    # ── Commands that queue messages onto _pending_input in the CLI ───
    # In the TUI the slash worker subprocess has no reader for that queue,
    # so we handle them here and return a structured payload.

    if name in {"queue", "q"}:
        if not arg:
            return _err(rid, 4004, "usage: /queue <prompt>")
        return _ok(rid, {"type": "send", "message": arg})

    if name == "learn":
        # Open-ended: build the standards-guided prompt and submit it as a
        # normal agent turn. The live agent gathers whatever the user
        # described (dirs, URLs, this conversation, pasted text) with its own
        # tools and authors the skill via skill_manage. Works on any backend.
        from agent.learn_prompt import build_learn_prompt

        return _ok(rid, {"type": "send", "message": build_learn_prompt(arg)})
    if name == "moa":
        # /moa is one-shot sugar only: run a single prompt through the default
        # MoA preset, then restore the prior model. To *switch* to a MoA preset
        # for the rest of the session, pick it from the model picker (MoA
        # presets surface as a virtual "Mixture of Agents" provider).
        try:
            from hermes_cli.moa_config import moa_usage, normalize_moa_config

            if not arg:
                return _err(rid, 4004, moa_usage())
            if not session:
                return _err(rid, 4001, "no active session")
            sid = params.get("session_id", "")
            moa_cfg = normalize_moa_config(_load_cfg().get("moa") or {})
            preset = moa_cfg["default_preset"]
            # Record the live model identity so it can be restored after the
            # one-shot turn, then swap the agent's client in place (#53444:
            # setting session["model_override"] alone never switched the
            # already-built agent, so the turn silently ran on the old model).
            agent = session.get("agent")
            session["moa_one_shot_restore"] = {
                "override": session.get("model_override"),
                "model": getattr(agent, "model", None) if agent else None,
                "provider": getattr(agent, "provider", None) if agent else None,
            }
            if agent is not None:
                # Live agent: swap its client in place so THIS turn runs MoA.
                try:
                    _apply_model_switch(
                        sid,
                        session,
                        f"{preset} --provider moa",
                        confirm_expensive_model=False,
                        pin_session_override=True,
                    )
                except Exception as exc:
                    session.pop("moa_one_shot_restore", None)
                    return _err(rid, 5030, f"moa unavailable: {exc}")
            else:
                # No agent built yet (lazy/fresh session): the override is
                # consumed by the first build, so the turn runs MoA without an
                # in-place switch.
                session["model_override"] = {
                    "provider": "moa",
                    "model": preset,
                    "base_url": "moa://local",
                    "api_key": "moa-virtual-provider",
                    "api_mode": "chat_completions",
                }
            return _ok(
                rid,
                {
                    "type": "send",
                    "notice": f"MoA one-shot queued with preset {preset}; previous model will be restored after this turn.",
                    "message": arg,
                },
            )
        except Exception as exc:
            return _err(rid, 5030, f"moa unavailable: {exc}")

    if name == "retry":
        if not session:
            return _err(rid, 4001, "no active session to retry")
        if session.get("running"):
            return _err(
                rid, 4009, "session busy — /interrupt the current turn before /retry"
            )
        history = session.get("history", [])
        if not history:
            return _err(rid, 4018, "no previous user message to retry")
        # Walk backwards to find the last user message
        last_user_idx = None
        for i in range(len(history) - 1, -1, -1):
            if history[i].get("role") == "user":
                last_user_idx = i
                break
        if last_user_idx is None:
            return _err(rid, 4018, "no previous user message to retry")
        content = history[last_user_idx].get("content", "")
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        if not content:
            return _err(rid, 4018, "last user message is empty")
        # Truncate history: remove everything from the last user message onward
        # (mirrors CLI retry_last() which strips the failed exchange)
        with session["history_lock"]:
            session["history"] = history[:last_user_idx]
            session["history_version"] = int(session.get("history_version", 0)) + 1
        return _ok(rid, {"type": "send", "message": content})

    if name == "steer":
        if not arg:
            return _err(rid, 4004, "usage: /steer <prompt>")
        agent = session.get("agent") if session else None
        if agent and hasattr(agent, "steer"):
            try:
                accepted = agent.steer(arg)
                if accepted:
                    return _ok(
                        rid,
                        {
                            "type": "exec",
                            "output": f"⏩ Steer queued — arrives after the next tool call: {arg[:80]}{'...' if len(arg) > 80 else ''}",
                        },
                    )
            except Exception:
                pass
        # Fallback: no active run, treat as next-turn message
        return _ok(rid, {"type": "send", "message": arg})

    if name == "goal":
        if not session:
            return _err(rid, 4001, "no active session")
        try:
            from hermes_cli.goals import GoalManager
        except Exception as exc:
            return _err(rid, 5030, f"goals unavailable: {exc}")

        sid_key = session.get("session_key") or ""
        if not sid_key:
            return _err(rid, 4001, "no session key")

        try:
            goals_cfg = _load_cfg().get("goals") or {}
            max_turns = int(goals_cfg.get("max_turns", 20) or 20)
        except Exception:
            max_turns = 20
        mgr = GoalManager(session_id=sid_key, default_max_turns=max_turns)

        lower = arg.strip().lower()
        if not arg.strip() or lower == "status":
            return _ok(rid, {"type": "exec", "output": mgr.status_line()})
        if lower == "pause":
            state = mgr.pause(reason="user-paused")
            out = "No goal set." if state is None else f"⏸ Goal paused: {state.goal}"
            return _ok(rid, {"type": "exec", "output": out})
        if lower == "resume":
            state = mgr.resume()
            if state is None:
                return _ok(rid, {"type": "exec", "output": "No goal to resume."})
            return _ok(
                rid,
                {
                    "type": "exec",
                    "output": (
                        f"▶ Goal resumed: {state.goal}\n"
                        "Send any message to continue, or wait — I'll take the next step on the next turn."
                    ),
                },
            )
        if lower in {"clear", "stop", "done"}:
            had = mgr.has_goal()
            mgr.clear()
            return _ok(
                rid,
                {
                    "type": "exec",
                    "output": "✓ Goal cleared." if had else "No active goal.",
                },
            )

        # Otherwise — treat the remaining text as the new goal.
        try:
            state = mgr.set(arg)
        except ValueError as exc:
            return _err(rid, 4004, f"invalid goal: {exc}")

        notice = (
            f"⊙ Goal set ({state.max_turns}-turn budget): {state.goal}\n"
            "I'll keep working until the goal is done, you pause/clear it, or the budget is exhausted.\n"
            "Controls: /goal status · /goal pause · /goal resume · /goal clear"
        )
        # Send the goal text as the kickoff prompt. The TUI client sees
        # {type: send, notice, message} → renders `notice` as a sys line,
        # then submits `message` as a user turn. The post-turn judge
        # wired in _run_prompt_submit takes over from there.
        return _ok(
            rid,
            {"type": "send", "notice": notice, "message": state.goal},
        )

    if name == "undo":
        # /undo [N]: back up N user turns (default 1), soft-delete the
        # truncated rows on disk, and prefill the composer with the text
        # of the user message we backed up to so it can be edited and
        # resubmitted. N=1 is the Claude-Code-style single-step undo;
        # /undo 3 backs up three user turns at once. See issue #21910.
        if not session:
            return _err(rid, 4001, "no active session to undo")
        if session.get("running"):
            return _err(
                rid, 4009, "session busy — /interrupt the current turn before /undo"
            )
        db = _get_db()
        if db is None:
            return _db_unavailable_error(rid, code=5008)
        session_key = session.get("session_key", "")
        if not session_key:
            return _err(rid, 4001, "no session key for undo")
        # Parse the optional count argument (e.g. "/undo 3" → 3).
        n = 1
        arg_str = (arg or "").strip()
        if arg_str:
            try:
                n = int(arg_str.split()[0])
            except (ValueError, IndexError):
                return _err(rid, 4004, f"undo: invalid count {arg_str!r} — use /undo or /undo N")
        if n < 1:
            n = 1
        try:
            recents = db.list_recent_user_messages(session_key, limit=max(n, 10))
        except Exception as e:
            return _err(rid, 5008, f"undo: failed to load history: {e}")
        if not recents:
            return _err(rid, 4018, "no user messages to undo")
        # recents[0] is the most-recent user turn; pick the Nth-from-last.
        # If N exceeds the number of user turns, back up to the oldest.
        target_idx = min(n - 1, len(recents) - 1)
        target_id = recents[target_idx]["id"]
        try:
            result = db.rewind_to_message(session_key, target_id)
        except ValueError as e:
            return _err(rid, 4004, f"undo: {e}")
        except Exception as e:
            return _err(rid, 5008, f"undo: {e}")
        # Reload the active-only transcript into the in-memory session
        # history so subsequent turns see the truncated view.
        try:
            active = db.get_messages_as_conversation(session_key)
        except Exception:
            active = []
        with session["history_lock"]:
            session["history"] = list(active)
            session["history_version"] = int(session.get("history_version", 0)) + 1
        # Notify memory providers — same hook /branch fires, plus the
        # rewound flag so providers caching per-turn document state
        # know to invalidate. See #6672 + #21910.
        agent = session.get("agent")
        if agent is not None:
            mm = getattr(agent, "_memory_manager", None)
            if mm is not None:
                try:
                    mm.on_session_switch(
                        session_key,
                        parent_session_id="",
                        reset=False,
                        rewound=True,
                    )
                except Exception:
                    pass
            if hasattr(agent, "_invalidate_system_prompt"):
                try:
                    agent._invalidate_system_prompt()
                except Exception:
                    pass
            if hasattr(agent, "_last_flushed_db_idx"):
                try:
                    agent._last_flushed_db_idx = len(active)
                except Exception:
                    pass
        target_msg = result.get("target_message") or {}
        target_text = target_msg.get("content") or ""
        if isinstance(target_text, list):
            parts = [
                p.get("text", "") for p in target_text
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            target_text = "\n".join(t for t in parts if t)
        if not isinstance(target_text, str):
            target_text = ""
        rewound_count = result.get("rewound_count", 0)
        turns_undone = target_idx + 1
        turn_word = "turn" if turns_undone == 1 else "turns"
        notice = (
            f"↶ Undid {turns_undone} {turn_word} ({rewound_count} message(s)). "
            "Edit and resubmit, or send a new message."
        )
        return _ok(
            rid,
            {"type": "prefill", "message": target_text, "notice": notice},
        )

    if name in {"snapshot", "snap"}:
        subcommand = arg.split(maxsplit=1)[0].lower() if arg else ""
        if subcommand in {"restore", "rewind"}:
            return _ok(
                rid,
                {
                    "type": "exec",
                    "output": (
                        "/snapshot restore is blocked in the TUI because it changes "
                        "config/state on disk while the live agent has cached settings. "
                        "Run it in the classic CLI, then restart the TUI."
                    ),
                },
            )

    return _err(rid, 4018, f"not a quick/plugin/skill command: {name}")


# ── Methods: paste ────────────────────────────────────────────────────

_paste_counter = 0


@method("paste.collapse")
def _(rid, params: dict) -> dict:
    global _paste_counter
    text = params.get("text", "")
    if not text:
        return _err(rid, 4004, "empty paste")

    _paste_counter += 1
    line_count = text.count("\n") + 1
    paste_dir = _hermes_home / "pastes"
    paste_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime

    paste_file = (
        paste_dir / f"paste_{_paste_counter}_{datetime.now().strftime('%H%M%S')}.txt"
    )
    paste_file.write_text(text, encoding="utf-8")

    placeholder = (
        f"[Pasted text #{_paste_counter}: {line_count} lines \u2192 {paste_file}]"
    )
    return _ok(
        rid, {"placeholder": placeholder, "path": str(paste_file), "lines": line_count}
    )


# ── Methods: complete ─────────────────────────────────────────────────

_FUZZY_CACHE_TTL_S = 5.0
_FUZZY_CACHE_MAX_FILES = 20000
_FUZZY_FALLBACK_EXCLUDES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".next",
        ".cache",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        "dist",
        "build",
        "target",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
)
_fuzzy_cache_lock = threading.Lock()
_fuzzy_cache: dict[str, tuple[float, list[str]]] = {}


def _list_repo_files(root: str) -> list[str]:
    """Return file paths relative to ``root``.

    Uses ``git ls-files`` from the repo top (resolved via
    ``rev-parse --show-toplevel``) so the listing covers tracked + untracked
    files anywhere in the repo, then converts each path back to be relative
    to ``root``. Files outside ``root`` (parent directories of cwd, sibling
    subtrees) are excluded so the picker stays scoped to what's reachable
    from the gateway's cwd. Falls back to a bounded ``os.walk(root)`` when
    ``root`` isn't inside a git repo. Result cached per-root for
    ``_FUZZY_CACHE_TTL_S`` so rapid keystrokes don't respawn git processes.
    """
    now = time.monotonic()
    with _fuzzy_cache_lock:
        cached = _fuzzy_cache.get(root)
        if cached and now - cached[0] < _FUZZY_CACHE_TTL_S:
            return cached[1]

    files: list[str] = []
    try:
        top_result = subprocess.run(
            ["git", "-C", root, "rev-parse", "--show-toplevel"],
            capture_output=True,
            timeout=2.0,
            check=False,
            stdin=subprocess.DEVNULL,
        )
        if top_result.returncode == 0:
            top = top_result.stdout.decode("utf-8", "replace").strip()
            list_result = subprocess.run(
                [
                    "git",
                    "-C",
                    top,
                    "ls-files",
                    "-z",
                    "--cached",
                    "--others",
                    "--exclude-standard",
                ],
                capture_output=True,
                timeout=2.0,
                check=False,
                stdin=subprocess.DEVNULL,
            )
            if list_result.returncode == 0:
                for p in list_result.stdout.decode("utf-8", "replace").split("\0"):
                    if not p:
                        continue
                    rel = os.path.relpath(os.path.join(top, p), root).replace(
                        os.sep, "/"
                    )
                    # Skip parents/siblings of cwd — keep the picker scoped
                    # to root-and-below, matching Cmd-P workspace semantics.
                    if rel.startswith("../"):
                        continue
                    files.append(rel)
                    if len(files) >= _FUZZY_CACHE_MAX_FILES:
                        break
    except (OSError, subprocess.TimeoutExpired):
        pass

    if not files:
        # Fallback walk: skip vendor/build dirs + dot-dirs so the walk stays
        # tractable. Dotfiles themselves survive — the ranker decides based
        # on whether the query starts with `.`.
        try:
            for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
                dirnames[:] = [
                    d
                    for d in dirnames
                    if d not in _FUZZY_FALLBACK_EXCLUDES and not d.startswith(".")
                ]
                rel_dir = os.path.relpath(dirpath, root)
                for f in filenames:
                    rel = f if rel_dir == "." else f"{rel_dir}/{f}"
                    files.append(rel.replace(os.sep, "/"))
                    if len(files) >= _FUZZY_CACHE_MAX_FILES:
                        break
                if len(files) >= _FUZZY_CACHE_MAX_FILES:
                    break
        except OSError:
            pass

    with _fuzzy_cache_lock:
        _fuzzy_cache[root] = (now, files)

    return files


def _fuzzy_basename_rank(name: str, query: str) -> tuple[int, int] | None:
    """Rank ``name`` against ``query``; lower is better. Returns None to reject.

    Tiers (kind):
      0 — exact basename
      1 — basename prefix (e.g. `app` → `appChrome.tsx`)
      2 — word-boundary / camelCase hit (e.g. `chrome` → `appChrome.tsx`)
      3 — substring anywhere in basename
      4 — subsequence match (every query char appears in order)

    Secondary key is `len(name)` so shorter names win ties.
    """
    if not query:
        return (3, len(name))

    nl = name.lower()
    ql = query.lower()

    if nl == ql:
        return (0, len(name))

    if nl.startswith(ql):
        return (1, len(name))

    # Word-boundary split: `foo-bar_baz.qux` → ["foo","bar","baz","qux"].
    # camelCase split: `appChrome` → ["app","Chrome"]. Cheap approximation;
    # falls through to substring/subsequence if it misses.
    parts: list[str] = []
    buf = ""
    for ch in name:
        if ch in "-_." or (ch.isupper() and buf and not buf[-1].isupper()):
            if buf:
                parts.append(buf)
            buf = ch if ch not in "-_." else ""
        else:
            buf += ch
    if buf:
        parts.append(buf)
    for p in parts:
        if p.lower().startswith(ql):
            return (2, len(name))

    if ql in nl:
        return (3, len(name))

    i = 0
    for ch in nl:
        if ch == ql[i]:
            i += 1
            if i == len(ql):
                return (4, len(name))

    return None


@method("complete.path")
def _(rid, params: dict) -> dict:
    word = params.get("word", "")
    if not word:
        return _ok(rid, {"items": []})

    items: list[dict] = []
    try:
        root = _completion_cwd(params)
        is_context = word.startswith("@")
        query = word[1:] if is_context else word

        if is_context and not query:
            items = [
                {"text": "@diff", "display": "@diff", "meta": "git diff"},
                {"text": "@staged", "display": "@staged", "meta": "staged diff"},
                {"text": "@file:", "display": "@file:", "meta": "attach file"},
                {"text": "@folder:", "display": "@folder:", "meta": "attach folder"},
                {"text": "@url:", "display": "@url:", "meta": "fetch url"},
                {"text": "@git:", "display": "@git:", "meta": "git log"},
            ]
            return _ok(rid, {"items": items})

        # Accept both `@folder:path` and the bare `@folder` form so the user
        # sees directory listings as soon as they finish typing the keyword,
        # without first accepting the static `@folder:` hint.
        if is_context and query in {"file", "folder"}:
            prefix_tag, path_part = query, ""
        elif is_context and query.startswith(("file:", "folder:")):
            prefix_tag, _, tail = query.partition(":")
            path_part = tail
        else:
            prefix_tag = ""
            path_part = query if is_context else query

        # Fuzzy basename search across the repo when the user types a bare
        # name with no path separator — `@appChrome` surfaces every file
        # whose basename matches, regardless of directory depth. Matches what
        # editors like Cursor / VS Code do for Cmd-P. Path-ish queries (with
        # `/`, `./`, `~/`, `/abs`) fall through to the directory-listing
        # path so explicit navigation intent is preserved.
        if (
            is_context
            and path_part
            and len(path_part.strip()) >= 2
            and "/" not in path_part
            and prefix_tag != "folder"
        ):
            ranked: list[tuple[tuple[int, int], str, str]] = []
            for rel in _list_repo_files(root):
                basename = os.path.basename(rel)
                if basename.startswith(".") and not path_part.startswith("."):
                    continue
                rank = _fuzzy_basename_rank(basename, path_part)
                if rank is None:
                    continue
                ranked.append((rank, rel, basename))

            ranked.sort(key=lambda r: (r[0], len(r[1]), r[1]))
            tag = prefix_tag or "file"
            for _, rel, basename in ranked[:30]:
                items.append(
                    {
                        "text": f"@{tag}:{rel}",
                        "display": basename,
                        "meta": os.path.dirname(rel),
                    }
                )

            return _ok(rid, {"items": items})

        expanded = _normalize_completion_path(path_part) if path_part else "."
        if expanded == "." or not expanded:
            search_dir, match = ".", ""
        elif expanded.endswith("/"):
            search_dir, match = expanded, ""
        else:
            search_dir = os.path.dirname(expanded) or "."
            match = os.path.basename(expanded)

        search_dir = (
            search_dir if os.path.isabs(search_dir) else os.path.join(root, search_dir)
        )
        if not os.path.isdir(search_dir):
            return _ok(rid, {"items": []})

        want_dir = prefix_tag == "folder"
        match_lower = match.lower()
        for entry in sorted(os.listdir(search_dir)):
            if match and not entry.lower().startswith(match_lower):
                continue
            if is_context and entry in _FUZZY_FALLBACK_EXCLUDES:
                continue
            if is_context and not prefix_tag and entry.startswith("."):
                continue
            full = os.path.join(search_dir, entry)
            is_dir = os.path.isdir(full)
            # Explicit `@folder:` / `@file:` — honour the user's filter.  Skip
            # the opposite kind instead of auto-rewriting the completion tag,
            # which used to defeat the prefix and let `@folder:` list files.
            if prefix_tag and want_dir != is_dir:
                continue
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            suffix = "/" if is_dir else ""

            if is_context and prefix_tag:
                text = f"@{prefix_tag}:{rel}{suffix}"
            elif is_context:
                kind = "folder" if is_dir else "file"
                text = f"@{kind}:{rel}{suffix}"
            elif word.startswith("~"):
                text = "~/" + os.path.relpath(full, os.path.expanduser("~")) + suffix
            elif word.startswith("./"):
                text = "./" + rel + suffix
            else:
                text = rel + suffix

            items.append(
                {
                    "text": text,
                    "display": entry + suffix,
                    "meta": "dir" if is_dir else "",
                }
            )
            if len(items) >= 30:
                break
    except Exception as e:
        return _err(rid, 5021, str(e))

    return _ok(rid, {"items": items})


def _details_completion_item(value: str, meta: str = "") -> dict:
    return {"text": value, "display": value, "meta": meta}


def _details_root_completion_item(
    value: str, meta: str, needs_leading_space: bool
) -> dict:
    return _details_completion_item(
        f" {value}" if needs_leading_space else value,
        meta,
    )


def _details_completions(text: str) -> list[dict] | None:
    if not text.lower().startswith("/details"):
        return None

    stripped = text.strip()
    if stripped and not "/details".startswith(stripped.lower().split()[0]):
        return None

    body = text[len("/details") :]
    if body.startswith(" "):
        body = body[1:]
    parts = body.split()
    has_trailing_space = text.endswith(" ")
    sections = ("thinking", "tools", "subagents", "activity")
    modes = ("hidden", "collapsed", "expanded")

    if not body or (len(parts) == 0 and has_trailing_space):
        return [
            *[
                _details_root_completion_item(
                    mode, "global mode", not has_trailing_space
                )
                for mode in modes
            ],
            _details_root_completion_item(
                "cycle", "cycle global mode", not has_trailing_space
            ),
            *[
                _details_root_completion_item(
                    section, "section override", not has_trailing_space
                )
                for section in sections
            ],
        ]

    if len(parts) == 1 and not has_trailing_space:
        prefix = parts[0].lower()
        candidates = [*modes, "cycle", *sections]
        return [
            _details_completion_item(
                candidate,
                (
                    "section override"
                    if candidate in sections
                    else "cycle global mode" if candidate == "cycle" else "global mode"
                ),
            )
            for candidate in candidates
            if candidate.startswith(prefix) and candidate != prefix
        ]

    if len(parts) == 1 and has_trailing_space and parts[0].lower() in sections:
        return [
            *[
                _details_completion_item(mode, f"set {parts[0].lower()}")
                for mode in modes
            ],
            _details_completion_item("reset", f"clear {parts[0].lower()} override"),
        ]

    if len(parts) == 2 and not has_trailing_space and parts[0].lower() in sections:
        prefix = parts[1].lower()
        return [
            _details_completion_item(
                candidate,
                (
                    f"clear {parts[0].lower()} override"
                    if candidate == "reset"
                    else f"set {parts[0].lower()}"
                ),
            )
            for candidate in (*modes, "reset")
            if candidate.startswith(prefix) and candidate != prefix
        ]

    return []


@method("complete.slash")
def _(rid, params: dict) -> dict:
    text = params.get("text", "")
    if not text.startswith("/"):
        return _ok(rid, {"items": []})

    try:
        from hermes_cli.commands import SlashCommandCompleter
        from prompt_toolkit.document import Document
        from prompt_toolkit.formatted_text import to_plain_text

        from agent.skill_commands import get_skill_commands
        from agent.skill_bundles import get_skill_bundles

        completer = SlashCommandCompleter(
            skill_commands_provider=lambda: get_skill_commands(),
            skill_bundles_provider=lambda: get_skill_bundles(),
        )
        doc = Document(text, len(text))
        items = [
            {
                "text": c.text,
                # prompt_toolkit gives us FormattedText (a list of (style,
                # text) tuples) for display/display_meta. Serialize both as
                # plain strings — the TUI's CompletionItem.display contract
                # is a string, and sending the raw list trips Ink's row
                # layout into 1-char truncation of the next column.
                "display": to_plain_text(c.display) if c.display else c.text,
                "meta": to_plain_text(c.display_meta) if c.display_meta else "",
            }
            for c in completer.get_completions(doc, None)
        ][:30]
        text_lower = text.lower()
        extras = [
            {
                "text": "/compact",
                "display": "/compact",
                "meta": "Toggle compact display mode",
            },
            {
                "text": "/details",
                "display": "/details",
                "meta": "Control agent detail visibility",
            },
            {
                "text": "/logs",
                "display": "/logs",
                "meta": "Show recent gateway log lines",
            },
            {
                "text": "/mouse",
                "display": "/mouse",
                "meta": "Set mouse tracking preset [on|off|toggle|wheel|buttons|all]",
            },
        ]
        for extra in extras:
            if extra["text"].startswith(text_lower) and not any(
                item["text"] == extra["text"] for item in items
            ):
                items.append(extra)

        details_items = _details_completions(text)
        if details_items is not None:
            return _ok(
                rid,
                {
                    "items": details_items,
                    "replace_from": text.rfind(" ") + 1 if " " in text else len(text),
                },
            )

        return _ok(
            rid,
            {"items": items, "replace_from": text.rfind(" ") + 1 if " " in text else 1},
        )
    except Exception as e:
        return _err(rid, 5020, str(e))


@method("model.options")
def _(rid, params: dict) -> dict:
    try:
        from hermes_cli.inventory import build_models_payload, load_picker_context

        session = _sessions.get(params.get("session_id", ""))
        agent = session.get("agent") if session else None
        # Layer agent-session state on top of disk config — once an agent
        # is spawned, IT owns the live provider/model/base_url. Empty
        # agent attributes must NOT clobber disk config (with_overrides
        # is truthy-only).
        ctx = load_picker_context().with_overrides(
            current_provider=getattr(agent, "provider", "") if agent else "",
            current_model=(
                (getattr(agent, "model", "") if agent else "") or _resolve_model()
            ),
            current_base_url=getattr(agent, "base_url", "") if agent else "",
        )
        # picker_hints + canonical_order produce the TUI's required shape:
        # `authenticated`/`auth_type`/`key_env`/`warning` per row, in
        # CANONICAL_PROVIDERS declaration order. include_unconfigured=True
        # so the picker can show the full provider universe (with the
        # setup-hint warning attached) instead of only authed rows.
        # Curated model lists are preserved — list_authenticated_providers
        # populates `models` from the curated catalog, not provider_model_ids
        # (which would pull non-agentic models like TTS/embeddings/etc.).
        payload = build_models_payload(
            ctx,
            include_unconfigured=True,
            picker_hints=True,
            canonical_order=True,
            pricing=True,
            capabilities=True,
            refresh=bool(params.get("refresh")),
        )
        return _ok(rid, payload)
    except Exception as e:
        return _err(rid, 5033, str(e))


@method("model.save_key")
def _(rid, params: dict) -> dict:
    """Save an API key for a provider, then return its refreshed model list.

    Params:
        slug: provider slug (e.g. "deepseek", "xai")
        api_key: the key value to save

    Returns the provider dict with models populated (same shape as
    model.options entries) on success.
    """
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY
        from hermes_cli.config import is_managed, save_env_value
        from hermes_cli.inventory import build_models_payload, load_picker_context

        slug = (params.get("slug") or "").strip()
        api_key = (params.get("api_key") or "").strip()
        if not slug or not api_key:
            return _err(rid, 4001, "slug and api_key are required")

        if is_managed():
            return _err(rid, 4006, "managed install — credentials are read-only")

        pconfig = PROVIDER_REGISTRY.get(slug)
        if not pconfig:
            return _err(rid, 4002, f"unknown provider: {slug}")
        if pconfig.auth_type != "api_key":
            return _err(
                rid,
                4003,
                f"{pconfig.name} uses {pconfig.auth_type} auth — "
                f"run `argus model` to configure",
            )
        if not pconfig.api_key_env_vars:
            return _err(rid, 4004, f"no env var defined for {pconfig.name}")

        # Save the key to ~/.argus/.env
        env_var = pconfig.api_key_env_vars[0]
        save_env_value(env_var, api_key)
        # Also set in current process so the refreshed inventory sees it.
        import os

        os.environ[env_var] = api_key

        # Refresh provider data via the shared inventory builder so this
        # surface stays in lock-step with model.options + dashboard
        # /api/model/options. picker_hints=True ensures the returned row
        # carries `authenticated` for the TUI frontend.
        session = _sessions.get(params.get("session_id", ""))
        agent = session.get("agent") if session else None
        ctx = load_picker_context().with_overrides(
            current_provider=getattr(agent, "provider", "") if agent else "",
            current_model=(
                (getattr(agent, "model", "") if agent else "") or _resolve_model()
            ),
            current_base_url=getattr(agent, "base_url", "") if agent else "",
        )
        payload = build_models_payload(
            ctx, picker_hints=True, max_models=50,
        )
        provider_data = next(
            (p for p in payload["providers"] if p["slug"] == slug), None
        )
        if provider_data is None:
            # Key was saved but provider didn't appear — still return success.
            provider_data = {
                "slug": slug,
                "name": pconfig.name,
                "is_current": False,
                "models": [],
                "total_models": 0,
                "authenticated": True,
            }
        # picker_hints sets `authenticated` from the row state, but the
        # synthetic fallback above doesn't go through that path.
        provider_data["authenticated"] = True
        return _ok(rid, {"provider": provider_data})
    except Exception as e:
        return _err(rid, 5034, str(e))


@method("model.disconnect")
def _(rid, params: dict) -> dict:
    """Remove credentials for a provider.

    Params:
        slug: provider slug (e.g. "deepseek", "xai")

    Returns success status and the provider's slug.
    """
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY, clear_provider_auth
        from hermes_cli.config import remove_env_value

        slug = (params.get("slug") or "").strip()
        if not slug:
            return _err(rid, 4001, "slug is required")

        pconfig = PROVIDER_REGISTRY.get(slug)
        cleared_env = False
        cleared_auth = False

        # Remove API key env vars from .env and process
        if pconfig and pconfig.api_key_env_vars:
            for ev in pconfig.api_key_env_vars:
                if remove_env_value(ev):
                    cleared_env = True

        # Clear OAuth / credential pool state
        cleared_auth = clear_provider_auth(slug)

        if not cleared_env and not cleared_auth:
            return _err(rid, 4005, f"no credentials found for {slug}")

        provider_name = pconfig.name if pconfig else slug
        return _ok(
            rid,
            {
                "slug": slug,
                "name": provider_name,
                "disconnected": True,
            },
        )
    except Exception as e:
        return _err(rid, 5035, str(e))


# ── Methods: slash.exec ──────────────────────────────────────────────


def _mirror_slash_side_effects(sid: str, session: dict, command: str) -> str:
    """Apply side effects that must also hit the gateway's live agent."""
    parts = command.lstrip("/").split(None, 1)
    if not parts:
        return ""
    name, arg, agent = (
        parts[0],
        (parts[1].strip() if len(parts) > 1 else ""),
        session.get("agent"),
    )

    # Reject agent-mutating commands during an in-flight turn.  These
    # all do read-then-mutate on live agent/session state that the
    # worker thread running agent.run_conversation is using.  Parity
    # with the session.compress / session.undo guards and the gateway
    # runner's running-agent /model guard.
    _MUTATES_WHILE_RUNNING = {"model", "personality", "prompt", "compress"}
    if name in _MUTATES_WHILE_RUNNING and session.get("running"):
        return f"session busy — /interrupt the current turn before running /{name}"

    try:
        if name == "model" and arg and agent:
            result = _apply_model_switch(sid, session, arg)
            return result.get("warning", "")
        elif name == "personality" and arg and agent:
            pname, new_prompt = _validate_personality(arg, _load_cfg())
            _apply_personality_to_session(sid, session, new_prompt, pname)
        elif name == "prompt" and agent:
            cfg = _load_cfg()
            new_prompt = _prompt_text((cfg.get("agent") or {}).get("system_prompt", ""))
            agent.ephemeral_system_prompt = new_prompt or None
            agent._cached_system_prompt = None
        elif name == "compress" and agent:
            # Mirror the session.compress RPC: build a before/after summary so
            # the user gets feedback (#46686). The slash path previously just
            # compressed + emitted session.info and returned "", so the TUI
            # showed no "compressed N → M messages / ~X → ~Y tokens" stats
            # while CLI and gateway both did.
            from agent.manual_compression_feedback import summarize_manual_compression
            from agent.model_metadata import estimate_request_tokens_rough

            with session["history_lock"]:
                _before_messages = list(session.get("history", []))
            _before_count = len(_before_messages)
            _sys_prompt = getattr(agent, "_cached_system_prompt", "") or ""
            _tools = getattr(agent, "tools", None) or None
            _before_tokens = (
                estimate_request_tokens_rough(
                    _before_messages, system_prompt=_sys_prompt, tools=_tools
                )
                if _before_count
                else 0
            )

            _compress_session_history(session, arg)
            _sync_session_key_after_compress(sid, session)

            with session["history_lock"]:
                _after_messages = list(session.get("history", []))
            _sys_prompt_after = getattr(agent, "_cached_system_prompt", "") or _sys_prompt
            _tools_after = getattr(agent, "tools", None) or _tools
            _after_tokens = (
                estimate_request_tokens_rough(
                    _after_messages, system_prompt=_sys_prompt_after, tools=_tools_after
                )
                if _after_messages
                else 0
            )
            _emit("session.info", sid, _session_info(agent, session))
            _fb = summarize_manual_compression(
                _before_messages, _after_messages, _before_tokens, _after_tokens
            )
            _lines = [_fb["headline"], _fb["token_line"]]
            if _fb.get("note"):
                _lines.append(_fb["note"])
            return "\n".join(_lines)
        elif name == "fast" and agent:
            mode = arg.lower()
            if mode in {"fast", "on"}:
                agent.service_tier = "priority"
            elif mode in {"normal", "off"}:
                agent.service_tier = None
            _emit("session.info", sid, _session_info(agent, session))
        elif name == "reload-mcp" and agent and hasattr(agent, "reload_mcp_tools"):
            agent.reload_mcp_tools()
        elif name == "stop":
            from tools.process_registry import process_registry

            process_registry.kill_all()
    except Exception as e:
        return f"live session sync failed: {e}"
    return ""


@method("slash.exec")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err

    cmd = params.get("command", "").strip()
    if not cmd:
        return _err(rid, 4004, "empty command")

    # Skill slash commands and _pending_input commands must NOT go through the
    # slash worker — see _PENDING_INPUT_COMMANDS definition above. Plugin
    # commands must also avoid the worker, but unlike skills/pending-input they
    # still return normal slash.exec output so the TUI keeps the pager path.
    _cmd_text = cmd.lstrip("/") if cmd.startswith("/") else cmd
    _cmd_parts = _cmd_text.split(maxsplit=1)
    _cmd_base = (_cmd_parts[0] if _cmd_parts else "").lower()
    _cmd_arg = _cmd_parts[1] if len(_cmd_parts) > 1 else ""

    if _cmd_base in _PENDING_INPUT_COMMANDS:
        # Route directly to command.dispatch instead of returning an error
        # that requires the frontend to retry.  Some TUI clients fail the
        # fallback, leaving the command empty and showing "empty command".
        return _methods["command.dispatch"](
            rid,
            {
                "name": _cmd_base,
                "arg": _cmd_arg,
                "session_id": params.get("session_id", ""),
            },
        )

    if _cmd_base in _WORKER_BLOCKED_COMMANDS:
        subcommand = _cmd_arg.split(maxsplit=1)[0].lower() if _cmd_arg else ""
        if subcommand in {"restore", "rewind"}:
            return _err(
                rid,
                4018,
                "snapshot restore mutates live config/state; use command.dispatch for /snapshot restore",
            )

    try:
        from agent.skill_commands import get_skill_commands

        _cmd_key = f"/{_cmd_base}"
        if _cmd_key in get_skill_commands():
            return _err(
                rid, 4018, f"skill command: use command.dispatch for {_cmd_key}"
            )
    except Exception:
        pass

    plugin_handler = None
    resolve_plugin_command_result = None
    if _cmd_base:
        try:
            from hermes_cli.plugins import (
                get_plugin_command_handler,
                resolve_plugin_command_result,
            )

            plugin_handler = get_plugin_command_handler(_cmd_base)
        except Exception:
            plugin_handler = None
            resolve_plugin_command_result = None

    if plugin_handler and resolve_plugin_command_result:
        try:
            result = resolve_plugin_command_result(plugin_handler(_cmd_arg))
            return _ok(rid, {"output": str(result or "(no output)")})
        except Exception as e:
            return _ok(rid, {"output": f"Plugin command error: {e}"})

    worker = session.get("slash_worker")
    if not worker:
        try:
            worker = _SlashWorker(
                session["session_key"],
                getattr(session.get("agent"), "model", _resolve_model()),
            )
            _attach_worker(params.get("session_id", ""), session, worker)
        except Exception as e:
            return _err(rid, 5030, f"slash worker start failed: {e}")

    try:
        output = worker.run(cmd)
        warning = _mirror_slash_side_effects(params.get("session_id", ""), session, cmd)
        payload = {"output": output or "(no output)"}
        if warning:
            payload["warning"] = warning
        return _ok(rid, payload)
    except Exception as e:
        try:
            worker.close()
        except Exception:
            pass
        session["slash_worker"] = None
        return _err(rid, 5030, str(e))


# ── Methods: voice ───────────────────────────────────────────────────


_voice_sid_lock = threading.Lock()
_voice_event_sid: str = ""


def _voice_emit(event: str, payload: dict | None = None) -> None:
    """Emit a voice event toward the session that most recently turned the
    mode on. Voice is process-global (one microphone), so there's only ever
    one sid to target; the TUI handler treats an empty sid as "active
    session". Kept separate from _emit to make the lack of per-call sid
    argument explicit."""
    with _voice_sid_lock:
        sid = _voice_event_sid
    _emit(event, sid, payload)


def _voice_mode_enabled() -> bool:
    """Current voice-mode flag (runtime-only, CLI parity).

    cli.py initialises ``_voice_mode = False`` at startup and only flips
    it via ``/voice on``; it never reads a persisted enable bit from
    config.yaml.  We match that: no config lookup, env var only.  This
    avoids the TUI auto-starting in REC the next time the user opens it
    just because they happened to enable voice in a prior session.
    """
    return os.environ.get("ARGUS_VOICE", "").strip() == "1"


def _voice_tts_enabled() -> bool:
    """Whether agent replies should be spoken back via TTS (runtime only)."""
    return os.environ.get("ARGUS_VOICE_TTS", "").strip() == "1"


def _voice_cfg_dict() -> dict:
    """Shape-safe accessor for the ``voice:`` block in config.yaml.

    ``_load_cfg()`` returns raw ``yaml.safe_load()`` output, so both the
    root AND ``voice`` may be any YAML scalar / list / None. A hand-edit
    like ``voice: true`` or a malformed top-level config that parses to
    a scalar would otherwise break ``.get("…")`` and take every
    ``voice.*`` branch down with it (Copilot round-3..7 review on
    #19835). Coerce through ``isinstance`` at every level so malformed
    config falls back to an empty dict instead of crashing /voice.
    """
    cfg = _load_cfg()
    voice_cfg = cfg.get("voice") if isinstance(cfg, dict) else None

    return voice_cfg if isinstance(voice_cfg, dict) else {}


def _voice_record_key() -> str:
    """Current ``voice.record_key`` value, documented default on error."""
    record_key = _voice_cfg_dict().get("record_key")

    return str(record_key) if isinstance(record_key, str) and record_key else "ctrl+b"


@method("voice.toggle")
def _(rid, params: dict) -> dict:
    """CLI parity for the ``/voice`` slash command.

    Subcommands:

    * ``status`` — report mode + TTS flags (default when action is unknown).
    * ``on`` / ``off`` — flip voice *mode* (the umbrella bit). Turning it
      off also tears down any active continuous recording loop. Does NOT
      start recording on its own; recording is driven by ``voice.record``
      (Ctrl+B) after mode is on, matching cli.py's enable/Ctrl+B split.
    * ``tts`` — toggle speech-output of agent replies. Requires mode on
      (mirrors CLI's _toggle_voice_tts guard).
    """
    action = params.get("action", "status")

    if action == "status":
        # Mirror CLI's _show_voice_status: include STT/TTS provider
        # availability so the user can tell at a glance *why* voice mode
        # isn't working ("STT provider: MISSING ..." is the common case).
        # ``record_key`` mirrors the configured ``voice.record_key`` so the
        # TUI can both bind it (frontend ``isVoiceToggleKey``) and display
        # it in /voice status — previously the TUI hardcoded Ctrl+B and
        # ignored the config (#18994).
        payload: dict = {
            "enabled": _voice_mode_enabled(),
            "record_key": _voice_record_key(),
            "tts": _voice_tts_enabled(),
        }
        try:
            from tools.voice_mode import check_voice_requirements

            reqs = check_voice_requirements()
            payload["available"] = bool(reqs.get("available"))
            payload["audio_available"] = bool(reqs.get("audio_available"))
            payload["stt_available"] = bool(reqs.get("stt_available"))
            payload["details"] = reqs.get("details") or ""
        except Exception as e:
            # check_voice_requirements pulls optional transcription deps —
            # swallow so /voice status always returns something useful.
            logger.warning("voice.toggle status: requirements probe failed: %s", e)

        return _ok(rid, payload)

    if action in {"on", "off"}:
        enabled = action == "on"
        # Runtime-only flag (CLI parity) — no _write_config_key, so the
        # next TUI launch starts with voice OFF instead of auto-REC from a
        # persisted stale toggle.
        os.environ["ARGUS_VOICE"] = "1" if enabled else "0"

        if not enabled:
            # Disabling the mode must tear the continuous loop down; the
            # loop holds the microphone and would otherwise keep running.
            try:
                from hermes_cli.voice import stop_continuous

                stop_continuous()
            except ImportError:
                pass
            except Exception as e:
                logger.warning("voice: stop_continuous failed during toggle off: %s", e)

            # Clear TTS so it can be toggled independently after voice is off.
            os.environ["ARGUS_VOICE_TTS"] = "0"

        return _ok(
            rid,
            {
                "enabled": enabled,
                "record_key": _voice_record_key(),
                "tts": _voice_tts_enabled(),
            },
        )

    if action == "tts":
        if not _voice_mode_enabled():
            return _err(rid, 4014, "enable voice mode first: /voice on")
        new_value = not _voice_tts_enabled()
        # Runtime-only flag (CLI parity) — see voice.toggle on/off above.
        os.environ["ARGUS_VOICE_TTS"] = "1" if new_value else "0"
        # Include ``record_key`` on every branch so a /voice tts toggle
        # doesn't reset the TUI's cached shortcut to the default when a
        # user has a custom binding configured (Copilot review, round 2
        # on #19835). Keeps parity with the status/on/off branches above.
        return _ok(
            rid,
            {
                "enabled": True,
                "record_key": _voice_record_key(),
                "tts": new_value,
            },
        )

    return _err(rid, 4013, f"unknown voice action: {action}")


@method("voice.record")
def _(rid, params: dict) -> dict:
    """VAD-bounded push-to-talk capture, CLI-parity.

    ``start`` begins one VAD-bounded capture and emits ``voice.transcript``
    after silence stops the recorder. ``stop`` forces transcription of the
    active buffer, matching classic CLI push-to-talk. The voice wrapper retains
    no-speech counts across single-shot starts, so three consecutive silent
    captures emit ``voice.transcript`` with ``no_speech_limit=True``.
    """
    action = params.get("action", "start")

    if action not in {"start", "stop"}:
        return _err(rid, 4019, f"unknown voice action: {action}")

    try:
        if action == "start":
            if not _voice_mode_enabled():
                return _err(rid, 4015, "voice mode is off — enable with /voice on")

            with _voice_sid_lock:
                global _voice_event_sid
                _voice_event_sid = params.get("session_id") or _voice_event_sid

            from hermes_cli.voice import start_continuous

            # Shape-safe lookups: malformed ``voice:`` YAML (bool/scalar/list)
            # must not crash /voice with a 5025 — fall back to VAD defaults.
            #
            # Exclude ``bool`` from the numeric check since Python's bool is
            # a subclass of int — a hand-edit like ``silence_threshold: true``
            # would otherwise forward as ``1`` instead of falling back to
            # the documented 200 / 3.0 defaults (Copilot round-12 on #19835).
            voice_cfg = _voice_cfg_dict()
            threshold = voice_cfg.get("silence_threshold")
            duration = voice_cfg.get("silence_duration")
            safe_threshold = (
                threshold
                if isinstance(threshold, (int, float))
                and not isinstance(threshold, bool)
                else 200
            )
            safe_duration = (
                duration
                if isinstance(duration, (int, float)) and not isinstance(duration, bool)
                else 3.0
            )
            started = start_continuous(
                on_transcript=lambda t: _voice_emit("voice.transcript", {"text": t}),
                on_status=lambda s: _voice_emit("voice.status", {"state": s}),
                on_silent_limit=lambda: _voice_emit(
                    "voice.transcript", {"no_speech_limit": True}
                ),
                silence_threshold=safe_threshold,
                silence_duration=safe_duration,
                auto_restart=False,
            )
            if started is False:
                return _ok(rid, {"status": "busy"})
            return _ok(rid, {"status": "recording"})

        # action == "stop"
        with _voice_sid_lock:
            _voice_event_sid = params.get("session_id") or _voice_event_sid

        from hermes_cli.voice import stop_continuous

        stop_continuous(force_transcribe=True)
        return _ok(rid, {"status": "stopped"})
    except ImportError:
        return _err(
            rid, 5025, "voice module not available — install audio dependencies"
        )
    except Exception as e:
        return _err(rid, 5025, str(e))


@method("voice.tts")
def _(rid, params: dict) -> dict:
    text = params.get("text", "")
    if not text:
        return _err(rid, 4020, "text required")
    try:
        from hermes_cli.voice import speak_text

        threading.Thread(target=speak_text, args=(text,), daemon=True).start()
        return _ok(rid, {"status": "speaking"})
    except ImportError:
        return _err(rid, 5026, "voice module not available")
    except Exception as e:
        return _err(rid, 5026, str(e))


# ── Methods: insights ────────────────────────────────────────────────


@method("insights.get")
def _(rid, params: dict) -> dict:
    days = params.get("days", 30)
    db = _get_db()
    if db is None:
        return _db_unavailable_error(rid, code=5017)
    try:
        cutoff = time.time() - days * 86400
        rows = [
            s
            for s in db.list_sessions_rich(limit=500)
            if (s.get("started_at") or 0) >= cutoff
        ]
        return _ok(
            rid,
            {
                "days": days,
                "sessions": len(rows),
                "messages": sum(s.get("message_count", 0) for s in rows),
            },
        )
    except Exception as e:
        return _err(rid, 5017, str(e))


# ── Methods: rollback ────────────────────────────────────────────────


@method("rollback.list")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    try:

        def go(mgr, cwd):
            if not mgr.enabled:
                return _ok(rid, {"enabled": False, "checkpoints": []})
            return _ok(
                rid,
                {
                    "enabled": True,
                    "checkpoints": [
                        {
                            "hash": c.get("hash", ""),
                            "timestamp": c.get("timestamp", ""),
                            "message": c.get("message", ""),
                        }
                        for c in mgr.list_checkpoints(cwd)
                    ],
                },
            )

        return _with_checkpoints(session, go)
    except Exception as e:
        return _err(rid, 5020, str(e))


@method("rollback.restore")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    target = params.get("hash", "")
    file_path = params.get("file_path", "")
    if not target:
        return _err(rid, 4014, "hash required")
    # Full-history rollback mutates session history.  Rejecting during
    # an in-flight turn prevents prompt.submit from silently dropping
    # the agent's output (version mismatch path) or clobbering the
    # rollback (version-matches path).  A file-scoped rollback only
    # touches disk, so we allow it.
    if not file_path and session.get("running"):
        return _err(
            rid,
            4009,
            "session busy — /interrupt the current turn before full rollback.restore",
        )
    try:

        def go(mgr, cwd):
            resolved = _resolve_checkpoint_hash(mgr, cwd, target)
            result = mgr.restore(cwd, resolved, file_path=file_path or None)
            if result.get("success") and not file_path:
                removed = 0
                with session["history_lock"]:
                    history = session.get("history", [])
                    while history and history[-1].get("role") in {"assistant", "tool"}:
                        history.pop()
                        removed += 1
                    if history and history[-1].get("role") == "user":
                        history.pop()
                        removed += 1
                    if removed:
                        session["history_version"] = (
                            int(session.get("history_version", 0)) + 1
                        )
                result["history_removed"] = removed
            return result

        return _ok(rid, _with_checkpoints(session, go))
    except Exception as e:
        return _err(rid, 5021, str(e))


@method("rollback.diff")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    target = params.get("hash", "")
    if not target:
        return _err(rid, 4014, "hash required")
    try:
        r = _with_checkpoints(
            session,
            lambda mgr, cwd: mgr.diff(cwd, _resolve_checkpoint_hash(mgr, cwd, target)),
        )
        raw = r.get("diff", "")[:4000]
        payload = {"stat": r.get("stat", ""), "diff": raw}
        rendered = render_diff(raw, session.get("cols", 80))
        if rendered:
            payload["rendered"] = rendered
        return _ok(rid, payload)
    except Exception as e:
        return _err(rid, 5022, str(e))


# ── Methods: browser / plugins / cron / skills ───────────────────────


def _resolve_browser_cdp_url() -> str:
    """Return the configured browser CDP override without network I/O.

    ``/browser status`` must be fast — calling
    ``tools.browser_tool._get_cdp_override`` would invoke
    ``_resolve_cdp_override``, which performs an HTTP probe to
    ``.../json/version`` for discovery-style URLs.  That probe has
    a multi-second timeout and would block the TUI on a slow or
    unreachable host even though status only needs to report whether
    an override is set.

    Mirrors the env/config precedence of ``_get_cdp_override`` (env
    var first, then ``browser.cdp_url`` from config.yaml) without the
    websocket-resolution step, so the answer reflects user intent
    even when the configured host is not currently reachable.  The
    actual WS normalization happens in ``browser_navigate`` on the
    next tool call.
    """
    env_url = os.environ.get("BROWSER_CDP_URL", "").strip()
    if env_url:
        return env_url
    try:
        from hermes_cli.config import read_raw_config

        cfg = read_raw_config()
        browser_cfg = cfg.get("browser", {}) if isinstance(cfg, dict) else {}
        if isinstance(browser_cfg, dict):
            return str(browser_cfg.get("cdp_url", "") or "").strip()
    except Exception:
        pass
    return ""


def _is_default_local_cdp(parsed) -> bool:
    """Match the discovery-style local default; never the concrete WS form.

    A user-supplied ``ws://127.0.0.1:9222/devtools/browser/<id>`` is a
    real, connectable endpoint — collapsing it to bare ``http://...:9222``
    would strip the path and break the connect.
    """
    try:
        port = parsed.port or 80
    except ValueError:
        return False

    discovery_path = parsed.path in {"", "/", "/json", "/json/version"}
    return (
        parsed.scheme in {"http", "ws"}
        and parsed.hostname in {"127.0.0.1", "localhost"}
        and port == 9222
        and discovery_path
    )


def _http_ok(url: str, timeout: float) -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= getattr(resp, "status", 200) < 300
    except Exception:
        return False


def _probe_urls(parsed) -> list[str]:
    scheme = {"ws": "http", "wss": "https"}.get(parsed.scheme, parsed.scheme)
    root = f"{scheme}://{parsed.netloc}".rstrip("/")
    return [f"{root}/json/version", f"{root}/json"]


def _normalize_cdp_url(parsed) -> str:
    # Concrete ``/devtools/browser/<id>`` endpoints (Browserbase et al.)
    # are connectable as-is. Discovery-style inputs collapse to bare
    # ``scheme://host:port`` so ``_resolve_cdp_override`` can append
    # ``/json/version`` later without doubling the path.
    if parsed.path.startswith("/devtools/browser/"):
        return parsed.geturl()
    return parsed._replace(path="", params="", query="", fragment="").geturl()


def _failure_messages(url: str, port: int, system: str) -> list[str]:
    from hermes_cli.browser_connect import manual_chrome_debug_command

    command = manual_chrome_debug_command(port, system)
    hint = (
        ["Start a Chromium-family browser with remote debugging, then retry /browser connect:", command]
        if command
        else [
            "No supported Chromium-family browser executable was found in this environment.",
            f"Install one or start a Chromium-family browser with --remote-debugging-port={port}, then retry /browser connect.",
        ]
    )
    return [
        f"Browser CDP is not reachable at {url}.",
        *hint,
        "Browser not connected — start a Chromium-family browser with remote debugging and retry /browser connect",
    ]


@method("browser.manage")
def _(rid, params: dict) -> dict:
    action = params.get("action", "status")

    if action == "status":
        url = _resolve_browser_cdp_url()
        return _ok(rid, {"connected": bool(url), "url": url})

    if action == "disconnect":
        return _browser_disconnect(rid)

    if action != "connect":
        return _err(rid, 4015, f"unknown action: {action}")

    return _browser_connect(rid, params)


def _browser_connect(rid, params: dict) -> dict:
    import platform

    from hermes_cli.browser_connect import DEFAULT_BROWSER_CDP_URL
    from tools.browser_tool import cleanup_all_browsers
    from urllib.parse import urlparse

    raw_url = params.get("url")
    if raw_url is not None and not isinstance(raw_url, str):
        return _err(
            rid, 4015, f"browser url must be a string, got {type(raw_url).__name__}"
        )
    url = (raw_url or "").strip() or DEFAULT_BROWSER_CDP_URL

    sid = params.get("session_id") or ""
    system = platform.system()
    messages: list[str] = []

    def announce(message: str, *, level: str = "info") -> None:
        messages.append(message)
        # Without a session id the TUI prints `messages` from the
        # response; emitting an event would double-render. Only stream
        # progress when there's a real session to scope it to.
        if sid:
            _emit("browser.progress", sid, {"message": message, "level": level})

    parsed = urlparse(url if "://" in url else f"http://{url}")
    if parsed.scheme not in {"http", "https", "ws", "wss"}:
        return _err(rid, 4015, f"unsupported browser url: {url}")
    if not parsed.hostname:
        return _err(rid, 4015, f"missing host in browser url: {url}")
    try:
        port = parsed.port or (443 if parsed.scheme in {"https", "wss"} else 80)
    except ValueError:
        return _err(rid, 4015, f"invalid port in browser url: {url}")

    # Always normalize default-local to 127.0.0.1:9222 so downstream
    # comparisons + messaging match what we'll actually persist.
    if _is_default_local_cdp(parsed):
        url = DEFAULT_BROWSER_CDP_URL
        parsed = urlparse(url)
        port = parsed.port or 9222

    try:
        # ws[s]://.../devtools/browser/<id> endpoints (hosted CDP
        # providers) don't serve the HTTP discovery path; just check
        # TCP-level reachability and let browser_navigate handshake.
        if parsed.scheme in {"ws", "wss"} and parsed.path.startswith(
            "/devtools/browser/"
        ):
            import socket

            try:
                with socket.create_connection((parsed.hostname, port), timeout=2.0):
                    pass
            except OSError as e:
                return _err(rid, 5031, f"could not reach browser CDP at {url}: {e}")
        else:
            probes = _probe_urls(parsed)
            ok = any(_http_ok(p, timeout=2.0) for p in probes)

            if not ok and _is_default_local_cdp(parsed):
                from hermes_cli.browser_connect import try_launch_chrome_debug

                announce(
                    "Chromium-family browser isn't running with remote debugging — attempting to launch..."
                )

                if try_launch_chrome_debug(port, system):
                    for _ in range(20):
                        time.sleep(0.5)
                        if any(_http_ok(p, timeout=1.0) for p in probes):
                            ok = True
                            break

                if ok:
                    announce(f"Chromium-family browser launched and listening on port {port}")
                else:
                    for line in _failure_messages(url, port, system)[1:]:
                        announce(line, level="error")
                    return _ok(
                        rid, {"connected": False, "url": url, "messages": messages}
                    )
            elif not ok:
                return _err(rid, 5031, f"could not reach browser CDP at {url}")
            elif _is_default_local_cdp(parsed):
                announce(f"Chromium-family browser is already listening on port {port}")

        normalized = _normalize_cdp_url(parsed)

        # Order matters: reap sessions BEFORE publishing the new env
        # so an in-flight tool call sees the old supervisor closed,
        # then again AFTER so the default task's cached supervisor
        # is drained against the new URL.
        cleanup_all_browsers()
        os.environ["BROWSER_CDP_URL"] = normalized
        cleanup_all_browsers()
    except Exception as e:
        return _err(rid, 5031, str(e))

    payload: dict[str, object] = {"connected": True, "url": normalized}
    if messages:
        payload["messages"] = messages
    return _ok(rid, payload)


def _browser_disconnect(rid) -> dict:
    # Reap, drop the env override, reap again — closes the same swap
    # window covered by ``_browser_connect``.
    def reap() -> None:
        try:
            from tools.browser_tool import cleanup_all_browsers

            cleanup_all_browsers()
        except Exception:
            pass

    reap()
    os.environ.pop("BROWSER_CDP_URL", None)
    reap()
    return _ok(rid, {"connected": False})


@method("plugins.list")
def _(rid, params: dict) -> dict:
    try:
        from hermes_cli.plugins import get_plugin_manager

        return _ok(
            rid,
            {
                "plugins": [
                    {
                        "name": n,
                        "version": getattr(i, "version", "?"),
                        "enabled": getattr(i, "enabled", True),
                    }
                    for n, i in get_plugin_manager()._plugins.items()
                ]
            },
        )
    except Exception as e:
        return _err(rid, 5032, str(e))


@method("config.show")
def _(rid, params: dict) -> dict:
    try:
        cfg = _load_cfg()
        model = _resolve_model()
        api_key = os.environ.get("ARGUS_API_KEY", "") or cfg.get("api_key", "")
        masked = f"****{api_key[-4:]}" if len(api_key) > 4 else "(not set)"
        base_url = os.environ.get("ARGUS_BASE_URL", "") or cfg.get("base_url", "")

        sections = [
            {
                "title": "Model",
                "rows": [
                    ["Model", model],
                    ["Base URL", base_url or "(default)"],
                    ["API Key", masked],
                ],
            },
            {
                "title": "Agent",
                "rows": [
                    ["Max Turns", str(_cfg_max_turns(cfg, 90))],
                    ["Toolsets", ", ".join(cfg.get("enabled_toolsets", [])) or "all"],
                    ["Verbose", str(cfg.get("verbose", False))],
                ],
            },
            {
                "title": "Environment",
                "rows": [
                    ["Working Dir", os.getcwd()],
                    ["Config File", str(_hermes_home / "config.yaml")],
                ],
            },
        ]
        return _ok(rid, {"sections": sections})
    except Exception as e:
        return _err(rid, 5030, str(e))


@method("tools.list")
def _(rid, params: dict) -> dict:
    try:
        from toolsets import get_all_toolsets, get_toolset_info

        session = _sessions.get(params.get("session_id", ""))
        enabled = (
            set(getattr(session["agent"], "enabled_toolsets", []) or [])
            if session
            else set(_load_enabled_toolsets() or [])
        )

        items = []
        for name in sorted(get_all_toolsets().keys()):
            info = get_toolset_info(name)
            if not info:
                continue
            items.append(
                {
                    "name": name,
                    "description": info["description"],
                    "tool_count": info["tool_count"],
                    "enabled": name in enabled if enabled else True,
                    "tools": info["resolved_tools"],
                }
            )
        return _ok(rid, {"toolsets": items})
    except Exception as e:
        return _err(rid, 5031, str(e))


@method("tools.show")
def _(rid, params: dict) -> dict:
    try:
        from model_tools import get_toolset_for_tool, get_tool_definitions

        session = _sessions.get(params.get("session_id", ""))
        enabled = (
            getattr(session["agent"], "enabled_toolsets", None)
            if session
            else _load_enabled_toolsets()
        )
        tools = get_tool_definitions(enabled_toolsets=enabled, quiet_mode=True)
        sections = {}

        for tool in sorted(tools, key=lambda t: t["function"]["name"]):
            name = tool["function"]["name"]
            desc = str(tool["function"].get("description", "") or "").split("\n")[0]
            if ". " in desc:
                desc = desc[: desc.index(". ") + 1]
            sections.setdefault(get_toolset_for_tool(name) or "unknown", []).append(
                {
                    "name": name,
                    "description": desc,
                }
            )

        return _ok(
            rid,
            {
                "sections": [
                    {"name": name, "tools": rows}
                    for name, rows in sorted(sections.items())
                ],
                "total": len(tools),
            },
        )
    except Exception as e:
        return _err(rid, 5034, str(e))


@method("tools.configure")
def _(rid, params: dict) -> dict:
    action = str(params.get("action", "") or "").strip().lower()
    targets = [
        str(name).strip() for name in params.get("names", []) or [] if str(name).strip()
    ]
    if action not in {"disable", "enable"}:
        return _err(rid, 4017, f"unknown tools action: {action}")
    if not targets:
        return _err(rid, 4018, "names required")

    try:
        from hermes_cli.config import load_config, save_config
        from hermes_cli.tools_config import (
            CONFIGURABLE_TOOLSETS,
            _apply_mcp_change,
            _apply_toolset_change,
            _get_platform_tools,
            _get_plugin_toolset_keys,
        )

        cfg = load_config()
        valid_toolsets = {
            ts_key for ts_key, _, _ in CONFIGURABLE_TOOLSETS
        } | _get_plugin_toolset_keys()
        toolset_targets = [name for name in targets if ":" not in name]
        mcp_targets = [name for name in targets if ":" in name]
        unknown = [name for name in toolset_targets if name not in valid_toolsets]
        toolset_targets = [name for name in toolset_targets if name in valid_toolsets]

        if toolset_targets:
            _apply_toolset_change(cfg, "cli", toolset_targets, action)

        missing_servers = (
            _apply_mcp_change(cfg, mcp_targets, action) if mcp_targets else set()
        )
        save_config(cfg)

        session = _sessions.get(params.get("session_id", ""))
        info = (
            _reset_session_agent(params.get("session_id", ""), session)
            if session
            else None
        )
        enabled = sorted(
            _get_platform_tools(load_config(), "cli", include_default_mcp_servers=False)
        )
        changed = [
            name
            for name in targets
            if name not in unknown
            and (":" not in name or name.split(":", 1)[0] not in missing_servers)
        ]

        return _ok(
            rid,
            {
                "changed": changed,
                "enabled_toolsets": enabled,
                "info": info,
                "missing_servers": sorted(missing_servers),
                "reset": bool(session),
                "unknown": unknown,
            },
        )
    except Exception as e:
        return _err(rid, 5035, str(e))


@method("toolsets.list")
def _(rid, params: dict) -> dict:
    try:
        from toolsets import get_all_toolsets, get_toolset_info

        session = _sessions.get(params.get("session_id", ""))
        enabled = (
            set(getattr(session["agent"], "enabled_toolsets", []) or [])
            if session
            else set(_load_enabled_toolsets() or [])
        )

        items = []
        for name in sorted(get_all_toolsets().keys()):
            info = get_toolset_info(name)
            if not info:
                continue
            items.append(
                {
                    "name": name,
                    "description": info["description"],
                    "tool_count": info["tool_count"],
                    "enabled": name in enabled if enabled else True,
                }
            )
        return _ok(rid, {"toolsets": items})
    except Exception as e:
        return _err(rid, 5032, str(e))


@method("agents.list")
def _(rid, params: dict) -> dict:
    try:
        from tools.process_registry import process_registry

        procs = process_registry.list_sessions()
        return _ok(
            rid,
            {
                "processes": [
                    {
                        "session_id": p["session_id"],
                        "command": p["command"][:80],
                        "status": p["status"],
                        "uptime": p["uptime_seconds"],
                    }
                    for p in procs
                ]
            },
        )
    except Exception as e:
        return _err(rid, 5033, str(e))


@method("cron.manage")
def _(rid, params: dict) -> dict:
    action, jid = params.get("action", "list"), params.get("name", "")
    try:
        from tools.cronjob_tools import cronjob

        if action == "list":
            return _ok(rid, json.loads(cronjob(action="list")))
        if action == "add":
            return _ok(
                rid,
                json.loads(
                    cronjob(
                        action="create",
                        name=jid,
                        schedule=params.get("schedule", ""),
                        prompt=params.get("prompt", ""),
                    )
                ),
            )
        if action in {"remove", "pause", "resume"}:
            return _ok(rid, json.loads(cronjob(action=action, job_id=jid)))
        return _err(rid, 4016, f"unknown cron action: {action}")
    except Exception as e:
        return _err(rid, 5023, str(e))


@method("skills.manage")
def _(rid, params: dict) -> dict:
    action, query = params.get("action", "list"), params.get("query", "")
    try:
        if action == "list":
            from hermes_cli.banner import get_available_skills

            return _ok(rid, {"skills": get_available_skills()})
        if action == "search":
            from tools.skills_hub import (
                GitHubAuth,
                create_source_router,
                unified_search,
            )

            raw = (
                unified_search(
                    query,
                    create_source_router(GitHubAuth()),
                    source_filter="all",
                    limit=20,
                )
                or []
            )
            return _ok(
                rid,
                {
                    "results": [
                        {"name": r.name, "description": r.description} for r in raw
                    ]
                },
            )
        if action == "install":
            from hermes_cli.skills_hub import do_install

            class _Q:
                def print(self, *a, **k):
                    pass

            do_install(query, skip_confirm=True, console=_Q())
            return _ok(rid, {"installed": True, "name": query})
        if action == "browse":
            from hermes_cli.skills_hub import browse_skills

            pg = int(params.get("page", 0) or 0) or (
                int(query) if query.isdigit() else 1
            )
            return _ok(
                rid, browse_skills(page=pg, page_size=int(params.get("page_size", 20)))
            )
        if action == "inspect":
            from hermes_cli.skills_hub import inspect_skill

            return _ok(rid, {"info": inspect_skill(query) or {}})
        return _err(rid, 4017, f"unknown skills action: {action}")
    except Exception as e:
        return _err(rid, 5024, str(e))


@method("skills.reload")
def _(rid, params: dict) -> dict:
    try:
        from agent.skill_commands import reload_skills

        result = reload_skills()
        added = result.get("added") or []
        removed = result.get("removed") or []
        total = int(result.get("total") or 0)

        lines = ["Reloading skills..."]
        if not added and not removed:
            lines.append("No new skills detected.")
        if added:
            lines.append("Added skills:")
            lines.extend(f"  - {item.get('name', '')}" for item in added)
        if removed:
            lines.append("Removed skills:")
            lines.extend(f"  - {item.get('name', '')}" for item in removed)
        lines.append(f"{total} skill(s) available")
        return _ok(rid, {"output": "\n".join(lines), "result": result})
    except Exception as e:
        return _err(rid, 5025, str(e))


@method("plugins.manage")
def _(rid, params: dict) -> dict:
    """List installed plugins with activation state, or toggle one on/off.

    Backs the TUI Plugins Hub. Uses the same disk-discovery + enable/disable
    primitives as ``hermes plugins`` / the dashboard, so the three surfaces
    agree on what's installed and what's enabled.

    Actions:
      - ``list``   → {"plugins": [{name, version, description, source,
                       status}], "user_count": N, "bundled_count": M}
      - ``toggle`` → flip ``name`` based on ``enable`` (bool). Returns the
                       refreshed row plus {"ok", "unchanged"}.
    """
    action = params.get("action", "list")
    try:
        from hermes_cli.plugins_cmd import (
            _discover_all_plugins,
            _get_disabled_set,
            _get_enabled_set,
            _plugin_status,
        )

        def _rows():
            enabled = _get_enabled_set()
            disabled = _get_disabled_set()
            out = []
            for name, version, desc, source, _dir, key in sorted(
                _discover_all_plugins()
            ):
                out.append(
                    {
                        "name": name,
                        "version": str(version or ""),
                        "description": desc or "",
                        "source": source,
                        "status": _plugin_status(name, enabled, disabled, key=key),
                    }
                )
            return out

        if action == "list":
            rows = _rows()
            user_count = sum(1 for r in rows if r["source"] != "bundled")
            return _ok(
                rid,
                {
                    "plugins": rows,
                    "user_count": user_count,
                    "bundled_count": len(rows) - user_count,
                },
            )

        if action == "toggle":
            from hermes_cli.plugins_cmd import dashboard_set_agent_plugin_enabled

            name = (params.get("name") or "").strip()
            if not name:
                return _err(rid, 4019, "plugins.toggle requires a 'name'")
            enable = bool(params.get("enable"))
            result = dashboard_set_agent_plugin_enabled(name, enabled=enable)
            if not result.get("ok"):
                return _err(rid, 5026, result.get("error") or "toggle failed")
            row = next((r for r in _rows() if r["name"] == name), None)
            return _ok(
                rid,
                {
                    "ok": True,
                    "unchanged": bool(result.get("unchanged")),
                    "name": name,
                    "plugin": row,
                },
            )

        return _err(rid, 4017, f"unknown plugins action: {action}")
    except Exception as e:
        return _err(rid, 5026, str(e))


@method("shell.exec")
def _(rid, params: dict) -> dict:
    cmd = params.get("command", "")
    if not cmd:
        return _err(rid, 4004, "empty command")
    try:
        from tools.approval import detect_dangerous_command, detect_hardline_command

        is_hardline, hardline_desc = detect_hardline_command(cmd)
        if is_hardline:
            return _err(
                rid, 4005, f"blocked (hardline): {hardline_desc}. Use the agent for dangerous commands."
            )
        is_dangerous, _, desc = detect_dangerous_command(cmd)
        if is_dangerous:
            return _err(
                rid, 4005, f"blocked: {desc}. Use the agent for dangerous commands."
            )
    except ImportError:
        return _err(rid, 5001, "shell.exec unavailable: approval safety module not importable")
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30, cwd=os.getcwd(),
            stdin=subprocess.DEVNULL,
        )
        return _ok(
            rid,
            {
                "stdout": r.stdout[-4000:],
                "stderr": r.stderr[-2000:],
                "code": r.returncode,
            },
        )
    except subprocess.TimeoutExpired:
        return _err(rid, 5002, "command timed out (30s)")
    except Exception as e:
        return _err(rid, 5003, str(e))
