"""On-demand attachment of the multimodal runtime to an already-live session.

``session.resume`` reuses a still-live session verbatim via a fast path that used
to ignore ``params["source"]``. A conversation first opened by the TUI/desktop
(``source=tui``) or by a delegated worker (``source=tool``) is built WITHOUT the
multimodal runtime, so reopening it from the dashboard's /multimodal page left
that page with no frame_buffer and no MonitorEngine — ``set_monitor`` then failed
with "Monitor backend 未就绪" no matter how many times the user retried.

``_promote_session_to_multimodal`` attaches the runtime after the fact. These
tests pin both directions: multimodal resumes promote, and everything else stays
lean (the regression the 2026-08-07 provenance fix was written to prevent — one
empty memory database per sub-agent run).

The return value is an ACK barrier, not a provenance echo: it is
``_multimodal_runtime_ready`` — capture, memory/QueryWorker, and Monitor all
resident and healthy. So the stubs below hand back fake engines that answer the
readiness markers the predicate reads (``is_ready``/``healthy``/``is_healthy``),
and the paths that deliberately end in a half-attached runtime keep asserting
False under a name that says so.
"""

import threading
import types

import pytest

import tui_gateway.server as gateway_server
from tui_gateway.server import (
    _is_multimodal_runtime_session,
    _promote_session_to_multimodal,
)


class _FakeMemoryBackend:
    """Resident memory backend stand-in, healthy unless a test says otherwise.

    ``_multimodal_runtime_ready`` reads ``is_ready``/``healthy`` and demands
    literal True, so a bare ``object()`` (or a Mock) reads as not-ready.
    """

    def __init__(self, *, is_ready: bool = True, healthy: bool = True):
        self.is_ready = is_ready
        self.healthy = healthy


class _FakeWatcher:
    """Live watcher stand-in; ``_mm_watcher_is_ready`` reads ``is_ready``."""

    def __init__(self, *, is_ready: bool = True, healthy: bool = True):
        self.is_ready = is_ready
        self.healthy = healthy


class _FakeMonitorEngine:
    """MonitorEngine stand-in; readiness calls ``is_healthy()``."""

    def __init__(self, *, healthy: bool = True):
        self._is_healthy = healthy

    def is_healthy(self) -> bool:
        return self._is_healthy


@pytest.fixture
def stub_mm_startup(monkeypatch):
    """Record which runtime pieces a promotion starts, without real engines.

    Each stub also publishes its engine into the session exactly like the real
    helper does, because that session state is what the readiness gate reads.
    """
    calls: list[str] = []

    def _fake_memory(sid, session_id, frame_buffer, *, session=None):
        calls.append("memory")
        backend = _FakeMemoryBackend()
        if session is not None:
            session["_mm_memory_backend"] = backend
        return backend

    def _fake_watcher(sid, frame_buffer, memory_backend, session=None):
        calls.append("watcher")
        watcher = _FakeWatcher()
        if session is not None:
            session["_mm_live_watcher_agent"] = watcher
        return watcher

    def _fake_monitor(sid, session, frame_buffer):
        calls.append("monitor")
        engine = _FakeMonitorEngine()
        session["_mm_monitor_engine"] = engine
        return engine

    monkeypatch.setattr(gateway_server, "_maybe_start_memory_backend", _fake_memory)
    monkeypatch.setattr(
        gateway_server, "_maybe_start_live_watcher_agent", _fake_watcher)
    monkeypatch.setattr(gateway_server, "_maybe_start_monitor_engine", _fake_monitor)
    monkeypatch.setattr(gateway_server, "_reconcile_stale_mm_jobs", lambda *a, **k: 0)
    monkeypatch.setattr(gateway_server, "_push_mm_registries", lambda *a, **k: None)
    return calls


def _built_session(source: str) -> dict:
    """A live session whose agent finished building without the MM runtime."""
    agent = types.SimpleNamespace(frame_buffer=None, mm_monitors=None)
    ready = threading.Event()
    ready.set()
    return {
        "agent": agent,
        "agent_ready": ready,
        "agent_build_started": True,
        "source": source,
        "session_key": "20260813_020231_79170c",
        "history": [],
        "history_lock": threading.Lock(),
    }


def _promoted_session() -> dict:
    """A live session that already owns a complete, healthy MM runtime."""
    session = _built_session("multimodal")
    session["agent"].frame_buffer = object()
    session["agent"].mm_monitors = {}
    session["agent"]._multimodal_session = True
    session["_mm_memory_backend"] = _FakeMemoryBackend()
    session["_mm_live_watcher_agent"] = _FakeWatcher()
    session["_mm_monitor_engine"] = _FakeMonitorEngine()
    return session


@pytest.mark.parametrize("source", ["tui", "tool", ""])
def test_promote_attaches_full_runtime(stub_mm_startup, source):
    """A tui/tool-built live session gains frame_buffer + all three engines."""
    session = _built_session(source)
    assert not _is_multimodal_runtime_session(session)

    # True is the readiness ACK, not merely "source was flipped": capture,
    # memory/QueryWorker, and Monitor are all resident and healthy.
    assert _promote_session_to_multimodal("sid1", session) is True

    # Provenance flipped, so the per-turn gates in _run_prompt_submit and the
    # trajectory routing in _emit now treat this as a multimodal session.
    assert _is_multimodal_runtime_session(session)
    # The monitor engine is what set_monitor(op=create) requires; without the
    # frame_buffer it is never even constructed.
    assert session["agent"].frame_buffer is not None
    assert session["agent"].mm_monitors == {}
    assert session["agent"]._multimodal_session is True
    assert session["_mm_monitor_engine"] is not None
    assert stub_mm_startup == ["memory", "watcher", "monitor"]


def test_promote_is_idempotent(stub_mm_startup):
    """An already-READY multimodal session must not build a second runtime.

    The short-circuit is provenance AND readiness, so the resident engines have
    to be in place for it to fire — see the repair test below for the session
    that merely claims to be multimodal.
    """
    session = _promoted_session()
    first_backend = session["_mm_memory_backend"]

    assert _promote_session_to_multimodal("sid1", session) is True

    assert stub_mm_startup == []
    assert session["_mm_memory_backend"] is first_backend


def test_promote_repairs_multimodal_session_without_runtime(stub_mm_startup):
    """source=multimodal alone is not readiness; the runtime is re-attached.

    A previous promotion that failed halfway deliberately leaves the provenance
    flipped. Resuming such a session must rebuild the missing engines instead of
    trusting the marker and handing the dashboard another "未就绪" monitor.
    """
    session = _built_session("multimodal")
    assert _is_multimodal_runtime_session(session)

    assert _promote_session_to_multimodal("sid1", session) is True

    assert session["agent"].frame_buffer is not None
    assert stub_mm_startup == ["memory", "watcher", "monitor"]


def test_promote_refuses_finalized_session(stub_mm_startup):
    """A session being torn down must not get a runtime nothing will stop."""
    session = _built_session("tui")
    session["_finalized"] = True

    assert _promote_session_to_multimodal("sid1", session) is False

    assert stub_mm_startup == []
    assert not _is_multimodal_runtime_session(session)


def test_lazy_session_only_flips_provenance(stub_mm_startup):
    """Before the build starts, the flip alone is enough.

    _start_agent_build reads ``source`` itself, so it will construct the runtime
    natively. Starting engines here would race that build.

    This is the one True that is not a readiness ACK: promotion returns before
    the readiness gate because there is nothing yet to measure.
    """
    session = _built_session("tui")
    session["agent"] = None
    session["agent_build_started"] = False

    assert _promote_session_to_multimodal("sid1", session) is True

    assert _is_multimodal_runtime_session(session)
    assert stub_mm_startup == []


def test_promote_waits_for_inflight_build(stub_mm_startup):
    """A build already in flight captured is_mm_session=False.

    Flipping ``source`` cannot retroactively change that decision, so promotion
    must wait for the agent to appear and then attach the runtime itself —
    otherwise the session silently stays non-multimodal.
    """
    session = _built_session("tui")
    session["agent"] = None  # mid-build
    ready = threading.Event()
    session["agent_ready"] = ready
    session["agent_build_started"] = True

    def _finish_build():
        session["agent"] = types.SimpleNamespace(
            frame_buffer=None, mm_monitors=None)
        ready.set()

    threading.Timer(0.05, _finish_build).start()

    assert _promote_session_to_multimodal("sid1", session) is True

    assert session["agent"].frame_buffer is not None
    assert stub_mm_startup == ["memory", "watcher", "monitor"]


def test_partial_promotion_on_engine_failure_reports_not_ready(
    stub_mm_startup, monkeypatch,
):
    """A broken engine yields a PARTIAL promotion: no raise, but False.

    The resume that asked for the runtime must survive (no exception escapes),
    yet the ACK has to stay negative — reporting success here is what let the
    dashboard open a /multimodal page on a session with no MonitorEngine.
    """
    def _boom(*_a, **_k):
        raise RuntimeError("monitor engine exploded")

    monkeypatch.setattr(gateway_server, "_maybe_start_monitor_engine", _boom)
    session = _built_session("tui")

    assert _promote_session_to_multimodal("sid1", session) is False

    # Provenance stays flipped: the caller asked for a multimodal session, and
    # the per-monitor guards report their own real cause rather than the
    # misleading "this isn't a multimodal session" silence.
    assert _is_multimodal_runtime_session(session)
    assert session.get("_mm_monitor_engine") is None


def test_partial_promotion_on_unhealthy_engine_reports_not_ready(
    stub_mm_startup, monkeypatch,
):
    """Engines that start but come up sick are still a PARTIAL promotion.

    Nothing raises and every session key is populated, so only the health
    markers distinguish this from a good promotion. False keeps source-start an
    honest barrier instead of an "the objects exist" tautology.
    """
    def _sick_monitor(sid, session, frame_buffer):
        engine = _FakeMonitorEngine(healthy=False)
        session["_mm_monitor_engine"] = engine
        return engine

    monkeypatch.setattr(
        gateway_server, "_maybe_start_monitor_engine", _sick_monitor)
    session = _built_session("tui")

    assert _promote_session_to_multimodal("sid1", session) is False

    # The runtime is attached — it just is not usable yet.
    assert session["agent"].frame_buffer is not None
    assert session["_mm_monitor_engine"] is not None
    assert _is_multimodal_runtime_session(session)
