"""set_live_watcher no longer has a simple branch (always background).

After the change, the main agent answers simple visual questions directly from
its injected 4 frames; set_live_watcher is only for complex analysis and
always kicks off background orchestration. We assert it NEVER calls
decide_complexity and ALWAYS returns mode=background.
"""
import json

import tools.live_watcher_tool as dt


class _FakeEngine:
    def __init__(self):
        self.decide_calls = 0
        self.submit_calls = []

    def decide_complexity(self, text, timeout=30.0):
        # If this is ever called, the simple branch wasn't removed.
        self.decide_calls += 1
        return {"simple": True, "answer": "SHOULD-NOT-BE-USED"}

    def submit_complex_async(self, task_instruction, request_id="", **_kw):
        # Matches the real WatcherAgent signature (single task_instruction).
        # **_kw swallows any legacy kwargs a caller might still pass.
        self.submit_calls.append((task_instruction,))
        return request_id or "req_deadbeef"

    def is_source_live(self) -> bool:
        # set_live_watcher now guards on a live video source (mm_stream_status →
        # engine.is_source_live()); report live so the background dispatch path
        # under test is actually exercised.
        return True


class _FakeAgent:
    pass


def _install_fake_session(monkeypatch, engine, agent):
    # set_live_watcher reads `_sessions` from tui_gateway.server at call time.
    # ★ Test-isolation fix: import the REAL module and patch its `_sessions`
    #   attribute via monkeypatch (auto-restored). The old code fabricated an
    #   empty fake module and assigned it into sys.modules directly, which LEAKED
    #   globally — a later test's `patch("tui_gateway.server._sessions", …)` then
    #   failed with AttributeError (the fake module had no _sessions). Importing
    #   the real module (which defines _sessions at module scope) + monkeypatch
    #   keeps every test isolated.
    fake_sessions = {"sid1": {"_mm_live_watcher_agent": engine, "agent": agent,
                             "session_key": "sid1"}}
    import tui_gateway.server as server_mod
    monkeypatch.setattr(server_mod, "_sessions", fake_sessions, raising=False)


def test_always_background_never_simple(monkeypatch):
    eng = _FakeEngine()
    agent = _FakeAgent()
    _install_fake_session(monkeypatch, eng, agent)

    out = json.loads(dt.set_live_watcher(task_instruction="屏幕上是什么",
                                               session_id="sid1"))
    # Monitor-parity receipt: success result with status running (no "mode").
    assert out["status"] == "running"
    assert out["op"] == "create"
    # v0.1: the handler generates the request_id and passes it into
    # submit_complex_async; just assert it's a proper req_ id.
    assert str(out["request_id"]).startswith("req_")
    assert eng.decide_calls == 0, "decide_complexity must NOT be called anymore"
    assert eng.submit_calls == [("屏幕上是什么",)]
    # main agent gets the background-stop signal
    assert getattr(agent, "_mm_route_background_stop", False) is True


def test_empty_user_text_errors(monkeypatch):
    eng = _FakeEngine()
    _install_fake_session(monkeypatch, eng, _FakeAgent())
    out = json.loads(dt.set_live_watcher(task_instruction="  ", session_id="sid1"))
    assert "error" in out


def test_no_engine_errors(monkeypatch):
    import tui_gateway.server as server_mod
    monkeypatch.setattr(server_mod, "_sessions", {}, raising=False)
    out = json.loads(dt.set_live_watcher(task_instruction="x", session_id="nope"))
    assert "error" in out
