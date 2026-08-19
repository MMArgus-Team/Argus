"""Thinking effort is a per-SESSION runtime preference, not a global config key.

``config.set {key: "reasoning"}`` historically did two things at once: write
``agent.reasoning_effort`` into ``config.yaml`` AND update the live agent. That
made the web UI's THINKING slider unusable for two independent reasons:

1. ``config.yaml`` in HERMES_HOME is overwritten on every dashboard start by
   ``sync_project_config()`` ("One-way, project → HERMES_HOME, always
   overwrites"), and the git-tracked project config has no ``agent:`` block —
   so the written key simply vanished and the slider snapped back to the
   ``normalizeEffort`` fallback ("medium").
2. The web client only wrote the config; it never updated the live agent. The
   agent is built once per session and ``_apply_mm_deep_thinking`` re-reads the
   ``agent.reasoning_config`` ATTRIBUTE every turn (never disk), so a config
   write could not affect the running session at all.

The fix adds an explicit ``scope``:

* ``scope="session"`` — set the live agent's ``reasoning_config`` (read fresh on
  every API call, so it lands on the next turn) and persist it into the session
  row's ``model_config``. ``config.yaml`` is NOT touched.
* ``scope="global"`` (default) — legacy behaviour, so the TUI ``/reasoning``
  command keeps writing the baseline for new sessions.

These tests pin both halves, plus the persist/restore round-trip that carries a
session's effort across resume.
"""

import json

import pytest

from hermes_constants import parse_reasoning_effort


# The six stops the web slider offers (web/src/lib/reasoning-effort.ts
# EFFORT_OPTIONS). Every one must survive parse_reasoning_effort — a value the
# backend normalizes away would silently do nothing.
EFFORT_VALUES = ["none", "minimal", "low", "medium", "high", "xhigh"]


class _FakeAgent:
    """Minimal stand-in for AIAgent: only the attributes the RPC touches."""

    def __init__(self):
        self.reasoning_config = {"enabled": True, "effort": "medium"}
        self.model = "glm-5.2"
        self.provider = "open.bigmodel.cn"
        self.base_url = "https://open.bigmodel.cn/api/paas/v4"
        self.api_mode = ""
        self.service_tier = None
        self.session_id = "sess-key"
        self._session_db = None


def _call_reasoning(srv, monkeypatch, *, value, scope=None, session=None):
    """Invoke the config.set RPC handler directly, stubbing its side effects."""
    writes = {}
    emitted = []
    persisted = []

    monkeypatch.setattr(
        srv, "_write_config_key", lambda k, v: writes.__setitem__(k, v)
    )
    monkeypatch.setattr(
        srv, "_persist_live_session_runtime", lambda s: persisted.append(s)
    )
    monkeypatch.setattr(srv, "_emit", lambda ev, sid, payload: emitted.append(ev))
    monkeypatch.setattr(srv, "_session_info", lambda agent, session: {"stub": True})

    sessions = {"sid-1": session} if session is not None else {}
    monkeypatch.setattr(srv, "_sessions", sessions)

    params = {"key": "reasoning", "value": value, "session_id": "sid-1"}
    if scope is not None:
        params["scope"] = scope

    resp = srv._methods["config.set"](1, params)
    return resp, writes, emitted, persisted


@pytest.fixture()
def srv():
    from tui_gateway import server

    return server


class TestScopeSession:
    def test_session_scope_updates_live_agent_and_skips_config(
        self, srv, monkeypatch
    ):
        agent = _FakeAgent()
        session = {"agent": agent, "session_key": "sess-key"}

        resp, writes, emitted, persisted = _call_reasoning(
            srv, monkeypatch, value="high", scope="session", session=session
        )

        assert "error" not in resp
        assert resp["result"]["scope"] == "session"
        # The live agent now carries the new effort → next turn uses it.
        assert agent.reasoning_config == {"enabled": True, "effort": "high"}
        # config.yaml must NOT be touched: sync_project_config would erase it,
        # and it would leak this session's choice into every other session.
        assert writes == {}
        # Persisted into the session row + broadcast so the UI can re-sync.
        assert persisted == [session]
        assert "session.info" in emitted

    def test_session_scope_none_disables_thinking(self, srv, monkeypatch):
        agent = _FakeAgent()
        session = {"agent": agent, "session_key": "sess-key"}

        resp, writes, _, _ = _call_reasoning(
            srv, monkeypatch, value="none", scope="session", session=session
        )

        assert "error" not in resp
        assert agent.reasoning_config == {"enabled": False}
        assert writes == {}

    def test_session_scope_without_agent_stashes_build_override(
        self, srv, monkeypatch
    ):
        """Lazy sessions build the agent on first prompt. Setting the dial before
        that must not be lost — it rides in as the build's reasoning override."""
        session = {"agent": None, "session_key": "sess-key"}

        resp, writes, emitted, persisted = _call_reasoning(
            srv, monkeypatch, value="low", scope="session", session=session
        )

        assert "error" not in resp
        assert session["create_reasoning_override"] == {
            "enabled": True,
            "effort": "low",
        }
        assert writes == {}
        # No live agent → nothing to persist or broadcast yet.
        assert persisted == []
        assert emitted == []

    def test_session_scope_requires_a_session(self, srv, monkeypatch):
        resp, writes, _, _ = _call_reasoning(
            srv, monkeypatch, value="high", scope="session", session=None
        )

        assert resp["error"]["code"] == 4001
        assert writes == {}


class TestScopeGlobal:
    def test_global_scope_still_writes_config(self, srv, monkeypatch):
        """The TUI /reasoning command must keep setting the new-session baseline."""
        agent = _FakeAgent()
        session = {"agent": agent, "session_key": "sess-key"}

        resp, writes, emitted, _ = _call_reasoning(
            srv, monkeypatch, value="xhigh", scope="global", session=session
        )

        assert "error" not in resp
        assert resp["result"]["scope"] == "global"
        assert writes == {"agent.reasoning_effort": "xhigh"}
        assert agent.reasoning_config == {"enabled": True, "effort": "xhigh"}
        assert "session.info" in emitted

    def test_scope_defaults_to_global(self, srv, monkeypatch):
        """Omitting scope keeps the pre-existing contract (no silent behaviour
        change for existing callers)."""
        agent = _FakeAgent()
        session = {"agent": agent, "session_key": "sess-key"}

        resp, writes, _, _ = _call_reasoning(
            srv, monkeypatch, value="high", session=session
        )

        assert resp["result"]["scope"] == "global"
        assert writes == {"agent.reasoning_effort": "high"}

    def test_unknown_value_rejected_before_any_write(self, srv, monkeypatch):
        agent = _FakeAgent()
        session = {"agent": agent, "session_key": "sess-key"}

        resp, writes, _, _ = _call_reasoning(
            srv, monkeypatch, value="turbo", scope="session", session=session
        )

        assert resp["error"]["code"] == 4002
        assert writes == {}
        # Live agent untouched by a rejected value.
        assert agent.reasoning_config == {"enabled": True, "effort": "medium"}


class TestSessionRoundTrip:
    """A session's effort must survive persist → resume."""

    def test_effort_round_trips_through_model_config(self, srv):
        agent = _FakeAgent()
        agent.reasoning_config = {"enabled": True, "effort": "high"}

        stored = srv._runtime_model_config(agent)
        assert stored["reasoning_config"] == {"enabled": True, "effort": "high"}

        row = {"model": agent.model, "model_config": json.dumps(stored)}
        overrides = srv._stored_session_runtime_overrides(row)

        assert overrides["reasoning_config_override"] == {
            "enabled": True,
            "effort": "high",
        }

    def test_thinking_off_round_trips(self, srv):
        agent = _FakeAgent()
        agent.reasoning_config = {"enabled": False}

        stored = srv._runtime_model_config(agent)
        row = {"model": agent.model, "model_config": json.dumps(stored)}
        overrides = srv._stored_session_runtime_overrides(row)

        assert overrides["reasoning_config_override"] == {"enabled": False}


class TestSessionInfoEffortEcho:
    """``session.info.reasoning_effort`` must be unambiguous about OFF.

    Reporting "" for a disabled config made "turn thinking off" impossible on
    every client: they optimistically flipped the switch off, this echo arrived
    with "", and "" means "no explicit level → default" — so desktop's
    isThinkingEnabled('') and web's normalizeEffort('') both said ON and the
    control snapped back. Screenshot symptom: the Thinking switch would not stay
    off. "none" round-trips through parse_reasoning_effort and both clients
    already had a label for it (desktop REASONING_LABELS.none → "Off").
    """

    @staticmethod
    def _echo(reasoning_config):
        """Mirror of the reasoning_effort computation in _session_info."""
        effort = ""
        if isinstance(reasoning_config, dict):
            if reasoning_config.get("enabled") is False:
                effort = "none"
            else:
                effort = str(reasoning_config.get("effort", "") or "")
        return effort

    def test_disabled_reports_none_not_empty(self, srv, monkeypatch):
        agent = _FakeAgent()
        session = {"agent": agent, "session_key": "sess-key"}
        _call_reasoning(
            srv, monkeypatch, value="none", scope="session", session=session
        )
        assert agent.reasoning_config == {"enabled": False}
        assert self._echo(agent.reasoning_config) == "none"

    def test_explicit_level_reports_that_level(self):
        assert self._echo({"enabled": True, "effort": "high"}) == "high"

    def test_enabled_without_level_still_reports_empty(self):
        """"" keeps meaning "use the default" — only OFF became explicit."""
        assert self._echo({"enabled": True}) == ""
        assert self._echo(None) == ""

    def test_none_echo_survives_a_reparse(self):
        """The echoed value must be something a client can send straight back."""
        assert parse_reasoning_effort(self._echo({"enabled": False})) == {
            "enabled": False
        }


class TestEffortVocabulary:
    @pytest.mark.parametrize("value", EFFORT_VALUES)
    def test_every_slider_stop_is_accepted(self, value):
        """A stop the backend normalizes away would be a dead notch on the dial."""
        assert parse_reasoning_effort(value) is not None

    def test_none_is_the_only_disabling_stop(self):
        assert parse_reasoning_effort("none") == {"enabled": False}
        for value in [v for v in EFFORT_VALUES if v != "none"]:
            parsed = parse_reasoning_effort(value)
            assert parsed["enabled"] is True
            assert parsed["effort"] == value
