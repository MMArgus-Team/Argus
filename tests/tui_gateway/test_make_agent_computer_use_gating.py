"""Computer use is opt-in at the gateway level too: ``_make_agent`` only
enables the ``computer_use`` toolset when the computer-use skill is explicitly
preloaded (ARGUS_TUI_SKILLS=computer-use). Otherwise it is stripped from the
enabled list — or excluded via ``disabled_toolsets`` when the resolver returns
None ("every toolset")."""

from unittest.mock import MagicMock

from tui_gateway import server

_FAKE_RUNTIME = {
    "provider": "openai",
    "base_url": "https://api.openai.com/v1",
    "api_key": "sk-test",
    "api_mode": "chat_completions",
    "command": None,
    "args": None,
    "credential_pool": None,
}


def _setup(monkeypatch, *, enabled_toolsets, startup_skills):
    captured = {}
    monkeypatch.setattr(server, "_load_cfg", lambda: {"agent": {"system_prompt": ""}})
    monkeypatch.setattr(server, "_get_db", lambda: MagicMock())
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested=None, target_model=None: dict(_FAKE_RUNTIME),
    )
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda: enabled_toolsets)
    monkeypatch.setattr(server, "_parse_tui_skills_env", lambda: list(startup_skills))
    monkeypatch.setattr(server, "_load_fallback_model", lambda: None)
    monkeypatch.setattr(server, "_load_provider_routing", lambda: {})
    monkeypatch.setattr(server, "_load_reasoning_config", lambda: None)
    monkeypatch.setattr(server, "_load_service_tier", lambda: None)
    monkeypatch.setattr(server, "_agent_cbs", lambda sid: {})
    # Skill preloading resolves identifiers through skill_view; stub the loader
    # so the skill-identity check runs without touching the real skills dir.
    monkeypatch.setattr(
        "agent.skill_commands.build_preloaded_skills_prompt",
        lambda identifiers, task_id=None: (
            "[skill prompt]",
            [str(i).replace("_", "-") for i in identifiers],
            [],
        ),
    )

    def fake_agent(**kwargs):
        captured.update(kwargs)
        return MagicMock(model=kwargs.get("model"))

    monkeypatch.setattr("run_agent.AIAgent", fake_agent)
    return captured


def test_make_agent_strips_computer_use_when_skill_not_preloaded(monkeypatch):
    captured = _setup(
        monkeypatch,
        enabled_toolsets=["web", "computer_use", "terminal"],
        startup_skills=[],
    )
    server._make_agent("sid", "key")
    assert captured["enabled_toolsets"] == ["web", "terminal"]
    assert captured["disabled_toolsets"] is None


def test_make_agent_keeps_computer_use_out_of_all_toolsets_expansion(monkeypatch):
    # enabled_toolsets=None means "every toolset" — the only way to keep the
    # opt-in tool out is an explicit disabled_toolsets entry.
    captured = _setup(
        monkeypatch,
        enabled_toolsets=None,
        startup_skills=[],
    )
    server._make_agent("sid", "key")
    assert captured["enabled_toolsets"] is None
    assert captured["disabled_toolsets"] == ["computer_use"]


def test_make_agent_adds_computer_use_when_skill_preloaded(monkeypatch):
    captured = _setup(
        monkeypatch,
        enabled_toolsets=["web", "terminal"],
        startup_skills=["computer-use"],
    )
    server._make_agent("sid", "key")
    assert captured["enabled_toolsets"] == ["web", "terminal", "computer_use"]
    assert captured["disabled_toolsets"] is None


def test_make_agent_all_toolsets_with_skill_preloaded_needs_no_disabled_entry(monkeypatch):
    captured = _setup(
        monkeypatch,
        enabled_toolsets=None,
        startup_skills=["computer-use"],
    )
    server._make_agent("sid", "key")
    assert captured["enabled_toolsets"] is None
    assert captured["disabled_toolsets"] is None
