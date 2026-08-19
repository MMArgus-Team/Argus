"""Tests for the `argus setup multimodal` section (hermes_cli.setup).

Drives the pure logic — anysearch-block discovery, key writes, and env-skip —
with mocked prompts so no TTY / network is needed. Guards against the
"implementation drifts, wizard silently stops writing the right key" failure
mode.
"""

from __future__ import annotations

import hermes_cli.setup as S


def _no_prompts(monkeypatch):
    """Decline every yes/no and return empty for every text prompt."""
    monkeypatch.setattr(S, "prompt_yes_no", lambda q, default=True: False)
    monkeypatch.setattr(S, "prompt", lambda q, default=None, password=False: "")


def _clear_env(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("ANYSEARCH_API_KEY", raising=False)


# --------------------------------------------------------------------------- #
# _find_anysearch_block — locates the nested block without hardcoding the path
# --------------------------------------------------------------------------- #
def test_find_anysearch_nested_two_levels():
    cfg = {"model": {"watcher": {"anysearch": {"api_key": "as_sk_x"}}}}
    blk = S._find_anysearch_block(cfg)
    assert blk is not None and blk["api_key"] == "as_sk_x"


def test_find_anysearch_top_level_under_model():
    cfg = {"model": {"anysearch": {"api_key": "k"}}}
    assert S._find_anysearch_block(cfg) is cfg["model"]["anysearch"]


def test_find_anysearch_absent():
    assert S._find_anysearch_block({"model": {"x": {}}}) is None
    assert S._find_anysearch_block({}) is None


# --------------------------------------------------------------------------- #
# section runs without crashing when everything is declined
# --------------------------------------------------------------------------- #
def test_section_declines_all_no_crash(monkeypatch):
    _clear_env(monkeypatch)
    _no_prompts(monkeypatch)
    cfg = {"model": {}, "audio": {}, "settings": {}}
    S.setup_multimodal(cfg)  # must not raise


# --------------------------------------------------------------------------- #
# key writes land in the paths flatten_mm_config reads
# --------------------------------------------------------------------------- #
def test_writes_dashscope_and_anysearch_keys(monkeypatch):
    _clear_env(monkeypatch)
    # Accept only the voice + deep-research questions; decline install/download.
    monkeypatch.setattr(
        S, "prompt_yes_no",
        lambda q, default=True: ("语音" in q or "深研" in q))
    monkeypatch.setattr(
        S, "prompt", lambda q, default=None, password=False: "NEWKEY")
    cfg = {"model": {"w": {"anysearch": {"api_key": ""}}},
           "audio": {}, "settings": {}}
    S.setup_multimodal(cfg)
    assert cfg["audio"]["dashscope_api_key"] == "NEWKEY"
    assert S._find_anysearch_block(cfg)["api_key"] == "NEWKEY"


# --------------------------------------------------------------------------- #
# env vars are honored — no secret prompt when the env key is already set
# --------------------------------------------------------------------------- #
def test_env_keys_skip_voice_and_search_prompts(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env")
    monkeypatch.setenv("ANYSEARCH_API_KEY", "env")
    asked = []
    monkeypatch.setattr(
        S, "prompt_yes_no", lambda q, default=True: (asked.append(q), False)[1])

    def _boom(*a, **k):
        raise AssertionError("secret prompt called despite env key present")
    monkeypatch.setattr(S, "prompt", _boom)
    S.setup_multimodal({"model": {}, "audio": {}, "settings": {}})
    assert not [q for q in asked if ("语音" in q or "深研" in q)]
