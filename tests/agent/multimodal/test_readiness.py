"""Tests for the multimodal readiness probe (agent.multimodal.readiness).

The probe is a pure function over a Config-like mapping + the environment +
installed modules, so these tests drive it with plain dicts and monkeypatched
env / module-presence — fully offline, no engine, no network.

Covers each capability's ok/missing/broken branch and the overall `ready`
aggregation (only REQUIRED capabilities gate readiness).
"""

from __future__ import annotations

import agent.multimodal.readiness as R


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _cap(report, key):
    for c in report["capabilities"]:
        if c["key"] == key:
            return c
    raise AssertionError(f"capability {key!r} not in report")


def _clear_mm_env(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("ANYSEARCH_API_KEY", raising=False)
    monkeypatch.delenv("ARGUS_HOME", raising=False)


# --------------------------------------------------------------------------- #
# voice (optional)
# --------------------------------------------------------------------------- #
def test_voice_missing_without_key(monkeypatch):
    _clear_mm_env(monkeypatch)
    r = R.probe_mm_readiness({})
    v = _cap(r, "voice")
    assert v["status"] == R.MISSING
    assert v["required"] is False
    assert "dashscope" in v["reason"].lower()
    assert v["fix"]


def test_voice_ok_via_config_key(monkeypatch):
    _clear_mm_env(monkeypatch)
    r = R.probe_mm_readiness({"dashscope_api_key": "sk-abc"})
    assert _cap(r, "voice")["status"] == R.OK


def test_voice_ok_via_env_key(monkeypatch):
    _clear_mm_env(monkeypatch)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-env")
    r = R.probe_mm_readiness({})
    assert _cap(r, "voice")["status"] == R.OK


def test_voice_missing_when_both_disabled(monkeypatch):
    _clear_mm_env(monkeypatch)
    r = R.probe_mm_readiness(
        {"dashscope_api_key": "sk-abc",
         "realtime_asr_enabled": False, "realtime_tts_enabled": False})
    # Even with a key, both toggles off → capability is off.
    assert _cap(r, "voice")["status"] == R.MISSING


# --------------------------------------------------------------------------- #
# deep_research (optional) — env takes priority over config
# --------------------------------------------------------------------------- #
def test_deep_research_missing(monkeypatch):
    _clear_mm_env(monkeypatch)
    assert _cap(R.probe_mm_readiness({}), "deep_research")["status"] == R.MISSING


def test_deep_research_ok_env_priority(monkeypatch):
    _clear_mm_env(monkeypatch)
    monkeypatch.setenv("ANYSEARCH_API_KEY", "as_sk_env")
    assert _cap(R.probe_mm_readiness({}), "deep_research")["status"] == R.OK


def test_deep_research_ok_config(monkeypatch):
    _clear_mm_env(monkeypatch)
    r = R.probe_mm_readiness({"anysearch_api_key": "as_sk_cfg"})
    assert _cap(r, "deep_research")["status"] == R.OK


# --------------------------------------------------------------------------- #
# memory (REQUIRED) — local OCR (rapidocr) is the only backend
# --------------------------------------------------------------------------- #
def test_memory_broken_when_rapidocr_absent(monkeypatch):
    _clear_mm_env(monkeypatch)
    monkeypatch.setattr(R, "_module_installed", lambda name: False)
    m = _cap(R.probe_mm_readiness({}), "memory")
    assert m["status"] == R.BROKEN
    assert m["required"] is True
    assert "rapidocr" in m["reason"].lower()


def test_memory_ok_when_rapidocr_present(monkeypatch):
    _clear_mm_env(monkeypatch)
    monkeypatch.setattr(R, "_module_installed", lambda name: name == "rapidocr")
    m = _cap(R.probe_mm_readiness({}), "memory")
    assert m["status"] == R.OK


# --------------------------------------------------------------------------- #
# capture_perms (optional) — always unknown (can't introspect OS grants)
# --------------------------------------------------------------------------- #
def test_capture_perms_unknown(monkeypatch):
    _clear_mm_env(monkeypatch)
    c = _cap(R.probe_mm_readiness({}), "capture_perms")
    assert c["status"] == R.UNKNOWN
    assert c["fix"]


# --------------------------------------------------------------------------- #
# overall readiness aggregation — only REQUIRED caps gate `ready`
# --------------------------------------------------------------------------- #
def test_ready_true_when_required_ok_despite_optional_missing(monkeypatch):
    _clear_mm_env(monkeypatch)
    # rapidocr present → memory (the only required cap) ok; everything else
    # (voice/search/weights/torch) missing but optional.
    monkeypatch.setattr(R, "_module_installed",
                        lambda name: name == "rapidocr")
    r = R.probe_mm_readiness({})
    assert r["ready"] is True


def test_ready_false_when_required_broken(monkeypatch):
    _clear_mm_env(monkeypatch)
    monkeypatch.setattr(R, "_module_installed", lambda name: False)
    r = R.probe_mm_readiness({})
    assert r["ready"] is False


def test_report_shape_is_stable(monkeypatch):
    _clear_mm_env(monkeypatch)
    r = R.probe_mm_readiness({})
    assert set(r.keys()) == {"ready", "capabilities"}
    keys = {c["key"] for c in r["capabilities"]}
    assert keys == {"voice", "deep_research", "memory", "capture_perms"}
    for c in r["capabilities"]:
        assert set(c.keys()) == {
            "key", "label", "status", "required", "reason", "fix"}
