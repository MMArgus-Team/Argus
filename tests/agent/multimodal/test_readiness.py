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
    # rapidocr present → memory + local OCR ok; main endpoint reachable →
    # llm_endpoints ok; voice/search missing but optional.
    monkeypatch.setattr(R, "_module_installed",
                        lambda name: name == "rapidocr")
    monkeypatch.setattr(R, "_tcp_reachable", lambda *_args: (True, ""))
    raw = {"model": {"base_url": "https://llm.example/v1"}}
    r = R.probe_mm_readiness({}, raw)
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
    assert keys == {
        "voice", "deep_research", "memory", "capture_perms",
        "aux_ocr_local", "llm_endpoints",
    }
    for c in r["capabilities"]:
        # extras (url / endpoints / tcp_error) are allowed on top of the base
        # six — assert the base shape as a subset.
        assert {"key", "label", "status", "required", "reason", "fix"} <= set(c.keys())


# --------------------------------------------------------------------------- #
# unified probe set — every consumer (CLI doctor / RPC / startup advisory)
# runs the SAME checks; only the wait-time knob and transport/output differ.
# --------------------------------------------------------------------------- #
def test_aux_ocr_and_llm_endpoints_run_in_the_common_probe_set(monkeypatch):
    _clear_mm_env(monkeypatch)
    monkeypatch.setattr(R, "_module_installed", lambda name: name == "rapidocr")
    probes = []
    monkeypatch.setattr(R, "_tcp_reachable",
                        lambda *args: (probes.append(args) or (True, "")))
    raw = {"model": {"base_url": "https://main.example/v1"}}
    report = R.probe_mm_readiness({}, raw)
    # aux_ocr_local + llm_endpoints are part of the ONE probe set — no deep flag.
    assert _cap(report, "aux_ocr_local")["status"] == R.OK
    llm = _cap(report, "llm_endpoints")
    assert llm["status"] == R.OK
    assert llm["endpoints"][0]["url"] == "https://main.example/v1"
    assert probes == [("https://main.example/v1", R._TCP_DEFAULT_TIMEOUT)]


def test_llm_endpoints_probe_honors_timeout(monkeypatch):
    _clear_mm_env(monkeypatch)
    monkeypatch.setattr(R, "_module_installed", lambda name: name == "rapidocr")
    calls = []
    monkeypatch.setattr(R, "_tcp_reachable",
                        lambda url, timeout: (calls.append((url, timeout)) or (True, "")))
    raw = {"model": {"base_url": "https://main.example/v1"}}
    R.probe_mm_readiness({}, raw, timeout=0.25)
    assert calls == [("https://main.example/v1", 0.25)]
