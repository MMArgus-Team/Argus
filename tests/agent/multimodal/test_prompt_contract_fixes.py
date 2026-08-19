"""Guards for prompt<->code contract fixes found in the prompt review.

Covers:
- #2  L2/L3 narrative_arc 's'-unit fallback regex must handle the "t" field,
      not only "ts" (_memory._TS_UNIT_FIX_RE).
- #3  RECALL_DISTILL sentinel "本轮工具无有效信息" must be filtered, not
      appended as a clue (logic guard).
- #4  distill observation truncation must be visible (marker appended).
- #6  monitor output normalization tolerates code fences / quotes / leading
      text so a real SPEAK isn't silently dropped.

Pure in-memory; no cloud, no hardware.
"""
import re

from agent.multimodal._memory import _TS_UNIT_FIX_RE
from agent.multimodal._workers import _truncate_obs, _DISTILL_OBS_CAP


# ── #2: s-unit fallback regex ───────────────────────────────────────────────
def test_ts_unit_fix_handles_ts_field():
    assert _TS_UNIT_FIX_RE.sub(r"\1", '{"ts": 130.0s}') == '{"ts": 130.0}'


def test_ts_unit_fix_now_handles_t_field():
    # This is the L2/L3 narrative_arc field that used to slip through.
    assert _TS_UNIT_FIX_RE.sub(r"\1", '{"t": 130.0s}') == '{"t": 130.0}'
    assert _TS_UNIT_FIX_RE.sub(r"\1", '{"t": -5s}') == '{"t": -5}'


def test_ts_unit_fix_leaves_clean_json_untouched():
    assert _TS_UNIT_FIX_RE.sub(r"\1", '{"t": 130.0}') == '{"t": 130.0}'


# ── #3: distill sentinel filtering (logic guard) ────────────────────────────
def test_distill_sentinel_is_filtered():
    def keep_clue(distilled):
        # mirrors _workers recall loop guard
        return bool(distilled and "本轮工具无有效信息" not in distilled)

    assert keep_clue("找到线索: 桌上有耳机") is True
    assert keep_clue("本轮工具无有效信息") is False
    assert keep_clue("本轮工具无有效信息。") is False   # tolerate punctuation
    assert keep_clue("") is False


# ── #4: visible truncation ──────────────────────────────────────────────────
def test_truncate_obs_short_passthrough():
    s = "hello world"
    assert _truncate_obs(s) == s


def test_truncate_obs_long_appends_visible_marker():
    s = "x" * (_DISTILL_OBS_CAP + 500)
    out = _truncate_obs(s)
    assert out.startswith("x" * _DISTILL_OBS_CAP)
    assert "截断" in out                      # visible marker present
    assert "500" in out                       # reports how much was cut


def test_truncate_obs_none_safe():
    assert _truncate_obs(None) == ""


# ── #6: monitor SPEAK/SILENT verdict parsing ───────────────────────────────
# 新协议为 SPEAK: <brief reason> / SILENT; 解析层兼容旧 JSON。
def _verdict(raw):
    from agent.multimodal.monitor_engine import parse_monitor_verdict
    return parse_monitor_verdict(raw)


def test_monitor_speak_hit():
    hit, reason = _verdict("SPEAK: 有人进来了")
    assert hit is True and reason == "有人进来了"


def test_monitor_json_code_fence_stripped():
    assert _verdict('```json\n{"status": true, "reason": "门开了"}\n```')[0] is True


def test_monitor_json_leading_text_stripped():
    assert _verdict('好的：{"status": true, "reason": "门开了"}')[0] is True


def test_monitor_json_status_false():
    hit, reason = _verdict('{"status": false, "reason": "无异常"}')
    assert hit is False and reason == ""


def test_monitor_silent_never_keeps_an_explanation():
    assert _verdict("SILENT: 目标未出现，因此不触发") == (False, "")


def test_monitor_non_json_is_miss():
    assert _verdict("SILENT") == (False, "")
    assert _verdict("") == (False, "")


def test_set_monitor_schema_forbids_narrowing_user_semantics():
    from tools.monitor_tool import SET_MONITOR_SCHEMA

    desc = SET_MONITOR_SCHEMA["parameters"]["properties"]["monitor_query"]["description"]
    assert "preserve the user's semantic scope exactly" in desc
    assert "Do NOT invent exclusions" in desc
    assert "Common subtypes and form variants" in desc
    assert "水杯" not in desc
