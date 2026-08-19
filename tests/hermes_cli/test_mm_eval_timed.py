# -*- coding: utf-8 -*-
"""mm-memory-eval 时序评测模式的 IO / 校验层测试 (纯逻辑, 不解码视频、不调 LLM)。

覆盖:
  * parse_timestamp: HH:MM:SS / MM:SS / 纯秒 / 非法 的解析。
  * load_eval_json 时序判定: 无 time → 非时序(向后兼容); 有任一 time → 时序且
    要求每题都有合法 time; 缺失/非法 time → ValueError。
  * write_eval_json 剥离内部 _time_sec, 保留 time + answer_predict。
"""
import json
import tempfile

import pytest

from hermes_cli.mm_eval_io import (
    parse_timestamp, load_eval_json, is_timed_eval, write_eval_json)


def _write(obj) -> str:
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(obj, f, ensure_ascii=False)
    f.close()
    return f.name


@pytest.mark.parametrize("raw,exp", [
    ("00:12:27", 747.0), ("12:27", 747.0), ("1:02:03", 3723.0),
    ("90", 90.0), (90, 90.0), (90.5, 90.5), (0, 0.0), ("0:00", 0.0),
    ("", None), (None, None), ("abc", None), ("00:99:00", None),
    ("1:2:3:4", None), ("-5", None), (-5, None),
])
def test_parse_timestamp(raw, exp):
    assert parse_timestamp(raw) == exp


def test_no_time_is_not_timed():
    p = _write({"title": "v", "qa_list": [{"query": "q", "answer": "a"}]})
    d = load_eval_json(p)
    assert is_timed_eval(d) is False


def test_all_time_is_timed_and_parsed():
    p = _write({"title": "v", "qa_list": [
        {"query": "q1", "answer": "a", "time": "00:00:10"},
        {"query": "q2", "answer": "b", "time": "00:01:00"},
    ]})
    d = load_eval_json(p)
    assert is_timed_eval(d) is True
    assert d["qa_list"][0]["_time_sec"] == 10.0
    assert d["qa_list"][1]["_time_sec"] == 60.0


def test_partial_time_raises():
    p = _write({"title": "v", "qa_list": [
        {"query": "q1", "answer": "a", "time": "00:00:10"},
        {"query": "q2", "answer": "b"},  # missing time → error in timed mode
    ]})
    with pytest.raises(ValueError):
        load_eval_json(p)


def test_bad_time_raises():
    p = _write({"title": "v", "qa_list": [
        {"query": "q", "answer": "a", "time": "garbage"},
    ]})
    with pytest.raises(ValueError):
        load_eval_json(p)


def test_write_strips_time_sec_keeps_time():
    p = _write({"title": "v", "qa_list": [
        {"query": "q", "answer": "a", "time": "00:00:10"},
    ]})
    d = load_eval_json(p)
    d["qa_list"][0]["answer_predict"] = "pred"
    out = write_eval_json(p, d)
    w = json.load(open(out, encoding="utf-8"))
    qa = w["qa_list"][0]
    assert "_time_sec" not in qa           # internal cache not leaked
    assert qa["time"] == "00:00:10"        # user-facing time preserved
    assert qa["answer_predict"] == "pred"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
