"""Regression guard for ASR segment parsing (finding C17).

_parse_asr_text concatenated ``str(s.get("text", "")).strip()`` over segments.
When a segment's ``text`` was JSON ``null``, ``str(None)`` produced the literal
string ``"None"``, which then polluted the transcript. The fix uses
``str(s.get("text") or "")`` and drops empty segments.
"""
from agent.multimodal._dual_agent import _parse_asr_text


def test_null_text_segment_is_skipped_not_stringified():
    js = {"segments": [{"text": None}, {"text": "hello"}, {"text": None}]}
    assert _parse_asr_text(js) == "hello"


def test_all_null_segments_yield_empty_string():
    js = {"segments": [{"text": None}, {"text": None}]}
    assert _parse_asr_text(js) == ""


def test_missing_text_key_segment_skipped():
    js = {"segments": [{}, {"text": "world"}]}
    assert _parse_asr_text(js) == "world"


def test_normal_segments_joined():
    js = {"segments": [{"text": "a"}, {"text": "b"}, {"text": " c "}]}
    assert _parse_asr_text(js) == "a b c"


def test_top_level_text_still_wins():
    # The top-level text/result/transcript keys take precedence over segments.
    js = {"text": "top", "segments": [{"text": "seg"}]}
    assert _parse_asr_text(js) == "top"
