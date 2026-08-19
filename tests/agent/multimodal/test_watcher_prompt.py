"""Prompt-contract checks for the deep-research ReAct brain (十项优化 #4/#6).

The #4 (prompt quality) and #6 (image-vs-search balance) optimizations live in
WATCHER_REACT_SYSTEM — there is no model in a unit test, so what we CAN
lock down is the prompt's binding invariants. These tests fail loudly if a future
edit removes the image-first mandate or the answer-quality rules, which is exactly
the regression that produced "过分依赖外部搜索、忽略画面".

They assert on the actual runtime prompt string the worker sends, not a copy.
"""

from __future__ import annotations

import agent.multimodal._workers as workers_mod

PROMPT = workers_mod.WATCHER_REACT_SYSTEM


# --------------------------------------------------------------------------- #
# #6 image-vs-external-search balance
# --------------------------------------------------------------------------- #
def test_prompt_declares_image_first_as_top_rule():
    # The #1 rule must be image-first, search-supplementary.
    assert "图像优先" in PROMPT
    assert "外部搜索是补充" in PROMPT
    # Explicit: search backfills BACKGROUND, doesn't replace looking at frames.
    assert "不是替代你看画面" in PROMPT


def test_prompt_has_negative_example_against_search_over_reliance():
    # A concrete "反面教材": seeing info on-screen but searching keywords anyway.
    assert "反面教材" in PROMPT
    assert "直接去搜" in PROMPT


def test_prompt_lets_it_answer_from_image_without_any_search():
    # Frame is clear -> answer directly, dispatch NO search at all.
    assert "完全不派 search" in PROMPT


def test_prompt_prefers_image_on_conflict():
    # When image and memory/search text disagree, image wins.
    assert "以画面为准" in PROMPT


def test_prompt_search_gated_on_only_when_image_insufficient():
    # search allowed ONLY when the frame can't show it AND external knowledge
    # is genuinely required.
    assert "画面里看不出" in PROMPT
    assert "需要外部知识" in PROMPT


# --------------------------------------------------------------------------- #
# #4 prompt quality: structured interpretation, no leakage, no hallucination.
# --------------------------------------------------------------------------- #
def test_prompt_demands_structured_interpretation_not_one_liner():
    # The output is a rich 解读, not a one-sentence chat reply.
    assert "解读" in PROMPT
    assert "不是一句口语答复" in PROMPT


def test_prompt_forbids_leakage_and_filler():
    # No "好的/经查询" openers, no "图搜结果/数据库显示" tool-leakage.
    assert "不要开场白" in PROMPT
    assert "不要露馅" in PROMPT


def test_prompt_forbids_hallucinating_numbers_and_names():
    # Numbers/dates/names/urls copied verbatim from frame/findings, not invented.
    assert "逐字复用" in PROMPT
    assert "不臆造" in PROMPT


def test_prompt_requires_surfacing_things_worth_exploring():
    # The user's stated ideal: proactively point out 值得深入的点.
    assert "值得进一步探索的点" in PROMPT


def test_prompt_requires_incremental_only_new_content():
    # Continuous research: only analyze what's NEW vs prior batches (no repeats).
    assert "增量" in PROMPT
    assert "新出现" in PROMPT


def test_prompt_graceful_when_search_unavailable():
    # #1/#6 interplay: if search is down, do NOT stall — answer from the frame.
    assert "外部搜索不可用" in PROMPT
    assert "不要卡住" in PROMPT


# --------------------------------------------------------------------------- #
# #4 output contract: native tool-calling (e874e8e0 起改用原生 function-calling,
# 不再是 JSON envelope; 终止语义 = 某轮不派工具即收尾, 无 can_answer 标志)。
# --------------------------------------------------------------------------- #
def test_prompt_advertises_native_tools_not_json_schema():
    # 只声明 text_search + recall_memory 两个原生工具; 图搜工具已废弃。
    assert "text_search" in PROMPT
    assert "recall_memory" in PROMPT
    assert "image_search" not in PROMPT
    # 不再要求模型输出 JSON envelope (thought/can_answer 作为 JSON key)。
    assert '"can_answer"' not in PROMPT
