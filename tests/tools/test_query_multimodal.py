"""Focused contracts for the model-visible ``query_multimodal`` entry."""

import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from tools.mm_memory_tool import (
    QUERY_MULTIMODAL_SCHEMA,
    query_multimodal,
    recall_multimodal_memory,
)
from tools.registry import registry
from toolsets import TOOLSETS, _HERMES_CORE_TOOLS


def test_query_multimodal_is_the_only_model_visible_entry():
    """The legacy Python alias must not create a second model schema."""
    assert callable(recall_multimodal_memory)
    assert QUERY_MULTIMODAL_SCHEMA["name"] == "query_multimodal"
    assert registry.get_entry("query_multimodal") is not None
    assert registry.get_entry("recall_multimodal_memory") is None

    assert "query_multimodal" in _HERMES_CORE_TOOLS
    assert "recall_multimodal_memory" not in _HERMES_CORE_TOOLS
    assert "query_multimodal" in TOOLSETS["live_watcher"]["tools"]
    assert "recall_multimodal_memory" not in TOOLSETS["live_watcher"]["tools"]


def test_fresh_live_watcher_schema_contains_only_the_new_entry_name():
    """Exercise the real schema assembly path used by a newly built agent."""
    from model_tools import get_tool_definitions

    definitions = get_tool_definitions(
        enabled_toolsets=["live_watcher"],
        quiet_mode=True,
        skip_tool_search_assembly=True,
    )
    names = {
        definition["function"]["name"]
        for definition in definitions
        if definition.get("type") == "function"
    }

    assert "query_multimodal" in names
    assert "recall_multimodal_memory" not in names
    # Guard the complete model-visible payload, including descriptions and
    # parameter help, so the retired call name cannot be re-primed indirectly.
    assert "recall_multimodal_memory" not in json.dumps(definitions)


def test_gateway_query_worker_handoff_does_not_require_recall_worker():
    """Current-frame VQA remains available when historical memory is off."""
    calls = []

    class _Engine:
        recall_worker = None

        def submit_query_async(self, instruction, **kwargs):
            calls.append((instruction, kwargs))
            return "qry_accepted"

    original_query = "画面里这家银行 2026-07-24 的收盘价是多少？"
    agent = SimpleNamespace(
        _active_parent_user_message_id="turn_42",
        _active_user_message_text=original_query,
    )

    with (
        patch(
            "tools.mm_memory_tool._resolve_mm_engine",
            return_value=(_Engine(), agent),
        ),
        patch(
            "tools.mm_memory_tool._resolve_send_anchor_ts",
            return_value=12.5,
        ),
    ):
        data = json.loads(query_multimodal(
            query="先识别银行，再查指定日期收盘价",
            session_id="sid",
        ))

    assert data["control"] == "handoff"
    assert data["reply_owner"] == "query_worker"
    assert data["handoff_mode"] == "deferred_reply"
    assert data["task_id"] == "qry_accepted"
    assert data["parent_user_message_id"] == "turn_42"

    assert len(calls) == 1
    instruction, kwargs = calls[0]
    assert original_query in instruction
    assert kwargs["original_user_query"] == original_query
    assert kwargs["ask_ts"] == 12.5
    assert kwargs["parent_user_message_id"] == "turn_42"


def test_evidence_mode_returns_normal_tool_result_and_keeps_main_ownership():
    engine = SimpleNamespace(
        query_visual_evidence=Mock(return_value={
            "ok": True,
            "evidence": (
                "Observed: a PDF viewer is open.\n"
                "Relevant text/identifiers: report.pdf, page 3/18."
            ),
            "ask_ts": 12.5,
            "n_frames": 3,
            "t_start": 11.5,
            "t_end": 12.5,
            "limitations": ["Visible frames do not establish the full document."],
            "elapsed_sec": 0.8,
        }),
    )
    agent = SimpleNamespace(
        _active_parent_user_message_id="turn_pdf",
        _active_user_message_text="帮我看看画面里的 PDF 讲了什么",
        # Evidence mode is allowed after ordinary tool work because it does not
        # create a second reply producer.
        _query_multimodal_handoff_block_reason="prior_tool_work",
    )
    with (
        patch(
            "tools.mm_memory_tool._resolve_mm_engine",
            return_value=(engine, agent),
        ),
        patch(
            "tools.mm_memory_tool._resolve_send_anchor_ts",
            return_value=12.5,
        ),
    ):
        data = json.loads(query_multimodal(
            query="定位 PDF 文件名、页码和可见主题，供后续 PDF skill 使用",
            response_mode="evidence",
            session_id="sid",
        ))

    assert data["mode"] == "evidence"
    assert "PDF viewer" in data["visual_evidence"]
    assert data["evidence_scope"] == {
        "source": "ask_time_frames",
        "ask_ts": 12.5,
        "n_frames": 3,
        "t_start": 11.5,
        "t_end": 12.5,
    }
    assert "Main Agent retains reply ownership" in data["next_action"]
    assert "control" not in data
    assert "reply_owner" not in data
    engine.query_visual_evidence.assert_called_once_with(
        "定位 PDF 文件名、页码和可见主题，供后续 PDF skill 使用",
        original_user_query="帮我看看画面里的 PDF 讲了什么",
        ask_ts=12.5,
        timeout=45.0,
    )


def test_evidence_mode_reports_missing_runtime_capability_without_handoff():
    agent = SimpleNamespace(
        _active_parent_user_message_id="turn_pdf",
        _active_user_message_text="read the PDF on screen",
    )
    with patch(
        "tools.mm_memory_tool._resolve_mm_engine",
        return_value=(SimpleNamespace(), agent),
    ):
        data = json.loads(query_multimodal(
            query="ground the visible PDF",
            response_mode="evidence",
            session_id="sid",
        ))

    assert data["success"] is False
    assert data["code"] == "query_worker_evidence_unavailable"
    assert "control" not in data


def test_query_multimodal_schema_exposes_explicit_dual_response_modes():
    mode = QUERY_MULTIMODAL_SCHEMA["parameters"]["properties"]["response_mode"]
    assert mode["enum"] == ["direct", "evidence"]
    assert mode["default"] == "direct"
    assert QUERY_MULTIMODAL_SCHEMA["parameters"]["required"] == ["query"]


def test_invalid_response_mode_fails_before_runtime_lookup():
    with patch("tools.mm_memory_tool._resolve_mm_engine") as resolve:
        data = json.loads(query_multimodal(
            query="inspect the screen",
            response_mode="delegate-everything",
            session_id="sid",
        ))

    assert data["success"] is False
    assert data["code"] == "invalid_query_multimodal_response_mode"
    resolve.assert_not_called()


@pytest.mark.parametrize("query", [None, "", "   "])
def test_query_is_required_before_any_runtime_lookup(query):
    with patch("tools.mm_memory_tool._resolve_mm_engine") as resolve:
        data = json.loads(query_multimodal(query=query, session_id="sid"))

    assert data["success"] is False
    assert "query is required" in data["error"]
    resolve.assert_not_called()


def test_missing_multimodal_runtime_fails_without_side_effects():
    with patch(
        "tools.mm_memory_tool._resolve_mm_engine", return_value=(None, None)
    ):
        data = json.loads(query_multimodal(query="what is visible?", session_id="sid"))

    assert data["success"] is False
    assert "not available" in data["error"]


def test_exact_original_question_is_forwarded_without_synthetic_rewrite():
    submitted = []

    class _Engine:
        def submit_query_async(self, instruction, **kwargs):
            submitted.append((instruction, kwargs))
            return "qry_exact"

    original = "这是什么银行？"
    agent = SimpleNamespace(
        _active_parent_user_message_id="turn_exact",
        _active_user_message_text=original,
    )
    with (
        patch(
            "tools.mm_memory_tool._resolve_mm_engine",
            return_value=(_Engine(), agent),
        ),
        patch("tools.mm_memory_tool._resolve_send_anchor_ts", return_value=7.0),
    ):
        data = json.loads(query_multimodal(query=original, session_id="sid"))

    assert data["task_id"] == "qry_exact"
    assert submitted[0][0] == original
    assert submitted[0][1]["original_user_query"] == original
    assert submitted[0][1]["ask_ts"] == 7.0


def test_main_agent_focus_preserves_original_question_and_external_constraints():
    submitted = []

    class _Engine:
        def submit_query_async(self, instruction, **kwargs):
            submitted.append((instruction, kwargs))
            return "qry_focus"

    original = "识别画面中的银行，并查询它在2026年07月24日的港股收盘价（港元）"
    focus = "先识别银行名称，再查指定日期的港股收盘价"
    agent = SimpleNamespace(
        _active_parent_user_message_id="turn_focus",
        _active_user_message_text=original,
    )
    with patch(
        "tools.mm_memory_tool._resolve_mm_engine",
        return_value=(_Engine(), agent),
    ):
        data = json.loads(query_multimodal(query=focus, session_id="sid"))

    instruction, kwargs = submitted[0]
    assert data["task_id"] == "qry_focus"
    assert original in instruction
    assert focus in instruction
    assert "2026年07月24日" in instruction
    assert "港元" in instruction
    assert instruction.index("AUTHORITATIVE ORIGINAL USER QUESTION") < instruction.index(
        "CONTEXT-RESOLVED MAIN AGENT DELEGATION BRIEF"
    )
    assert "sole authoritative answer contract" in instruction
    assert "must not narrow, replace, contradict, or reinterpret" in instruction
    assert "answers every requested part" in instruction
    assert kwargs["original_user_query"] == original


def test_context_resolved_brief_preserves_approximate_anchor_and_all_subtasks():
    original = "再看看这个面包店呢，离我家最近的一家在哪？"
    resolved = (
        "识别当前画面中的面包店品牌；上一轮只能将用户住处定位到岗厦城附近，"
        "完整楼栋和门牌不明；地图搜索该品牌距岗厦城最近的门店，并说明精度限制。"
    )
    engine = SimpleNamespace(
        submit_query_async=Mock(return_value="qry_follow_up"),
    )
    agent = SimpleNamespace(_active_parent_user_message_id="turn_follow_up",
                            _active_user_message_text=original)
    with patch(
        "tools.mm_memory_tool._resolve_mm_engine",
        return_value=(engine, agent),
    ):
        data = json.loads(query_multimodal(query=resolved, session_id="sid"))

    instruction = engine.submit_query_async.call_args.args[0]
    kwargs = engine.submit_query_async.call_args.kwargs
    assert data["task_id"] == "qry_follow_up"
    for evidence in (
        original,
        resolved,
        "识别当前画面中的面包店品牌",
        "岗厦城附近",
        "完整楼栋和门牌不明",
        "地图搜索",
        "preserving uncertainty",
    ):
        assert evidence in instruction
    assert kwargs["original_user_query"] == original


def test_bad_visible_price_focus_cannot_replace_original_market_price_question():
    submitted = []

    class _Engine:
        def submit_query_async(self, instruction, **kwargs):
            submitted.append((instruction, kwargs))
            return "qry_price"

    original = "我拿的饮料是啥，多少钱一瓶？"
    narrowed_focus = "读取包装上可见的价格；如果看不清价格就说无法确认"
    agent = SimpleNamespace(
        _active_parent_user_message_id="turn_price",
        _active_user_message_text=original,
    )
    with patch(
        "tools.mm_memory_tool._resolve_mm_engine",
        return_value=(_Engine(), agent),
    ):
        data = json.loads(query_multimodal(query=narrowed_focus, session_id="sid"))

    instruction, kwargs = submitted[0]
    assert data["task_id"] == "qry_price"
    assert original in instruction
    assert narrowed_focus in instruction
    assert "sole authoritative answer contract" in instruction
    assert "Ignore any part of the brief that does" in instruction
    assert kwargs["original_user_query"] == original


def test_rejected_dispatch_returns_terminal_queryworker_receipt_without_recall():
    class _Engine:
        recall_memory = Mock(side_effect=AssertionError("must not recall synchronously"))

        def submit_query_async(self, _instruction, **_kwargs):
            return ""

    agent = SimpleNamespace(
        _active_parent_user_message_id="turn_rejected",
        _active_user_message_text="what is visible?",
    )
    engine = _Engine()
    with patch(
        "tools.mm_memory_tool._resolve_mm_engine",
        return_value=(engine, agent),
    ):
        data = json.loads(query_multimodal(query="what is visible?", session_id="sid"))

    assert data["success"] is False
    assert data["control"] == "handoff"
    assert data["reply_owner"] == "query_worker"
    assert data["handoff_mode"] == "receipt"
    assert data["parent_user_message_id"] == "turn_rejected"
    engine.recall_memory.assert_not_called()


def test_query_multimodal_never_silently_degrades_to_recall_only():
    recall_calls = []

    class _Engine:
        recall_worker = object()

        def submit_query_async(self, _instruction, **_kwargs):
            raise AssertionError("no parent slot means no async submission")

        def recall_memory(self, **kwargs):
            recall_calls.append(kwargs)
            return {"ok": True, "found": True, "findings": "recall-only"}

    with patch(
        "tools.mm_memory_tool._resolve_mm_engine",
        return_value=(_Engine(), None),
    ):
        data = json.loads(query_multimodal(
            query="what is on screen now?",
            session_id="sid",
        ))

    assert data["success"] is False
    assert data["code"] == "query_worker_handoff_unavailable"
    assert data["missing"] == ["parent_user_message_id"]
    assert "will not fall back to Recall-only" in data["error"]
    assert recall_calls == []


def test_missing_queryworker_dispatcher_is_reported_explicitly():
    engine = SimpleNamespace(recall_worker=object())
    agent = SimpleNamespace(
        _active_parent_user_message_id="turn_1",
        _active_user_message_text="what is visible?",
    )
    with patch(
        "tools.mm_memory_tool._resolve_mm_engine",
        return_value=(engine, agent),
    ):
        data = json.loads(query_multimodal(query="what is visible?", session_id="sid"))

    assert data["success"] is False
    assert data["code"] == "query_worker_handoff_unavailable"
    assert data["missing"] == ["query_worker_dispatcher"]


@pytest.mark.parametrize(
    ("reason", "reason_fragment"),
    [
        ("mixed_tool_batch", "same model response"),
        ("prior_tool_work", "already ran earlier"),
    ],
)
def test_query_handoff_block_is_checked_before_worker_submission(
    reason, reason_fragment,
):
    """A non-solo QueryWorker call cannot create an async reply producer."""

    class _Engine:
        def submit_query_async(self, *_args, **_kwargs):
            raise AssertionError("blocked QueryWorker must not be submitted")

    agent = SimpleNamespace(
        _active_parent_user_message_id="turn_blocked",
        _active_user_message_text="what is visible?",
        _query_multimodal_handoff_block_reason=reason,
    )
    with patch(
        "tools.mm_memory_tool._resolve_mm_engine",
        return_value=(_Engine(), agent),
    ):
        data = json.loads(query_multimodal(
            query="what is visible?",
            session_id="sid",
        ))

    assert data["success"] is False
    assert data["code"] == "query_multimodal_requires_solo_turn"
    assert data["reason"] == reason
    assert reason_fragment in data["error"]
    assert "control" not in data


def test_legacy_python_alias_keeps_synchronous_recall_compatibility():
    calls = []

    class _Engine:
        recall_worker = object()

        def recall_memory(self, **kwargs):
            calls.append(kwargs)
            return {
                "ok": True,
                "found": True,
                "findings": "legacy result",
                "frame_ids": [],
                "rounds": 1,
                "elapsed_sec": 0.1,
            }

    with patch(
        "tools.mm_memory_tool._resolve_mm_engine",
        return_value=(_Engine(), None),
    ):
        data = json.loads(recall_multimodal_memory(
            query="earlier object",
            session_id="sid",
        ))

    assert data["found"] is True
    assert data["findings"] == "legacy result"
    assert calls[0]["brief"] == "earlier object"
