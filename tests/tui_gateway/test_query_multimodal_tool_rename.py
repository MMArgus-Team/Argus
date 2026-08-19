"""Gateway contracts for the model-visible QueryWorker tool rename."""

from __future__ import annotations

import json
import threading
from unittest.mock import patch

import tui_gateway.server as server


def _handoff_result() -> str:
    return json.dumps({
        "success": True,
        "control": "handoff",
        "reply_owner": "query_worker",
        "handoff_mode": "deferred_reply",
        "task_id": "qry_1",
        "parent_user_message_id": "turn_1",
        "original_user_text": "这是什么银行？",
        "query": "识别银行",
        "status": "running",
        "note": "accepted",
    })


def test_live_trajectory_labels_only_the_new_tool_as_queryworker():
    assert server._trajectory_worker(
        "tool.start", {"name": "query_multimodal"}
    ) == "QueryWorker"
    assert server._trajectory_worker(
        "tool.complete", {"tool_name": "query_multimodal"}
    ) == "QueryWorker"
    assert server._trajectory_worker(
        "tool.start", {"name": "recall_multimodal_memory"}
    ) == "MainTool:recall_multimodal_memory"


def test_new_tool_handoff_populates_queryworker_live_state():
    emitted = []
    session = {
        "history_lock": threading.Lock(),
        "tool_started_at": {},
        "edit_snapshots": {},
    }
    with (
        patch.dict(server._sessions, {"sid": session}, clear=True),
        patch.object(server, "_tool_progress_enabled", return_value=True),
        patch.object(server, "_append_mm_context") as append_context,
        patch.object(server, "_emit", side_effect=lambda event, sid, payload: emitted.append(
            (event, sid, payload)
        )),
    ):
        server._on_tool_complete(
            "sid", "call_1", "query_multimodal", {"query": "识别银行"},
            _handoff_result(),
        )

    task = session["_mm_worker_tasks"]["qry_1"]
    assert task["parent_user_message_id"] == "turn_1"
    assert task["query"] == "这是什么银行？"
    append_context.assert_called_once_with(
        session,
        kind="query_user",
        text="这是什么银行？",
        event_id="turn_1",
        label="这是什么银行？",
        status="running",
    )
    [(_, _, payload)] = emitted
    assert payload["task_id"] == "qry_1"
    assert payload["request_id"] == "turn_1"
    assert "QueryWorker" in payload["dispatch_label"]


def test_legacy_tool_name_is_not_treated_as_a_new_live_handoff():
    emitted = []
    session = {
        "history_lock": threading.Lock(),
        "tool_started_at": {},
        "edit_snapshots": {},
    }
    with (
        patch.dict(server._sessions, {"sid": session}, clear=True),
        patch.object(server, "_tool_progress_enabled", return_value=True),
        patch.object(server, "_append_mm_context") as append_context,
        patch.object(server, "_emit", side_effect=lambda event, sid, payload: emitted.append(
            (event, sid, payload)
        )),
    ):
        server._on_tool_complete(
            "sid", "call_old", "recall_multimodal_memory", {"query": "old"},
            _handoff_result(),
        )

    assert "_mm_worker_tasks" not in session
    append_context.assert_not_called()
    [(_, _, payload)] = emitted
    assert "dispatch_label" not in payload
    assert "task_id" not in payload
    assert "request_id" not in payload
