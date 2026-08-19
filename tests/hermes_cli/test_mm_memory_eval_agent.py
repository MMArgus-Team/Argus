"""Contracts for the headless ``mm-memory-eval --mode agent`` path."""

from __future__ import annotations

import base64
import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

from hermes_cli.mm_memory_eval import _make_agent_answerer


def _frame(ts: float, payload: bytes = b"jpeg") -> SimpleNamespace:
    return SimpleNamespace(
        ts=ts,
        jpeg_b64=base64.b64encode(payload).decode("ascii"),
    )


def test_agent_mode_prefetches_recall_then_synthesizes_without_live_tools():
    backend = Mock()
    backend.recall.return_value = {
        "ok": True,
        "found": True,
        "findings": "画面中的车是蓝色。",
        "frame_ids": ["frame-1"],
        "rounds": 2,
        "elapsed_sec": 0.4,
    }
    buf = Mock()
    buf.latest.return_value = [_frame(1.0), _frame(2.0)]
    captured = {}

    def _fake_run_agent(prompt, **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        for path in kwargs["image_paths"]:
            assert os.path.isfile(path)
        return "这辆车是蓝色的。"

    with patch("hermes_cli.oneshot._run_agent", side_effect=_fake_run_agent) as run_agent:
        answer, trace = _make_agent_answerer(backend, buf, 17.0)("车是什么颜色？")

    assert answer == "这辆车是蓝色的。"
    backend.recall.assert_called_once()
    recall_kwargs = backend.recall.call_args.kwargs
    assert recall_kwargs["brief"] == "车是什么颜色？"
    assert recall_kwargs["user_text"] == "车是什么颜色？"
    assert recall_kwargs["timeout"] == 17.0
    assert callable(recall_kwargs["on_progress"])
    run_agent.assert_called_once()
    assert captured["kwargs"]["toolsets"] == ["web", "vision"]
    assert captured["kwargs"]["use_config_toolsets"] is False
    assert "车是什么颜色？" in captured["prompt"]
    assert "画面中的车是蓝色。" in captured["prompt"]
    assert "query_multimodal" not in captured["prompt"]
    assert "recall_multimodal_memory" not in captured["prompt"]
    assert trace["ok"] is True
    assert trace["agent_mode"] is True
    assert trace["interactive_query_worker"] is False
    assert trace["execution_path"] == "offline_prefetched_recall_agent_synthesis"
    assert trace["agent_synthesis"] == "complete"
    assert trace["attached_recent_frames"] == 2
    assert all(not os.path.exists(path) for path in captured["kwargs"]["image_paths"])


def test_agent_mode_skips_synthesis_when_recall_itself_failed():
    backend = Mock()
    backend.recall.return_value = {
        "ok": False,
        "error": "recall timed out",
    }
    buf = Mock()

    with patch("hermes_cli.oneshot._run_agent") as run_agent:
        answer, trace = _make_agent_answerer(backend, buf, 3.0)("之前发生了什么？")

    assert answer == "(记忆召回失败: recall timed out)"
    run_agent.assert_not_called()
    buf.latest.assert_not_called()
    assert trace["ok"] is False
    assert trace["interactive_query_worker"] is False
    assert trace["agent_synthesis"] == "skipped_recall_error"
