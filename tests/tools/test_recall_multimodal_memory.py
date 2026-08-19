"""Tests for recall_multimodal_memory — the main agent's sync entry into the
RecallWorker memory-recall loop.

The tool resolves the per-session WatcherAgent and calls
``engine.recall_memory(brief, user_text, timeout)`` (which runs RecallWorker on
the engine loop and returns a dict). These tests stub that method + the shared
engine resolver, so no real engine/event loop is needed.
"""

import json
import types
import unittest
from unittest.mock import patch

from tools.mm_memory_tool import recall_multimodal_memory


def _engine(recall_ret, *, has_worker=True):
    """Fake engine whose recall_memory returns a canned dict."""
    calls = []

    def _recall_memory(brief, user_text="", timeout=25.0):
        calls.append({"brief": brief, "user_text": user_text, "timeout": timeout})
        return recall_ret

    eng = types.SimpleNamespace(
        recall_worker=(object() if has_worker else None),
        recall_memory=_recall_memory,
    )
    eng._calls = calls
    return eng


class TestRecallMultimodalMemory(unittest.TestCase):
    def _run(self, engine, **kw):
        with patch("tools.mm_memory_tool._resolve_mm_engine",
                   return_value=(engine, None)):
            return json.loads(recall_multimodal_memory(session_id="sess1", **kw))

    def test_found_returns_findings(self):
        eng = _engine({
            "ok": True, "found": True,
            "findings": "穿红衣服的人在 02:10 拿起了咖啡杯。",
            "clues": ["c1"], "frame_ids": ["f1", "f2"],
            "rounds": 2, "elapsed_sec": 3.4,
        })
        data = self._run(eng, query="穿红衣服的人")
        self.assertTrue(data["found"])
        self.assertIn("红衣服", data["findings"])
        self.assertEqual(data["frame_ids"], ["f1", "f2"])
        self.assertEqual(data["rounds"], 2)
        # The receipt language is presentation, not the behavior contract.
        # The contract is that reliable findings are returned as evidence and
        # the caller is directed to answer from them.
        self.assertIn("answer the user from this evidence", data["note"])
        # brief threaded through; user_text defaults to query
        self.assertEqual(eng._calls[0]["brief"], "穿红衣服的人")
        self.assertEqual(eng._calls[0]["user_text"], "穿红衣服的人")

    def test_user_text_passed_through(self):
        eng = _engine({"ok": True, "found": True, "findings": "x",
                       "frame_ids": [], "clues": [], "rounds": 1,
                       "elapsed_sec": 1.0})
        self._run(eng, query="那个人是谁", user_text="刚才屏幕左边那个人是谁")
        self.assertEqual(eng._calls[0]["user_text"], "刚才屏幕左边那个人是谁")

    def test_not_found_surfaces_clues(self):
        eng = _engine({"ok": True, "found": False, "findings": "",
                       "clues": ["看到过一个杯子但不确定颜色"],
                       "frame_ids": [], "rounds": 3, "elapsed_sec": 5.0})
        data = self._run(eng, query="蓝色的杯子")
        self.assertFalse(data["found"])
        self.assertEqual(data["clues"], ["看到过一个杯子但不确定颜色"])
        self.assertIn("no reliable memory", data["note"])

    def test_timeout_soft_result(self):
        eng = _engine({"ok": False, "error": "recall timed out after 25s",
                       "timed_out": True})
        data = self._run(eng, query="很久以前的事")
        self.assertFalse(data["found"])
        self.assertTrue(data["timed_out"])
        self.assertIn("timed out", data["note"])

    def test_engine_error_is_tool_error(self):
        eng = _engine({
            "ok": False,
            "error": "submit failed: loop dead",
            "recall_trace": [{"phase": "error", "stage": "decision"}],
            "rounds": 1,
            "elapsed_sec": 0.4,
        })
        data = self._run(eng, query="x")
        self.assertFalse(data.get("success", True))
        self.assertIn("submit failed", data.get("error", ""))
        self.assertEqual(data["recall_trace"][0]["phase"], "error")
        self.assertEqual(data["rounds"], 1)

    def test_empty_query_errors(self):
        eng = _engine({"ok": True, "found": True, "findings": "x"})
        data = self._run(eng, query="   ")
        self.assertFalse(data.get("success", True))
        self.assertIn("query is required", data.get("error", ""))

    def test_no_engine_errors(self):
        with patch("tools.mm_memory_tool._resolve_mm_engine",
                   return_value=(None, None)):
            data = json.loads(recall_multimodal_memory(query="x", session_id="nope"))
        self.assertFalse(data.get("success", True))
        self.assertIn("not available", data.get("error", ""))

    def test_no_recall_worker_errors(self):
        eng = _engine({"ok": True, "found": True, "findings": "x"},
                      has_worker=False)
        data = self._run(eng, query="x")
        self.assertFalse(data.get("success", True))
        self.assertIn("not available", data.get("error", ""))

    def test_frame_ids_capped_at_20(self):
        eng = _engine({"ok": True, "found": True, "findings": "y",
                       "frame_ids": [f"f{i}" for i in range(50)],
                       "clues": [], "rounds": 1, "elapsed_sec": 1.0})
        data = self._run(eng, query="x")
        self.assertEqual(len(data["frame_ids"]), 20)


if __name__ == "__main__":
    unittest.main()
