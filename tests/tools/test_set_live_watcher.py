"""Tests for set_live_watcher lifecycle (create/update/delete) + registry.

The tool is now a CRUD surface over agent.mm_watchers (mirrors set_monitor):
all research is CONTINUOUS (query_type removed from the public API). The main
agent passes op / task_instruction / label / hook_* ; the tool generates the
request_id, pre-creates the analyse file, registers the task, and returns its
label. update/delete mutate the registry (batch-boundary effect in the engine).
"""

import json
import pathlib
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import hermes_constants
from tools.live_watcher_tool import (
    set_live_watcher, list_live_watcher, get_live_watcher,
)


def _agent():
    """A plain object (NOT MagicMock) so `mm_watchers` starts absent and the
    tool creates a real dict — a MagicMock would fake dict membership."""
    return types.SimpleNamespace()


class TestSetWatcherLifecycle(unittest.TestCase):
    def setUp(self):
        self._tmp = pathlib.Path(tempfile.mkdtemp())
        self._orig_home = hermes_constants.get_hermes_home
        hermes_constants.get_hermes_home = lambda: self._tmp

    def tearDown(self):
        hermes_constants.get_hermes_home = self._orig_home

    def _mock_engine(self):
        engine = MagicMock()
        engine.is_source_live.return_value = True
        engine.submit_complex_async.side_effect = (
            lambda task_instruction, request_id="", **_kw: request_id)
        engine.stop_delegation.return_value = True
        return engine

    def _sessions(self, engine, agent):
        return {"sess1": {"_mm_live_watcher_agent": engine, "agent": agent}}

    # ── create ────────────────────────────────────────────────────────────────
    def test_no_live_stream_returns_tool_error(self):
        agent = _agent()
        engine = self._mock_engine()
        engine.is_source_live.return_value = False
        dead = MagicMock()
        dead.size = 0
        dead._last_push_wall = None
        engine.frame_buffer = dead
        with patch("tui_gateway.server._sessions", self._sessions(engine, agent)):
            raw = set_live_watcher(task_instruction="持续帮我分析屏幕", session_id="sess1")
        data = json.loads(raw)
        self.assertFalse(data.get("success", True))
        self.assertIn("video stream", data.get("error", "").lower())
        engine.submit_complex_async.assert_not_called()

    def test_create_registers_task_and_returns_label(self):
        agent = _agent()
        engine = self._mock_engine()
        with patch("tui_gateway.server._sessions", self._sessions(engine, agent)):
            raw = set_live_watcher(
                task_instruction="调研汇总这场会议的知识点",
                label="会议纪要", session_id="sess1")
        data = json.loads(raw)
        # Monitor-parity receipt: clean success result (status running + note),
        # no "mode"/"render_hint". request_id retained so the later report can
        # backfill this turn's tool result.
        self.assertEqual(data["op"], "create")
        self.assertEqual(data["status"], "running")
        self.assertTrue(data["request_id"].startswith("req_"))
        self.assertEqual(data["watcher_id"], data["request_id"])
        self.assertEqual(data["label"], "会议纪要")
        # Turn-2 receipt: confirms the research was CREATED (real id present).
        self.assertIn("was created and is running", data["note"])
        # ★ 重构后 note 只 carry 数据事实; 行为指令 (reply short / 别写报告) 已挪到
        #   MM_LIVE_GUIDANCE 系统提示, 不再在 note 里。故不再断言"只回一句"文案。
        self.assertNotIn("mode", data)
        # registry populated
        self.assertIn(data["request_id"], agent.mm_watchers)
        ent = agent.mm_watchers[data["request_id"]]
        self.assertEqual(ent["label"], "会议纪要")
        self.assertEqual(ent["status"], "running")
        # submit was called (no query_type — the watcher has one mode)
        _, kwargs = engine.submit_complex_async.call_args
        self.assertNotIn("query_type", kwargs)
        self.assertEqual(kwargs.get("request_id"), data["request_id"])

    def test_create_ttl_explicit_maps_to_pacing(self):
        agent = _agent()
        engine = self._mock_engine()
        with patch("tui_gateway.server._sessions", self._sessions(engine, agent)):
            raw = set_live_watcher(
                task_instruction="盯比赛", ttl="1min", session_id="sess1")
        data = json.loads(raw)
        ent = agent.mm_watchers[data["request_id"]]
        # 1min → medium 档 (_PACE_TIERS): ttl_sec=60, target_frames=60。
        #   (重构后 target_frames 由 60/60 的 medium 档决定, 原测试的 30 是旧值。)
        self.assertEqual(ent["ttl"], "1min")
        self.assertEqual(ent["ttl_sec"], 60)
        self.assertEqual(ent["target_frames"], 60)

    def test_create_ttl_defaults_from_current_scene(self):
        from agent.multimodal._memory import FrameBuffer
        from types import SimpleNamespace as _NS
        agent = _agent()
        engine = self._mock_engine()
        # No explicit ttl → derive from agent.frame_buffer.current_scene (pace=slow).
        fb = FrameBuffer(_NS(buffer_seconds=60, buffer_capture_fps=2,
                             framebuffer_dhash_threshold_init=6,
                             framebuffer_dhash_threshold_min=2,
                             framebuffer_dhash_threshold_max=20))
        fb.set_current_scene({"scene": "meeting", "pace": "slow",
                              "ttl_sec": 360, "target_frames": 150})
        agent.frame_buffer = fb
        with patch("tui_gateway.server._sessions", self._sessions(engine, agent)):
            raw = set_live_watcher(task_instruction="记会议纪要", session_id="sess1")
        data = json.loads(raw)
        ent = agent.mm_watchers[data["request_id"]]
        self.assertEqual(ent["ttl_sec"], 360)
        self.assertEqual(ent["target_frames"], 150)

    def test_create_label_autoderived_when_omitted(self):
        agent = _agent()
        engine = self._mock_engine()
        with patch("tui_gateway.server._sessions", self._sessions(engine, agent)):
            raw = set_live_watcher(
                task_instruction="请持续研究屏幕上的这份长技术文档并解读", session_id="sess1")
        data = json.loads(raw)
        self.assertTrue(data["label"])  # non-empty auto-derived label
        self.assertLessEqual(len(data["label"]), 20)

    def test_create_with_hook(self):
        agent = _agent()
        engine = self._mock_engine()
        with patch("tui_gateway.server._sessions", self._sessions(engine, agent)):
            raw = set_live_watcher(
                task_instruction="研究屏幕", label="屏幕研究",
                hook_main_agent=True, hook_instruction="把要点记到笔记里",
                session_id="sess1")
        data = json.loads(raw)
        ent = agent.mm_watchers[data["request_id"]]
        self.assertTrue(ent["hook_main_agent"])
        self.assertEqual(ent["hook_instruction"], "把要点记到笔记里")
        self.assertTrue(data["hook_main_agent"])

    def test_background_sets_agent_stop_flags(self):
        agent = _agent()
        engine = self._mock_engine()
        with patch("tui_gateway.server._sessions", self._sessions(engine, agent)):
            raw = set_live_watcher(task_instruction="分析屏幕", session_id="sess1")
        data = json.loads(raw)
        self.assertEqual(agent._mm_route_last_mode, "background")
        self.assertTrue(agent._mm_route_background_stop)
        self.assertEqual(agent._mm_route_background_rid, data["request_id"])

    def test_watch_file_created(self):
        agent = _agent()
        engine = self._mock_engine()
        with patch("tui_gateway.server._sessions", self._sessions(engine, agent)):
            raw = set_live_watcher(task_instruction="调研这场会议", session_id="sess1")
        data = json.loads(raw)
        p = pathlib.Path(data["watch_file"])
        self.assertTrue(p.exists())
        head = p.read_text(encoding="utf-8")
        # Header carries the task + rid (no query-type classification anymore).
        self.assertIn("调研这场会议", head)
        self.assertIn(data["request_id"], head)

    # ── update ────────────────────────────────────────────────────────────────
    def test_update_changes_text_label_hook(self):
        agent = _agent()
        engine = self._mock_engine()
        with patch("tui_gateway.server._sessions", self._sessions(engine, agent)):
            raw = set_live_watcher(task_instruction="旧目标", label="旧", session_id="sess1")
            rid = json.loads(raw)["request_id"]
            raw2 = set_live_watcher(
                op="update", watcher_id=rid, task_instruction="新目标",
                label="新标签", hook_main_agent=True, hook_instruction="做X",
                session_id="sess1")
        data2 = json.loads(raw2)
        self.assertEqual(data2["op"], "update")
        self.assertEqual(data2["request_id"], rid)
        ent = agent.mm_watchers[rid]
        self.assertEqual(ent["task_instruction"], "新目标")
        self.assertEqual(ent["label"], "新标签")
        self.assertTrue(ent["hook_main_agent"])
        self.assertTrue(ent.get("_pending_update"))  # engine picks up next round
        from agent.multimodal import watch_file
        persisted = watch_file.read_state(rid)
        self.assertEqual(persisted["task_instruction"], "新目标")
        self.assertEqual(persisted["label"], "新标签")
        self.assertTrue(persisted["hook_main_agent"])
        # update must NOT spawn a second delegation
        engine.submit_complex_async.assert_called_once()

    def test_update_unknown_id_errors(self):
        agent = _agent()
        engine = self._mock_engine()
        with patch("tui_gateway.server._sessions", self._sessions(engine, agent)):
            raw = set_live_watcher(op="update", watcher_id="req_nope",
                                    task_instruction="x", session_id="sess1")
        self.assertFalse(json.loads(raw).get("success", True))

    def test_update_cancel_hook(self):
        agent = _agent()
        engine = self._mock_engine()
        with patch("tui_gateway.server._sessions", self._sessions(engine, agent)):
            rid = json.loads(set_live_watcher(
                task_instruction="研究", hook_main_agent=True,
                hook_instruction="做X", session_id="sess1"))["request_id"]
            set_live_watcher(op="update", watcher_id=rid,
                              hook_main_agent=False, session_id="sess1")
        self.assertFalse(agent.mm_watchers[rid]["hook_main_agent"])

    # ── delete ────────────────────────────────────────────────────────────────
    def test_delete_marks_deleted_and_stops(self):
        agent = _agent()
        engine = self._mock_engine()
        with patch("tui_gateway.server._sessions", self._sessions(engine, agent)):
            rid = json.loads(set_live_watcher(
                task_instruction="研究", session_id="sess1"))["request_id"]
            raw = set_live_watcher(op="delete", watcher_id=rid, session_id="sess1")
        data = json.loads(raw)
        self.assertEqual(data["op"], "delete")
        self.assertTrue(agent.mm_watchers[rid]["_deleted"])
        engine.stop_delegation.assert_called_once_with(rid, reason="deleted")

    def test_delete_unknown_id_errors(self):
        agent = _agent()
        engine = self._mock_engine()
        with patch("tui_gateway.server._sessions", self._sessions(engine, agent)):
            raw = set_live_watcher(op="delete", watcher_id="req_nope",
                                    session_id="sess1")
        self.assertFalse(json.loads(raw).get("success", True))

    def test_delete_inactive_task_is_removed_synchronously(self):
        agent = _agent()
        engine = self._mock_engine()
        with patch("tui_gateway.server._sessions", self._sessions(engine, agent)):
            rid = json.loads(set_live_watcher(
                task_instruction="研究", session_id="sess1"))["request_id"]
            engine.stop_delegation.return_value = False
            data = json.loads(set_live_watcher(
                op="delete", watcher_id=rid, session_id="sess1"))

        self.assertNotIn(rid, agent.mm_watchers)
        from agent.multimodal import watch_file
        self.assertEqual(watch_file.read_status(rid)["status"], "deleted")
        self.assertEqual(watch_file.read_state(rid)["stop_reason"], "deleted")
        self.assertIn("deleted immediately", data["note"])

    # ── enable / disable (interrupted-job on/off) ──────────────────────────────
    def test_disable_pauses_and_stops(self):
        agent = _agent()
        engine = self._mock_engine()
        with patch("tui_gateway.server._sessions", self._sessions(engine, agent)):
            rid = json.loads(set_live_watcher(
                task_instruction="研究", session_id="sess1"))["request_id"]
            raw = set_live_watcher(op="disable", watcher_id=rid, session_id="sess1")
        data = json.loads(raw)
        self.assertEqual(data["op"], "disable")
        # ★ 五态统一重构后: disable = 先 stopping (当前轮收尾, UI 显"正在停止"),
        #   收尾后引擎才落 interrupted。不再即时 disabled。
        self.assertEqual(agent.mm_watchers[rid]["status"], "stopping")
        engine.stop_delegation.assert_called_with(rid, reason="disabled")

    def test_enable_requires_live_stream(self):
        """Enabling an interrupted research with NO live stream fails — so the UI
        toggle rolls back to off."""
        agent = _agent()
        engine = self._mock_engine()
        engine.is_source_live.return_value = False   # stream off
        dead = MagicMock(); dead.size = 0; dead._last_push_wall = None
        engine.frame_buffer = dead
        # Pre-seed an interrupted research (as a reopen would).
        agent.mm_watchers = {"req_x": {"id": "req_x", "watcher_id": "req_x",
                                         "task_instruction": "研究",
                                         "status": "interrupted", "_interrupted": True}}
        with patch("tui_gateway.server._sessions", self._sessions(engine, agent)):
            raw = set_live_watcher(op="enable", watcher_id="req_x", session_id="sess1")
        self.assertFalse(json.loads(raw).get("success", True))
        # Still interrupted (not flipped to running) since enable failed.
        self.assertNotEqual(agent.mm_watchers["req_x"]["status"], "running")
        engine.submit_complex_async.assert_not_called()

    def test_enable_with_live_stream_restarts(self):
        agent = _agent()
        engine = self._mock_engine()   # is_source_live True
        agent.mm_watchers = {"req_x": {"id": "req_x", "watcher_id": "req_x",
                                         "task_instruction": "研究",
                                         "status": "interrupted", "_interrupted": True}}
        with patch("tui_gateway.server._sessions", self._sessions(engine, agent)):
            raw = set_live_watcher(op="enable", watcher_id="req_x", session_id="sess1")
        data = json.loads(raw)
        self.assertEqual(data["op"], "enable")
        self.assertEqual(agent.mm_watchers["req_x"]["status"], "running")
        self.assertNotIn("_interrupted", agent.mm_watchers["req_x"])
        engine.submit_complex_async.assert_called_once()

    # ── list / get ──────────────────────────────────────────────────────────
    def test_list_live_watcher(self):
        agent = _agent()
        engine = self._mock_engine()
        with patch("tui_gateway.server._sessions", self._sessions(engine, agent)):
            r1 = json.loads(set_live_watcher(
                task_instruction="研究A", label="A", session_id="sess1"))["request_id"]
            set_live_watcher(task_instruction="研究B", label="B", session_id="sess1")
            raw = list_live_watcher(session_id="sess1")
        data = json.loads(raw)
        self.assertTrue(data["found"])
        labels = {t["label"] for t in data["watchers"]}
        self.assertEqual(labels, {"A", "B"})
        ids = {t["watcher_id"] for t in data["watchers"]}
        self.assertIn(r1, ids)

    def test_list_empty(self):
        agent = _agent()
        engine = self._mock_engine()
        with patch("tui_gateway.server._sessions", self._sessions(engine, agent)):
            raw = list_live_watcher(session_id="sess1")
        self.assertFalse(json.loads(raw)["found"])

    def test_get_accepts_watcher_id_and_legacy_request_id(self):
        agent = _agent()
        engine = self._mock_engine()
        with patch("tui_gateway.server._sessions", self._sessions(engine, agent)):
            rid = json.loads(set_live_watcher(
                task_instruction="研究这个文档", session_id="sess1"))["request_id"]
            # watcher_id (new) and request_id (legacy) both resolve the same task
            raw_new = get_live_watcher(watcher_id=rid, session_id="sess1")
            raw_old = get_live_watcher(request_id=rid, session_id="sess1")
        d_new, d_old = json.loads(raw_new), json.loads(raw_old)
        self.assertTrue(d_new["found"])
        self.assertTrue(d_old["found"])
        self.assertEqual(d_new["request_id"], rid)

    def test_get_is_session_scoped_for_specific_and_list_reads(self):
        agent1, agent2 = _agent(), _agent()
        engine1, engine2 = self._mock_engine(), self._mock_engine()
        sessions = {
            "gw1": {"_mm_live_watcher_agent": engine1, "agent": agent1,
                    "session_key": "sess1"},
            "gw2": {"_mm_live_watcher_agent": engine2, "agent": agent2,
                    "session_key": "sess2"},
        }
        with patch("tui_gateway.server._sessions", sessions):
            rid1 = json.loads(set_live_watcher(
                task_instruction="会话一", session_id="sess1"))["request_id"]
            rid2 = json.loads(set_live_watcher(
                task_instruction="会话二", session_id="sess2"))["request_id"]
            foreign = json.loads(get_live_watcher(
                watcher_id=rid2, session_id="sess1"))
            own_list = json.loads(get_live_watcher(session_id="sess1"))

        self.assertFalse(foreign["found"])
        listed = {row["watcher_id"] for row in own_list["watchers"]}
        self.assertEqual(listed, {rid1})
        self.assertNotIn(rid2, listed)

    def test_get_rejects_invalid_watcher_id_before_file_access(self):
        agent = _agent()
        engine = self._mock_engine()
        with patch("tui_gateway.server._sessions", self._sessions(engine, agent)):
            data = json.loads(get_live_watcher(
                watcher_id="../../watch_req_secret", session_id="sess1"))
        self.assertFalse(data.get("success", True))
        self.assertIn("invalid watcher_id", data["error"])


if __name__ == "__main__":
    unittest.main()
