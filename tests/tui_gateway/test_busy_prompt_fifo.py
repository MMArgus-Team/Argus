# -*- coding: utf-8 -*-
"""Busy prompt intake keeps independent FIFO turns."""

import threading
import unittest
from types import SimpleNamespace
from unittest import mock


class TestBusyPromptFifo(unittest.TestCase):
    def setUp(self):
        import tui_gateway.server as server

        self.server = server
        self.orig_submit = server._run_prompt_submit
        self.orig_emit = server._emit
        self.dispatched = []
        self.correlations = []
        self.internal_origins = []

        def fake_submit(
            rid, sid, session, text, *, user_originated=False,
            client_request_id="", internal_origin="",
        ):
            self.dispatched.append((text, user_originated))
            self.correlations.append(client_request_id)
            self.internal_origins.append(internal_origin)
            with session["history_lock"]:
                session["running"] = False
                if internal_origin in {"monitor_hook", "watcher_hook"}:
                    session["_monitor_hook_running"] = False

        server._run_prompt_submit = fake_submit
        server._emit = lambda *args, **kwargs: None

    def tearDown(self):
        self.server._run_prompt_submit = self.orig_submit
        self.server._emit = self.orig_emit

    @staticmethod
    def session(running=True):
        return {
            "history_lock": threading.Lock(),
            "running": running,
            "queued_prompt": None,
            "queued_prompts": [],
        }

    def test_three_submits_remain_three_turns_in_order(self):
        session = self.session(running=True)
        self.assertEqual(self.server._enqueue_prompt(session, "one", None), 1)
        self.assertEqual(self.server._enqueue_prompt(session, "two", None), 2)
        self.assertEqual(self.server._enqueue_prompt(session, "three", None), 3)
        self.assertEqual(
            [item["text"] for item in session["queued_prompts"]],
            ["one", "two", "three"],
        )

        session["running"] = False
        while self.server._drain_queued_prompt("rid", "sid", session):
            pass
        self.assertEqual(
            self.dispatched,
            [("one", True), ("two", True), ("three", True)],
        )
        self.assertIsNone(session["queued_prompt"])

    def test_queue_only_does_not_interrupt_live_answer(self):
        class Agent:
            interrupts = 0

            def interrupt(self):
                self.interrupts += 1

        session = self.session(running=True)
        agent = Agent()
        session["agent"] = agent
        result = self.server._handle_busy_submit(
            "rid", "sid", session, "next question", None, queue_only=True)
        self.assertEqual(result["result"]["status"], "queued")
        self.assertEqual(result["result"]["queue_position"], 1)
        self.assertEqual(agent.interrupts, 0)

    def test_backend_followup_echoes_when_dequeued(self):
        session = self.session(running=True)
        self.server._enqueue_prompt(
            session, "monitor follow-up", None,
            user_originated=False, origin="monitor_hook")
        session["running"] = False
        self.assertTrue(self.server._drain_queued_prompt("rid", "sid", session))
        self.assertEqual(self.dispatched, [("monitor follow-up", False)])
        self.assertEqual(self.internal_origins, ["monitor_hook"])

    def test_query_id_and_turn_options_survive_fifo(self):
        # NB: deep_thinking / _mm_deep_thinking is no longer a session-level
        # option; thinking on/off is derived from agent.reasoning_config
        # (agent.reasoning_effort in config.yaml). This test now only guards
        # the client_request_id correlation surviving the FIFO drain.
        session = self.session(running=True)
        self.server._enqueue_prompt(
            session, "correlated question", None,
            metadata={"client_request_id": "turn_abc"},
        )
        session["running"] = False

        self.assertTrue(self.server._drain_queued_prompt("rid", "sid", session))
        self.assertEqual(self.correlations, ["turn_abc"])
        self.assertNotIn("_mm_deep_thinking", session)


class TestMultimodalMemorySessionBinding(unittest.TestCase):
    def setUp(self):
        import tui_gateway.server as server

        # These tests replace MemoryBackend with bare Mock objects, which do
        # not run a real thread/finalizer callback. Keep their process-local
        # ownership entries isolated from sibling cases.
        with server._MM_ACTIVE_MEMORY_BACKENDS_LOCK:
            server._MM_ACTIVE_MEMORY_BACKENDS.clear()
            server._MM_ACTIVE_WATCHERS.clear()

    def tearDown(self):
        import tui_gateway.server as server

        with server._MM_ACTIVE_MEMORY_BACKENDS_LOCK:
            server._MM_ACTIVE_MEMORY_BACKENDS.clear()
            server._MM_ACTIVE_WATCHERS.clear()

    def test_only_canonical_multimodal_source_owns_runtime(self):
        import tui_gateway.server as server

        self.assertTrue(server._is_multimodal_runtime_session(
            {"source": "multimodal"}))
        self.assertTrue(server._is_multimodal_runtime_session(
            {"source": " MULTIMODAL "}))
        self.assertFalse(server._is_multimodal_runtime_session(
            {"source": "tool"}))
        self.assertFalse(server._is_multimodal_runtime_session(
            {"source": "tui"}))

    def test_backend_persists_durable_session_key_not_live_transport_id(self):
        import tui_gateway.server as server

        backend = mock.Mock()
        backend.start.return_value = True
        runtime_config = object()
        session = {}
        with (
            mock.patch(
                "agent.multimodal.hermes_glue.flatten_mm_config",
                return_value={"memory_enabled": True},
            ),
            mock.patch(
                "agent.multimodal.hermes_glue.build_config",
                return_value=runtime_config,
            ) as build_config,
            mock.patch(
                "agent.multimodal.memory_backend.MemoryBackend",
                return_value=backend,
            ) as backend_cls,
        ):
            result = server._maybe_start_memory_backend(
                "live-sid", "durable-session-key", object(), session=session)

        self.assertIs(result, backend)
        self.assertIs(session["_mm_memory_backend"], backend)
        self.assertEqual(
            backend_cls.call_args.kwargs["session_id"],
            "durable-session-key",
        )
        self.assertIs(
            backend_cls.call_args.kwargs["runtime_config"], runtime_config)
        build_config.assert_called_once()
        self.assertEqual(
            build_config.call_args.kwargs["session_id"],
            "durable-session-key",
        )
        backend.start.assert_called_once_with(
            timeout=server._MM_MEMORY_STARTUP_TIMEOUT_SEC)

    def test_failed_backend_is_stopped_and_never_returned_to_watcher(self):
        import tui_gateway.server as server

        backend = mock.Mock()
        backend.start.return_value = False
        backend.startup_error = RuntimeError("broken provider")
        with (
            mock.patch(
                "agent.multimodal.hermes_glue.flatten_mm_config",
                return_value={"memory_enabled": True},
            ),
            mock.patch(
                "agent.multimodal.hermes_glue.build_config",
                return_value=object(),
            ),
            mock.patch(
                "agent.multimodal.memory_backend.MemoryBackend",
                return_value=backend,
            ),
        ):
            result = server._maybe_start_memory_backend(
                "live-sid", "durable-session-key", object())

        self.assertIsNone(result)
        backend.stop.assert_called_once_with(
            timeout=server._MM_MEMORY_STOP_TIMEOUT_SEC)

    def test_watcher_fails_closed_when_enabled_memory_backend_is_missing(self):
        import tui_gateway.server as server

        with (
            mock.patch(
                "agent.multimodal.hermes_glue.flatten_mm_config",
                return_value={"enabled": True, "memory_enabled": True},
            ),
            mock.patch(
                "agent.multimodal.watcher_engine.WatcherAgent",
            ) as watcher_cls,
        ):
            result = server._maybe_start_live_watcher_agent(
                "live-sid", object(), None, {})

        self.assertIsNone(result)
        watcher_cls.assert_not_called()

    def test_legacy_memory_disabled_flag_does_not_allow_standalone_watcher(self):
        import tui_gateway.server as server

        with (
            mock.patch(
                "agent.multimodal.hermes_glue.flatten_mm_config",
                return_value={"enabled": True, "memory_enabled": False},
            ),
            mock.patch(
                "agent.multimodal.watcher_engine.WatcherAgent",
            ) as watcher_cls,
        ):
            result = server._maybe_start_live_watcher_agent(
                "live-sid", object(), None, {})

        self.assertIsNone(result)
        watcher_cls.assert_not_called()

    def test_unready_watcher_is_stopped_and_not_returned(self):
        import tui_gateway.server as server

        watcher = mock.Mock()
        watcher.start.return_value = False
        with (
            mock.patch(
                "agent.multimodal.hermes_glue.flatten_mm_config",
                return_value={"enabled": True, "memory_enabled": False},
            ),
            mock.patch(
                "agent.multimodal.watcher_engine.WatcherAgent",
                return_value=watcher,
            ),
        ):
            result = server._maybe_start_live_watcher_agent(
                "live-sid", object(), object(), {})

        self.assertIsNone(result)
        watcher.start.assert_called_once_with(
            timeout=server._MM_WATCHER_STARTUP_TIMEOUT_SEC)
        watcher.stop.assert_called_once_with(
            timeout=server._MM_WATCHER_STOP_TIMEOUT_SEC)

    def test_watcher_hook_is_completion_only_not_per_round(self):
        import tui_gateway.server as server

        watcher = mock.Mock()
        watcher.start.return_value = True
        agent = SimpleNamespace(
            session_id="durable-watcher",
            mm_watchers={
                "req_once": {
                    "label": "视频总结",
                    "task_instruction": "看到播放器结束时总结",
                    "hook_main_agent": True,
                    "hook_instruction": "向用户呈现最终总结",
                    "status": "running",
                }
            },
        )
        session = {
            "agent": agent,
            "history": [],
            "history_lock": threading.Lock(),
            "history_version": 0,
            "running": True,  # keep completion hook queued for inspection
            "session_key": "",
        }
        emitted = []
        with (
            mock.patch(
                "agent.multimodal.hermes_glue.flatten_mm_config",
                return_value={"enabled": True, "memory_enabled": False},
            ),
            mock.patch(
                "agent.multimodal.watcher_engine.WatcherAgent",
                return_value=watcher,
            ) as watcher_cls,
            mock.patch.object(
                server,
                "_emit",
                side_effect=lambda event, sid, payload=None: emitted.append(
                    (event, sid, payload or {})),
            ),
        ):
            result = server._maybe_start_live_watcher_agent(
                "live-hook-once", object(), object(), session)
            self.assertIs(result, watcher)
            callbacks = watcher_cls.call_args.kwargs

            callbacks["on_round_report"](
                "req_once", 1, "第一段分析结果")
            self.assertEqual(session.get("_watcher_hook_queue", []), [])

            callbacks["on_delegation_complete"](
                "req_once",
                "看到播放器结束时总结",
                "完整累积后的最终总结",
                "task_complete",
            )

            # Pausing/deleting is not successful task completion and must never
            # synthesize a hidden main-agent turn.
            agent.mm_watchers["req_paused"] = {
                "label": "暂停任务",
                "task_instruction": "继续观察",
                "hook_main_agent": True,
                "hook_instruction": "返回总结",
                "status": "running",
            }
            callbacks["on_delegation_complete"](
                "req_paused", "继续观察", "未完成的分段内容", "disabled")

            # Closing the actual camera/screen source is the other successful
            # completion path and also queues exactly one consolidated hook.
            agent.mm_watchers["req_source_end"] = {
                "label": "流结束总结",
                "task_instruction": "观看到视频源结束",
                "hook_main_agent": True,
                "hook_instruction": "返回总结",
                "status": "running",
            }
            callbacks["on_delegation_complete"](
                "req_source_end", "观看到视频源结束", "源结束完整总结", "source_end")

        queued = session.get("_watcher_hook_queue") or []
        self.assertEqual(len(queued), 2)
        self.assertEqual(queued[0]["rid"], "req_once")
        self.assertEqual(queued[0]["report"], "完整累积后的最终总结")
        self.assertEqual(queued[1]["rid"], "req_source_end")
        self.assertEqual(queued[1]["report"], "源结束完整总结")
        self.assertFalse(any(h["rid"] == "req_paused" for h in queued))
        final_ids = {
            payload.get("request_id")
            for event, _sid, payload in emitted
            if event == "watcher.final"
        }
        self.assertEqual(final_ids, {"req_once", "req_source_end"})


class TestQueryWorkerTurnProjection(unittest.TestCase):
    @staticmethod
    def session(results=None, history=None):
        return {
            "history_lock": threading.Lock(),
            "history": list(history or []),
            "_mm_query_results": list(results or []),
        }

    @staticmethod
    def completed(query, answer, *, task_id="q1", originated_at=1.0,
                  completed_at=2.0):
        return {
            "task_id": task_id,
            "parent_user_message_id": f"turn_{task_id}",
            "projection_key": task_id,
            "query": query,
            "answer": answer,
            "status": "complete",
            "originated_at": originated_at,
            "completed_at": completed_at,
            "projection_state": "pending",
        }

    def test_incomplete_or_answerless_result_projects_neither_q_nor_a(self):
        import tui_gateway.server as server

        session = self.session(results=[
            {
                "task_id": "running",
                "query": "Q running",
                "answer": "",
                "status": "running",
                "projection_state": "pending",
            },
            {
                "task_id": "empty",
                "query": "Q empty",
                "answer": "",
                "status": "complete",
                "projection_state": "pending",
            },
            {
                "task_id": "failed",
                "query": "Q failed",
                "answer": "failure detail",
                "status": "error",
                "projection_state": "pending",
            },
        ])

        projected = server._reserve_mm_query_turn_projection(session, "turn2")
        self.assertEqual(projected, [])
        self.assertTrue(all(
            row.get("projection_state") == "pending"
            for row in session["_mm_query_results"]
        ))

    def test_complete_result_is_plain_user_assistant_pair(self):
        import tui_gateway.server as server

        session = self.session(results=[
            self.completed("第二个物品多少钱？", "价格是 12 元。")
        ])
        projected = server._reserve_mm_query_turn_projection(session, "turn2")

        self.assertEqual(
            [(msg["role"], msg["content"]) for msg in projected],
            [("user", "第二个物品多少钱？"),
             ("assistant", "价格是 12 元。")],
        )
        self.assertNotIn("worker", str(projected).lower())
        row = session["_mm_query_results"][0]
        self.assertEqual(row["projection_state"], "reserved")

        self.assertTrue(server._commit_mm_query_turn_projection(
            session, "turn2", projected))
        self.assertEqual(row["projection_state"], "committed")
        self.assertEqual(session["history"], projected)
        self.assertEqual(
            server._reserve_mm_query_turn_projection(session, "turn3"), [])

    def test_aborted_projection_reservation_releases_whole_pair(self):
        import tui_gateway.server as server

        session = self.session(results=[self.completed("Q1", "A1")])
        first = server._reserve_mm_query_turn_projection(session, "turn2")
        self.assertEqual(len(first), 2)
        server._finish_mm_query_turn_projection(
            session, "turn2", committed=False)

        row = session["_mm_query_results"][0]
        self.assertEqual(row["projection_state"], "pending")
        second = server._reserve_mm_query_turn_projection(session, "turn3")
        self.assertEqual(
            [(msg["role"], msg["content"]) for msg in second],
            [("user", "Q1"), ("assistant", "A1")],
        )

    def test_available_pairs_follow_original_query_order(self):
        import tui_gateway.server as server

        session = self.session(results=[
            self.completed("Q2", "A2", task_id="q2",
                           originated_at=2.0, completed_at=3.0),
            self.completed("Q1", "A1", task_id="q1",
                           originated_at=1.0, completed_at=4.0),
        ])
        projected = server._reserve_mm_query_turn_projection(session, "turn3")
        self.assertEqual(
            [(msg["role"], msg["content"]) for msg in projected],
            [("user", "Q1"), ("assistant", "A1"),
             ("user", "Q2"), ("assistant", "A2")],
        )

    def test_resume_rehydrates_complete_pair_from_query_notice(self):
        import tui_gateway.server as server

        history = [{
            "role": "assistant",
            "content": {
                "type": "mm_notice",
                "mm_kind": "query",
                "mm_event_id": "turn_q1",
                "mm_label": "用户的完整问题",
                "text": "最终答案",
            },
        }]
        session = self.session(history=history)
        projected = server._reserve_mm_query_turn_projection(session, "turn2")
        self.assertEqual(
            [(msg["role"], msg["content"]) for msg in projected],
            [("user", "用户的完整问题"), ("assistant", "最终答案")],
        )

    def test_live_result_and_its_notice_do_not_duplicate_the_pair(self):
        import tui_gateway.server as server

        row = self.completed("用户原问", "答案", task_id="task_q1")
        row["parent_user_message_id"] = "turn_q1"
        row["projection_key"] = "turn_q1"
        notice = {
            "role": "assistant",
            "content": {
                "type": "mm_notice",
                "mm_kind": "query",
                "mm_event_id": "turn_q1",
                "mm_label": "用户原问",
                "mm_status": "complete",
                "text": "答案",
            },
        }
        session = self.session(results=[row], history=[notice])
        projected = server._reserve_mm_query_turn_projection(session, "turn2")
        self.assertEqual(len(projected), 2)
        self.assertEqual(len(session["_mm_query_results"]), 1)

    def test_error_notice_is_not_rehydrated_as_an_answer(self):
        import tui_gateway.server as server

        notice = {
            "role": "assistant",
            "content": {
                "type": "mm_notice",
                "mm_kind": "query",
                "mm_event_id": "turn_q1",
                "mm_label": "Q",
                "mm_status": "error",
                "text": "QueryWorker 执行失败",
            },
        }
        session = self.session(history=[notice])
        self.assertEqual(
            server._reserve_mm_query_turn_projection(session, "turn2"), [])

    def test_pending_query_user_notice_is_ui_only(self):
        import tui_gateway.server as server

        notice = {
            "role": "assistant",
            "content": {
                "type": "mm_notice",
                "mm_kind": "query_user",
                "mm_event_id": "turn_q1",
                "mm_label": "Q1",
                "mm_status": "running",
                "text": "Q1",
            },
        }
        session = self.session(history=[notice])
        self.assertEqual(server._strip_mm_context([notice]), [])
        self.assertEqual(server._reserve_mm_query_turn_projection(
            session, "turn2"), [])
        self.assertEqual(server._history_to_messages([notice]), [{
            "role": "user",
            "text": "Q1",
            "monitorLabel": "Q1",
            "eventId": "turn_q1",
            "requestId": "turn_q1",
        }])

    def test_projection_is_hidden_from_ui_but_marker_is_not_sent_to_model(self):
        import tui_gateway.server as server
        from agent.agent_runtime_helpers import sanitize_api_messages

        marker = server._MM_QUERY_PROJECTION_MARKER
        internal = [
            {"role": "user", "content": "Q", marker: "q1"},
            {"role": "assistant", "content": "A", marker: "q1"},
        ]
        self.assertEqual(server._history_to_messages(internal), [])

        wire = sanitize_api_messages(internal)
        self.assertEqual(wire, [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ])
        # Sanitizing the provider copy must not erase the internal UI marker.
        self.assertTrue(all(marker in msg for msg in internal))

        restored = [
            {
                "role": "user",
                "content": {
                    "type": "mm_query_projection",
                    "projection_key": "q1",
                    "text": "Q",
                },
            },
            {
                "role": "assistant",
                "content": {
                    "type": "mm_query_projection",
                    "projection_key": "q1",
                    "text": "A",
                },
            },
        ]
        self.assertEqual(server._history_to_messages(restored), [])
        normalized = server._strip_mm_context(restored)
        self.assertEqual(
            [(msg["role"], msg["content"]) for msg in normalized],
            [("user", "Q"), ("assistant", "A")],
        )
        self.assertTrue(all(marker in msg for msg in normalized))

    def test_projected_history_pair_is_persisted_in_order_with_hidden_envelope(self):
        from run_agent import AIAgent

        class RecordingDb:
            def __init__(self):
                self.rows = []

            def append_message(self, **kwargs):
                self.rows.append(kwargs)

        class MinimalAgent:
            def __init__(self):
                self._session_db = RecordingDb()
                self._session_db_created = True
                self._last_flushed_db_idx = 1
                self._flushed_db_message_ids = set()
                self._flushed_db_message_session_id = "session_1"
                self.session_id = "session_1"
                self.model = "test-model"

            @staticmethod
            def _apply_persist_user_message_override(messages):
                return None

            @staticmethod
            def _ensure_db_session():
                return None

        qa = [
            {"role": "user", "content": "Q", "_mm_query_projection": "q1"},
            {"role": "assistant", "content": "A", "_mm_query_projection": "q1"},
        ]
        current = {"role": "user", "content": "Q2"}
        agent = MinimalAgent()
        AIAgent._flush_messages_to_session_db(
            agent, qa + [current], conversation_history=qa)

        self.assertEqual(
            [row["role"] for row in agent._session_db.rows],
            ["user", "assistant", "user"],
        )
        self.assertEqual(
            [row["content"] for row in agent._session_db.rows[:2]],
            [
                {"type": "mm_query_projection",
                 "projection_key": "q1", "text": "Q"},
                {"type": "mm_query_projection",
                 "projection_key": "q1", "text": "A"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
