"""Reply-ownership handoff regressions.

These tests are deliberately stdlib ``unittest`` so the repository's minimal
``mm_hermes`` environment can execute them without installing pytest.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from agent.conversation_loop import _collect_tool_handoffs
from agent.monitor_routing import direct_monitor_request_text
from agent.multimodal.watcher_engine import WatcherAgent
from tools.monitor_tool import (
    SET_MONITOR_SCHEMA,
    _resolve_silent,
    _set_monitor_handoff,
)
from tools.registry import tool_handoff, tool_result
from tools.mm_memory_tool import query_multimodal


def _tool_defs(*names: str) -> list[dict]:
    return [{
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    } for name in names]


def _tool_call(name: str, call_id: str = "call_1") -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments="{}"),
    )


def _prime_query_runtime(engine: WatcherAgent, loop) -> None:
    """Initialize the QueryWorker-only state for __new__ unit fixtures."""
    engine._stop = threading.Event()
    engine._healthy = True
    engine._ready = threading.Event()
    engine._ready.set()
    engine._query_lock = threading.RLock()
    engine._active_queries = {}
    engine._query_by_parent = {}
    engine._query_pending = set()
    engine._query_running = set()
    engine._query_max_concurrency = 2
    engine._query_max_pending = 8
    engine._query_semaphore = asyncio.Semaphore(2)


def _response(*, tool_calls=None, content="", finish_reason="tool_calls"):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        model="test/model",
        usage=None,
    )


def _monitor_fast_path_agent(response, *, session_db=None, session_id=None):
    from run_agent import AIAgent

    with (
        patch("run_agent.get_tool_definitions",
              return_value=_tool_defs("set_monitor")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("hermes_logging.setup_logging"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://example.invalid/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_db=session_db,
            session_id=session_id,
        )
    agent.client = MagicMock()
    agent._multimodal_session = True
    agent.mm_monitors = {}
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.client.chat.completions.create.return_value = response
    return agent


class HandoffConversationTests(unittest.TestCase):
    def test_explicit_new_monitor_requests_are_directly_recognized(self):
        agent = SimpleNamespace(
            _multimodal_session=True,
            valid_tool_names={"set_monitor"},
            mm_monitors={},
        )
        cases = [
            "你可以帮我监控一下画面吗 第一次看到手机告诉我",
            "帮我监控一下画面，第一次有人出现就告诉我",
            "第一次看到水杯的时候告诉我",
            "持续盯着门口，有人出现就通知我",
            "再加一个监控，画面第一次出现猫就提醒我",
            "新建一个手机监控，画面第一次出现手机就提醒我",
            "再加一个猫咪监控，画面第一次出现猫就提醒我",
            "监控屏幕上的日志，第一次出现 ERROR 就提醒我",
            "画面第一次出现‘删除失败’就提醒我",
            "画面里第一次有猫就提醒我",
            "帮我监控一下我点开第一个视频就告诉我这个视频的标题",
            "摄像头第一次拍到快递员时告诉我",
            "屏幕第一次变成红色时提醒我",
            "进度条第一次从画面消失后告诉我",
            "桌面第一次弹出通知时提醒我",
            "Watch my screen and alert me the first time an error dialog appears.",
            "Alert me the first time an update dialog appears on screen.",
            "Start another phone monitor and alert me the first time it appears on screen.",
            "从现在开始，每次看到手机都告诉我",
            "Starting right now, tell me every time a phone appears on screen.",
            "Tell me the first time you see a phone.",
            "Message me the first time the popup disappears from my display.",
            "Warn me the first time video playback finishes.",
            "Create a new screen monitor that alerts me the first time a QR code appears.",
            "Let me know the first time the UI shows that installation is complete.",
        ]
        for text in cases:
            with self.subTest(text=text):
                routed_text = direct_monitor_request_text(agent, text)
                self.assertEqual(routed_text, text)

    def test_monitor_gate_rejects_non_create_and_non_visual_intents(self):
        agent = SimpleNamespace(
            _multimodal_session=True,
            valid_tool_names={"set_monitor"},
            mm_monitors={},
        )
        cases = [
            "我开了哪些监控",
            "暂停门口监控",
            "视频流开着吗",
            "现在画面里有手机吗",
            "刚才出现过手机吗",
            "明天提醒我开会",
            "不要再监控手机",
            "看到手机就告诉我",
            "看到手机也告诉我",
            "如果画面出现手机就提醒我",
            "Tell me when you see a phone.",
            "帮我监控8080端口，挂了告诉我",
            "持续分析画面并总结整体变化",
            "监控为什么这么慢",
            "我刚才在画面看到报错，告诉我它是什么意思",
            "我在画面看到报错，告诉我它是什么意思",
            "屏幕弹出了报错，告诉我怎么修",
            "画面出现报错，告诉我该怎么办",
            "我在屏幕看到一个弹窗，告诉我怎么关闭",
            "Tell me if there is a phone on screen right now.",
            "Tell me if my screen share is on.",
            "Let me know if the camera is active.",
            "Analyze the video and alert me when it changes.",
            "Monitor my database and alert me if it fails.",
            "Watch the deployment and tell me if it fails.",
            "Tell me if there is a visible difference.",
            "Tell me if the object is in the center of the screen.",
            "An error appeared on screen; tell me how to fix it.",
            "Alert me when an error appears in my script.",
            "Tell me when the compiler error appears.",
            "You said 'tell me when you see a phone', correct?",
            "Did you just say 'tell me when you see a phone'?",
            "Watch the Kafka stream and tell me when lag rises.",
            "看到手机时告诉我，并给 Bob 发消息",
            "Alert me when a phone appears and send a message to Bob.",
            "看到手机时告诉我，截图保存",
            "Tell me when a phone appears and save a screenshot.",
            "告诉我屏幕监控有没有开",
            "告诉我摄像头监控有没有开",
            "Tell me whether the screen monitor is running.",
            "Is the screen watcher running? Let me know.",
            "Tell me when the screen watcher sees ERROR.",
            "Enable the screen watcher and tell me.",
            "如果刚才画面上那个报错再次出现就提醒我",
            "If that error appears again on screen, alert me.",
            "盯着屏幕，告诉我现在有什么",
            "监控一下画面并告诉我它的分辨率",
            "Watch my screen and tell me what you see.",
            "Monitor this video and tell me how many people are visible.",
            "更新门口的摄像头监控，看到人就提醒我",
            "Update the screen monitor to alert me when a tablet appears.",
            "监控摄像头 API，失败时提醒我",
            "Monitor the video upload process and alert me if it stalls.",
            "屏幕出现 ERROR 时告诉我并点击确定",
            "Alert me when the login button appears on screen, then click it.",
            "翻译成英文：看到手机就告诉我",
            "Write a prompt: tell me when a phone appears on screen.",
            "Tell me when the error dialog appeared on screen.",
            "告诉我屏幕上的错误提示是什么时候出现的",
            "Keep an eye on the screen and tell me if sharing is enabled.",
            "给已有监控加一个手机条件，看到手机告诉我",
            "Make the old camera monitor also tell me when it sees a package.",
            "分析视频，发生重要变化就通知我",
            "看到包裹时提醒我，顺便把窗口关掉",
            "举个包含‘看到手机就告诉我’的例子",
            "我可以说‘监控屏幕，出现弹窗就通知我’吗",
            "Is 'watch my screen and alert me when ERROR appears' natural?",
            "Next time that dialog appears on screen, notify me.",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(
                    direct_monitor_request_text(agent, text), "")

    def test_explicit_lifecycle_stays_fast_with_unrelated_existing_monitors(self):
        agent = SimpleNamespace(
            _multimodal_session=True,
            valid_tool_names={"set_monitor"},
            mm_monitors={"mon_existing": {"monitor_query": "看到水杯就提醒"}},
        )
        ambiguous_new = direct_monitor_request_text(agent, "看到手机就告诉我")
        explicit_new = direct_monitor_request_text(
            agent, "新建一个独立监控，第一次看到手机就告诉我")

        self.assertEqual(ambiguous_new, "")
        self.assertEqual(
            explicit_new,
            "新建一个独立监控，第一次看到手机就告诉我",
        )

    def test_monitor_gate_requires_multimodal_session_and_visible_tool(self):
        agent = SimpleNamespace(
            _multimodal_session=False,
            valid_tool_names={"set_monitor"},
            mm_monitors={},
        )
        self.assertEqual(
            direct_monitor_request_text(agent, "看到手机就告诉我"), "")

        agent._multimodal_session = True
        agent.valid_tool_names = set()
        self.assertEqual(
            direct_monitor_request_text(agent, "看到手机就告诉我"), "")

    def test_monitor_silent_default_never_calls_clarify(self):
        clarify = MagicMock(side_effect=AssertionError("must not clarify"))
        agent = SimpleNamespace(clarify_callback=clarify)

        self.assertEqual(
            _resolve_silent(agent, report_interval=None, silent_arg=None),
            (False, ""),
        )
        self.assertEqual(
            _resolve_silent(agent, report_interval=None, silent_arg=True),
            (True, ""),
        )
        self.assertEqual(
            _resolve_silent(agent, report_interval=60, silent_arg=True),
            (False, ""),
        )
        self.assertIs(
            SET_MONITOR_SCHEMA["parameters"]["properties"]["silent"]["default"],
            False,
        )
        clarify.assert_not_called()

    def test_handoff_parser_reads_only_current_tool_batch(self):
        payload = tool_handoff(
            tool_result({"note": "accepted"}),
            reply_owner="monitor",
            task_id="mon_1",
        )
        messages = [
            {"role": "assistant", "tool_calls": [{
                "id": "call_1", "function": {"name": "set_monitor"},
            }]},
            {"role": "tool", "tool_call_id": "call_1", "content": payload},
        ]
        found = _collect_tool_handoffs(messages, [_tool_call("set_monitor")])
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["reply_owner"], "monitor")
        self.assertEqual(found[0]["task_id"], "mon_1")

    def test_direct_monitor_handoff_uses_zero_model_calls(self):
        # Even a client prepared to return slow/wrong routes must never be
        # called: the deterministic local path executes set_monitor directly.
        agent = _monitor_fast_path_agent(_response(
            tool_calls=[
                _tool_call("check_video_stream", "call_wrong_check"),
                _tool_call("list_monitor", "call_wrong_list"),
            ]))
        handoff = tool_handoff(
            tool_result({"op": "create", "monitor_id": "mon_1",
                         "note": "监控已启动。"}),
            reply_owner="monitor",
            history_policy="ephemeral_control",
            task_id="mon_1",
        )

        with (
            patch("run_agent.handle_function_call", return_value=handoff) as handler,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("第一次看到有人进门就告诉我")

        self.assertEqual(result["api_calls"], 0)
        self.assertEqual(agent.client.chat.completions.create.call_count, 0)
        self.assertEqual(result["final_response"], "监控已启动。")
        self.assertEqual(result["handoff"]["reply_owner"], "monitor")
        self.assertEqual(result["handoff"]["history_policy"], "ephemeral_control")
        self.assertEqual(result["messages"], [])
        self.assertFalse(any(
            message.get("role") == "tool" or message.get("tool_calls")
            for message in result["messages"]
        ))
        handler.assert_called_once()
        self.assertEqual(handler.call_args.args[0], "set_monitor")
        fast_args = handler.call_args.args[1]
        self.assertEqual(fast_args["op"], "create")
        self.assertEqual(
            fast_args["monitor_query"], "第一次看到有人进门就告诉我")
        self.assertEqual(fast_args["trigger_mode"], "once")
        self.assertIs(fast_args["silent"], False)
        self.assertIs(fast_args["hook_main_agent"], False)

    def test_successful_direct_monitor_is_absent_from_real_session_db(self):
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmp:
            session_id = "ephemeral-monitor-success"
            db = SessionDB(db_path=Path(tmp) / "state.db")
            agent = _monitor_fast_path_agent(
                _response(), session_db=db, session_id=session_id)
            handoff = tool_handoff(
                tool_result({
                    "op": "create",
                    "monitor_id": "mon_db",
                    "note": "监控已启动。",
                }),
                reply_owner="monitor",
                history_policy="ephemeral_control",
                task_id="mon_db",
            )

            with (
                patch("run_agent.handle_function_call", return_value=handoff),
                patch.object(agent, "_save_session_log"),
                patch.object(agent, "_save_trajectory"),
                patch.object(agent, "_cleanup_task_resources"),
            ):
                result = agent.run_conversation("第一次看到手机就告诉我")

            self.assertEqual(result["messages"], [])
            self.assertEqual(db.get_messages(session_id), [])

    def test_failed_direct_monitor_remains_in_real_session_db(self):
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmp:
            session_id = "ephemeral-monitor-failure"
            db = SessionDB(db_path=Path(tmp) / "state.db")
            agent = _monitor_fast_path_agent(
                _response(), session_db=db, session_id=session_id)
            handoff = tool_handoff(
                json.dumps({
                    "success": False,
                    "error": "视频流未开启",
                }, ensure_ascii=False),
                reply_owner="main",
                history_policy="persist",
                ack="视频流未开启",
            )

            with (
                patch("run_agent.handle_function_call", return_value=handoff),
                patch.object(agent, "_save_session_log"),
                patch.object(agent, "_save_trajectory"),
                patch.object(agent, "_cleanup_task_resources"),
            ):
                result = agent.run_conversation("第一次看到手机就告诉我")

            rows = db.get_messages(session_id)
            self.assertTrue(result["failed"])
            self.assertEqual([row["role"] for row in rows], ["user", "assistant"])
            self.assertEqual(rows[0]["content"], "第一次看到手机就告诉我")
            self.assertIn("视频流未开启", rows[1]["content"])

    def test_direct_monitor_fast_path_never_surfaces_prepared_model_prose(self):
        agent = _monitor_fast_path_agent(_response(
            tool_calls=None,
            content="I will check the stream first.",
            finish_reason="stop",
        ))
        handoff = tool_handoff(
            tool_result({
                "op": "create",
                "monitor_id": "mon_prose",
                "note": "监控已启动。",
            }),
            reply_owner="monitor",
            history_policy="ephemeral_control",
            task_id="mon_prose",
        )

        with (
            patch("run_agent.handle_function_call", return_value=handoff) as handler,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("第一次看到猫进门就提醒我")

        self.assertEqual(result["api_calls"], 0)
        self.assertEqual(agent.client.chat.completions.create.call_count, 0)
        handler.assert_called_once()
        args = handler.call_args.args[1]
        self.assertEqual(args["op"], "create")
        self.assertEqual(args["trigger_mode"], "once")
        self.assertIs(args["silent"], False)
        self.assertNotIn("check_video_stream", json.dumps(result, ensure_ascii=False))

    def test_each_new_object_monitor_is_direct_and_continuous(self):
        agent = _monitor_fast_path_agent(_response())
        handoff = tool_handoff(
            tool_result({"op": "create", "monitor_id": "mon_score",
                         "note": "监控已启动。"}),
            reply_owner="monitor",
            history_policy="ephemeral_control",
            task_id="mon_score",
        )

        with (
            patch("run_agent.handle_function_call", return_value=handoff) as handler,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation(
                "监控我的画面，每看到有什么新物品就告诉我名字")

        self.assertEqual(result["api_calls"], 0)
        self.assertEqual(handler.call_args.args[1]["trigger_mode"], "continuous")
        self.assertEqual(
            handler.call_args.args[1]["monitor_query"],
            "监控我的画面，每看到有什么新物品就告诉我名字",
        )
        self.assertEqual(result["messages"], [])

    def test_direct_monitor_fast_path_reaches_real_monitor_handler(self):
        agent = _monitor_fast_path_agent(_response())
        agent.session_id = "sid_direct_monitor"
        agent.frame_buffer = SimpleNamespace(size=1, _last_push_wall=1.0)
        engine = MagicMock()
        engine.is_healthy.return_value = True
        engine.add_monitor.return_value = True
        entry = {
            "agent": agent,
            "session_key": agent.session_id,
            "_mm_monitor_engine": engine,
        }

        with (
            patch("tools.monitor_tool._find_agent_by_session",
                  return_value=(entry, agent.session_id)),
            patch("tools.monitor_tool._push_monitors_event"),
            patch("agent.multimodal.monitor_agent.init_event_file",
                  return_value="/tmp/monitor_direct.md"),
            patch("agent.multimodal.monitor_agent.read_event_ids",
                  return_value=[]),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("第一次看到有人进画面就提醒我")

        self.assertEqual(result["api_calls"], 0)
        self.assertEqual(agent.client.chat.completions.create.call_count, 0)
        self.assertEqual(len(agent.mm_monitors), 1)
        monitor_id, monitor = next(iter(agent.mm_monitors.items()))
        self.assertEqual(
            monitor["monitor_query"], "第一次看到有人进画面就提醒我")
        self.assertIs(monitor["silent"], False)
        self.assertEqual(monitor["trigger_mode"], "once")
        engine.add_monitor.assert_called_once_with(monitor_id)
        self.assertEqual(result["handoff"]["reply_owner"], "monitor")
        self.assertEqual(result["messages"], [])

    def test_model_routed_monitor_control_is_one_call_and_ephemeral(self):
        call = SimpleNamespace(
            id="call_manage",
            type="function",
            function=SimpleNamespace(
                name="set_monitor",
                arguments=json.dumps({
                    "op": "disable",
                    "monitor_ref": "手机监控",
                }, ensure_ascii=False),
            ),
        )
        agent = _monitor_fast_path_agent(_response(tool_calls=[call]))
        handoff = tool_handoff(
            tool_result({"op": "disable", "monitor_id": "mon_phone",
                         "note": "已暂停。"}),
            reply_owner="monitor",
            history_policy="ephemeral_control",
            task_id="mon_phone",
        )
        history = [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
        ]

        with (
            patch("run_agent.handle_function_call", return_value=handoff) as handler,
            patch.object(agent, "_persist_session") as persist,
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation(
                "暂停手机监控", conversation_history=history)

        self.assertEqual(result["api_calls"], 1)
        self.assertEqual(agent.client.chat.completions.create.call_count, 1)
        self.assertEqual(result["messages"], history)
        self.assertEqual(result["handoff"]["history_policy"], "ephemeral_control")
        handler.assert_called_once()
        self.assertEqual(handler.call_args.args[0], "set_monitor")
        persist.assert_called_once()
        self.assertEqual(persist.call_args.args[0], history)

    def test_conflicting_lifecycle_cues_are_decided_by_main_model(self):
        text = "整场监控库里下一次进球时告诉我"
        call = SimpleNamespace(
            id="call_lifecycle_decision",
            type="function",
            function=SimpleNamespace(
                name="set_monitor",
                arguments=json.dumps({
                    "op": "create",
                    "monitor_query": text,
                    "trigger_mode": "once",
                    "silent": False,
                    "hook_main_agent": False,
                }, ensure_ascii=False),
            ),
        )
        agent = _monitor_fast_path_agent(_response(tool_calls=[call]))
        handoff = tool_handoff(
            tool_result({"op": "create", "monitor_id": "mon_next_goal",
                         "note": "监控已启动。"}),
            reply_owner="monitor",
            history_policy="ephemeral_control",
            task_id="mon_next_goal",
        )

        with (
            patch("run_agent.handle_function_call", return_value=handoff) as handler,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation(text)

        self.assertEqual(result["api_calls"], 1)
        self.assertEqual(agent.client.chat.completions.create.call_count, 1)
        handler.assert_called_once()
        self.assertEqual(handler.call_args.args[0], "set_monitor")
        self.assertEqual(handler.call_args.args[1]["trigger_mode"], "once")
        self.assertEqual(result["messages"], [])

    def test_missing_lifecycle_cardinality_is_decided_by_main_model(self):
        text = "看到手机就告诉我"
        call = SimpleNamespace(
            id="call_lifecycle_ambiguous",
            type="function",
            function=SimpleNamespace(
                name="set_monitor",
                arguments=json.dumps({
                    "op": "create",
                    "monitor_query": text,
                    "trigger_mode": "continuous",
                    "silent": False,
                    "hook_main_agent": False,
                }, ensure_ascii=False),
            ),
        )
        agent = _monitor_fast_path_agent(_response(tool_calls=[call]))
        handoff = tool_handoff(
            tool_result({"op": "create", "monitor_id": "mon_phone",
                         "note": "监控已启动。"}),
            reply_owner="monitor",
            history_policy="ephemeral_control",
            task_id="mon_phone",
        )

        with (
            patch("run_agent.handle_function_call", return_value=handoff) as handler,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation(text)

        self.assertEqual(result["api_calls"], 1)
        self.assertEqual(agent.client.chat.completions.create.call_count, 1)
        handler.assert_called_once()
        self.assertEqual(handler.call_args.args[0], "set_monitor")
        self.assertEqual(
            handler.call_args.args[1]["trigger_mode"], "continuous")
        self.assertEqual(result["messages"], [])

    def test_model_create_without_valid_trigger_mode_does_not_mutate(self):
        with patch("tools.monitor_tool.set_monitor") as mutate:
            for trigger_mode in (None, "sometimes"):
                with self.subTest(trigger_mode=trigger_mode):
                    payload = json.loads(_set_monitor_handoff(
                        op="create",
                        monitor_query="看到手机就告诉我",
                        trigger_mode=trigger_mode,
                        session_id="sid_missing_mode",
                    ))
                    self.assertFalse(payload["success"])
                    self.assertEqual(
                        payload["code"], "monitor_trigger_mode_required")
                    self.assertNotIn("control", payload)
            mutate.assert_not_called()

    def test_model_can_correct_missing_mode_without_persisting_control_turn(self):
        text = "看到手机就告诉我"
        missing = SimpleNamespace(
            id="call_missing_mode",
            type="function",
            function=SimpleNamespace(
                name="set_monitor",
                arguments=json.dumps({
                    "op": "create",
                    "monitor_query": text,
                    "silent": False,
                }, ensure_ascii=False),
            ),
        )
        corrected = SimpleNamespace(
            id="call_corrected_mode",
            type="function",
            function=SimpleNamespace(
                name="set_monitor",
                arguments=json.dumps({
                    "op": "create",
                    "monitor_query": text,
                    "trigger_mode": "continuous",
                    "silent": False,
                }, ensure_ascii=False),
            ),
        )
        agent = _monitor_fast_path_agent(_response())
        agent.client.chat.completions.create.side_effect = [
            _response(tool_calls=[missing]),
            _response(tool_calls=[corrected]),
        ]
        validation_error = tool_result({
            "error": "trigger_mode is required",
            "success": False,
            "code": "monitor_trigger_mode_required",
        })
        handoff = tool_handoff(
            tool_result({"op": "create", "monitor_id": "mon_corrected",
                         "note": "监控已启动。"}),
            reply_owner="monitor",
            history_policy="ephemeral_control",
            task_id="mon_corrected",
        )

        with (
            patch("run_agent.handle_function_call",
                  side_effect=[validation_error, handoff]) as handler,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation(text)

        self.assertEqual(result["api_calls"], 2)
        self.assertEqual(handler.call_count, 2)
        self.assertEqual(
            handler.call_args_list[1].args[1]["trigger_mode"], "continuous")
        self.assertEqual(
            result["handoff"]["history_policy"], "ephemeral_control")
        self.assertEqual(result["messages"], [])

    def test_model_routed_monitor_control_is_absent_from_real_session_db(self):
        from hermes_state import SessionDB

        call = SimpleNamespace(
            id="call_manage_db",
            type="function",
            function=SimpleNamespace(
                name="set_monitor",
                arguments=json.dumps({
                    "op": "disable",
                    "monitor_ref": "手机监控",
                }, ensure_ascii=False),
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            session_id = "ephemeral-monitor-model-routed"
            db = SessionDB(db_path=Path(tmp) / "state.db")
            agent = _monitor_fast_path_agent(
                _response(tool_calls=[call]),
                session_db=db,
                session_id=session_id,
            )
            handoff = tool_handoff(
                tool_result({
                    "op": "disable",
                    "monitor_id": "mon_phone",
                    "note": "已暂停。",
                }),
                reply_owner="monitor",
                history_policy="ephemeral_control",
                task_id="mon_phone",
            )

            with (
                patch("run_agent.handle_function_call", return_value=handoff),
                patch.object(agent, "_save_session_log"),
                patch.object(agent, "_save_trajectory"),
                patch.object(agent, "_cleanup_task_resources"),
            ):
                result = agent.run_conversation("暂停手机监控")

            self.assertEqual(result["api_calls"], 1)
            self.assertEqual(result["messages"], [])
            self.assertEqual(db.get_messages(session_id), [])

    def test_ephemeral_handoff_after_ordinary_work_persists_complete_turn(self):
        """A later control handoff must not erase prior substantive tool work."""
        from hermes_state import SessionDB

        ordinary_call = _tool_call("set_monitor", "call_ordinary_first")
        ephemeral_call = _tool_call("set_monitor", "call_ephemeral_last")
        with tempfile.TemporaryDirectory() as tmp:
            session_id = "ephemeral-after-ordinary"
            db = SessionDB(db_path=Path(tmp) / "state.db")
            agent = _monitor_fast_path_agent(
                _response(), session_db=db, session_id=session_id
            )
            agent.client.chat.completions.create.side_effect = [
                _response(tool_calls=[ordinary_call]),
                _response(tool_calls=[ephemeral_call]),
            ]
            handoff = tool_handoff(
                tool_result({
                    "op": "disable",
                    "monitor_id": "mon_phone",
                    "note": "已暂停。",
                }),
                reply_owner="monitor",
                history_policy="ephemeral_control",
                task_id="mon_phone",
            )

            with (
                patch(
                    "run_agent.handle_function_call",
                    side_effect=[
                        tool_result({"note": "ordinary result must survive"}),
                        handoff,
                    ],
                ),
                patch.object(agent, "_save_session_log"),
                patch.object(agent, "_save_trajectory"),
                patch.object(agent, "_cleanup_task_resources"),
            ):
                result = agent.run_conversation("先检查，再暂停手机监控")

            self.assertEqual(result["api_calls"], 2)
            self.assertEqual(result["handoff"]["history_policy"], "persist")
            self.assertTrue(any(
                m.get("role") == "tool"
                and "ordinary result must survive" in str(m.get("content"))
                for m in result["messages"]
            ))
            durable = db.get_messages(session_id)
            self.assertTrue(any(
                m.get("role") == "tool"
                and "ordinary result must survive" in str(m.get("content"))
                for m in durable
            ))

    def test_voice_ephemeral_reply_resolves_waiter_without_speaking(self):
        from agent.multimodal.voice_agent import UserTask, VoiceAgent

        async def exercise(*, speak: bool):
            voice = VoiceAgent.__new__(VoiceAgent)
            voice._loop = asyncio.get_running_loop()
            voice._pending_main_tasks = {}
            voice._user_utterance_seq = 0
            voice._emit_progress = MagicMock()
            voice._speak_task_result_with_expiry_check = AsyncMock()

            def submit(_text, seq):
                voice.notify_main_reply(seq, "监控操作已处理。", speak=speak)

            voice._submit_main_agent = submit
            await voice._handle_user_task(UserTask(
                seq=1,
                user_text="暂停手机监控",
                user_seq_at_submit=0,
            ))
            self.assertEqual(voice._pending_main_tasks, {})
            return voice._speak_task_result_with_expiry_check

        silent_speaker = asyncio.run(exercise(speak=False))
        silent_speaker.assert_not_awaited()

        normal_speaker = asyncio.run(exercise(speak=True))
        normal_speaker.assert_awaited_once()

    def test_mixed_tool_batch_does_not_drop_ordinary_result_history(self):
        def _call(call_id, op):
            return SimpleNamespace(
                id=call_id,
                type="function",
                function=SimpleNamespace(
                    name="set_monitor",
                    arguments=json.dumps({"op": op}),
                ),
            )

        agent = _monitor_fast_path_agent(_response())
        agent.client.chat.completions.create.side_effect = [
            _response(tool_calls=[
                _call("call_ephemeral", "create"),
                _call("call_ordinary", "update"),
            ]),
            _response(tool_calls=None, content="combined answer", finish_reason="stop"),
        ]
        successful_handoff = tool_handoff(
            tool_result({"op": "create", "monitor_id": "mon_new"}),
            reply_owner="monitor",
            history_policy="ephemeral_control",
            task_id="mon_new",
        )

        def _execute(_name, args, *_positional, **_kwargs):
            if args.get("op") == "create":
                return successful_handoff
            return tool_result({"op": "update", "note": "ordinary result"})

        with (
            patch("run_agent.handle_function_call", side_effect=_execute),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("请同时调整两个已有监控")

        self.assertEqual(result["api_calls"], 2)
        self.assertEqual(result["final_response"], "combined answer")
        self.assertNotIn("handoff", result)
        self.assertTrue(any(m.get("role") == "tool" for m in result["messages"]))

    def test_mixed_query_batch_never_starts_worker_and_preserves_normal_tool(self):
        """The loop rejects QueryWorker before dispatch, not after it races."""
        query_call = SimpleNamespace(
            id="call_query",
            type="function",
            function=SimpleNamespace(
                name="query_multimodal",
                arguments=json.dumps({"query": "what is visible?"}),
            ),
        )
        normal_call = SimpleNamespace(
            id="call_normal",
            type="function",
            function=SimpleNamespace(name="normal_tool", arguments="{}"),
        )

        from run_agent import AIAgent
        with (
            patch("run_agent.get_tool_definitions",
                  return_value=_tool_defs("query_multimodal", "normal_tool")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
            patch("hermes_logging.setup_logging"),
        ):
            agent = AIAgent(
                api_key="test-key-1234567890",
                base_url="https://example.invalid/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
        agent.client = MagicMock()
        agent._multimodal_session = True
        agent._active_parent_user_message_id = "turn_mixed"
        agent._active_user_message_text = "what is visible?"
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.tool_delay = 0
        agent.compression_enabled = False
        agent.save_trajectories = False
        agent.client.chat.completions.create.side_effect = [
            _response(tool_calls=[query_call, normal_call]),
            _response(tool_calls=None, content="normal result handled",
                      finish_reason="stop"),
        ]

        class _Engine:
            def submit_query_async(self, *_args, **_kwargs):
                raise AssertionError("mixed batch must not start QueryWorker")

        def _execute(name, args, *_positional, **_kwargs):
            if name == "query_multimodal":
                with patch(
                    "tools.mm_memory_tool._resolve_mm_engine",
                    return_value=(_Engine(), agent),
                ):
                    return query_multimodal(
                        query=args.get("query"), session_id="sid")
            return tool_result({"note": "ordinary result survives"})

        with (
            patch("run_agent.handle_function_call", side_effect=_execute),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("what is visible?")

        self.assertEqual(result["api_calls"], 2)
        self.assertEqual(result["final_response"], "normal result handled")
        self.assertNotIn("handoff", result)
        query_results = [
            json.loads(m["content"])
            for m in result["messages"]
            if m.get("role") == "tool" and m.get("name") == "query_multimodal"
        ]
        self.assertEqual(
            query_results[0]["code"],
            "query_multimodal_requires_solo_turn",
        )
        self.assertTrue(any(
            m.get("role") == "tool"
            and m.get("name") == "normal_tool"
            and "ordinary result survives" in str(m.get("content"))
            for m in result["messages"]
        ))

    def test_query_after_prior_tool_round_never_starts_worker(self):
        """Solo query_multimodal is still unsafe after ordinary same-turn work."""
        normal_call = SimpleNamespace(
            id="call_normal_first",
            type="function",
            function=SimpleNamespace(name="normal_tool", arguments="{}"),
        )
        query_call = SimpleNamespace(
            id="call_query_second",
            type="function",
            function=SimpleNamespace(
                name="query_multimodal",
                arguments=json.dumps({"query": "what is visible?"}),
            ),
        )

        from run_agent import AIAgent
        with (
            patch("run_agent.get_tool_definitions",
                  return_value=_tool_defs("query_multimodal", "normal_tool")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
            patch("hermes_logging.setup_logging"),
        ):
            agent = AIAgent(
                api_key="test-key-1234567890",
                base_url="https://example.invalid/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
        agent.client = MagicMock()
        agent._multimodal_session = True
        agent._active_parent_user_message_id = "turn_prior_work"
        agent._active_user_message_text = "inspect something, then answer visually"
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.tool_delay = 0
        agent.compression_enabled = False
        agent.save_trajectories = False
        agent.client.chat.completions.create.side_effect = [
            _response(tool_calls=[normal_call]),
            _response(tool_calls=[query_call]),
            _response(tool_calls=None, content="handled after rejection",
                      finish_reason="stop"),
        ]

        class _Engine:
            def submit_query_async(self, *_args, **_kwargs):
                raise AssertionError("post-work QueryWorker must not start")

        def _execute(name, args, *_positional, **_kwargs):
            if name == "query_multimodal":
                with patch(
                    "tools.mm_memory_tool._resolve_mm_engine",
                    return_value=(_Engine(), agent),
                ):
                    return query_multimodal(
                        query=args.get("query"), session_id="sid")
            return tool_result({"note": "first-round ordinary result"})

        with (
            patch("run_agent.handle_function_call", side_effect=_execute),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation(
                "inspect something, then answer visually")

        self.assertEqual(result["api_calls"], 3)
        self.assertNotIn("handoff", result)
        self.assertTrue(any(
            m.get("role") == "tool"
            and m.get("name") == "normal_tool"
            and "first-round ordinary result" in str(m.get("content"))
            for m in result["messages"]
        ))
        query_result = next(
            json.loads(m["content"])
            for m in result["messages"]
            if m.get("role") == "tool"
            and m.get("name") == "query_multimodal"
        )
        self.assertEqual(query_result["reason"], "prior_tool_work")

    def test_failed_monitor_handoff_persists_and_marks_turn_failed(self):
        call = SimpleNamespace(
            id="call_failed",
            type="function",
            function=SimpleNamespace(
                name="set_monitor",
                arguments=json.dumps({
                    "op": "disable",
                    "monitor_ref": "不存在",
                }, ensure_ascii=False),
            ),
        )
        agent = _monitor_fast_path_agent(_response(tool_calls=[call]))
        failed_handoff = tool_handoff(
            json.dumps({"success": False, "error": "监控目标不存在"}, ensure_ascii=False),
            reply_owner="main",
            history_policy="persist",
            ack="监控目标不存在",
        )

        with (
            patch("run_agent.handle_function_call", return_value=failed_handoff),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("暂停不存在的监控")

        self.assertEqual(result["api_calls"], 1)
        self.assertTrue(result["failed"])
        self.assertFalse(result["completed"])
        self.assertNotIn("handoff", result)
        self.assertIn("不存在", result["final_response"])
        self.assertTrue(any(m.get("role") == "tool" for m in result["messages"]))

    def test_direct_monitor_skips_model_dependent_prologue_work(self):
        agent = _monitor_fast_path_agent(_response())
        agent.compression_enabled = True
        agent._compress_context = MagicMock(
            side_effect=AssertionError("must not compress a direct tool turn"))
        memory_manager = MagicMock()
        agent._memory_manager = memory_manager
        agent._spawn_background_review = MagicMock(
            side_effect=AssertionError("must not review a direct tool turn"))
        handoff = tool_handoff(
            tool_result({"op": "create", "monitor_id": "mon_fast",
                         "note": "监控已启动。"}),
            reply_owner="monitor",
            task_id="mon_fast",
        )
        history = [
            {"role": "user" if i % 2 == 0 else "assistant",
             "content": "old context " * 500}
            for i in range(20)
        ]

        with (
            patch("run_agent.handle_function_call", return_value=handoff),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation(
                "第一次看到手机就告诉我", conversation_history=history)

        self.assertEqual(result["api_calls"], 0)
        agent._compress_context.assert_not_called()
        memory_manager.on_turn_start.assert_not_called()
        memory_manager.prefetch_all.assert_not_called()
        memory_manager.sync_turn.assert_not_called()

    def test_direct_monitor_interrupt_before_execution_is_not_success(self):
        agent = _monitor_fast_path_agent(_response())
        original_build = agent._build_assistant_message

        def interrupt_after_selection(*args, **kwargs):
            message = original_build(*args, **kwargs)
            agent._interrupt_requested = True
            return message

        with (
            patch.object(agent, "_build_assistant_message",
                         side_effect=interrupt_after_selection),
            patch("run_agent.handle_function_call") as handler,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("第一次看到手机就告诉我")

        handler.assert_not_called()
        self.assertTrue(result["interrupted"])
        self.assertTrue(result["failed"])
        self.assertFalse(result["completed"])
        self.assertEqual(result["api_calls"], 0)

    def test_direct_monitor_preserves_steer_for_the_next_turn(self):
        agent = _monitor_fast_path_agent(_response())
        handoff = tool_handoff(
            tool_result({"op": "create", "monitor_id": "mon_steer",
                         "note": "监控已启动。"}),
            reply_owner="monitor",
            task_id="mon_steer",
        )

        def create_and_steer(*_args, **_kwargs):
            agent.steer("下一条用户指令")
            return handoff

        with (
            patch("run_agent.handle_function_call",
                  side_effect=create_and_steer),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("第一次看到手机就告诉我")

        self.assertEqual(result["pending_steer"], "下一条用户指令")
        self.assertFalse(any(
            message.get("role") == "tool" or message.get("tool_calls")
            for message in result["messages"]
        ))

    def test_non_monitor_intent_keeps_the_normal_model_path(self):
        agent = _monitor_fast_path_agent(_response(
            tool_calls=None,
            content="normal answer",
            finish_reason="stop",
        ))

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("现在画面里有手机吗")

        self.assertEqual(result["api_calls"], 1)
        self.assertEqual(agent.client.chat.completions.create.call_count, 1)
        self.assertEqual(result["final_response"], "normal answer")

    def test_direct_monitor_reports_missing_stream_without_model_fallback(self):
        agent = _monitor_fast_path_agent(_response())
        agent.session_id = "sid_no_stream"
        agent.frame_buffer = None
        engine = MagicMock()
        engine.is_healthy.return_value = True
        engine.add_monitor.return_value = True
        entry = {
            "agent": agent,
            "session_key": agent.session_id,
            "_mm_monitor_engine": engine,
        }

        with (
            patch("tools.monitor_tool._find_agent_by_session",
                  return_value=(entry, agent.session_id)),
            patch("tools.monitor_tool._push_monitors_event"),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("第一次看到手机就告诉我")

        self.assertEqual(result["api_calls"], 0)
        self.assertEqual(agent.client.chat.completions.create.call_count, 0)
        self.assertEqual(agent.mm_monitors, {})
        self.assertIn("not ready", result["final_response"].lower())
        self.assertTrue(result["failed"])
        self.assertFalse(result["completed"])
        self.assertNotIn("handoff", result)

    def test_deferred_query_handoff_leaves_no_half_turn_in_main_history(self):
        from run_agent import AIAgent

        with (
            patch("run_agent.get_tool_definitions",
                  return_value=_tool_defs("query_multimodal")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
            patch("hermes_logging.setup_logging"),
        ):
            agent = AIAgent(
                api_key="test-key-1234567890",
                base_url="https://example.invalid/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
        agent.client = MagicMock()
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.tool_delay = 0
        agent.compression_enabled = False
        agent.save_trajectories = False
        agent._defer_current_turn_persistence = True
        agent.client.chat.completions.create.return_value = _response(
            tool_calls=[_tool_call("query_multimodal")])
        handoff = tool_handoff(
            tool_result({"status": "running", "note": "accepted"}),
            reply_owner="query_worker",
            handoff_mode="deferred_reply",
            task_id="qry_1",
            parent_user_message_id="turn_1",
        )
        base_history = [
            {"role": "user", "content": "earlier Q"},
            {"role": "assistant", "content": "earlier A"},
        ]

        with (
            patch("run_agent.handle_function_call", return_value=handoff),
            patch.object(agent, "_persist_session") as persist,
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation(
                "Q1", conversation_history=base_history)

        self.assertEqual(result["handoff"]["handoff_mode"], "deferred_reply")
        self.assertEqual(result["messages"], base_history)
        self.assertEqual(persist.call_count, 1)
        self.assertEqual(persist.call_args.args[0], base_history)

    def test_internal_completion_hook_returns_text_without_persisting_turn(self):
        from run_agent import AIAgent

        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
            patch("hermes_logging.setup_logging"),
        ):
            agent = AIAgent(
                api_key="test-key-1234567890",
                base_url="https://example.invalid/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
        agent.client = MagicMock()
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.compression_enabled = False
        agent.save_trajectories = False
        agent._ephemeral_internal_turn = True
        agent._memory_manager = MagicMock()
        agent._memory_manager.prefetch_all.return_value = ""
        agent._spawn_background_review = MagicMock()
        agent.client.chat.completions.create.return_value = _response(
            content="最终视频总结",
            finish_reason="stop",
        )
        base_history = [
            {"role": "user", "content": "开始观看"},
            {"role": "assistant", "content": "已启动"},
        ]
        invoked_hooks = []

        with (
            patch(
                "hermes_cli.plugins.invoke_hook",
                side_effect=lambda name, **_kwargs: invoked_hooks.append(name) or [],
            ),
            patch.object(agent, "_persist_session") as persist,
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation(
                "hidden watcher completion prompt",
                conversation_history=base_history,
            )

        self.assertEqual(result["final_response"], "最终视频总结")
        self.assertEqual(result["messages"], base_history)
        self.assertEqual(persist.call_count, 1)
        self.assertEqual(persist.call_args.args[0], base_history)
        agent._memory_manager.on_turn_start.assert_not_called()
        agent._memory_manager.prefetch_all.assert_not_called()
        agent._memory_manager.sync_turn.assert_not_called()
        agent._spawn_background_review.assert_not_called()
        self.assertNotIn("pre_llm_call", invoked_hooks)
        self.assertNotIn("post_llm_call", invoked_hooks)
        self.assertNotIn("on_session_end", invoked_hooks)

    def test_deferred_query_scheduler_receipt_skips_completion_side_effects(self):
        """Only the later projected QueryWorker Q/A may enter memory/hooks."""
        from run_agent import AIAgent

        with (
            patch("run_agent.get_tool_definitions",
                  return_value=_tool_defs("query_multimodal")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
            patch("hermes_logging.setup_logging"),
        ):
            agent = AIAgent(
                api_key="test-key-1234567890",
                base_url="https://example.invalid/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
        agent.client = MagicMock()
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.tool_delay = 0
        agent.compression_enabled = False
        agent.save_trajectories = False
        agent._defer_current_turn_persistence = True
        agent._skill_nudge_interval = 1
        agent._iters_since_skill = 1
        agent.valid_tool_names.add("skill_manage")
        agent._memory_manager = MagicMock()
        agent._memory_manager.prefetch_all.return_value = ""
        agent._memory_manager.has_tool.return_value = False
        agent._spawn_background_review = MagicMock()
        agent.client.chat.completions.create.return_value = _response(
            tool_calls=[_tool_call("query_multimodal")])
        handoff = tool_handoff(
            tool_result({"status": "running", "note": "scheduler receipt"}),
            reply_owner="query_worker",
            handoff_mode="deferred_reply",
            task_id="qry_side_effect",
            parent_user_message_id="turn_side_effect",
        )
        invoked_hooks = []

        def _hooks(name, **_kwargs):
            invoked_hooks.append(name)
            return ["MUTATED"] if name == "transform_llm_output" else []

        prior_history = [
            {"role": "user", "content": "old question"},
            {
                "role": "assistant",
                "content": "old answer",
                "reasoning": "old reasoning must not leak",
            },
        ]

        with (
            patch("run_agent.handle_function_call", return_value=handoff),
            patch("hermes_cli.plugins.invoke_hook", side_effect=_hooks),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation(
                "Q1", conversation_history=prior_history)

        self.assertEqual(result["final_response"], "scheduler receipt")
        self.assertFalse(result["response_transformed"])
        self.assertIsNone(result["last_reasoning"])
        self.assertNotIn("transform_llm_output", invoked_hooks)
        self.assertNotIn("post_llm_call", invoked_hooks)
        self.assertNotIn("on_session_end", invoked_hooks)
        agent._memory_manager.sync_turn.assert_not_called()
        agent._spawn_background_review.assert_not_called()
        # It is deferred_reply metadata only, never ephemeral_control: the
        # frontend must keep the original user bubble/answer slot alive.
        self.assertEqual(result["handoff"]["history_policy"], "persist")
        self.assertNotIn("ephemeral_control", json.dumps(result["handoff"]))

    def test_gateway_recall_dispatches_query_worker_instead_of_blocking(self):
        calls = []

        class _Engine:
            recall_worker = object()

            def submit_query_async(self, instruction, **kwargs):
                calls.append((instruction, kwargs))
                return kwargs["task_id"]

            def recall_memory(self, *_args, **_kwargs):
                raise AssertionError("gateway handoff must not block on recall_memory")

        agent = SimpleNamespace(
            _active_parent_user_message_id="turn_7",
            _active_user_message_text="第二个物品的价格是多少",
        )
        with patch(
            "tools.mm_memory_tool._resolve_mm_engine",
            return_value=(_Engine(), agent),
        ), patch(
            "tools.mm_memory_tool._resolve_send_anchor_ts",
            return_value=42.5,
        ):
            data = json.loads(query_multimodal(
                query="召回第二个物品并查价格",
                session_id="sid",
            ))

        self.assertEqual(data["control"], "handoff")
        self.assertEqual(data["handoff_mode"], "deferred_reply")
        self.assertEqual(data["parent_user_message_id"], "turn_7")
        self.assertEqual(len(calls), 1)
        self.assertIn("第二个物品的价格", calls[0][0])
        self.assertEqual(
            calls[0][1]["original_user_query"], "第二个物品的价格是多少")
        self.assertEqual(calls[0][1]["ask_ts"], 42.5)

    def test_gateway_receipt_uses_deduplicated_query_task_id(self):
        class _Engine:
            recall_worker = object()

            def submit_query_async(self, _instruction, **_kwargs):
                return "qry_already_running"

        agent = SimpleNamespace(
            _active_parent_user_message_id="turn_same",
            _active_user_message_text="同一个问题",
        )
        with (
            patch("tools.mm_memory_tool._resolve_mm_engine",
                  return_value=(_Engine(), agent)),
            patch("tools.mm_memory_tool.secrets.token_hex",
                  return_value="new_random_id"),
        ):
            data = json.loads(query_multimodal(
                query="同一个问题", session_id="sid"))

        self.assertEqual(data["task_id"], "qry_already_running")
        self.assertIn("qry_already_running", data["note"])
        self.assertNotIn("new_random_id", data["note"])


class QueryWorkerRuntimeTests(unittest.TestCase):
    def test_query_worker_streams_to_parent_slot_and_completes(self):
        loop = asyncio.new_event_loop()
        loop_ready = threading.Event()

        def _run_loop():
            asyncio.set_event_loop(loop)
            loop_ready.set()
            loop.run_forever()

        thread = threading.Thread(target=_run_loop, daemon=True)
        thread.start()
        self.assertTrue(loop_ready.wait(2))

        captured_kwargs = {}

        class _Responder:
            async def _spawn_delegation(self, *, sink, on_event, **_kwargs):
                captured_kwargs.update(_kwargs)
                async def _drive():
                    await on_event({
                        "type": "bg_progress",
                        "channel": "recall",
                        "phase": "start",
                        "brief": "先召回物品",
                    })
                    await on_event({
                        "type": "bg_progress",
                        "channel": "recall",
                        "phase": "done",
                        "frame_ids": ["f_1"],
                        "frames": [{"frame_id": "f_1", "jpeg_b64": "AA=="}],
                    })
                    await on_event({
                        "type": "recall_done",
                        "frame_ids": ["f_1"],
                        "frames": [{"frame_id": "f_1", "jpeg_b64": "AA=="}],
                    })
                    await sink("第二个物品")
                    await sink("价格是 12 元")
                    await on_event({
                        "type": "answer_ready",
                        "answer_full": "第二个物品价格是 12 元",
                    })
                return asyncio.create_task(_drive())

        events = []
        completed = []
        done = threading.Event()
        engine = WatcherAgent.__new__(WatcherAgent)
        engine._loop = loop
        engine.responder = _Responder()
        _prime_query_runtime(engine, loop)
        engine.cfg = SimpleNamespace(cont_recent_frames=3)
        frame1 = SimpleNamespace(ts=1.0)
        frame2 = SimpleNamespace(ts=2.0)
        frame3 = SimpleNamespace(ts=3.0)
        frame4 = SimpleNamespace(ts=4.0)
        engine.frame_buffer = SimpleNamespace(
            all_le=lambda _ts: [frame1, frame2, frame3, frame4],
            latest=lambda _n: [frame2, frame3, frame4],
            latest_ts=9.0,
        )
        engine._emit_cb = lambda event, payload: events.append((event, payload))

        def _complete(task_id, parent_id, query, text, status):
            completed.append((task_id, parent_id, query, text, status))
            done.set()

        engine._on_query_complete = _complete
        qid = engine.submit_query_async(
            "用户原问：第二个物品的价格\n主 Agent 解析：先召回物品",
            task_id="qry_1",
            parent_user_message_id="turn_2",
            original_user_query="第二个物品的价格",
            ask_ts=4.0,
        )
        self.assertEqual(qid, "qry_1")
        self.assertTrue(done.wait(3))
        self.assertEqual(completed[0][0:2], ("qry_1", "turn_2"))
        self.assertEqual(completed[0][2], "第二个物品的价格")
        self.assertEqual(completed[0][3], "第二个物品价格是 12 元")
        deltas = [p for event, p in events if event == "message.delta"]
        self.assertTrue(deltas)
        self.assertTrue(all(p["request_id"] == "turn_2" for p in deltas))
        traces = [p for event, p in events
                  if event == "multimodal.trajectory"]
        self.assertTrue(any(p.get("worker") == "RecallWorker" for p in traces))
        self.assertTrue(any(
            p.get("worker") == "RecallWorker"
            and (p.get("event") or {}).get("phase") == "start"
            for p in traces
        ))
        self.assertTrue(any(
            (p.get("frames") or [{}])[0].get("frame_id") == "f_1"
            for p in traces if p.get("frames")
        ))
        self.assertEqual(
            len([p for p in traces if p.get("frames")]), 1,
            "Recall thumbnails should be serialized once on recall_done",
        )
        self.assertEqual(captured_kwargs["ask_ts"], 4.0)
        self.assertEqual(captured_kwargs["ask_frames_override"],
                         [frame2, frame3, frame4])
        self.assertFalse(captured_kwargs["force_initial_recall"])
        self.assertTrue(captured_kwargs["query_worker_mode"])

        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        loop.close()

    def test_two_query_workers_finish_out_of_order_without_crossing_slots(self):
        loop = asyncio.new_event_loop()
        loop_ready = threading.Event()

        def _run_loop():
            asyncio.set_event_loop(loop)
            loop_ready.set()
            loop.run_forever()

        thread = threading.Thread(target=_run_loop, daemon=True)
        thread.start()
        self.assertTrue(loop_ready.wait(2))

        class _Responder:
            async def _spawn_delegation(
                self, *, task_instruction, sink, on_event, **_kwargs
            ):
                async def _drive():
                    # Q2 intentionally wins the race although Q1 was submitted
                    # first. Each stream still carries its own parent turn id.
                    await asyncio.sleep(0.08 if task_instruction == "Q1" else 0.01)
                    answer = f"{task_instruction}-answer"
                    await sink(answer)
                    await on_event({
                        "type": "answer_ready", "answer_full": answer,
                    })
                return asyncio.create_task(_drive())

        events = []
        completed = []
        done = threading.Event()
        engine = WatcherAgent.__new__(WatcherAgent)
        engine._loop = loop
        engine.responder = _Responder()
        _prime_query_runtime(engine, loop)
        engine.cfg = SimpleNamespace(cont_recent_frames=3)
        engine.frame_buffer = SimpleNamespace(
            latest=lambda _n: [], latest_ts=1.0)
        engine._emit_cb = lambda event, payload: events.append((event, payload))

        def _complete(task_id, parent_id, query, text, status):
            completed.append((task_id, parent_id, query, text, status))
            if len(completed) == 2:
                done.set()

        engine._on_query_complete = _complete
        engine.submit_query_async(
            "Q1", task_id="qry_1", parent_user_message_id="turn_1")
        engine.submit_query_async(
            "Q2", task_id="qry_2", parent_user_message_id="turn_2")

        self.assertTrue(done.wait(3))
        self.assertEqual([row[0] for row in completed], ["qry_2", "qry_1"])
        self.assertEqual(
            {(row[1], row[3]) for row in completed},
            {("turn_1", "Q1-answer"), ("turn_2", "Q2-answer")},
        )
        streamed = {
            (payload["request_id"], payload["text"])
            for event, payload in events if event == "message.delta"
        }
        self.assertEqual(
            streamed,
            {("turn_1", "Q1-answer"), ("turn_2", "Q2-answer")},
        )

        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        loop.close()


if __name__ == "__main__":
    unittest.main()
