"""Recall must keep its role-specific model independent from Watcher."""

from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


def _response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class TestRecallAgentRoleModel(unittest.IsolatedAsyncioTestCase):
    def test_recall_evidence_times_are_structured_and_bounded_by_ask_ts(self):
        from agent.multimodal._workers import _extract_recall_evidence_segments

        frame_store = SimpleNamespace(get=lambda fid: (
            SimpleNamespace(ts=76.25) if fid == "f_1234567890" else None
        ))
        segments = _extract_recall_evidence_segments(
            "\n".join([
                "- t=84.5s: 店主提到联系探店人",
                "- [02:22-02:28] 画面中出现攀岩鞋",
                "- frame_id=f_1234567890: 历史关键帧",
                "- entity=ent_shop first=01:10 last=01:25",
                "- ent_shop --[联系]--> ent_guest @02:05",
                "- [00:42] 单点屏幕证据",
                "- t=999.0s: 提问之后的内容",
            ]),
            tool_name="search_audio",
            ask_ts=148.0,
            frame_store=frame_store,
        )

        self.assertEqual(
            [(item["t_start"], item["t_end"]) for item in segments],
            [
                (84.5, 84.5),
                (142.0, 148.0),
                (76.25, 76.25),
                (70.0, 85.0),
                (125.0, 125.0),
                (42.0, 42.0),
            ],
        )
        self.assertEqual(segments[2]["frame_ids"], ["f_1234567890"])
        self.assertTrue(all(item["t_end"] <= 148.0 for item in segments))

    async def test_fast_table_reports_the_actual_tool_arguments_and_result(self):
        from agent.multimodal._config import Config
        from agent.multimodal._workers import RecallAgent

        recall = RecallAgent(
            Config(), MagicMock(), MagicMock(), MagicMock(),
            screen_text_store=MagicMock(),
        )
        recall.mem_tools.call = MagicMock(return_value=(
            "[search_screen_text query='表 2']\n"
            "[structured_tables query='表 2'] 共 1 张表\n"
            "[00:42-00:45] frame_id=f_1234567890 | 表 2 | Model | Score\n"
            "[00:42-00:45] frame_id=f_1234567890 | Argus | 91.2"
        ))
        events = []

        async def emit(event):
            events.append(event)

        result = await recall._table_fast_path(
            brief="表 2 里 Argus 的 Score 是多少",
            user_text="刚才论文表格里的数字",
            ask_ts=60.0,
            emit=emit,
        )

        self.assertIsNotNone(result)
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["phase"], "fast_table")
        self.assertEqual(event["tool_name"], "search_screen_text")
        self.assertEqual(event["args"]["limit"], 8)
        self.assertIn("表 2", event["args"]["query"])
        self.assertIn("Argus | 91.2", event["obs_summary"])
        self.assertIn("row-level evidence", event["findings_preview"])
        self.assertGreater(event["obs_len"], 0)
        self.assertGreaterEqual(event["elapsed_sec"], 0)
        self.assertEqual(event["frame_ids"], ["f_1234567890"])
        self.assertEqual(event["evidence_segments"][0]["t_start"], 42.0)
        self.assertEqual(event["evidence_segments"][0]["t_end"], 45.0)

    async def test_all_recall_llm_calls_use_resolved_recall_model(self):
        from agent.multimodal._config import Config
        from agent.multimodal._workers import RecallAgent

        cfg = Config()
        cfg.model = "qwen3.7-plus"
        cfg.recall_model = "gpt-5.6-luna"
        mem = MagicMock()
        mem.get_recent_entities.return_value = []
        create = AsyncMock(side_effect=[
            _response("distilled clue"),
            _response('{"keep":["f1"],"visual_correction":""}'),
            _response(
                '{"thought":"enough","can_answer":true,'
                '"useful_info":"done","tool_calls":[]}'
            ),
        ])
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        from agent.multimodal.hermes_glue import wrap_kimi_client
        wrap_kimi_client(client, dialect="openai")
        frame_store = SimpleNamespace(get_many=lambda _fids: [
            SimpleNamespace(frame_id="f1", ts=1.0, jpeg_b64="ZmFrZQ==")
        ])
        recall = RecallAgent(
            cfg,
            mem,
            client,
            MagicMock(),
            frame_store=frame_store,
            model="gpt-5.6-luna",
        )

        class _CountingChannel:
            def __init__(self):
                self.acquires = 0
                self.releases = 0

            async def acquire(self):
                self.acquires += 1

            def release(self):
                self.releases += 1

        channel = _CountingChannel()
        recall.recall_limiter = channel

        self.assertEqual(
            await recall._distill(
                raw_obs="observation", brief="find it", user_text="where was it?"
            ),
            "distilled clue",
        )
        kept, _ = await recall._verify_frames_with_grounding(
            ["f1"], query="find it"
        )
        self.assertEqual(kept, ["f1"])
        decision = await recall._decide_next(
            brief="find it",
            user_text="where was it?",
            ask_ts=1.0,
            clues=[],
            round_idx=0,
        )
        self.assertTrue(decision["can_answer"])

        self.assertEqual(create.await_count, 3)
        self.assertEqual(
            [call.kwargs["model"] for call in create.await_args_list],
            ["gpt-5.6-luna"] * 3,
        )
        for call in create.await_args_list:
            self.assertEqual(
                call.kwargs.get("extra_body"),
                {"thinking": {"type": "disabled"}},
            )
            self.assertNotIn("top_p", call.kwargs)
            self.assertNotIn("temperature", call.kwargs)
            self.assertGreaterEqual(call.kwargs["max_completion_tokens"], 8192)
        self.assertEqual(channel.acquires, 3)
        self.assertEqual(channel.releases, 3)

    async def test_decision_transport_error_is_not_reported_as_memory_miss(self):
        from agent.multimodal._config import Config
        from agent.multimodal._workers import RecallAgent

        cfg = Config()
        cfg.recall_model = "gpt-5.6-luna"
        mem = MagicMock()
        mem.get_recent_entities.return_value = []
        create = AsyncMock(side_effect=RuntimeError("HTTP 400 unknown parameter"))
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        recall = RecallAgent(
            cfg, mem, client, MagicMock(), model="gpt-5.6-luna")
        events = []

        async def on_progress(event):
            events.append(event)

        with self.assertRaisesRegex(RuntimeError, "Recall decision failed"):
            await recall.run(
                initial_calls=[], brief="店主找谁探店",
                user_text="店主找谁探店？", ask_ts=10.0,
                on_progress=on_progress,
            )

        errors = [event for event in events if event.get("phase") == "error"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["stage"], "decision")
        self.assertIn("HTTP 400", errors[0]["error"])
        self.assertFalse(any(event.get("phase") == "done" for event in events))

    async def test_identical_memory_tool_read_is_not_repeated_in_snapshot(self):
        from agent.multimodal._config import Config
        from agent.multimodal._workers import RecallAgent

        cfg = Config()
        cfg.recall_max_rounds = 4
        cfg.recall_verify_enabled = False
        mem = MagicMock()
        mem.get_recent_entities.return_value = []
        recall = RecallAgent(
            cfg, mem, MagicMock(), MagicMock(), model="gpt-5.6-luna")
        recall.mem_tools.call = MagicMock(return_value="[search_audio] (空)")
        recall._distill = AsyncMock(return_value="")
        repeated = {
            "thought": "",
            "can_answer": False,
            "useful_info": "",
            "tool_calls": [{
                "name": "search_audio",
                "args": {"query": "探店", "top_k": 8},
            }],
        }
        recall._decide_next = AsyncMock(side_effect=[repeated, repeated])
        events = []

        async def on_progress(event):
            events.append(event)

        result = await recall.run(
            initial_calls=[], brief="查探店对话",
            user_text="刚才说了谁来探店", ask_ts=10.0,
            on_progress=on_progress,
        )

        self.assertEqual(recall.mem_tools.call.call_count, 1)
        from agent.multimodal._sentinels import RECALL_NO_CLUES
        self.assertEqual(result.findings, RECALL_NO_CLUES)
        skipped = [event for event in events
                   if event.get("phase") == "tool_skipped"]
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["reason"], "duplicate_snapshot_read")
        starts = [event for event in events if event.get("phase") == "start"]
        self.assertEqual(starts[0]["model"], "gpt-5.6-luna")
        decisions = [event for event in events
                     if str(event.get("phase", "")).endswith("_decision")]
        self.assertTrue(decisions)
        for event in decisions:
            # Public trajectory is structured evidence, never raw CoT/prompt.
            self.assertIn("decision_summary", event)
            self.assertNotIn("thought", event)
            self.assertNotIn("user_text", event)
            self.assertNotIn("clues", event)
        observations = [event for event in events
                        if event.get("phase") == "tool_obs"]
        self.assertEqual(len(observations), 1)
        self.assertIn("obs_summary", observations[0]["observations"][0])
        self.assertGreaterEqual(
            observations[0]["observations"][0]["elapsed_sec"], 0.0)
        self.assertNotIn("obs_full", observations[0]["observations"][0])

    def test_configured_recall_model_beats_watcher_model(self):
        from agent.multimodal._config import Config
        from agent.multimodal._workers import RecallAgent

        cfg = Config()
        cfg.model = "qwen3.7-plus"
        cfg.recall_model = "gpt-5.6-luna"
        recall = RecallAgent(cfg, MagicMock(), MagicMock(), MagicMock())
        self.assertEqual(recall.model, "gpt-5.6-luna")


class TestWatcherFallbackRecallRouting(unittest.TestCase):
    def _engine(self):
        from agent.multimodal.watcher_engine import WatcherAgent

        engine = WatcherAgent.__new__(WatcherAgent)
        engine._hermes_cfg = {}
        engine.frame_buffer = object()
        # A local RecallAgent is allowed only in explicit standalone mode.  An
        # existing-but-incomplete backend must fail instead of spawning a second
        # memory stack.
        engine._memory_backend = None
        import threading
        engine._stop = threading.Event()
        return engine

    def test_fallback_uses_dedicated_recall_pair_not_watcher_pair(self):
        from agent.multimodal._config import Config

        cfg = Config()
        cfg.model = "qwen3.7-plus"
        cfg.recall_model = "gpt-5.6-luna"
        cfg._worker_model_explicit = True
        watcher_client = object()
        recall_client = object()
        factory = MagicMock()
        factory.worker_client.return_value = (watcher_client, "qwen3.7-plus")
        factory.recall_client.return_value = (recall_client, "gpt-5.6-luna")
        engine = self._engine()

        with (
            patch("agent.multimodal.hermes_glue.build_config", return_value=cfg),
            patch(
                "agent.multimodal.hermes_glue.HermesClientFactory",
                return_value=factory,
            ),
            patch("agent.multimodal._memory.MemoryStore", return_value=MagicMock()),
            patch("agent.multimodal._memory.SearchFactStore", return_value=MagicMock()),
            patch("agent.multimodal._memory.ConversationLog", return_value=MagicMock()),
            patch("agent.multimodal._memory.FrameStore", return_value=MagicMock()),
            patch("agent.multimodal._memory.ScreenTextStore", return_value=MagicMock()),
            patch("agent.multimodal._memory.ScreenTableStore", return_value=MagicMock()),
            patch("agent.multimodal._memory.TaskStateStore", return_value=MagicMock()),
            patch("agent.multimodal._workers.ToolBox", return_value=MagicMock()),
            patch("agent.multimodal._workers.RecallAgent") as recall_cls,
            patch("agent.multimodal._workers.WatcherWorker", return_value=MagicMock()),
        ):
            self.assertTrue(engine._build())

        factory.recall_client.assert_called_once_with()
        args = recall_cls.call_args.args
        kwargs = recall_cls.call_args.kwargs
        self.assertIs(args[2], recall_client)
        self.assertEqual(kwargs["model"], "gpt-5.6-luna")
        self.assertIsNot(args[2], watcher_client)
        self.assertIsNotNone(kwargs["screen_table_store"])


if __name__ == "__main__":
    unittest.main()
