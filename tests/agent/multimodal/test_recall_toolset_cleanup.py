"""2026-08-19 recall 工具集瘦身的运行时断言。

覆盖四项改动:
  1. get_subgraph 从分发/白名单/RECALL_SYSTEM prompt 三处彻底移除 (它需要
     macro_id, 而格式化层只有 _fmt_audio_evidence 回显过 macro_id → 不可达)。
  2. get_events_by_entity / get_frames_by_entity 两个死分发分支移除 (白名单
     早就把它们过滤掉了)。
  3. _fmt_entity_context 的 resolution 行与 canonical entity 块各只打印一次,
     且 entity=None / resolution_chain=None 不再崩;_fmt_artifact_context 不再
     叠加第三份实体属性。
  4. entity_quotes 空表时, 两个 quote 工具返回明确的 "do not retry" 观测。

这些测试用真实 SQLite (tmp 文件), 不 mock store。
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from agent.multimodal._config import Config  # noqa: E402
from agent.multimodal._memory import Entity, MemoryStore  # noqa: E402
from agent.multimodal._workers import (  # noqa: E402
    MemoryToolBox, RECALL_SYSTEM, RecallAgent,
)


def _mk_store(tmpdir):
    cfg = Config()
    cfg.mem_db_path = os.path.join(tmpdir, "mem.sqlite3")
    return MemoryStore(cfg)


class TestGetSubgraphRemoved(unittest.TestCase):
    def test_not_in_whitelist(self):
        self.assertNotIn("get_subgraph", RecallAgent._RECALL_TOOL_NAMES)

    def test_not_advertised_in_prompt(self):
        # prompt 里连"不要这样调"的告警也一并撤掉了, 否则等于继续付预算
        self.assertNotIn("get_subgraph", RECALL_SYSTEM)

    def test_normalizer_now_aliases_the_call(self):
        """2026-08-19 二阶段: 不再静默丢弃, 而是映射到 search_events(macro_id=)。

        macro_id 现在是 search_events 的合法入参 (且 _fmt_micros 会回显 macro),
        所以老名字带来的意图是可执行的 —— 丢掉它只会让模型白烧一轮。
        """
        parsed, _ = RecallAgent._normalize_decision_tool_calls({
            "can_answer": False,
            "tool_calls": [
                {"name": "get_subgraph", "args": {"macro_id": "mac_1"}},
                {"name": "search_micro", "args": {"query": "x"}},
            ],
        })
        self.assertEqual([c["name"] for c in parsed["tool_calls"]],
                         ["search_events", "search_events"])
        self.assertEqual(parsed["tool_calls"][0]["args"]["macro_id"], "mac_1")
        self.assertEqual(parsed["tool_calls"][1]["args"]["query"], "x")

    def test_dispatch_reports_unknown(self):
        with tempfile.TemporaryDirectory() as d:
            box = MemoryToolBox(_mk_store(d))
            obs = box.call("get_subgraph", {"macro_id": "mac_1"}, ask_ts=10.0)
            self.assertIn("unknown tool", obs)

    def test_store_helper_kept_for_future_revival(self):
        # 注释里承诺 store 侧保留, 别让将来复活的人白找
        self.assertTrue(hasattr(MemoryStore, "get_subgraph_for_macro"))


class TestDeadEntityDispatchRemoved(unittest.TestCase):
    def test_both_dead_branches_gone(self):
        with tempfile.TemporaryDirectory() as d:
            box = MemoryToolBox(_mk_store(d))
            for name in ("get_events_by_entity", "get_frames_by_entity"):
                obs = box.call(name, {"entity_id": "ent_1"}, ask_ts=10.0)
                self.assertIn("unknown tool", obs, name)

    def test_they_were_never_reachable_anyway(self):
        for name in ("get_events_by_entity", "get_frames_by_entity"):
            self.assertNotIn(name, RecallAgent._RECALL_TOOL_NAMES, name)


class TestEntityContextNoDuplication(unittest.TestCase):
    def _fmt(self, box, **kw):
        base = dict(
            entity=None, requested_id="", resolution_chain=None,
            events=[], frame_ids=[], states=[], header="get_entity_context x",
        )
        base.update(kw)
        return box._fmt_entity_context("ent_a", **base)

    def test_canonical_block_printed_once(self):
        ent = Entity(id="ent_a", type="OBJECT", name="红色耳机",
                     attributes={"color": "red"}, aliases=["耳机"],
                     seen_count=3, first_seen=1.0, last_seen=9.0)
        with tempfile.TemporaryDirectory() as d:
            box = MemoryToolBox(_mk_store(d))
            out = self._fmt(box, entity=ent, requested_id="ent_a",
                            resolution_chain=["ent_a"])
        self.assertEqual(
            out.count("canonical entity (authoritative current state):"), 1,
            "canonical entity 块只应出现一次:\n" + out)
        self.assertEqual(out.count("name='红色耳机'"), 1)

    def test_resolution_line_printed_once_on_merge(self):
        ent = Entity(id="ent_new", type="PERSON", name="A",
                     first_seen=1.0, last_seen=2.0)
        with tempfile.TemporaryDirectory() as d:
            box = MemoryToolBox(_mk_store(d))
            out = box._fmt_entity_context(
                "ent_new", entity=ent, requested_id="ent_old",
                resolution_chain=["ent_old", "ent_new"],
                events=[], frame_ids=[], states=[], header="h")
        self.assertEqual(out.count("resolution:"), 1, out)
        self.assertIn("ent_old -> ent_new", out)
        self.assertIn("old entity was merged", out)

    def test_no_crash_on_default_none_args(self):
        # 旧实现在 entity=None / resolution_chain=None 时会 AttributeError /
        # TypeError, 只因两处调用点都实传才没炸
        with tempfile.TemporaryDirectory() as d:
            box = MemoryToolBox(_mk_store(d))
            out = self._fmt(box)
        self.assertIn("summary: entity_id=ent_a", out)
        self.assertNotIn("canonical entity", out)

    def test_screen_text_section_does_not_add_third_entity_block(self):
        """原 _fmt_artifact_context 已并入 _fmt_entity_context(screen_hits=)。

        它当年多打的那份 "artifact entity:" 是 canonical 块的截短劣化版, 合并后
        不该以任何形式回来。
        """
        ent = Entity(id="ent_f", type="FILE", name="report.docx",
                     attributes={"path": "/tmp/report.docx"},
                     aliases=["report"], first_seen=1.0, last_seen=2.0)
        with tempfile.TemporaryDirectory() as d:
            box = MemoryToolBox(_mk_store(d))
            out = box._fmt_entity_context(
                "ent_f", entity=ent, requested_id="ent_f",
                resolution_chain=["ent_f"], events=[], frame_ids=[],
                states=[], screen_hits=[], header="get_entity_context ent_f")
        self.assertNotIn("artifact entity:", out)
        self.assertEqual(out.count("name='report.docx'"), 1, out)
        self.assertIn("related_screen_text", out)
        self.assertFalse(hasattr(MemoryToolBox, "_fmt_artifact_context"))

    def test_real_dispatch_path_stays_deduped(self):
        """走真实 store + call(), 确认线上路径也只打一份实体块。"""
        with tempfile.TemporaryDirectory() as d:
            mem = _mk_store(d)
            ent = Entity(id="ent_z", type="OBJECT", name="背包",
                         attributes={"color": "black"}, aliases=[],
                         first_seen=1.0, last_seen=5.0)
            mem.upsert_entity(ent)
            box = MemoryToolBox(mem)
            obs = box.call("get_entity_context", {"entity_id": ent.id},
                           ask_ts=100.0)
        self.assertNotIn("exception:", obs)
        self.assertLessEqual(
            obs.count("canonical entity (authoritative current state):"), 1,
            obs)


class TestQuotesUnwiredGuard(unittest.TestCase):
    def test_has_any_quotes_false_on_fresh_store(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(_mk_store(d).has_any_quotes())

    def test_search_quotes_by_text_says_do_not_retry(self):
        with tempfile.TemporaryDirectory() as d:
            box = MemoryToolBox(_mk_store(d))
            obs = box.call("search_quotes_by_text", {"query": "谁说的"},
                           ask_ts=100.0)
        self.assertIn("NOT wired yet", obs)
        self.assertIn("Do NOT retry", obs)
        self.assertIn("search_audio", obs)

    def test_get_quotes_by_entity_says_do_not_retry(self):
        with tempfile.TemporaryDirectory() as d:
            mem = _mk_store(d)
            mem.upsert_entity(Entity(id="ent_p", type="PERSON", name="张三",
                                     first_seen=1.0, last_seen=2.0))
            box = MemoryToolBox(mem)
            obs = box.call("get_quotes_by_entity", {"entity_id": "ent_p"},
                           ask_ts=100.0)
        self.assertIn("NOT wired yet", obs)

    def test_guard_is_sticky_positive_only(self):
        """未接通时不缓存否定结论, 人脸/声纹上线后无需重启即可恢复正常文案。"""
        with tempfile.TemporaryDirectory() as d:
            box = MemoryToolBox(_mk_store(d))
            self.assertIsNotNone(box._quotes_unwired_hint("h"))
            self.assertFalse(getattr(box, "_quotes_seen_rows", False))

            box.mem.has_any_quotes = lambda: True      # 模拟写入侧接通
            self.assertIsNone(box._quotes_unwired_hint("h"))
            self.assertTrue(box._quotes_seen_rows)

            box.mem.has_any_quotes = lambda: False     # 之后不再重复探表
            self.assertIsNone(box._quotes_unwired_hint("h"))


class TestSurvivingToolsetShape(unittest.TestCase):
    def test_whitelist_is_10_and_all_dispatchable(self):
        self.assertEqual(len(RecallAgent._RECALL_TOOL_NAMES), 10,
                         sorted(RecallAgent._RECALL_TOOL_NAMES))
        with tempfile.TemporaryDirectory() as d:
            box = MemoryToolBox(_mk_store(d))
            for name in RecallAgent._RECALL_TOOL_NAMES:
                obs = box.call(name, {}, ask_ts=10.0)
                self.assertNotIn("unknown tool", obs, f"{name} 无分发实现")

    def test_prompt_and_whitelist_fully_agree(self):
        """合并后不再有"能调但 prompt 没写"的工具 —— 缺口应为空集。"""
        missing = {n for n in RecallAgent._RECALL_TOOL_NAMES
                   if n not in RECALL_SYSTEM}
        self.assertEqual(missing, set(), missing)

    def test_prompt_does_not_advertise_any_retired_name(self):
        for old in RecallAgent._LEGACY_TOOL_ALIASES:
            self.assertNotIn(f"- {old}(", RECALL_SYSTEM, old)


class TestEntityTimelineStillReachable(unittest.TestCase):
    def test_timeline_via_legacy_alias(self):
        """get_entity_timeline 现在是 get_entity_context 的一种参数组合。"""
        with tempfile.TemporaryDirectory() as d:
            mem = _mk_store(d)
            mem.upsert_entity(Entity(id="ent_t", type="OBJECT", name="杯子",
                                     first_seen=1.0, last_seen=5.0))
            box = MemoryToolBox(mem)
            name, args = RecallAgent._apply_legacy_tool_alias(
                "get_entity_timeline", {"entity_id": "ent_t", "limit": 7})
            self.assertEqual(name, "get_entity_context")
            self.assertEqual(args["timeline_limit"], 7)
            self.assertEqual(args["events_limit"], 0)
            obs = box.call(name, args, ask_ts=100.0)
        self.assertNotIn("unknown tool", obs)
        self.assertNotIn("exception:", obs)
        self.assertIn("timeline (", obs)
        self.assertIn("events: (not requested", obs)


if __name__ == "__main__":
    unittest.main()
