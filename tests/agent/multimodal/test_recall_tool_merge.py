"""2026-08-19 recall 工具集合并 (16 → 10) 的运行时断言。

被合并的三组:
  1. search_events   ← search_micro + search_by_time + get_subgraph
     新增 macro_id 入参, 且 _fmt_micros / _fmt_entity_context 的事件行开始回显
     `macro=mac_xxx` —— 这是"展开整个宏事件"这条路从结构性不可达变为可达的
     关键, 缺了它 search_events(macro_id=) 跟老 get_subgraph 一样没人能调。
  2. get_entity_context ← get_artifact_context + get_entity_timeline
                          + get_relations
     用 include_screen_text / include_relations / *_limit=0 表达差异。limit=0 的
     段落必须渲染成 "(not requested)" 而不是 "(empty)" —— 后者会被模型读成
     "这个实体没有事件", 是最贵的一类假阴性。
  3. search_audio    ← get_audio_around
     query 模式 / t+window_sec 模式 / 两者同时给。

外加别名层: 旧名必须被映射而不是被白名单静默丢弃。

全部用真实 SQLite (tmp 文件), 不 mock store。
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from agent.multimodal._config import Config  # noqa: E402
from agent.multimodal._memory import (  # noqa: E402
    Entity, MacroEvent, MemoryStore, MicroEvent,
)
from agent.multimodal._workers import (  # noqa: E402
    MemoryToolBox, RECALL_SYSTEM, RecallAgent,
)


def _mk_store(tmpdir):
    cfg = Config()
    cfg.mem_db_path = os.path.join(tmpdir, "mem.sqlite3")
    return MemoryStore(cfg)


def _seed(mem):
    """两条 micro 挂在同一个 macro 下, 时间上分开, 方便区分三种检索模式。"""
    mac = MacroEvent(id="mac_1", t_start=0.0, t_end=60.0,
                     label="厨房片段", summary="有人在厨房做饭")
    mem.insert_macro(mac)
    m1 = MicroEvent(
        id="mic_1", t_start=5.0, t_end=8.0, subject="男人", action="拿起",
        object="红色水壶", description="男人从灶台上拿起红色水壶",
        macro_id="mac_1", frame_ids=["fr_1"])
    m2 = MicroEvent(
        id="mic_2", t_start=40.0, t_end=44.0, subject="男人", action="倒",
        object="水", description="男人把水倒进杯子里", macro_id="mac_1")
    mem.insert_micro(m1)
    mem.insert_micro(m2)
    return mac, m1, m2


class TestSearchEventsModes(unittest.TestCase):
    def test_time_window_mode(self):
        with tempfile.TemporaryDirectory() as d:
            mem = _mk_store(d)
            _seed(mem)
            box = MemoryToolBox(mem)
            obs = box.call("search_events",
                           {"t_start": 0.0, "t_end": 20.0}, ask_ts=100.0)
        self.assertNotIn("exception:", obs)
        self.assertIn("mic_1", obs)
        self.assertNotIn("mic_2", obs)      # 40s 那条在窗口外

    def test_macro_id_mode_expands_the_whole_segment(self):
        with tempfile.TemporaryDirectory() as d:
            mem = _mk_store(d)
            _seed(mem)
            box = MemoryToolBox(mem)
            obs = box.call("search_events", {"macro_id": "mac_1"},
                           ask_ts=100.0)
        self.assertNotIn("exception:", obs)
        self.assertIn("window from macro mac_1", obs)
        self.assertIn("mic_1", obs)
        self.assertIn("mic_2", obs)

    def test_unknown_macro_id_is_actionable_not_silent(self):
        with tempfile.TemporaryDirectory() as d:
            box = MemoryToolBox(_mk_store(d))
            obs = box.call("search_events", {"macro_id": "mac_nope"},
                           ask_ts=100.0)
        self.assertIn("macro not found", obs)
        self.assertIn("t_start", obs)       # 告诉模型下一步怎么走

    def test_no_args_returns_usage_hint_not_silent_recent_rows(self):
        with tempfile.TemporaryDirectory() as d:
            box = MemoryToolBox(_mk_store(d))
            obs = box.call("search_events", {}, ask_ts=100.0)
        self.assertIn("needs at least one of", obs)

    def test_time_upper_bound_is_clamped_to_ask_ts(self):
        """D3 防脏读: t_end 超过 ask_ts 必须被夹, 而不是撞 store 的 assert。"""
        with tempfile.TemporaryDirectory() as d:
            mem = _mk_store(d)
            _seed(mem)
            box = MemoryToolBox(mem)
            obs = box.call("search_events",
                           {"t_start": 0.0, "t_end": 9999.0}, ask_ts=10.0)
        self.assertNotIn("out of bounds", obs)
        self.assertNotIn("exception:", obs)
        self.assertIn("mic_1", obs)
        self.assertNotIn("mic_2", obs)      # 40s > ask_ts=10

    def test_query_plus_window_labels_filler_rows(self):
        """query + 窗口是合并后独有的组合; 补齐行必须标出来。

        否则模型会把"窗口里凑数的时间序行"当成 query 命中, 高估相关性。
        """
        with tempfile.TemporaryDirectory() as d:
            mem = _mk_store(d)
            _seed(mem)
            box = MemoryToolBox(mem)
            obs = box.call(
                "search_events",
                {"query": "红色水壶", "t_start": 0.0, "t_end": 60.0,
                 "top_k": 5},
                ask_ts=100.0)
        self.assertNotIn("exception:", obs)
        self.assertIn("window=[0.0,60.0]", obs)
        self.assertIn("filler row", obs)


class TestMacroIdEchoed(unittest.TestCase):
    """没有回显就没有可达性 —— 这是老 get_subgraph 死掉的直接原因。"""

    def test_fmt_micros_echoes_macro(self):
        with tempfile.TemporaryDirectory() as d:
            mem = _mk_store(d)
            _seed(mem)
            box = MemoryToolBox(mem)
            obs = box.call("search_events",
                           {"t_start": 0.0, "t_end": 60.0}, ask_ts=100.0)
        self.assertIn("macro=mac_1", obs)

    def test_entity_context_events_echo_macro(self):
        with tempfile.TemporaryDirectory() as d:
            mem = _mk_store(d)
            _, m1, _ = _seed(mem)
            ent = Entity(id="ent_k", type="OBJECT", name="红色水壶",
                         first_seen=5.0, last_seen=8.0)
            mem.upsert_entity(ent)
            mem.link_entity_event(ent.id, m1.id, 5.0)
            box = MemoryToolBox(mem)
            obs = box.call("get_entity_context", {"entity_id": ent.id},
                           ask_ts=100.0)
        self.assertNotIn("exception:", obs)
        self.assertIn("micro_id=mic_1", obs)
        self.assertIn("macro=mac_1", obs)


class TestEntityContextSections(unittest.TestCase):
    def _box_with_entity(self, d):
        mem = _mk_store(d)
        _seed(mem)
        mem.upsert_entity(Entity(id="ent_a", type="OBJECT", name="红色水壶",
                                 first_seen=5.0, last_seen=8.0))
        return MemoryToolBox(mem)

    def test_zero_limit_says_not_requested_not_empty(self):
        with tempfile.TemporaryDirectory() as d:
            box = self._box_with_entity(d)
            obs = box.call("get_entity_context",
                           {"entity_id": "ent_a", "events_limit": 0,
                            "frames_limit": 0}, ask_ts=100.0)
        self.assertIn("events: (not requested", obs)
        self.assertIn("frames: (not requested", obs)
        self.assertIn("events=(not requested)", obs)   # summary 行也要一致
        self.assertIn("timeline (", obs)               # 这一段还是要出

    def test_include_relations_adds_the_section(self):
        with tempfile.TemporaryDirectory() as d:
            box = self._box_with_entity(d)
            off = box.call("get_entity_context", {"entity_id": "ent_a"},
                           ask_ts=100.0)
            on = box.call("get_entity_context",
                          {"entity_id": "ent_a", "include_relations": True},
                          ask_ts=100.0)
        self.assertNotIn("relations of ent_a", off)
        self.assertIn("relations of ent_a (1 hop)", on)
        self.assertIn("relations=", on)

    def test_include_screen_text_adds_the_section(self):
        with tempfile.TemporaryDirectory() as d:
            box = self._box_with_entity(d)
            off = box.call("get_entity_context", {"entity_id": "ent_a"},
                           ask_ts=100.0)
            on = box.call("get_entity_context",
                          {"entity_id": "ent_a",
                           "include_screen_text": True}, ask_ts=100.0)
        self.assertNotIn("related_screen_text", off)
        self.assertIn("related_screen_text", on)

    def test_all_sections_off_is_an_explicit_error(self):
        with tempfile.TemporaryDirectory() as d:
            box = self._box_with_entity(d)
            obs = box.call("get_entity_context",
                           {"entity_id": "ent_a", "events_limit": 0,
                            "frames_limit": 0, "timeline_limit": 0},
                           ask_ts=100.0)
        self.assertIn("every section was disabled", obs)

    def test_node_id_is_accepted_as_entity_id(self):
        """get_relations 用的是 node_id; 别名层映射后这里必须收得住。"""
        with tempfile.TemporaryDirectory() as d:
            box = self._box_with_entity(d)
            obs = box.call("get_entity_context", {"node_id": "ent_a"},
                           ask_ts=100.0)
        self.assertNotIn("exception:", obs)
        self.assertIn("entity_id=ent_a", obs)


class TestSearchAudioModes(unittest.TestCase):
    def test_no_args_returns_usage_hint(self):
        with tempfile.TemporaryDirectory() as d:
            box = MemoryToolBox(_mk_store(d))
            obs = box.call("search_audio", {}, ask_ts=100.0)
        self.assertIn("needs a query", obs)

    def test_t_mode_reports_window_clamp(self):
        with tempfile.TemporaryDirectory() as d:
            box = MemoryToolBox(_mk_store(d))
            obs = box.call("search_audio", {"t": 50.0, "window_sec": 9999.0},
                           ask_ts=1000.0)
        self.assertNotIn("exception:", obs)
        self.assertIn("window_sec clamped", obs)

    def test_t_mode_header_shape(self):
        with tempfile.TemporaryDirectory() as d:
            box = MemoryToolBox(_mk_store(d))
            obs = box.call("search_audio", {"t": 50.0, "window_sec": 30.0},
                           ask_ts=1000.0)
        self.assertIn("search_audio around t=50.0s", obs)

    def test_query_mode_does_not_crash_on_empty_transcript(self):
        with tempfile.TemporaryDirectory() as d:
            box = MemoryToolBox(_mk_store(d))
            obs = box.call("search_audio", {"query": "水壶"}, ask_ts=100.0)
        self.assertNotIn("exception:", obs)

    def test_query_plus_t_distinguishes_window_miss_from_global_miss(self):
        with tempfile.TemporaryDirectory() as d:
            box = MemoryToolBox(_mk_store(d))
            obs = box.call("search_audio",
                           {"query": "水壶", "t": 50.0, "window_sec": 10.0},
                           ask_ts=100.0)
        self.assertNotIn("exception:", obs)
        # 空库 → 全片都没有, 应该说"全片没有"而不是"窗口内没有"
        self.assertIn("no hit anywhere in the transcript", obs)


class TestLegacyAliasLayer(unittest.TestCase):
    def test_every_retired_name_maps_into_the_whitelist(self):
        for old, (new, _m) in RecallAgent._LEGACY_TOOL_ALIASES.items():
            self.assertIn(new, RecallAgent._RECALL_TOOL_NAMES, old)
            self.assertNotIn(old, RecallAgent._RECALL_TOOL_NAMES, old)

    def test_alias_survives_the_whitelist_filter(self):
        """核心回归点: 别名必须在白名单过滤**之前**生效。

        放在之后等于没放 —— 旧名会在 `if name not in _RECALL_TOOL_NAMES:
        continue` 处被静默丢掉, 模型只看到"空了一轮"然后原样重试, 4 轮预算
        直接烧光。
        """
        parsed, repairs = RecallAgent._normalize_decision_tool_calls({
            "can_answer": False,
            "tool_calls": [
                {"name": "search_by_time", "args": {"t_start": 1, "t_end": 2}},
                {"name": "get_audio_around", "args": {"t": 5}},
                {"name": "get_relations", "args": {"node_id": "ent_a"}},
                {"name": "get_artifact_context", "args": {"entity_id": "ent_f"}},
            ],
        })
        self.assertEqual([c["name"] for c in parsed["tool_calls"]],
                         ["search_events", "search_audio",
                          "get_entity_context", "get_entity_context"])
        self.assertTrue(any("legacy_alias" in str(r) for r in repairs))

    def test_relations_alias_carries_the_intent(self):
        """映射不能只改名: get_relations 的意图是"要边", 得把开关打开。"""
        name, args = RecallAgent._apply_legacy_tool_alias(
            "get_relations", {"node_id": "ent_a", "max_hops": 2})
        self.assertEqual(name, "get_entity_context")
        self.assertEqual(args["entity_id"], "ent_a")
        self.assertTrue(args["include_relations"])
        self.assertEqual(args["events_limit"], 0)

    def test_artifact_alias_turns_on_screen_text(self):
        name, args = RecallAgent._apply_legacy_tool_alias(
            "get_artifact_context", {"entity_id": "ent_f"})
        self.assertEqual(name, "get_entity_context")
        self.assertTrue(args["include_screen_text"])

    def test_explicit_arg_wins_over_alias_default(self):
        _n, args = RecallAgent._apply_legacy_tool_alias(
            "get_entity_timeline", {"entity_id": "e", "events_limit": 5})
        self.assertEqual(args["events_limit"], 5)

    def test_shorthand_object_form_also_aliased(self):
        parsed, _ = RecallAgent._normalize_decision_tool_calls({
            "can_answer": False,
            "tool_calls": [{"search_micro": "红色水壶"}],
        })
        self.assertEqual(parsed["tool_calls"],
                         [{"name": "search_events",
                           "args": {"query": "红色水壶"}}])

    def test_fail_closed_still_triggers_on_retired_names(self):
        """提到 search_micro 同样说明模型想检索, 不许翻成 can_answer=true。"""
        out = RecallAgent._apply_decision_fail_closed(
            {"can_answer": True}, "I should call search_micro next")
        self.assertFalse(out["can_answer"])

    def test_text_recovery_maps_retired_names(self):
        rec = RecallAgent._recover_decision_from_text(
            'can_answer: false\nsearch_micro("红色水壶")')
        self.assertIsNotNone(rec)
        self.assertEqual([c["name"] for c in rec["tool_calls"]],
                         ["search_events"])


class TestMergedSurfaceIsAdvertised(unittest.TestCase):
    def test_prompt_documents_the_new_params(self):
        for token in ("search_events(query?", "macro_id",
                      "include_screen_text", "include_relations",
                      "search_audio(query?"):
            self.assertIn(token, RECALL_SYSTEM, token)

    def test_prompt_no_longer_routes_through_retired_names(self):
        for old in ("search_micro(", "search_by_time(", "get_audio_around(",
                    "get_artifact_context(", "get_entity_timeline(",
                    "get_relations(", "get_subgraph("):
            self.assertNotIn(old, RECALL_SYSTEM, old)

    def test_merged_formatters_are_gone(self):
        for gone in ("_fmt_subgraph", "_fmt_artifact_context",
                     "_fmt_entity_timeline"):
            self.assertFalse(hasattr(MemoryToolBox, gone), gone)


if __name__ == "__main__":
    unittest.main()
