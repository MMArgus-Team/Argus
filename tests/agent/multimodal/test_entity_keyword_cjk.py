"""2026-08-19: search_entity_by_keyword 补 CJK 二元组兜底 + 相关性池化。

背景: mm_tokenize_query 把整块中文保留为**一个** token, 所以自然语言提问
("红色外套的男人把背包放哪了") 会要求整句原样出现在字段里 —— 永不命中。
search_micro_by_keyword 早就有二元组兜底 (M~4743), search_entity_by_keyword
一直没有, 而后者是 RECALL_SYSTEM "Standard object lookup" 的第一步。

顺带修的第二处漂移: entity 侧候选池此前是 `ORDER BY last_seen DESC LIMIT
pool_cap`, 字段权重打分只在"最近 pool_cap 条命中"里生效 —— 久远但精确命中的
实体进不了池子。micro 侧早已把相关性下推到 ORDER BY。两处现在共用
mm_keyword_pool_sql。

真实 SQLite, 不 mock。
"""

import os
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from agent.multimodal._config import Config  # noqa: E402
from agent.multimodal._memory import (  # noqa: E402
    Entity, MemoryStore, MicroEvent, mm_cjk_bigram_groups,
    mm_expand_groups, mm_expand_terms, mm_keyword_pool_sql,
    mm_tokenize_query,
)


def _mk_store(tmpdir):
    cfg = Config()
    cfg.mem_db_path = os.path.join(tmpdir, "mem.sqlite3")
    return MemoryStore(cfg)


# 自然语言中文提问: 分词后是一个超长 token, 严格 LIKE 必然打空。
NL_QUERY = "红色外套的男人把背包放哪了"


class TestTokenizerPremise(unittest.TestCase):
    """先把"为什么需要兜底"钉死, 否则下面的测试可能在错误的前提上通过。"""

    def test_natural_language_query_collapses_to_one_token(self):
        self.assertEqual(mm_tokenize_query(NL_QUERY), [NL_QUERY])

    def test_that_token_yields_bigrams(self):
        groups = mm_cjk_bigram_groups(mm_tokenize_query(NL_QUERY))
        self.assertTrue(groups)
        flat = [v for g in groups for v in g]
        self.assertIn("背包", flat)
        self.assertIn("红色", flat)

    def test_short_noun_phrase_is_not_bigrammed(self):
        """"红色耳机" 这种直接 LIKE 是准的, 不该退化 (会引入噪声)。"""
        self.assertEqual(mm_cjk_bigram_groups(["红色耳机"]), [])


class TestEntityKeywordCJKFallback(unittest.TestCase):
    def _seed(self, mem):
        mem.upsert_entity(Entity(id="ent_bag", type="OBJECT", name="背包",
                                 attributes={"color": "black"},
                                 aliases=["书包"], first_seen=1.0,
                                 last_seen=5.0))
        mem.upsert_entity(Entity(id="ent_coat", type="OBJECT", name="红色外套",
                                 first_seen=2.0, last_seen=6.0))
        mem.upsert_entity(Entity(id="ent_cup", type="OBJECT", name="玻璃杯",
                                 first_seen=3.0, last_seen=7.0))

    def test_natural_language_query_now_hits(self):
        """核心回归: 改之前这里返回 []。"""
        with tempfile.TemporaryDirectory() as d:
            mem = _mk_store(d)
            self._seed(mem)
            rows = mem.search_entity_by_keyword(NL_QUERY, ask_ts=100.0,
                                                top_k=5)
        ids = [e.id for e in rows]
        self.assertTrue(rows, "CJK 自然提问仍然打空, 兜底没生效")
        self.assertIn("ent_bag", ids)
        self.assertIn("ent_coat", ids)
        self.assertNotIn("ent_cup", ids, "无关实体不该被二元组捞进来")

    def test_coverage_ranks_above_recency(self):
        """覆盖更多二元组的实体要排在"更新但只沾一个"的前面。"""
        with tempfile.TemporaryDirectory() as d:
            mem = _mk_store(d)
            # ent_multi 覆盖 红色/色外/外套 三个二元组, 但 last_seen 最早
            mem.upsert_entity(Entity(
                id="ent_multi", type="OBJECT", name="红色外套",
                aliases=["背包"], first_seen=1.0, last_seen=2.0))
            # ent_one 只沾 "男人", 但是最新的
            mem.upsert_entity(Entity(
                id="ent_one", type="PERSON", name="男人",
                first_seen=1.0, last_seen=99.0))
            rows = mem.search_entity_by_keyword(NL_QUERY, ask_ts=1000.0,
                                                top_k=5)
        self.assertEqual(rows[0].id, "ent_multi",
                         [e.id for e in rows])

    def test_strict_match_still_wins_when_it_exists(self):
        """兜底只在严格匹配为空时启用, 别把精确查询也拖进二元组噪声。"""
        with tempfile.TemporaryDirectory() as d:
            mem = _mk_store(d)
            self._seed(mem)
            rows = mem.search_entity_by_keyword("背包", ask_ts=100.0, top_k=5)
        self.assertEqual([e.id for e in rows], ["ent_bag"])

    def test_ascii_query_unaffected(self):
        with tempfile.TemporaryDirectory() as d:
            mem = _mk_store(d)
            mem.upsert_entity(Entity(id="ent_pr", type="TICKET",
                                     name="PR-4213", first_seen=1.0,
                                     last_seen=2.0))
            hit = mem.search_entity_by_keyword("PR-4213", ask_ts=100.0)
            miss = mem.search_entity_by_keyword("zzz-nonexistent",
                                                ask_ts=100.0)
        self.assertEqual([e.id for e in hit], ["ent_pr"])
        self.assertEqual(miss, [], "ASCII 查询无 CJK run, 不该有兜底行")

    def test_merged_and_ask_ts_invariants_survive_the_fallback(self):
        """兜底路径必须和严格路径共用同一套 merged_into / ask_ts 约束。"""
        with tempfile.TemporaryDirectory() as d:
            mem = _mk_store(d)
            mem.upsert_entity(Entity(
                id="ent_future", type="OBJECT", name="背包",
                first_seen=500.0, last_seen=600.0))   # 晚于 ask_ts
            rows = mem.search_entity_by_keyword(NL_QUERY, ask_ts=100.0)
        self.assertEqual(rows, [], "ask_ts 快照被兜底路径绕过了 (D3 违规)")


class TestRelevancePoolPushdown(unittest.TestCase):
    """久远但精确命中的实体, 不能被 pool_cap 按 last_seen 截掉。

    改之前 entity 侧候选池是 `ORDER BY last_seen DESC LIMIT pool_cap`, 字段权重
    只在"最近 pool_cap 条命中"里排序。pool_cap = max(top_k*6, 60), 所以只要有
    >60 条"更新但只在 attributes 弱命中"的实体, 旧的精确命中就进不了池子。

    注意: 这里绕过 upsert_entity 直接写表 —— upsert_entity 会按 name 模糊度
    (mem_entity_alias_threshold=0.85) 合并同 type 实体, "杂物0..杂物79" 会被
    折叠成 10 行, 根本堆不出淹没池子所需的规模。被测的是读路径, 直接 seed 合法。
    """

    NOISE_N = 80

    def _seed_pool(self, mem, cfg_path):
        rows = [("ent_target", "红色保温水壶", "OBJECT", "{}", 1.0, 1.0)]
        for i in range(self.NOISE_N):
            # attributes 命中"水壶"(权重 1.5, 最低), 且 last_seen 全部更新
            rows.append((f"ent_noise{i}", f"N{i}-unrelated-{i * 7}", "OBJECT",
                         '{"note": "水壶旁边"}', 10.0 + i, 10.0 + i))
        with sqlite3.connect(cfg_path) as c:
            c.executemany(
                """INSERT INTO entities
                   (id, name, type, attributes, aliases, first_seen,
                    last_seen, seen_count, updated_at)
                   VALUES (?,?,?,?,'[]',?,?,1,0)""", rows)
            c.commit()

    def test_exact_hit_survives_a_pool_full_of_newer_noise(self):
        with tempfile.TemporaryDirectory() as d:
            mem = _mk_store(d)
            self._seed_pool(mem, mem.cfg.mem_db_path)
            # 多 token 查询: "水壶" 会命中噪声的 attributes, "红色" 只命中目标
            rows = mem.search_entity_by_keyword("水壶 红色", ask_ts=1000.0,
                                                top_k=3)
        self.assertTrue(rows)
        self.assertEqual(rows[0].id, "ent_target", [e.id for e in rows])

    def test_old_recency_only_pool_would_have_dropped_it(self):
        """钉死上面那条不是空跑: 复现旧 SQL, 断言它确实丢了目标。"""
        with tempfile.TemporaryDirectory() as d:
            mem = _mk_store(d)
            self._seed_pool(mem, mem.cfg.mem_db_path)
            fields = ("name", "aliases", "attributes")
            variants = mm_expand_terms(mm_tokenize_query("水壶 红色"))
            where = " OR ".join(f"{f} LIKE ?" for f in fields
                                for _ in variants)
            likes = [f"%{v}%" for _ in fields for v in variants]
            with sqlite3.connect(mem.cfg.mem_db_path) as c:
                old = [r[0] for r in c.execute(
                    f"""SELECT id FROM entities WHERE first_seen <= ?
                        AND (merged_into IS NULL OR merged_into='')
                        AND ({where})
                        ORDER BY last_seen DESC LIMIT 60""",
                    (1000.0, *likes)).fetchall()]
        self.assertEqual(len(old), 60, "噪声不足以填满 pool_cap, 用例失效")
        self.assertNotIn("ent_target", old,
                         "旧行为没有丢掉目标, 说明这个用例证明不了什么")

    def test_micro_side_behavior_unchanged_by_the_refactor(self):
        """micro 侧改成共用 builder 后行为必须不变 (它本来就是对的)。"""
        with tempfile.TemporaryDirectory() as d:
            mem = _mk_store(d)
            mem.insert_micro(MicroEvent(
                id="mic_hit", t_start=1.0, t_end=2.0, subject="男人",
                action="放下", object="背包",
                description="男人把背包放在椅子上"))
            mem.insert_micro(MicroEvent(
                id="mic_other", t_start=3.0, t_end=4.0, subject="女人",
                action="喝", object="水", description="女人在喝水"))
            strict = mem.search_micro_by_keyword("背包", ask_ts=100.0)
            nl = mem.search_micro_by_keyword(NL_QUERY, ask_ts=100.0)
        self.assertEqual([m.id for m in strict], ["mic_hit"])
        self.assertIn("mic_hit", [m.id for m in nl])


class TestKeywordPoolSqlBuilder(unittest.TestCase):
    FIELDS = ("name", "aliases")
    FW = {"name": 3.0, "aliases": 2.0}

    def test_empty_variants_short_circuits(self):
        where_or, likes, order_by, rank = mm_keyword_pool_sql(
            self.FIELDS, self.FW, [], [], recency_col="last_seen")
        self.assertEqual((where_or, likes, order_by, rank), ("", [], "", []))

    def test_relevance_is_pushed_into_order_by(self):
        terms = mm_tokenize_query("红色水壶")
        _w, _l, order_by, rank = mm_keyword_pool_sql(
            self.FIELDS, self.FW, mm_expand_groups(terms),
            mm_expand_terms(terms), recency_col="last_seen")
        self.assertIn("CASE WHEN", order_by)
        self.assertIn("last_seen DESC", order_by)
        self.assertTrue(rank)

    def test_bind_param_count_matches_placeholders(self):
        terms = mm_tokenize_query("红色水壶 backpack")
        where_or, likes, order_by, rank = mm_keyword_pool_sql(
            self.FIELDS, self.FW, mm_expand_groups(terms),
            mm_expand_terms(terms), recency_col="last_seen")
        self.assertEqual(where_or.count("?"), len(likes))
        self.assertEqual(order_by.count("?"), len(rank))

    def test_variable_budget_degrades_to_recency_not_error(self):
        """词表爆炸时放弃 rank 表达式而不是撞 SQLite 的 999 变量上限。"""
        groups = [[f"t{i}"] for i in range(400)]
        variants = [f"t{i}" for i in range(400)]
        _w, _l, order_by, rank = mm_keyword_pool_sql(
            self.FIELDS, self.FW, groups, variants, recency_col="t_end")
        self.assertEqual(order_by, "t_end DESC")
        self.assertEqual(rank, [])

    def test_recency_col_is_the_only_difference_between_callers(self):
        terms = mm_tokenize_query("背包")
        a = mm_keyword_pool_sql(self.FIELDS, self.FW, mm_expand_groups(terms),
                                mm_expand_terms(terms), recency_col="t_end")
        b = mm_keyword_pool_sql(self.FIELDS, self.FW, mm_expand_groups(terms),
                                mm_expand_terms(terms),
                                recency_col="last_seen")
        self.assertEqual(a[0], b[0])
        self.assertEqual(a[1], b[1])
        self.assertEqual(a[3], b[3])
        self.assertEqual(a[2].replace("t_end", "X"),
                         b[2].replace("last_seen", "X"))


if __name__ == "__main__":
    unittest.main()
