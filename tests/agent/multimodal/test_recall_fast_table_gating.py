"""Gating for the cheap screen-text arm of RecallAgent.

Two invariants are pinned here, both from the 2026-08-26 redesign:

1. The short-circuit (``_table_fast_path``, ``rounds=0``, no ReAct at all) is
   only allowed when the query carries an *explicit numbered reference*
   (``Table 3`` / ``表3`` / ``图2``) **and** that reference actually shows up in
   the matched screen text. Before this, ``_table_ref_terms`` was only a filter
   that ran when a number happened to exist, so number-less queries reached the
   short-circuit with no check at all.

2. The trigger regex must respect ASCII word boundaries. The real regression:
   every QueryWorker delegation carries the boilerplate "The brief must not
   **narrow**, replace, ..." injected by ``tools/mm_memory_tool.py``, and the
   bare substring ``row`` matched inside ``narrow`` — so effectively *every*
   delegation took the table short-circuit.

Everything that no longer short-circuits must still get the cheap retrieval,
but as a round-0 seed observation feeding the ReAct loop.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock


# Reproduced from tools/mm_memory_tool.py's injected delegation template. Kept
# verbatim so this test fails if the template regains a table-ish trigger word.
DELEGATION_BOILERPLATE = """### AUTHORITATIVE ORIGINAL USER QUESTION
他刚才说了什么

Answer every requested part of this question.

### CONTEXT-RESOLVED MAIN AGENT DELEGATION BRIEF
用户想知道刚才那个人说的话

Use this brief for directly useful prior-QA context, referent binding, and task
planning, while preserving uncertainty. The brief must not narrow, replace,
contradict, or reinterpret the original question. Ignore any part of the brief
that does.

Inspect the ask-time frames first.
"""


def _recall(*, screen_text_store=None, tool_result=""):
    from agent.multimodal._config import Config
    from agent.multimodal._workers import RecallAgent

    agent = RecallAgent(
        Config(), MagicMock(), MagicMock(), MagicMock(),
        screen_text_store=screen_text_store or MagicMock(),
    )
    agent.frame_store = None
    agent.mem_tools.call = MagicMock(return_value=tool_result)
    return agent


class TestTableTriggerWordBoundaries(unittest.TestCase):
    def setUp(self):
        from agent.multimodal._workers import RecallAgent
        self.match = RecallAgent._looks_like_table_recall_query

    def test_substrings_of_ordinary_words_do_not_trigger(self):
        for text in [
            "the brief must not narrow the question",   # row  ⊂ narrow
            "revenue growth last quarter",              # row  ⊂ growth
            "he was reading a newspaper",               # paper ⊂ newspaper
            "tablecloth on the desk",                   # table ⊂ tablecloth
            "he slid the papercup across",              # paper ⊂ papercup
        ]:
            with self.subTest(text=text):
                self.assertFalse(self.match(text))

    def test_the_injected_delegation_boilerplate_never_triggers(self):
        self.assertFalse(self.match(DELEGATION_BOILERPLATE))

    def test_real_table_questions_still_trigger(self):
        for text in [
            "what does Table 3 say",
            "Fig. 2 caption",
            "表3 里 Argus 的分数",
            "图 2 的纵轴是什么",
            "这篇论文的 benchmark 结果",
            "slide 里那一列数字",
            "which rows are in the dataset",
        ]:
            with self.subTest(text=text):
                self.assertTrue(self.match(text))


class TestExplicitRefIsAPrecondition(unittest.IsolatedAsyncioTestCase):
    async def test_no_numbered_reference_means_no_tool_call_at_all(self):
        """Number-less table-ish queries must not even pay for the fast path.

        The seed observation covers them instead, so spending a retrieval here
        would just duplicate it.
        """
        agent = _recall()
        result = await agent._table_fast_path(
            brief="这篇论文说了什么", user_text="benchmark 结果",
            ask_ts=60.0, emit=None,
        )
        self.assertIsNone(result)
        agent.mem_tools.call.assert_not_called()

    async def test_reference_absent_from_matched_text_blocks_short_circuit(self):
        agent = _recall(tool_result=(
            "[search_screen_text query='表 2']\n"
            "[00:42-00:45] frame_id=f_1234567890 | 一段完全无关的屏幕文字"
        ))
        result = await agent._table_fast_path(
            brief="表 2 里的分数", user_text="",
            ask_ts=60.0, emit=None,
        )
        self.assertIsNone(result)
        agent.mem_tools.call.assert_called_once()

    async def test_reference_present_in_matched_text_short_circuits(self):
        agent = _recall(tool_result=(
            "[search_screen_text query='表 2']\n"
            "[00:42-00:45] frame_id=f_1234567890 | 表 2 | Model | Score\n"
            "[00:42-00:45] frame_id=f_1234567890 | Argus | 91.2"
        ))
        result = await agent._table_fast_path(
            brief="表 2 里 Argus 的 Score 是多少", user_text="",
            ask_ts=60.0, emit=None,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.rounds, 0)
        self.assertEqual(result.origin, "fast_table")


class TestSeedObservation(unittest.IsolatedAsyncioTestCase):
    def test_boilerplate_only_delegation_yields_no_seed(self):
        agent = _recall()
        self.assertEqual(
            agent._screen_text_seed_calls(
                brief="用户想知道刚才那个人说的话",
                user_text=DELEGATION_BOILERPLATE),
            [],
        )

    def test_table_question_yields_one_screen_text_seed(self):
        agent = _recall()
        calls = agent._screen_text_seed_calls(
            brief="这篇论文的 benchmark 表格里 Argus 多少分", user_text="")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "search_screen_text")
        self.assertEqual(calls[0]["args"]["limit"], 8)
        self.assertIn("benchmark", calls[0]["args"]["query"])

    def test_seed_becomes_round_zero_observation_not_final_findings(self):
        """The number-less table question keeps the cheap retrieval, but the
        LLM sees it as evidence and stays free to escalate."""
        import asyncio

        agent = _recall(tool_result=(
            "[search_screen_text query='benchmark']\n"
            "[00:42-00:45] frame_id=f_1234567890 | Argus | 91.2"
        ))
        agent._distill = AsyncMock(return_value="屏幕文字里 Argus 是 91.2")
        agent._decide_next = AsyncMock(return_value={
            "can_answer": True, "useful_info": "Argus 的分数是 91.2",
            "tool_calls": [],
        })

        result = asyncio.run(agent.run(
            initial_calls=[],
            brief="这篇论文的 benchmark 表格里 Argus 多少分",
            user_text="", ask_ts=60.0,
        ))

        # The seed ran as a tool call, not as a short-circuit.
        first_call = agent.mem_tools.call.call_args_list[0]
        self.assertEqual(first_call.args[0], "search_screen_text")
        self.assertGreaterEqual(result.rounds, 1)
        self.assertEqual(result.origin, "react+seed")
        # Distillation ran over the seed observation (the raw obs is no longer
        # handed back verbatim as findings, which is what the short-circuit did).
        agent._distill.assert_awaited()
        self.assertIn("91.2", result.findings)
        self.assertNotIn("[search_screen_text", result.findings)

    def test_seed_is_ordered_before_initial_calls(self):
        import asyncio

        agent = _recall(tool_result="[search_screen_text] nothing useful")
        agent._distill = AsyncMock(return_value="")
        agent._decide_next = AsyncMock(return_value={
            "can_answer": True, "useful_info": "done", "tool_calls": [],
        })
        asyncio.run(agent.run(
            initial_calls=[{"name": "search_events", "args": {"query": "表格"}}],
            brief="这篇论文的表格里第二列", user_text="", ask_ts=60.0,
        ))
        names = [c.args[0] for c in agent.mem_tools.call.call_args_list]
        self.assertEqual(names[:2], ["search_screen_text", "search_events"])

    def test_failed_short_circuit_falls_through_to_the_seed(self):
        """Documents the one accepted duplicate: an explicit ``表 3`` pays for
        the fast-path retrieval, and when the reference is absent from the
        matched text the same local FTS query runs again as the r0 seed.
        It is ~0.01s of SQLite with no LLM involved, so no cache is warranted.
        """
        import asyncio

        agent = _recall(tool_result="[search_screen_text] nothing useful")
        agent._distill = AsyncMock(return_value="")
        agent._decide_next = AsyncMock(return_value={
            "can_answer": True, "useful_info": "done", "tool_calls": [],
        })
        result = asyncio.run(agent.run(
            initial_calls=[], brief="表 3 的第二列", user_text="", ask_ts=60.0,
        ))
        names = [c.args[0] for c in agent.mem_tools.call.call_args_list]
        self.assertEqual(names, ["search_screen_text", "search_screen_text"])
        self.assertEqual(result.origin, "react+seed")


class TestOriginIsReportedForDiag(unittest.TestCase):
    def test_default_origin_is_plain_react(self):
        from agent.multimodal._workers import RecallResult
        self.assertEqual(RecallResult(findings="x").origin, "react")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
