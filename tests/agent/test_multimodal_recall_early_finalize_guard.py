from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock

from agent.multimodal._config import Config
from agent.multimodal._workers import RecallAgent


class TestRecallEarlyFinalizeGuard(unittest.IsolatedAsyncioTestCase):
    def test_exact_id_uncertain_clue_does_not_early_finalize(self):
        bad_clue = (
            "历史画面中可见白蓝涂装的出租车，说明其从事出租车客运业务"
            "（证据帧：f_3ad1d04e62）。但这些帧无法清晰确认车牌"
            "“粤B70463D”对应车辆的具体品牌，需进一步复核车身或车尾标识。"
        )

        assert not RecallAgent._distilled_clue_seems_answerable(
            brief="请回忆视频中车牌为粤B70463D的车辆：它在做什么业务？",
            user_text="视频中粤B70463D这辆车是做什么业务的",
            clue=bad_clue,
        )

    def test_exact_id_positive_clue_needs_identifier_and_frame(self):
        assert RecallAgent._distilled_clue_seems_answerable(
            brief="请回忆视频中粤B70463D这辆车：它在做什么业务？",
            user_text="视频中粤B70463D这辆车是做什么业务的",
            clue=(
                "证据帧 f_c45c2b25b5 显示车尾文字“东部公交 70463D”，"
                "因此该车从事公交运营业务。"
            ),
        )
        assert not RecallAgent._distilled_clue_seems_answerable(
            brief="请回忆视频中粤B70463D这辆车：它在做什么业务？",
            user_text="视频中粤B70463D这辆车是做什么业务的",
            clue="画面显示一辆公交车从事公交运营业务。",
        )

    async def test_exact_id_verify_failure_drops_unverified_clue_from_findings(self):
        cfg = Config()
        cfg.recall_max_rounds = 1
        cfg.recall_verify_enabled = True
        cfg.recall_verify_max_frames = 4
        mem = MagicMock()
        mem.get_recent_entities.return_value = []
        recall = RecallAgent(
            cfg, mem, MagicMock(), MagicMock(),
            frame_store=SimpleNamespace(), model="stub-model")
        recall._frames_payload = lambda *args, **kwargs: []
        bad_clue = (
            "历史画面中可见白蓝涂装的出租车（证据帧：f_3ad1d04e62）。"
            "但这些帧无法清晰确认车牌“粤B70463D”对应车辆，需进一步复核。"
        )
        recall.mem_tools.call = MagicMock(return_value=(
            "[search_micro query='粤B70463D']\n"
            "[02:24] frame_id=f_3ad1d04e62 蓝白涂装出租车"
        ))
        recall._distill = AsyncMock(return_value=bad_clue)
        recall._decide_next = AsyncMock(return_value={
            "thought": "enough",
            "can_answer": True,
            "useful_info": bad_clue,
            "tool_calls": [],
        })
        recall._verify_frames_with_grounding = AsyncMock(return_value=(
            None,
            "精确文字/编号的视觉验收失败，不能采信先前基于地点或上下文补全出的字符。",
        ))

        result = await recall.run(
            initial_calls=[{
                "name": "search_micro",
                "args": {"query": "粤B70463D 车辆 业务 品牌"},
            }],
            brief="请回忆视频中车牌为粤B70463D的车辆：它在做什么业务？",
            user_text="视频中粤B70463D这辆车是做什么业务的",
            ask_ts=200.0,
        )

        assert "视觉复核未完成" in result.findings
        assert "出租车" not in result.findings
        assert "不能采信" in result.findings
