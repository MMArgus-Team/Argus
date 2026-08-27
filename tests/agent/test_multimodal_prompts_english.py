import ast
import re
from pathlib import Path

from agent.multimodal import _workers
from agent.multimodal.monitor_agent import MONITOR_AGENT_SYSTEM
from agent.multimodal.scene_dhash import SCENE_DHASH_SYSTEM
from agent.multimodal.voice_agent_context import (
    _DECIDE_ROUTE_SYSTEM,
    _DECIDE_SPEAK_SYSTEM,
    _INTENT_ADDRESSED_SYSTEM,
    _INTENT_EOU_SYSTEM,
    _PHRASE_UTTERANCE_SYSTEM,
)
from agent.multimodal.voice_rewrite import _JUDGE_SYSTEM, build_system_prompt
from agent.prompt_builder import MM_LIVE_GUIDANCE
from agent.prompt_i18n import LANGUAGE_POLICY


_CJK_RE = re.compile(r"[\u3400-\u9fff]")


def test_runtime_multimodal_prompts_are_authored_in_english():
    prompts = {
        "main_agent_multimodal_guidance": MM_LIVE_GUIDANCE,
        "language_policy": LANGUAGE_POLICY,
        "monitor": MONITOR_AGENT_SYSTEM,
        "scene_probe": SCENE_DHASH_SYSTEM,
        "voice_decide_speak": _DECIDE_SPEAK_SYSTEM,
        "voice_decide_route": _DECIDE_ROUTE_SYSTEM,
        "voice_phrase": _PHRASE_UTTERANCE_SYSTEM,
        "voice_intent": _INTENT_ADDRESSED_SYSTEM,
        "voice_intent_eou": _INTENT_EOU_SYSTEM,
        "voice_tts_rewrite": build_system_prompt(),
        "voice_dedup": _JUDGE_SYSTEM,
        "memory_writer": _workers.MEMORY_WRITER_SYSTEM,
        "memory_ocr": _workers.OCR_SYSTEM,
        "memory_reviewer": _workers.MEMORY_REVIEWER_SYSTEM,
        "recall": _workers.RECALL_SYSTEM,
        "recall_distill": _workers.RECALL_DISTILL_SYSTEM,
        "recall_verify": _workers.RECALL_VERIFY_SYSTEM,
        "watcher_react": _workers.WATCHER_REACT_SYSTEM,
        "watcher_answer": _workers.WATCHER_ANSWER_SYSTEM,
        "watcher_summary": _workers.WATCHER_SUMMARY_SYSTEM,
    }

    for name, prompt in prompts.items():
        assert not _CJK_RE.search(prompt), f"{name} contains CJK instructions"


def test_tool_schema_descriptions_are_authored_in_english():
    repo_root = Path(__file__).resolve().parents[2]
    schema_files = (
        repo_root / "tools" / "yuanbao_tools.py",
        repo_root / "tools" / "computer_use" / "schema.py",
    )

    for path in schema_files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values):
                if not (
                    isinstance(key, ast.Constant)
                    and key.value == "description"
                ):
                    continue
                try:
                    description = ast.literal_eval(value)
                except (ValueError, TypeError):
                    continue
                if isinstance(description, str):
                    assert not _CJK_RE.search(description), (
                        f"{path.name}:{value.lineno} contains CJK tool instructions"
                    )
