"""Thinking on/off semantics for the multimodal main agent.

Source of truth: ``agent.reasoning_config`` (built from
``agent.reasoning_effort`` in config.yaml via ``parse_reasoning_effort``).
The multimodal turn reads it to build the provider-specific extra_body
thinking payload:
  1. thinking derivation rule (enabled=False → OFF; else ON; missing → ON)
  2. extra_body reflects the flag, in the PROVIDER-CORRECT form:
     - Kimi/Moonshot endpoints: ``thinking: {"type": "enabled"|"disabled"}``
       (kimi ignores enable_thinking, so the OFF switch needs this form).
     - qwen custom / vLLM: ``enable_thinking`` (top-level + nested).
     - non-Qwen OpenAI-compatible models: no private thinking kwargs.

The old ``session["_mm_deep_thinking"]`` and ``prompt.submit`` ``deep_thinking``
parameter are gone end-to-end (frontend and backend); this file no longer
exercises them.
"""
import unittest

from utils import base_url_host_matches


def _think_from_reasoning_config(reasoning_config) -> bool:
    # Mirror of tui_gateway.server._apply_mm_deep_thinking's derivation.
    if isinstance(reasoning_config, dict) and reasoning_config.get("enabled") is False:
        return False
    return True


def _is_kimi(base_url: str) -> bool:
    return (base_url_host_matches(base_url, "moonshot.cn")
            or base_url_host_matches(base_url, "moonshot.ai")
            or base_url_host_matches(base_url, "kimi.com"))


def _build_extra_body(reasoning_config, base_url: str = "", model: str = "qwen") -> dict:
    # Mirror of the multimodal turn's provider-aware extra_body assembly.
    think = _think_from_reasoning_config(reasoning_config)
    if _is_kimi(base_url):
        return {"thinking": {"type": "enabled" if think else "disabled"}}
    if "qwen" not in model.lower() and "dashscope" not in base_url.lower() \
            and "aliyuncs" not in base_url.lower():
        return {}
    return {"enable_thinking": think,
            "chat_template_kwargs": {"enable_thinking": think}}


KIMI = "https://api.moonshot.cn/v1"
QWEN = "https://llm-xxx.maas.aliyuncs.com/compatible-mode/v1"

OFF = {"enabled": False}
MEDIUM = {"enabled": True, "effort": "medium"}


class TestThinkingFlag(unittest.TestCase):
    # ---- derivation rule (provider-independent) ----
    def test_missing_config_defaults_on(self):
        # No reasoning_config on the agent → thinking ON (backward compatible).
        self.assertTrue(_think_from_reasoning_config(None))
        self.assertTrue(_think_from_reasoning_config({}))

    def test_enabled_false_turns_off(self):
        self.assertFalse(_think_from_reasoning_config(OFF))

    def test_enabled_true_or_effort_only_is_on(self):
        self.assertTrue(_think_from_reasoning_config(MEDIUM))
        self.assertTrue(_think_from_reasoning_config({"enabled": True}))

    # ---- Kimi path: thinking:{type} ----
    def test_kimi_off_emits_disabled(self):
        eb = _build_extra_body(OFF, KIMI)
        self.assertEqual(eb["thinking"], {"type": "disabled"})
        self.assertNotIn("enable_thinking", eb)  # kimi ignores it

    def test_kimi_on_emits_enabled(self):
        self.assertEqual(
            _build_extra_body(MEDIUM, KIMI)["thinking"],
            {"type": "enabled"},
        )

    # ---- qwen path ----
    def test_qwen_uses_enable_thinking(self):
        eb = _build_extra_body(MEDIUM, QWEN)
        self.assertTrue(eb["enable_thinking"])
        self.assertTrue(eb["chat_template_kwargs"]["enable_thinking"])
        self.assertNotIn("thinking", eb)

    def test_qwen_off_flips_both_flags(self):
        eb = _build_extra_body(OFF, QWEN)
        self.assertFalse(eb["enable_thinking"])
        self.assertFalse(eb["chat_template_kwargs"]["enable_thinking"])

    # ---- GPT-style proxy: no private kwargs regardless of thinking state ----
    def test_gpt_proxy_gets_no_vllm_private_fields(self):
        eb = _build_extra_body(
            MEDIUM,
            "https://doc.devops.beta.xiaohongshu.com/u/example",
            model="gpt-5.6-luna",
        )
        self.assertEqual(eb, {})


if __name__ == "__main__":
    unittest.main()
