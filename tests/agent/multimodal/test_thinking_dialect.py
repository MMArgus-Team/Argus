"""Multimodal submodule provider adaptation (defects D2/D6/D9).

The multimodal workers/monitor call chat.completions.create() directly, bypassing
the main agent's provider machinery. They hardcode the vLLM-style
``extra_body.chat_template_kwargs.enable_thinking`` shape. These tests pin the
per-dialect translation (DeepSeek / DashScope) done at client-build/call time,
and the reasoning-content fallback for thinking models.
"""

from __future__ import annotations

import types

from agent.multimodal.hermes_glue import (
    _detect_thinking_provider as detect,
    normalize_thinking_kwargs as norm,
)
from agent.multimodal._workers import _msg_text


class TestDialectDetection:
    def test_by_provider(self):
        assert detect("deepseek", "", "") == "deepseek"
        assert detect("dashscope", "", "") == "dashscope"
        assert detect("alibaba", "", "") == "dashscope"
        assert detect("moonshot", "", "") == "moonshot"
        assert detect("kimi", "", "") == "moonshot"

    def test_by_url(self):
        assert detect("custom", "https://api.deepseek.com/v1", "x") == "deepseek"
        assert detect("custom", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "x") == "dashscope"
        assert detect("openai", "https://api.moonshot.ai/v1", "kimi-k2.6") == "moonshot"

    def test_default_vllm(self):
        assert detect("custom", "http://localhost:8000/v1", "qwen3.5") == "vllm"
        assert detect("", "", "") == "vllm"

    def test_gpt_model_behind_custom_proxy_uses_portable_openai_dialect(self):
        assert detect(
            "custom", "https://private.example.test/compatible/v1",
            "gpt-5.6-luna",
        ) == "openai"


class TestNormalizeThinking:
    def test_deepseek_translation(self):
        out = norm({"model": "deepseek-v4-pro",
                    "extra_body": {"top_k": 20,
                                   "chat_template_kwargs": {"enable_thinking": True}}},
                   "deepseek")
        assert out["extra_body"] == {"thinking": {"type": "enabled"}}

    def test_deepseek_disabled(self):
        out = norm({"model": "deepseek-v4-pro",
                    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
                   "deepseek")
        assert out["extra_body"] == {"thinking": {"type": "disabled"}}

    def test_deepseek_v3_strips_flag(self):
        out = norm({"model": "deepseek-chat",
                    "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
                   "deepseek")
        # V3 has no thinking mode → no thinking key, extra_body dropped if empty.
        assert "extra_body" not in out or "thinking" not in out.get("extra_body", {})

    def test_dashscope_top_level_flag(self):
        out = norm({"model": "qwen3.6-flash",
                    "extra_body": {"top_k": 20,
                                   "chat_template_kwargs": {"enable_thinking": False}}},
                   "dashscope")
        assert out["extra_body"] == {"enable_thinking": False}

    def test_vllm_untouched(self):
        eb = {"top_k": 20, "chat_template_kwargs": {"enable_thinking": True}}
        out = norm({"model": "qwen3.5", "extra_body": dict(eb)}, "vllm")
        assert out["extra_body"] == eb

    def test_moonshot_noop(self):
        eb = {"chat_template_kwargs": {"enable_thinking": True}}
        out = norm({"model": "kimi-k2.6", "extra_body": dict(eb)}, "moonshot")
        assert out["extra_body"] == eb

    def test_openai_strips_vllm_only_keys(self):
        out = norm({
            "model": "gpt-5.6-luna",
            "temperature": 0.2,
            "top_p": 0.8,
            "extra_body": {
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        }, "openai")
        assert "extra_body" not in out
        # Standard OpenAI-compatible sampling keys remain ordinary top-level
        # kwargs; only the vLLM-only extension fields are removed.
        assert out["temperature"] == 0.2
        assert out["top_p"] == 0.8


def _resp(content=None, reasoning_content=None, reasoning=None):
    msg = types.SimpleNamespace(
        content=content, reasoning_content=reasoning_content, reasoning=reasoning)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


class TestMsgTextReasoningFallback:
    def test_content(self):
        assert _msg_text(_resp(content="hello")) == "hello"

    def test_reasoning_content_fallback(self):
        assert _msg_text(_resp(content=None, reasoning_content="thought")) == "thought"

    def test_reasoning_fallback(self):
        assert _msg_text(_resp(content="", reasoning="r")) == "r"

    def test_empty(self):
        assert _msg_text(_resp()) == ""
