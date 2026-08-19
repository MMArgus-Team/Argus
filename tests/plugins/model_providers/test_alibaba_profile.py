"""Unit tests for the Alibaba DashScope provider profile's thinking wiring.

DashScope's OpenAI-compatible endpoint gates reasoning on the non-standard
``enable_thinking`` flag passed at the TOP LEVEL of ``extra_body`` (not nested
under ``chat_template_kwargs``, which is the vLLM/self-hosted convention).
Without an override the unified reasoning toggle never reached the wire
(defect D6). These tests pin the wire-shape contract.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def alibaba_profile():
    import model_tools  # noqa: F401  (triggers plugin discovery/registration)
    import providers

    profile = providers.get_provider_profile("alibaba")
    assert profile is not None, "alibaba provider profile must be registered"
    return profile


class TestAlibabaThinkingWireShape:
    def test_hybrid_default_enables_thinking(self, alibaba_profile):
        eb, tl = alibaba_profile.build_api_kwargs_extras(
            reasoning_config=None, model="qwen3.6-flash")
        assert eb == {"enable_thinking": True}
        assert tl == {}

    def test_explicit_disable(self, alibaba_profile):
        eb, tl = alibaba_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}, model="qwen3.6-flash")
        assert eb == {"enable_thinking": False}
        assert tl == {}

    def test_thinking_budget_passthrough(self, alibaba_profile):
        eb, _ = alibaba_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "thinking_budget": 2000},
            model="qwen-plus")
        assert eb == {"enable_thinking": True, "thinking_budget": 2000}

    def test_budget_tokens_alias(self, alibaba_profile):
        eb, _ = alibaba_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "budget_tokens": 512},
            model="qwen3-max")
        assert eb == {"enable_thinking": True, "thinking_budget": 512}

    def test_disabled_omits_budget(self, alibaba_profile):
        eb, _ = alibaba_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False, "thinking_budget": 2000},
            model="qwen3.6-flash")
        assert eb == {"enable_thinking": False}


class TestAlibabaModelGating:
    @pytest.mark.parametrize("model", [
        "qwen3.6-flash", "qwen3.5-plus", "qwen3-max", "qwen-plus",
        "qwen-flash", "qwen-turbo", "qwen-max", "qwen3.7-plus",
    ])
    def test_hybrid_models_emit_flag(self, alibaba_profile, model):
        eb, _ = alibaba_profile.build_api_kwargs_extras(
            reasoning_config=None, model=model)
        assert "enable_thinking" in eb

    @pytest.mark.parametrize("model", [
        "qwq-32b", "qwen3-235b-a22b-thinking-2507",  # thinking-only
        "qwen-vl-max", "qwen-max-2024-09-19", "qwen2.5-72b",  # non-thinking/legacy
        "", None,
    ])
    def test_non_toggle_models_emit_nothing(self, alibaba_profile, model):
        eb, tl = alibaba_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True}, model=model)
        assert eb == {}
        assert tl == {}


class TestAlibabaFullKwargs:
    def test_full_wire_shape(self, alibaba_profile):
        from agent.transports.chat_completions import ChatCompletionsTransport

        kwargs = ChatCompletionsTransport().build_kwargs(
            model="qwen3.6-flash",
            messages=[{"role": "user", "content": "ping"}],
            tools=None,
            provider_profile=alibaba_profile,
            reasoning_config={"enabled": True},
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            provider_name="alibaba",
        )
        assert kwargs["extra_body"] == {"enable_thinking": True}
