"""Unit tests for the Z.AI / GLM provider profile's thinking wiring.

GLM-4.5+/GLM-5+ gate reasoning with ``thinking: {"type": "enabled"|"disabled"}``
(same shape as DeepSeek), passed via ``extra_body`` on the OpenAI-compatible
endpoint. Without an override the unified reasoning toggle never reached the wire
and GLM's default-on thinking could trip the reasoning-echo contract on later
turns (defect D3). These tests pin the wire-shape contract.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def zai_profile():
    import model_tools  # noqa: F401  (triggers plugin discovery/registration)
    import providers

    profile = providers.get_provider_profile("zai")
    assert profile is not None, "zai provider profile must be registered"
    return profile


class TestZaiThinkingWireShape:
    def test_default_enables_thinking(self, zai_profile):
        eb, tl = zai_profile.build_api_kwargs_extras(
            reasoning_config=None, model="glm-4.6")
        assert eb == {"thinking": {"type": "enabled"}}
        assert tl == {}

    def test_explicit_disable(self, zai_profile):
        eb, tl = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}, model="glm-5.2")
        assert eb == {"thinking": {"type": "disabled"}}
        assert tl == {}


class TestZaiModelGating:
    @pytest.mark.parametrize("model", [
        "glm-4.5", "glm-4.6", "glm-5", "glm-5.2", "glm-4-5-air", "glm-6",
    ])
    def test_thinking_capable_emit_flag(self, zai_profile, model):
        eb, _ = zai_profile.build_api_kwargs_extras(
            reasoning_config=None, model=model)
        assert eb == {"thinking": {"type": "enabled"}}

    @pytest.mark.parametrize("model", [
        "glm-4-9b", "glm-4-flash", "glm-4v-plus", "", None, "gpt-5",
    ])
    def test_non_thinking_emit_nothing(self, zai_profile, model):
        eb, tl = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True}, model=model)
        assert eb == {}
        assert tl == {}


class TestZaiFullKwargs:
    def test_full_wire_shape(self, zai_profile):
        from agent.transports.chat_completions import ChatCompletionsTransport

        kwargs = ChatCompletionsTransport().build_kwargs(
            model="glm-4.6",
            messages=[{"role": "user", "content": "ping"}],
            tools=None,
            provider_profile=zai_profile,
            reasoning_config={"enabled": False},
            base_url="https://api.z.ai/api/paas/v4",
            provider_name="zai",
        )
        assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
