"""The UI must not offer a thinking switch the backend cannot honour.

``supports_reasoning`` (model catalog) says the model REASONS.
``can_toggle_reasoning`` (this contract) says WE CAN CHANGE THAT.

Those were conflated, so the thinking control rendered live for models whose
endpoint accepts no toggle — it looked functional and did nothing. Three real
cases:

  * thinking-only models (``kimi-k2.7-code``) that 400 on an explicit off
  * pre-toggle generations (``glm-4-flash``) that predate the parameter
  * aggregators serving another vendor's model (GLM/Kimi on the DashScope
    gateway) where the endpoint's own thinking params don't apply

Whether those endpoints SHOULD support a toggle is the operator's problem, not
ours. Ours is to not lie about it in the UI.
"""

import pytest

from providers import get_provider_profile, resolve_provider_profile

GLM_URL = "https://open.bigmodel.cn/api/paas/v4/"
KIMI_URL = "https://api.moonshot.ai/v1"


class TestToggleable:
    @pytest.mark.parametrize(
        "provider,model",
        [
            ("kimi-coding", "kimi-k2.6"),  # hybrid: on/off both legal
            ("zai", "glm-5-turbo"),
            ("zai", "glm-4.7"),  # forced-thinking, but `disabled` IS accepted
            ("deepseek", "deepseek-reasoner"),
            ("alibaba", "qwen3-max"),
            ("custom", "qwen3"),  # ollama `think: false`
        ],
    )
    def test_reports_true(self, provider, model):
        assert get_provider_profile(provider).can_toggle_reasoning(model) is True

    def test_openrouter_unified_param(self):
        assert get_provider_profile("openrouter").can_toggle_reasoning("z-ai/glm-5") is True

    def test_works_through_host_resolution(self):
        """A vendor reached via a custom endpoint reports its real capability."""
        profile = resolve_provider_profile("custom:open.bigmodel.cn", GLM_URL)
        assert profile.can_toggle_reasoning("glm-5v-turbo") is True


class TestNotToggleable:
    def test_thinking_only_model(self):
        """k2.7-code answers HTTP 400 to an explicit off."""
        assert (
            get_provider_profile("kimi-coding").can_toggle_reasoning("kimi-k2.7-code")
            is False
        )

    @pytest.mark.parametrize("model", ["glm-4-flash", "glm-4v-plus", "glm-4-9b"])
    def test_pre_toggle_glm_generations(self, model):
        assert get_provider_profile("zai").can_toggle_reasoning(model) is False

    @pytest.mark.parametrize("model", ["glm-5", "glm-4.7", "kimi-k2.5", "MiniMax-M2.5"])
    def test_third_party_models_on_the_dashscope_gateway(self, model):
        """The gateway hosts other vendors' models; its own `enable_thinking`
        does not apply to them, so the switch would be dead."""
        assert get_provider_profile("alibaba").can_toggle_reasoning(model) is False

    @pytest.mark.parametrize("model", ["qwq-plus", "qwen3-thinking-plus"])
    def test_always_reasoning_qwen_models(self, model):
        assert get_provider_profile("alibaba").can_toggle_reasoning(model) is False

    def test_deepseek_v3_has_no_thinking_mode(self):
        assert (
            get_provider_profile("deepseek").can_toggle_reasoning("deepseek-chat")
            is False
        )


class TestDerivationIsHonest:
    """The answer is derived from the payload we really send, so it cannot drift.

    In particular it must not be fooled by a profile that emits a field for ON
    while emitting NOTHING for OFF — every vendor here defaults to reasoning on,
    so "nothing" means "still thinking", not "turned off".
    """

    def test_empty_off_payload_is_not_controllable(self):
        from providers.base import ProviderProfile

        class OnlyEnables(ProviderProfile):
            def build_api_kwargs_extras(self, *, reasoning_config=None, **ctx):
                if reasoning_config and reasoning_config.get("enabled") is False:
                    return {}, {}
                return {"thinking": {"type": "enabled"}}, {}

        assert OnlyEnables(name="t").can_toggle_reasoning("m") is False

    def test_a_profile_that_raises_reports_false(self):
        from providers.base import ProviderProfile

        class Broken(ProviderProfile):
            def build_api_kwargs_extras(self, **ctx):
                raise RuntimeError("boom")

        assert Broken(name="t").can_toggle_reasoning("m") is False

    @pytest.mark.parametrize(
        "off_payload",
        [
            {"thinking": {"type": "disabled"}},
            {"enable_thinking": False},
            {"chat_template_kwargs": {"enable_thinking": False}},
            {"reasoning": {"enabled": False}},
            {"think": False},
        ],
    )
    def test_every_vendor_spelling_of_off_is_recognised(self, off_payload):
        from providers.base import ProviderProfile

        class Spelled(ProviderProfile):
            def build_api_kwargs_extras(self, *, reasoning_config=None, **ctx):
                if reasoning_config and reasoning_config.get("enabled") is False:
                    return dict(off_payload), {}
                return {}, {}

        assert Spelled(name="t").can_toggle_reasoning("m") is True
