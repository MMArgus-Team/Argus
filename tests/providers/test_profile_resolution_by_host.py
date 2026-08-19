"""A vendor reached via a custom endpoint must still get its own profile.

WHY THIS EXISTS
===============
``ProviderProfile`` is where every per-vendor request quirk lives (thinking
toggle, temperature limits, reasoning placement). Those quirks are properties of
the ENDPOINT, but profiles were looked up by the CONFIGURED NAME alone. So a
vendor reached as a hand-written custom endpoint got the generic ``custom``
profile and none of its quirks — even though the repo already had the hostname
→ provider mapping needed to know better (``model_metadata._URL_TO_PROVIDER``,
which has carried ``open.bigmodel.cn: zai`` all along, used only for
context-length lookups).

That is how "turn thinking off" silently did nothing for GLM:
``custom:open.bigmodel.cn`` → ``CustomProfile``, whose disable-thinking emits
Ollama's ``think: false``, while the correct ``thinking: {"type": "disabled"}``
sat in the never-consulted ``zai`` profile.

Pointing ``custom`` at a vendor URL (or at a proxy in front of one) is a mainstream
setup, not an edge case, so this is a base-layer contract and gets base-layer tests.
"""

import pytest

from providers import get_provider_profile, resolve_provider_profile

GLM_URL = "https://open.bigmodel.cn/api/paas/v4/"
KIMI_URL = "https://api.moonshot.ai/v1"
OLLAMA_URL = "http://localhost:11434/v1"


def _name(profile):
    return type(profile).__name__ if profile is not None else None


class TestHostFallback:
    @pytest.mark.parametrize(
        "provider",
        ["custom:open.bigmodel.cn", "custom", "", "local"],
    )
    def test_glm_via_custom_endpoint_resolves_to_zai(self, provider):
        assert _name(resolve_provider_profile(provider, GLM_URL)) == "ZaiProfile"

    def test_other_vendors_behind_custom_resolve_too(self):
        assert _name(resolve_provider_profile("custom", KIMI_URL)) == "KimiProfile"
        assert (
            _name(resolve_provider_profile("custom", "https://api.deepseek.com/v1"))
            == "DeepSeekProfile"
        )

    def test_a_url_specific_vendor_beats_a_generic_name(self):
        """provider=openai pointed at Moonshot is Moonshot."""
        assert _name(resolve_provider_profile("openai", KIMI_URL)) == "KimiProfile"


class TestExplicitChoiceWins:
    def test_named_provider_is_never_overridden_by_url(self):
        """An explicitly chosen non-generic provider must survive, even when the
        URL points somewhere else (proxies legitimately do this)."""
        assert (
            _name(resolve_provider_profile("deepseek", GLM_URL)) == "DeepSeekProfile"
        )

    def test_named_provider_without_url_still_works(self):
        assert _name(resolve_provider_profile("zai", "")) == "ZaiProfile"
        assert _name(resolve_provider_profile("moonshot", "")) == "KimiProfile"


class TestGenericStaysGeneric:
    def test_local_ollama_keeps_the_custom_profile(self):
        """A genuinely custom/local endpoint must NOT be mapped to a vendor."""
        assert _name(resolve_provider_profile("custom", OLLAMA_URL)) == "CustomProfile"

    def test_unknown_host_keeps_the_custom_profile(self):
        assert (
            _name(resolve_provider_profile("custom", "https://mystery.example/v1"))
            == "CustomProfile"
        )

    def test_unknown_name_and_unknown_host_is_none(self):
        assert resolve_provider_profile("no-such-vendor", "") is None

    def test_matches_plain_lookup_when_no_url_is_given(self):
        for name in ("zai", "deepseek", "custom", "no-such-vendor"):
            assert _name(resolve_provider_profile(name, "")) == _name(
                get_provider_profile(name)
            )


class TestThinkingTogglePayload:
    """End-to-end: the resolved profile produces the right wire shape."""

    def _extras(self, provider, base_url, model, *, off):
        profile = resolve_provider_profile(provider, base_url)
        assert profile is not None
        extra_body, _top = profile.build_api_kwargs_extras(
            reasoning_config={"enabled": not off},
            model=model,
            base_url=base_url,
        )
        return extra_body

    def test_glm_off_via_custom_endpoint(self):
        assert self._extras(
            "custom:open.bigmodel.cn", GLM_URL, "glm-5-turbo", off=True
        ) == {"thinking": {"type": "disabled"}}

    def test_glm_on_via_custom_endpoint(self):
        assert self._extras(
            "custom:open.bigmodel.cn", GLM_URL, "glm-5-turbo", off=False
        ) == {"thinking": {"type": "enabled"}}

    def test_glm_without_thinking_mode_is_left_alone(self):
        """glm-4-flash / glm-4v have no thinking mode — don't perturb their wire
        format. (The zai profile already knew this; a hand-rolled `"glm" in name`
        rule would have got it wrong.)"""
        assert (
            self._extras("custom", GLM_URL, "glm-4-flash", off=True) == {}
        )

    def test_ollama_still_uses_its_own_spelling(self):
        assert self._extras("custom", OLLAMA_URL, "qwen3", off=True) == {"think": False}
