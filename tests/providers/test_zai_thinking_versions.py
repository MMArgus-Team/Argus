"""Z.AI / GLM thinking-capability detection is a VERSION test, not an allow-list.

Zhipu gates the ``thinking`` parameter on the model generation — "仅 GLM-4.5 及以上
模型支持此参数配置". The profile originally enumerated known releases
(``glm-4.5``/``glm-4.6``/``glm-5``…), which meant every new version was silently
excluded until someone remembered to add it: ``glm-4.7`` shipped and could not
have its thinking turned off, for exactly that reason.

The parametrisation below therefore includes versions that do not exist yet. They
are the point of the test: a correct implementation answers them without edits.
"""

import importlib.util
from pathlib import Path

import pytest

_ZAI_PATH = (
    Path(__file__).resolve().parents[2]
    / "plugins"
    / "model-providers"
    / "zai"
    / "__init__.py"
)


def _load_zai():
    spec = importlib.util.spec_from_file_location("_zai_under_test", _ZAI_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


zai_mod = _load_zai()


@pytest.mark.parametrize(
    "model",
    [
        "glm-4.5",
        "glm-4-5",  # dash separator used by some aggregators
        "glm-4.5v",
        "glm-4.6",
        "glm-4.7",  # ★ the version the old allow-list missed
        "glm-5",
        "glm-5-turbo",
        "glm-5v-turbo",
        "glm-5.2",
        # Future generations must work with no code change — that is the whole
        # reason this is a version comparison.
        "glm-6",
        "glm-7.3",
        "glm-10.1",
        "zai/glm-5",  # vendor-prefixed form from aggregators
    ],
)
def test_thinking_capable(model):
    assert zai_mod._model_supports_thinking(model) is True


@pytest.mark.parametrize(
    "model",
    [
        # Pre-4.5 families predate the `thinking` parameter — leave their wire
        # format untouched rather than sending a field they don't know.
        "glm-4",
        "glm-4-flash",
        "glm-4-9b",
        "glm-4-air",
        "glm-4-plus",
        "glm-4v-plus",
        "glm-3-turbo",
        # Not GLM at all.
        "gpt-5.5",
        "kimi-k2.6",
        "",
        None,
    ],
)
def test_not_thinking_capable(model):
    assert zai_mod._model_supports_thinking(model) is False


class TestWireShape:
    def _extras(self, model, *, off):
        extra_body, top_level = zai_mod.zai.build_api_kwargs_extras(
            reasoning_config={"enabled": not off}, model=model
        )
        assert top_level == {}
        return extra_body

    def test_off(self):
        assert self._extras("glm-4.7", off=True) == {"thinking": {"type": "disabled"}}

    def test_on(self):
        assert self._extras("glm-4.7", off=False) == {"thinking": {"type": "enabled"}}

    def test_no_reasoning_config_defaults_to_enabled(self):
        extra_body, _ = zai_mod.zai.build_api_kwargs_extras(model="glm-5")
        assert extra_body == {"thinking": {"type": "enabled"}}

    def test_pre_thinking_model_gets_nothing(self):
        assert self._extras("glm-4-flash", off=True) == {}


class TestReasoningEffort:
    """Depth (``reasoning_effort``) — narrower support than the on/off toggle.

    Zhipu: "仅 GLM-5.2 支持", "thinking 开启时生效", and absent from the vision
    request schema. Before this was wired, all five "on" tiers produced a byte
    -identical request — the dial was decoration.
    """

    def _extras(self, model, effort):
        config = {"enabled": True, "effort": effort} if effort else {"enabled": True}
        extra_body, top_level = zai_mod.zai.build_api_kwargs_extras(
            reasoning_config=config, model=model
        )
        assert top_level == {}, "GLM carries effort in extra_body, not top-level"
        return extra_body

    @pytest.mark.parametrize(
        "model,supported",
        [
            ("glm-5.2", True),
            ("glm-5-2", True),  # dash separator
            ("glm-5.2-preview", True),  # aggregator suffix
            ("glm-6", True),  # future generation
            # Narrower than `thinking`: these take the toggle but not the depth.
            ("glm-5", False),
            ("glm-5-turbo", False),
            ("glm-4.7", False),
            # Vision schema has `thinking` but NOT `reasoning_effort`.
            ("glm-5.2v", False),
            ("glm-5v-turbo", False),
        ],
    )
    def test_effort_only_where_supported(self, model, supported):
        has_effort = "reasoning_effort" in self._extras(model, "high")
        assert has_effort is supported
        # The on/off toggle must survive either way.
        assert self._extras(model, "high")["thinking"] == {"type": "enabled"}

    @pytest.mark.parametrize(
        "tier,expected",
        [
            # Zhipu remaps low/medium up to high, and xhigh to max, on its side.
            # We send the remapped value so the request states what it means.
            ("low", "high"),
            ("medium", "high"),
            ("high", "high"),
            ("xhigh", "max"),
            # ★ `minimal` is passed through, NOT promoted: it is Hermes' lowest
            #   "on" tier and Zhipu treats it as skip-reasoning. Promoting it
            #   would make Min and High byte-identical.
            ("minimal", "minimal"),
        ],
    )
    def test_tier_mapping(self, tier, expected):
        assert self._extras("glm-5.2", tier)["reasoning_effort"] == expected

    def test_minimal_is_distinguishable_from_high(self):
        """The regression guard: the dial must actually move the request."""
        assert self._extras("glm-5.2", "minimal") != self._extras("glm-5.2", "high")

    def test_no_effort_requested_sends_no_depth(self):
        """Absent a tier, let the vendor default (``max``) stand."""
        assert "reasoning_effort" not in self._extras("glm-5.2", "")

    def test_unknown_tier_is_ignored_rather_than_guessed(self):
        assert "reasoning_effort" not in self._extras("glm-5.2", "turbo")

    def test_thinking_off_sends_no_depth(self):
        """A depth is meaningless when thinking is off ("thinking 开启时生效")."""
        extra_body, _ = zai_mod.zai.build_api_kwargs_extras(
            reasoning_config={"enabled": False, "effort": "high"}, model="glm-5.2"
        )
        assert extra_body == {"thinking": {"type": "disabled"}}
