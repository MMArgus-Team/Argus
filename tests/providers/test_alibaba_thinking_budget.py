"""DashScope depth control: the effort tier must reach ``thinking_budget``.

Regression: ``AlibabaProfile`` set ``enable_thinking`` from ``reasoning_config``
but derived NO depth. It read a numeric budget from
``reasoning_config["thinking_budget"]`` / ``["budget_tokens"]`` — keys that
nothing in Hermes ever writes (``parse_reasoning_effort`` emits only ``enabled``
and ``effort``). So that branch was unreachable and all five "on" tiers produced a
byte-identical request: the thinking dial did nothing on DashScope.

DashScope takes a TOKEN COUNT, not a named level ("Use thinking_budget to cap
reasoning tokens... When the limit is reached, the model stops reasoning and
responds immediately"), supported by "Qwen3 (in thinking mode) and Kimi models".
"""

import pytest

from providers import get_provider_profile

alibaba = get_provider_profile("alibaba")

HYBRID = "qwen3-max"


def _extras(model=HYBRID, *, enabled=True, effort=None, **extra):
    config = {"enabled": enabled}
    if effort is not None:
        config["effort"] = effort
    config.update(extra)
    extra_body, top_level = alibaba.build_api_kwargs_extras(
        reasoning_config=config, model=model
    )
    assert top_level == {}, "DashScope carries these in extra_body"
    return extra_body


class TestDepthReachesTheWire:
    @pytest.mark.parametrize(
        "tier,expected",
        [
            ("minimal", 1024),
            ("low", 4000),
            ("medium", 8000),
            ("high", 16000),
            ("xhigh", 32000),
        ],
    )
    def test_tier_maps_to_budget(self, tier, expected):
        assert _extras(effort=tier) == {
            "enable_thinking": True,
            "thinking_budget": expected,
        }

    def test_every_tier_is_distinguishable(self):
        """The regression guard: the dial must actually move the request."""
        shapes = {
            str(_extras(effort=t))
            for t in ("minimal", "low", "medium", "high", "xhigh")
        }
        assert len(shapes) == 5

    def test_budget_ladder_matches_the_anthropic_one(self):
        """"High" should mean a comparable amount of reasoning across vendors."""
        from agent.anthropic_adapter import THINKING_BUDGET

        for tier in ("low", "medium", "high", "xhigh"):
            assert _extras(effort=tier)["thinking_budget"] == THINKING_BUDGET[tier]


class TestPrecedenceAndDefaults:
    def test_explicit_numeric_budget_still_wins(self):
        assert _extras(effort="low", thinking_budget=777)["thinking_budget"] == 777

    def test_budget_tokens_alias_still_wins(self):
        assert _extras(effort="low", budget_tokens=555)["thinking_budget"] == 555

    def test_no_tier_sends_no_cap(self):
        """DashScope's own default is the model's max chain-of-thought length, so
        omitting the key is the right "user expressed no preference" behaviour."""
        assert _extras() == {"enable_thinking": True}

    def test_unknown_tier_sends_no_cap(self):
        assert _extras(effort="turbo") == {"enable_thinking": True}

    def test_non_numeric_budget_is_ignored_not_crashed(self):
        assert _extras(effort="low", thinking_budget="lots") == {
            "enable_thinking": True
        }


class TestOffAndUnsupported:
    def test_thinking_off_sends_no_budget(self):
        """A cap is meaningless with no reasoning to cap."""
        assert _extras(enabled=False, effort="high") == {"enable_thinking": False}

    @pytest.mark.parametrize("model", ["qwq-plus", "qwen3-thinking-plus"])
    def test_thinking_only_models_are_left_untouched(self, model):
        """They always reason; the toggle is a no-op or rejected there."""
        assert _extras(model, effort="high") == {}

    @pytest.mark.parametrize("model", ["qwen-vl-max", "qwen2.5-7b", "qwen-max-2024-09-19"])
    def test_pre_hybrid_models_are_left_untouched(self, model):
        assert _extras(model, effort="high") == {}
