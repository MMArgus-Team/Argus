"""Turning thinking OFF must actually reach Zhipu/GLM.

Regression: ``_apply_mm_deep_thinking`` hand-rolled an ``if is_kimi / elif
is_qwenish / else`` chain, and GLM matched neither branch — so it fell into the
catch-all that emitted ``{}``. Because Zhipu documents ``thinking`` as defaulting
to ``enabled``, sending nothing means GLM keeps reasoning: the UI showed "Off"
while the model still emitted a chain of thought.

The correct handling already existed in the ``zai`` provider profile (whose own
docstring even names this defect), but profiles were resolved by CONFIGURED NAME
only, so ``custom:open.bigmodel.cn`` got the generic ``CustomProfile`` — whose
disable-thinking emits Ollama's ``think: false``. This function now asks
``providers.resolve_provider_profile``, which also resolves by hostname; see
``tests/providers/test_profile_resolution_by_host.py``.

★ Do NOT add a Kimi-style "thinking-only" carve-out for GLM-4.7 / GLM-4.5V. The
docs label those 强制思考, but that describes whether the model reasons *once
thinking is on*; it is not a claim that they reject ``disabled``, and there is no
evidence they 400 on it (unlike Moonshot's k2.7-code, which genuinely does).
Suppressing the flag for them would silently re-break the off switch.
"""

import types

import pytest

from tui_gateway.server import _apply_mm_deep_thinking

GLM_URL = "https://open.bigmodel.cn/api/paas/v4/"


def _agent(*, model, base_url, provider="", reasoning_config=None):
    return types.SimpleNamespace(
        base_url=base_url,
        model=model,
        provider=provider,
        reasoning_config=reasoning_config,
    )


@pytest.mark.parametrize(
    "model,base_url,provider",
    [
        # The reported case: GLM configured as a hand-written custom endpoint.
        ("glm-5v-turbo", GLM_URL, "custom:open.bigmodel.cn"),
        ("glm-5-turbo", GLM_URL, "custom"),
        # An explicit provider name needs no recognisable URL.
        ("glm-4.6", "", "zhipu"),
        # ★ The 强制思考 models get the flag too — see the module docstring.
        ("glm-4.7", GLM_URL, ""),
        ("glm-4.5v", GLM_URL, ""),
    ],
)
def test_thinking_off_reaches_glm(model, base_url, provider):
    agent = _agent(
        model=model,
        base_url=base_url,
        provider=provider,
        reasoning_config={"enabled": False},
    )

    _apply_mm_deep_thinking(agent, {})

    assert agent._extra_body_additions == {"thinking": {"type": "disabled"}}


def test_unidentifiable_endpoint_is_left_untouched():
    """A GLM-looking model behind an unbranded gateway is NOT assumed to be GLM.

    Sending a vendor-private field to an endpoint we cannot identify risks a 400
    that kills the whole request before the model runs (the observed failure of
    this class was ``Unknown parameter: 'top_k'``). Identifying by model-name
    substring is a guess; the user can state the provider explicitly.

    This documents a deliberate limit, not an oversight — an earlier draft DID
    guess from the name, and that is the same "assume, don't verify" instinct
    that produced the original bug.
    """
    agent = _agent(
        model="glm-4.6",
        base_url="https://gateway.internal.example/v1",
        provider="",
        reasoning_config={"enabled": False},
    )

    _apply_mm_deep_thinking(agent, {})

    assert agent._extra_body_additions == {}


def test_known_provider_with_unknown_model_alias_is_left_untouched():
    """provider=zhipu but an aliased model name we cannot version-check: don't
    guess a thinking contract for it."""
    agent = _agent(
        model="internal-alias-v2",
        base_url="",
        provider="zhipu",
        reasoning_config={"enabled": False},
    )

    _apply_mm_deep_thinking(agent, {})

    assert agent._extra_body_additions == {}


def test_thinking_on_is_explicit_for_glm():
    agent = _agent(
        model="glm-5v-turbo",
        base_url=GLM_URL,
        reasoning_config={"enabled": True, "effort": "high"},
    )

    _apply_mm_deep_thinking(agent, {})

    assert agent._extra_body_additions == {"thinking": {"type": "enabled"}}


def test_glm_never_gets_the_qwen_shape():
    """`enable_thinking` is not part of Zhipu's API — sending it does nothing."""
    agent = _agent(
        model="glm-5v-turbo", base_url=GLM_URL, reasoning_config={"enabled": False}
    )

    _apply_mm_deep_thinking(agent, {})

    assert "enable_thinking" not in agent._extra_body_additions
    assert "chat_template_kwargs" not in agent._extra_body_additions


def test_non_glm_openai_proxy_still_gets_no_thinking_keys():
    """The catch-all must stay intact for strict GPT-compatible gateways, which
    reject both private thinking dialects."""
    agent = _agent(
        model="gpt-5.5",
        base_url="https://api.openai.com/v1",
        reasoning_config={"enabled": False},
    )

    _apply_mm_deep_thinking(agent, {})

    assert agent._extra_body_additions == {}
