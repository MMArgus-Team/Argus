"""Override match for a hand-configured custom provider is case-insensitive.

Regression: config entries are named in whatever case the user typed
("Open.bigmodel.cn"), but provider slugs are lowercased downstream
("custom:open.bigmodel.cn"). The override that exists to accept
explicitly-configured models compared them verbatim, so it never fired for the
exact configs it protects — leaving the switch rejected.

★ Isolation: ``switch_model`` resolves credentials through
``resolve_runtime_provider``, which reads the on-disk config rather than the
``custom_providers`` argument passed here. Under the suite's autouse
``_hermetic_environment`` fixture ``ARGUS_HOME`` points at an empty tempdir, so
that lookup finds no provider and the switch fails with "Could not resolve
credentials" — before ever reaching the slug-matching logic under test. Stub the
credential layer (same pattern as
test_model_switch_custom_providers.py::test_switch_model_accepts_explicit_named_custom_provider)
so this test exercises slug matching and nothing else. Without the stub it also
passed/failed depending on the developer's real ~/.argus/config.yaml.
"""

import pytest

from hermes_cli.model_switch import switch_model

CUSTOM = [
    {
        "name": "Open.bigmodel.cn",          # capitalised, as a user would write it
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "model": "glm-5v-turbo",
        "api_key": "k",
    }
]

# ★ REJECTED on purpose. The override under test only runs when validation says
# no — it exists precisely because an endpoint's /v1/models often omits a model
# the endpoint serves (open.bigmodel.cn omits glm-5v-turbo). Stubbing this
# `accepted: True` would skip the override branch entirely and the test would
# pass even with the slug-matching bug reintroduced (verified: it did).
_MOCK_REJECTION = {
    "accepted": False,
    "persist": False,
    "recognized": False,
    "message": "not in /v1/models",
}


@pytest.fixture(autouse=True)
def _stub_credential_resolution(monkeypatch):
    """Pin the credential/validation layer so only slug matching is under test."""
    # ★ The resolved base_url deliberately DIFFERS from the config entry's.
    #   The override accepts a model if EITHER the slug or the base_url matches;
    #   returning the same URL here would satisfy the base_url branch and the
    #   test would pass even with the slug comparison broken (verified: it did).
    #   A differing URL is also realistic — a gateway can resolve credentials to
    #   a rotated/normalised endpoint. This pins the SLUG path specifically.
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: {
            "api_key": "k",
            "base_url": "https://resolved-elsewhere.example/v1",
            "api_mode": "chat_completions",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.models.validate_requested_model", lambda *a, **k: _MOCK_REJECTION
    )
    monkeypatch.setattr("hermes_cli.model_switch.get_model_info", lambda *a, **k: None)
    monkeypatch.setattr(
        "hermes_cli.model_switch.get_model_capabilities", lambda *a, **k: None
    )


def _switch(explicit_provider):
    return switch_model(
        raw_input="glm-5v-turbo",
        current_provider="moa",
        current_model="default",
        current_base_url="moa://local",
        current_api_key="moa-virtual-provider",
        is_global=False,
        explicit_provider=explicit_provider,
        user_providers=None,
        custom_providers=CUSTOM,
    )


def test_lowercase_slug_matches_capitalised_config_entry():
    result = _switch("custom:open.bigmodel.cn")
    assert result.success, result.error_message
    assert result.new_model == "glm-5v-turbo"


def test_exact_case_slug_still_matches():
    result = _switch("custom:Open.bigmodel.cn")
    assert result.success, result.error_message
