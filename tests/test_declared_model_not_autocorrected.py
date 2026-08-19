"""A model declared in the user's own provider config must survive the switch verbatim.

``validate_requested_model()`` auto-corrects a requested model to the closest
entry in the endpoint's ``/v1/models`` listing at a 0.9 difflib ratio.  That is
right for typos, but model families that differ by one character sail past the
threshold: ``glm-5v-turbo`` → ``glm-5-turbo`` scores 0.957.  So selecting the
configured *vision* model silently switched it to the non-vision one whenever
the endpoint's listing omitted it — a wrong model, no error, no warning.

``switch_model()`` already had an override for config-declared models, but it
was gated on ``not validation["accepted"]`` while auto-correction returns
``accepted: True``, so it never ran on this path.
"""

from unittest.mock import patch

import pytest

from hermes_cli import model_switch as ms

BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

# The endpoint lists glm-5-turbo but NOT the configured glm-5v-turbo.
LISTED = {
    "models": ["glm-5-turbo", "glm-4.5", "glm-5"],
    "probed_url": f"{BASE_URL}/models",
    "resolved_base_url": BASE_URL,
    "suggested_base_url": None,
    "used_fallback": False,
}

# Mixed-case provider name, as a hand-written config.yaml really contains.
CUSTOM_PROVIDERS = [{
    "name": "Open.bigmodel.cn",
    "base_url": BASE_URL,
    "api_key": "k",
    "model": "glm-5v-turbo",
}]


@pytest.fixture
def probe():
    with patch("hermes_cli.models.probe_api_models", return_value=LISTED):
        yield


def switch(model, *, custom_providers=CUSTOM_PROVIDERS, user_providers=None,
           provider="custom:Open.bigmodel.cn"):
    return ms.switch_model(
        model,
        current_provider=provider,
        current_model="glm-5v-turbo",
        current_base_url=BASE_URL,
        current_api_key="k",
        user_providers=user_providers,
        custom_providers=custom_providers,
    )


class TestDeclaredModelNotAutocorrected:
    def test_declared_model_survives_when_endpoint_omits_it(self, probe):
        """The regression: config says glm-5v-turbo, so glm-5v-turbo is used."""
        result = switch("glm-5v-turbo")
        assert result.success is True
        assert result.new_model == "glm-5v-turbo"

    def test_provider_name_case_does_not_defeat_the_override(self, probe):
        """Slugs get lowercased downstream; the config spelling must still match."""
        result = switch("glm-5v-turbo", provider="custom:open.bigmodel.cn")
        assert result.success is True
        assert result.new_model == "glm-5v-turbo"

    def test_bare_custom_provider_still_reaches_its_config_entry(self, probe):
        result = switch("glm-5v-turbo", provider="custom")
        assert result.success is True
        assert result.new_model == "glm-5v-turbo"

    def test_trailing_slash_does_not_defeat_the_override(self, probe):
        entry = dict(CUSTOM_PROVIDERS[0], base_url=BASE_URL + "/")
        result = switch("glm-5v-turbo", custom_providers=[entry])
        assert result.new_model == "glm-5v-turbo"

    def test_declared_via_models_mapping(self, probe):
        """``models:`` mapping is honoured, not just the singular ``model:``."""
        entry = {"name": "Open.bigmodel.cn", "base_url": BASE_URL, "api_key": "k",
                 "models": {"glm-5v-turbo": {"context_length": 200000}}}
        result = switch("glm-5v-turbo", custom_providers=[entry])
        assert result.new_model == "glm-5v-turbo"

    def test_declared_under_user_providers(self, probe):
        """``providers.<slug>`` entries get the same protection."""
        user = {"custom:Open.bigmodel.cn": {"base_url": BASE_URL, "api_key": "k",
                                            "models": ["glm-5v-turbo"]}}
        result = switch("glm-5v-turbo", custom_providers=None, user_providers=user)
        assert result.new_model == "glm-5v-turbo"

    # -- auto-correction must still work for everything NOT declared ----------

    def test_undeclared_typo_is_still_autocorrected(self, probe):
        """The feature is preserved: a real typo still snaps to the listed model."""
        result = switch("glm-5-turb")
        assert result.success is True
        assert result.new_model == "glm-5-turbo"

    def test_autocorrect_applies_when_no_config_is_passed(self, probe):
        result = switch("glm-5v-turbo", custom_providers=None)
        assert result.new_model == "glm-5-turbo"

    def test_exactly_listed_model_is_untouched(self, probe):
        result = switch("glm-4.5")
        assert result.success is True
        assert result.new_model == "glm-4.5"


class TestDeclaredModelForTarget:
    """Unit coverage for the shared lookup both decisions now use."""

    def test_matches_provider_slug_case_insensitively(self):
        """Slugs get lowercased downstream; the config spelling must still match."""
        assert ms._declared_model_for_target(
            "glm-5v-turbo", "custom:open.bigmodel.cn", BASE_URL, None, CUSTOM_PROVIDERS,
        ) == "glm-5v-turbo"

    def test_bare_custom_provider_matches_by_base_url(self):
        """A bare ``custom`` target identifies its config entry only by URL."""
        assert ms._declared_model_for_target(
            "glm-5v-turbo", "custom", BASE_URL, None, CUSTOM_PROVIDERS,
        ) == "glm-5v-turbo"

    def test_returns_canonical_config_spelling(self):
        entry = dict(CUSTOM_PROVIDERS[0], model="GLM-5V-Turbo")
        assert ms._declared_model_for_target(
            "glm-5v-turbo", "custom:Open.bigmodel.cn", BASE_URL, None, [entry],
        ) == "GLM-5V-Turbo"

    def test_none_when_model_not_declared(self):
        assert ms._declared_model_for_target(
            "glm-5-turbo", "custom:Open.bigmodel.cn", BASE_URL, None, CUSTOM_PROVIDERS,
        ) is None

    def test_none_when_another_provider_declares_it(self):
        """Declaration by a *different* endpoint must not protect this switch."""
        other = [{"name": "elsewhere", "base_url": "https://other.example/v1",
                  "model": "glm-5v-turbo"}]
        assert ms._declared_model_for_target(
            "glm-5v-turbo", "custom:Open.bigmodel.cn", BASE_URL, None, other,
        ) is None
