"""A ``custom:<name>`` provider must resolve credentials from the config it was
handed, not only from config on disk.

``switch_model()`` rewrites ``target_provider`` to the config's own slug in
several places (configured-provider routing, differently-cased provider names).
That makes ``provider_changed`` true, so the credential block re-resolves — and
``resolve_runtime_provider()`` resolves by NAME against config on DISK.  It
therefore could not see a ``custom_providers`` entry passed in as an argument
and raised ``Unknown provider``, so the switch failed with "Could not resolve
credentials" before validation ever ran.

The ``providers:`` branch already resolved this by feeding the entry's endpoint
in as an explicit hint; ``custom_providers`` entries now do the same.
"""

from unittest.mock import patch

import pytest

from hermes_cli import model_switch as ms

BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

LISTED = {
    "models": ["glm-5v-turbo", "glm-4.5"],
    "probed_url": f"{BASE_URL}/models",
    "resolved_base_url": BASE_URL,
    "suggested_base_url": None,
    "used_fallback": False,
}

# Mixed-case display name, as a hand-written config.yaml really contains.
CUSTOM_PROVIDERS = [{
    "name": "Open.bigmodel.cn",
    "base_url": BASE_URL,
    "api_key": "sk-from-config",
    "model": "glm-5v-turbo",
}]


@pytest.fixture
def probe():
    with patch("hermes_cli.models.probe_api_models", return_value=LISTED):
        yield


def switch(provider, *, custom_providers=CUSTOM_PROVIDERS, api_key="sk-from-config"):
    return ms.switch_model(
        "glm-5v-turbo",
        current_provider=provider,
        current_model="glm-5v-turbo",
        current_base_url=BASE_URL,
        current_api_key=api_key,
        custom_providers=custom_providers,
    )


class TestCustomProviderCredentialResolution:
    @pytest.mark.parametrize("provider", [
        "custom:open.bigmodel.cn",   # lowercased slug (the failing case)
        "custom:Open.bigmodel.cn",   # config's own spelling
        "custom",                    # bare custom → self-heals to the entry
    ])
    def test_switch_succeeds_for_every_slug_spelling(self, probe, provider):
        result = switch(provider)
        assert result.success is True, result.error_message
        assert result.new_model == "glm-5v-turbo"

    def test_endpoint_comes_from_the_config_entry(self, probe):
        result = switch("custom:open.bigmodel.cn")
        assert result.base_url.rstrip("/") == BASE_URL

    def test_api_key_comes_from_the_config_entry(self, probe):
        """The entry's own key is used even when no key was passed in."""
        result = switch("custom:open.bigmodel.cn", api_key="")
        assert result.api_key == "sk-from-config"

    def test_key_env_indirection_is_honoured(self, probe, monkeypatch):
        monkeypatch.setenv("MY_GLM_KEY", "sk-from-env")
        entry = {"name": "Open.bigmodel.cn", "base_url": BASE_URL,
                 "key_env": "MY_GLM_KEY", "model": "glm-5v-turbo"}
        result = switch("custom:open.bigmodel.cn", custom_providers=[entry], api_key="")
        assert result.api_key == "sk-from-env"

    def test_dollar_brace_key_reference_is_expanded(self, probe, monkeypatch):
        monkeypatch.setenv("MY_GLM_KEY", "sk-expanded")
        entry = {"name": "Open.bigmodel.cn", "base_url": BASE_URL,
                 "api_key": "${MY_GLM_KEY}", "model": "glm-5v-turbo"}
        result = switch("custom:open.bigmodel.cn", custom_providers=[entry], api_key="")
        assert result.api_key == "sk-expanded"

    def test_slug_not_in_config_routes_to_the_provider_that_declares_the_model(self, probe):
        """The new branch only claims slugs config actually backs.

        A slug with no matching entry falls through to configured-provider
        routing, which sends the model to the entry that declares it — the same
        behaviour as before this branch existed (verified against a pristine
        checkout).  Unknown custom slugs are deliberately soft-accepted here
        rather than rejected, so this asserts the routing, not an error.
        """
        result = switch("custom:nowhere.example", custom_providers=CUSTOM_PROVIDERS)
        assert result.success is True
        assert result.target_provider == "custom:Open.bigmodel.cn"


class TestCustomProviderEntryLookup:
    def test_matches_display_name_case_insensitively(self):
        entry = ms._custom_provider_entry("custom:open.bigmodel.cn", CUSTOM_PROVIDERS)
        assert entry is not None and entry["name"] == "Open.bigmodel.cn"

    def test_matches_without_the_custom_prefix(self):
        assert ms._custom_provider_entry("Open.bigmodel.cn", CUSTOM_PROVIDERS) is not None

    def test_none_for_unknown_or_malformed_input(self):
        assert ms._custom_provider_entry("custom:nope", CUSTOM_PROVIDERS) is None
        assert ms._custom_provider_entry("custom:x", None) is None
        assert ms._custom_provider_entry("", CUSTOM_PROVIDERS) is None
        assert ms._custom_provider_entry("custom:x", [{"no_name": 1}, "junk"]) is None


class TestConfigEntryApiKey:
    def test_inline_key(self):
        assert ms._config_entry_api_key({"api_key": " k "}) == "k"

    def test_env_reference_forms(self, monkeypatch):
        monkeypatch.setenv("V", "sk-v")
        assert ms._config_entry_api_key({"api_key": "${V}"}) == "sk-v"
        assert ms._config_entry_api_key({"key_env": "V"}) == "sk-v"

    def test_empty_when_absent_or_unset(self, monkeypatch):
        monkeypatch.delenv("MISSING_KEY", raising=False)
        assert ms._config_entry_api_key({}) == ""
        assert ms._config_entry_api_key(None) == ""
        assert ms._config_entry_api_key({"key_env": "MISSING_KEY"}) == ""
