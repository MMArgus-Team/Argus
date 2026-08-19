"""Monitor-only auth compatibility for MaaS OpenAI-compatible endpoints."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agent.multimodal._config import Config
from agent.multimodal.hermes_glue import HermesClientFactory, build_config


def _fake_openai_client() -> SimpleNamespace:
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock()),
        ),
    )


def _monitor_config(*, send_api_key_header: object = False) -> SimpleNamespace:
    return SimpleNamespace(
        monitor_provider="custom",
        monitor_base_url="https://maas.example.test/v1",
        monitor_api_key="monitor-secret",
        monitor_model="qwen3-vl",
        monitor_send_api_key_header=send_api_key_header,
    )


def test_monitor_api_key_header_defaults_off() -> None:
    assert Config().monitor_send_api_key_header is False


def test_nested_monitor_api_key_header_flattens_and_coerces(tmp_path) -> None:
    hermes_cfg = {
        "model": {
            "monitor": {
                "send_api_key_header": "true",
            },
        },
    }

    with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
        cfg = build_config(hermes_cfg, runtime_id="monitor-header-test")

    assert cfg.monitor_send_api_key_header is True


def test_monitor_opt_in_reuses_api_key_in_custom_header(caplog) -> None:
    fake_client = _fake_openai_client()
    caplog.set_level(logging.INFO, logger="hermes.multimodal.glue")

    with (
        patch("openai.AsyncOpenAI", return_value=fake_client) as async_openai,
        patch(
            "agent.multimodal.hermes_glue._submodule_http_client",
            return_value=None,
        ),
    ):
        client, model = HermesClientFactory(
            _monitor_config(send_api_key_header=True)
        ).monitor_client()

    assert client is fake_client
    assert model == "qwen3-vl"
    async_openai.assert_called_once_with(
        base_url="https://maas.example.test/v1",
        api_key="monitor-secret",
        default_headers={"api-key": "monitor-secret"},
    )
    assert "monitor-secret" not in caplog.text


def test_monitor_default_keeps_original_client_kwargs() -> None:
    fake_client = _fake_openai_client()

    with (
        patch("openai.AsyncOpenAI", return_value=fake_client) as async_openai,
        patch(
            "agent.multimodal.hermes_glue._submodule_http_client",
            return_value=None,
        ),
    ):
        HermesClientFactory(_monitor_config()).monitor_client()

    async_openai.assert_called_once_with(
        base_url="https://maas.example.test/v1",
        api_key="monitor-secret",
    )


def test_monitor_header_flag_does_not_mutate_shared_main_client() -> None:
    """Without a dedicated base URL the original shared-client path wins."""
    from agent.multimodal.hermes_glue import build_submodule_client

    shared_client = _fake_openai_client()
    with patch("openai.AsyncOpenAI") as async_openai:
        client, model = build_submodule_client(
            provider="custom",
            base_url="",
            api_key="monitor-secret",
            model="qwen3-vl",
            resolve_main=lambda: (shared_client, "main-model"),
            label="monitor",
            send_api_key_header=True,
        )

    assert client is shared_client
    assert model == "main-model"
    async_openai.assert_not_called()


def test_non_monitor_label_cannot_enable_api_key_header() -> None:
    """The low-level shared factory must not widen this flag to Watcher."""
    from agent.multimodal.hermes_glue import build_submodule_client

    fake_client = _fake_openai_client()
    with (
        patch("openai.AsyncOpenAI", return_value=fake_client) as async_openai,
        patch(
            "agent.multimodal.hermes_glue._submodule_http_client",
            return_value=None,
        ),
    ):
        build_submodule_client(
            provider="custom",
            base_url="https://watcher.example.test/v1",
            api_key="watcher-secret",
            model="qwen3-vl",
            resolve_main=lambda: (fake_client, "main-model"),
            label="watcher",
            send_api_key_header=True,
        )

    async_openai.assert_called_once_with(
        base_url="https://watcher.example.test/v1",
        api_key="watcher-secret",
    )
