from __future__ import annotations


def _coerce_timeout(raw: object) -> float | None:
    try:
        timeout = float(raw)
    except (TypeError, ValueError):
        return None
    if timeout <= 0:
        return None
    return timeout


def _get_global_llm_timeout(config: object) -> float | None:
    """Return the universal ``llm_timeout_seconds`` fallback, if configured.

    This is the single knob shared by the main agent AND every submodule LLM
    (recall/monitor/watcher/memory).  It sits *below* the per-provider and
    per-model overrides in :func:`get_provider_request_timeout` so a specific
    provider/model can still override it, but *above* returning ``None`` so
    that setting one value covers every LLM at once.
    """
    if not isinstance(config, dict):
        return None
    return _coerce_timeout(config.get("llm_timeout_seconds"))


def get_provider_request_timeout(
    provider_id: str, model: str | None = None
) -> float | None:
    """Return a configured provider request timeout in seconds, if any.

    Priority:
      1. ``providers.<id>.models.<model>.timeout_seconds`` (per-model override)
      2. ``providers.<id>.request_timeout_seconds`` (provider-wide)
      3. ``llm_timeout_seconds`` (universal fallback shared by every LLM)
    """
    try:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly()
    except Exception:
        return None

    if provider_id:
        providers = config.get("providers", {}) if isinstance(config, dict) else {}
        provider_config = (
            providers.get(provider_id, {}) if isinstance(providers, dict) else {}
        )
        if isinstance(provider_config, dict):
            model_config = _get_model_config(provider_config, model)
            if model_config is not None:
                timeout = _coerce_timeout(model_config.get("timeout_seconds"))
                if timeout is not None:
                    return timeout

            timeout = _coerce_timeout(
                provider_config.get("request_timeout_seconds")
            )
            if timeout is not None:
                return timeout

    return _get_global_llm_timeout(config)


def get_provider_stale_timeout(
    provider_id: str, model: str | None = None
) -> float | None:
    """Return a configured non-stream stale timeout in seconds, if any."""
    if not provider_id:
        return None

    try:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly()
    except Exception:
        return None

    providers = config.get("providers", {}) if isinstance(config, dict) else {}
    provider_config = (
        providers.get(provider_id, {}) if isinstance(providers, dict) else {}
    )
    if not isinstance(provider_config, dict):
        return None

    model_config = _get_model_config(provider_config, model)
    if model_config is not None:
        timeout = _coerce_timeout(model_config.get("stale_timeout_seconds"))
        if timeout is not None:
            return timeout

    return _coerce_timeout(provider_config.get("stale_timeout_seconds"))


def _get_model_config(
    provider_config: dict[str, object], model: str | None
) -> dict[str, object] | None:
    if not model:
        return None

    models = provider_config.get("models", {})
    model_config = models.get(model, {}) if isinstance(models, dict) else {}
    if isinstance(model_config, dict):
        return model_config
    return None
