"""Tests for agent.models_dev — models.dev registry integration."""
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from unittest.mock import patch, MagicMock

from agent.models_dev import (
    PROVIDER_TO_MODELS_DEV,
    _extract_context,
    fetch_models_dev,
    get_model_capabilities,
    lookup_models_dev_context,
)


SAMPLE_REGISTRY = {
    "anthropic": {
        "id": "anthropic",
        "name": "Anthropic",
        "models": {
            "claude-opus-4-6": {
                "id": "claude-opus-4-6",
                "limit": {"context": 1000000, "output": 128000},
            },
            "claude-sonnet-4-6": {
                "id": "claude-sonnet-4-6",
                "limit": {"context": 1000000, "output": 64000},
            },
            "claude-sonnet-4-0": {
                "id": "claude-sonnet-4-0",
                "limit": {"context": 200000, "output": 64000},
            },
        },
    },
    "github-copilot": {
        "id": "github-copilot",
        "name": "GitHub Copilot",
        "models": {
            "claude-opus-4.6": {
                "id": "claude-opus-4.6",
                "limit": {"context": 128000, "output": 32000},
            },
        },
    },
    "xai": {
        "id": "xai",
        "name": "xAI",
        "models": {
            "grok-build-0.1": {
                "id": "grok-build-0.1",
                "limit": {"context": 256000, "output": 64000},
            },
        },
    },
    "kilo": {
        "id": "kilo",
        "name": "Kilo Gateway",
        "models": {
            "anthropic/claude-sonnet-4.6": {
                "id": "anthropic/claude-sonnet-4.6",
                "limit": {"context": 1000000, "output": 128000},
            },
        },
    },
    "deepseek": {
        "id": "deepseek",
        "name": "DeepSeek",
        "models": {
            "deepseek-chat": {
                "id": "deepseek-chat",
                "limit": {"context": 128000, "output": 8192},
            },
        },
    },
    "audio-only": {
        "id": "audio-only",
        "models": {
            "tts-model": {
                "id": "tts-model",
                "limit": {"context": 0, "output": 0},
            },
        },
    },
}


class TestProviderMapping:
    def test_xai_oauth_uses_xai_catalog(self):
        assert PROVIDER_TO_MODELS_DEV["xai"] == "xai"
        assert PROVIDER_TO_MODELS_DEV["xai-oauth"] == "xai"

    def test_unmapped_provider_not_in_dict(self):
        assert "nous" not in PROVIDER_TO_MODELS_DEV

    def test_openai_codex_mapped_to_openai(self):
        assert PROVIDER_TO_MODELS_DEV["openai"] == "openai"
        assert PROVIDER_TO_MODELS_DEV["openai-codex"] == "openai"


class TestExtractContext:
    def test_valid_entry(self):
        assert _extract_context({"limit": {"context": 128000}}) == 128000

    def test_zero_context_returns_none(self):
        assert _extract_context({"limit": {"context": 0}}) is None

    def test_missing_limit_returns_none(self):
        assert _extract_context({"id": "test"}) is None

    def test_missing_context_returns_none(self):
        assert _extract_context({"limit": {"output": 8192}}) is None

    def test_non_dict_returns_none(self):
        assert _extract_context("not a dict") is None

    def test_float_context_coerced_to_int(self):
        assert _extract_context({"limit": {"context": 131072.0}}) == 131072


class TestLookupModelsDevContext:
    @patch("agent.models_dev.fetch_models_dev")
    def test_exact_match(self, mock_fetch):
        mock_fetch.return_value = SAMPLE_REGISTRY
        assert lookup_models_dev_context("anthropic", "claude-opus-4-6") == 1000000

    @patch("agent.models_dev.fetch_models_dev")
    def test_case_insensitive_match(self, mock_fetch):
        mock_fetch.return_value = SAMPLE_REGISTRY
        assert lookup_models_dev_context("anthropic", "Claude-Opus-4-6") == 1000000

    @patch("agent.models_dev.fetch_models_dev")
    def test_provider_not_mapped(self, mock_fetch):
        mock_fetch.return_value = SAMPLE_REGISTRY
        assert lookup_models_dev_context("nous", "some-model") is None

    @patch("agent.models_dev.fetch_models_dev")
    def test_model_not_found(self, mock_fetch):
        mock_fetch.return_value = SAMPLE_REGISTRY
        assert lookup_models_dev_context("anthropic", "nonexistent-model") is None

    @patch("agent.models_dev.fetch_models_dev")
    def test_provider_aware_context(self, mock_fetch):
        """Same model, different context per provider."""
        mock_fetch.return_value = SAMPLE_REGISTRY
        # Anthropic direct: 1M
        assert lookup_models_dev_context("anthropic", "claude-opus-4-6") == 1000000
        # GitHub Copilot: only 128K for same model
        assert lookup_models_dev_context("copilot", "claude-opus-4.6") == 128000

    @patch("agent.models_dev.fetch_models_dev")
    def test_xai_oauth_resolves_xai_context(self, mock_fetch):
        """xAI OAuth is an auth path, not a separate model catalog."""
        mock_fetch.return_value = SAMPLE_REGISTRY
        assert lookup_models_dev_context("xai-oauth", "grok-build-0.1") == 256000

    @patch("agent.models_dev.fetch_models_dev")
    def test_zero_context_filtered(self, mock_fetch):
        mock_fetch.return_value = SAMPLE_REGISTRY
        # audio-only is not a mapped provider, but test the filtering directly
        data = SAMPLE_REGISTRY["audio-only"]["models"]["tts-model"]
        assert _extract_context(data) is None

    @patch("agent.models_dev.fetch_models_dev")
    def test_empty_registry(self, mock_fetch):
        mock_fetch.return_value = {}
        assert lookup_models_dev_context("anthropic", "claude-opus-4-6") is None


class TestFetchModelsDev:
    @patch("agent.models_dev.requests.get")
    def test_fetch_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_REGISTRY
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        # Clear caches
        import agent.models_dev as md
        md._models_dev_cache = {}
        md._models_dev_cache_time = 0

        with patch.object(md, "_save_disk_cache"):
            result = fetch_models_dev(force_refresh=True)

        assert "anthropic" in result
        assert len(result) == len(SAMPLE_REGISTRY)

    @patch("agent.models_dev.requests.get")
    def test_fetch_failure_returns_stale_cache(self, mock_get):
        mock_get.side_effect = Exception("network error")

        import agent.models_dev as md
        md._models_dev_cache = SAMPLE_REGISTRY
        md._models_dev_cache_time = 0  # expired

        with patch.object(md, "_load_disk_cache", return_value=SAMPLE_REGISTRY):
            result = fetch_models_dev(force_refresh=True)

        assert "anthropic" in result

    @patch("agent.models_dev.requests.get")
    def test_in_memory_cache_used(self, mock_get):
        import agent.models_dev as md
        import time
        md._models_dev_cache = SAMPLE_REGISTRY
        md._models_dev_cache_time = time.time()  # fresh

        result = fetch_models_dev()
        mock_get.assert_not_called()
        assert result == SAMPLE_REGISTRY

    @patch("agent.models_dev.requests.get")
    def test_fresh_disk_cache_skips_network(self, mock_get):
        """When in-mem cache is empty but disk cache exists and is fresh by
        mtime (< TTL), fetch_models_dev returns disk data without ever
        making the network call.

        This is the cold-start fast path: every fresh process previously
        paid ~500 ms re-fetching a registry that was already on disk
        from an earlier run.
        """
        import agent.models_dev as md
        # Empty in-mem cache so stage 1 doesn't short-circuit.
        md._models_dev_cache = {}
        md._models_dev_cache_time = 0

        with patch.object(md, "_disk_cache_age_seconds", return_value=60.0), \
             patch.object(md, "_load_disk_cache", return_value=SAMPLE_REGISTRY):
            result = fetch_models_dev()

        # The whole point: no network call.
        mock_get.assert_not_called()
        assert "anthropic" in result
        # In-mem cache populated so subsequent calls within the same
        # process stay on stage 1.
        assert md._models_dev_cache == SAMPLE_REGISTRY

    def test_stale_disk_cache_serves_stale_and_refreshes_in_background(self):
        """Expired disk data is returned while one background refresh starts."""
        import agent.models_dev as md
        md._models_dev_cache = {}
        md._models_dev_cache_time = 0

        with patch.object(md, "_disk_cache_age_seconds",
                          return_value=md._MODELS_DEV_CACHE_TTL + 60), \
             patch.object(md, "_load_disk_cache", return_value=SAMPLE_REGISTRY), \
             patch.object(md, "_trigger_background_refresh") as mock_refresh, \
             patch.object(md.requests, "get") as mock_get:
            result = fetch_models_dev()

        assert result == SAMPLE_REGISTRY
        mock_refresh.assert_called_once_with()
        mock_get.assert_not_called()

    @patch("agent.models_dev.requests.get")
    def test_force_refresh_skips_disk_cache(self, mock_get):
        """force_refresh=True bypasses BOTH the in-mem cache AND the
        disk-cache fast path. Used by ``argus config refresh`` and
        anywhere else the user explicitly asked for fresh data.
        """
        import agent.models_dev as md
        md._models_dev_cache = {}
        md._models_dev_cache_time = 0

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_REGISTRY
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        # Disk cache is fresh, but force_refresh must override it.
        with patch.object(md, "_disk_cache_age_seconds", return_value=60.0), \
             patch.object(md, "_load_disk_cache", return_value=SAMPLE_REGISTRY), \
             patch.object(md, "_save_disk_cache"):
            result = fetch_models_dev(force_refresh=True)

        mock_get.assert_called_once()
        assert "anthropic" in result

    def test_missing_disk_cache_uses_bundled_snapshot(self):
        """First-run lookups return bundled data without foreground HTTP."""
        import agent.models_dev as md
        md._models_dev_cache = {}
        md._models_dev_cache_time = 0

        with patch.object(md, "_disk_cache_age_seconds", return_value=None), \
             patch.object(md, "_load_disk_cache", return_value={}), \
             patch.object(md, "_load_bundled_snapshot", return_value=SAMPLE_REGISTRY), \
             patch.object(md, "_trigger_background_refresh") as mock_refresh, \
             patch.object(md.requests, "get") as mock_get:
            result = fetch_models_dev()

        mock_get.assert_not_called()
        mock_refresh.assert_called_once_with()
        assert result == SAMPLE_REGISTRY
        assert md._models_dev_cache == SAMPLE_REGISTRY

    def test_corrupt_bundled_snapshot_returns_empty(self, tmp_path):
        """A broken package asset is tolerated instead of escaping JSON errors."""
        import agent.models_dev as md

        snapshot = tmp_path / "models_dev_api.json"
        snapshot.write_text("{not-json", encoding="utf-8")

        with patch.object(md, "_get_bundled_snapshot_path", return_value=snapshot):
            assert md._load_bundled_snapshot() == {}

    def test_missing_bundled_snapshot_uses_rate_limited_background_refresh(self):
        """No fallback data still never repeats blocking HTTP per lookup.

        Run the daemon target inline to model a failed refresh that has already
        completed. The retry timestamp must suppress the second lookup's
        network attempt even though the in-memory registry remains empty.
        """
        import agent.models_dev as md

        class _ImmediateThread:
            def __init__(self, *, target, **_kwargs):
                self._target = target

            def start(self):
                self._target()

        old_cache = md._models_dev_cache
        old_cache_time = md._models_dev_cache_time
        old_inflight = md._bg_refresh_inflight
        old_last_attempt = md._bg_refresh_last_attempt
        try:
            md._models_dev_cache = {}
            md._models_dev_cache_time = 0
            md._bg_refresh_inflight = False
            md._bg_refresh_last_attempt = 0

            with patch.object(md, "_disk_cache_age_seconds", return_value=None), \
                 patch.object(md, "_load_disk_cache", return_value={}), \
                 patch.object(md, "_load_bundled_snapshot", return_value={}), \
                 patch.object(md, "_fetch_from_network", return_value={}) as mock_fetch, \
                 patch.object(md.threading, "Thread", side_effect=_ImmediateThread):
                assert fetch_models_dev() == {}
                assert fetch_models_dev() == {}

            mock_fetch.assert_called_once_with()
        finally:
            md._models_dev_cache = old_cache
            md._models_dev_cache_time = old_cache_time
            md._bg_refresh_inflight = old_inflight
            md._bg_refresh_last_attempt = old_last_attempt

    @patch("agent.models_dev.requests.get")
    def test_force_refresh_failure_falls_back_to_bundled_snapshot(self, mock_get):
        """Explicit refresh still blocks on HTTP, then degrades to bundled data."""
        import agent.models_dev as md

        mock_get.side_effect = Exception("network error")
        md._models_dev_cache = {}
        md._models_dev_cache_time = 0

        with patch.object(md, "_load_disk_cache", return_value={}), \
             patch.object(md, "_load_bundled_snapshot", return_value=SAMPLE_REGISTRY):
            result = fetch_models_dev(force_refresh=True)

        mock_get.assert_called_once()
        assert result == SAMPLE_REGISTRY

    def test_cold_start_background_network_does_not_block_foreground(self):
        """A genuinely blocked refresh thread cannot delay the caller."""
        import agent.models_dev as md

        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()

        def blocked_fetch():
            started.set()
            try:
                assert release.wait(timeout=5)
                return {}
            finally:
                finished.set()

        old_cache = md._models_dev_cache
        old_cache_time = md._models_dev_cache_time
        old_inflight = md._bg_refresh_inflight
        old_last_attempt = md._bg_refresh_last_attempt
        try:
            md._models_dev_cache = {}
            md._models_dev_cache_time = 0
            md._bg_refresh_inflight = False
            md._bg_refresh_last_attempt = 0

            with patch.object(md, "_disk_cache_age_seconds", return_value=None), \
                 patch.object(md, "_load_disk_cache", return_value={}), \
                 patch.object(md, "_load_bundled_snapshot", return_value=SAMPLE_REGISTRY), \
                 patch.object(md, "_fetch_from_network", side_effect=blocked_fetch):
                before = time.perf_counter()
                result = fetch_models_dev()
                elapsed = time.perf_counter() - before

                assert result == SAMPLE_REGISTRY
                assert elapsed < 1.0
                assert started.wait(timeout=1)
        finally:
            release.set()
            assert finished.wait(timeout=2)
            deadline = time.monotonic() + 2
            while md._bg_refresh_inflight and time.monotonic() < deadline:
                time.sleep(0.001)
            md._models_dev_cache = old_cache
            md._models_dev_cache_time = old_cache_time
            md._bg_refresh_inflight = old_inflight
            md._bg_refresh_last_attempt = old_last_attempt

    def test_concurrent_cold_start_callers_start_one_background_refresh(self):
        """N simultaneous snapshot readers share one blocked network refresh."""
        import agent.models_dev as md

        callers = 8
        all_loading = threading.Barrier(callers)
        network_started = threading.Event()
        release_network = threading.Event()
        network_finished = threading.Event()
        count_lock = threading.Lock()
        network_calls = 0

        def synchronized_snapshot_load():
            all_loading.wait(timeout=5)
            return SAMPLE_REGISTRY

        def blocked_fetch():
            nonlocal network_calls
            with count_lock:
                network_calls += 1
            network_started.set()
            try:
                assert release_network.wait(timeout=5)
                return {}
            finally:
                network_finished.set()

        old_cache = md._models_dev_cache
        old_cache_time = md._models_dev_cache_time
        old_inflight = md._bg_refresh_inflight
        old_last_attempt = md._bg_refresh_last_attempt
        try:
            md._models_dev_cache = {}
            md._models_dev_cache_time = 0
            md._bg_refresh_inflight = False
            md._bg_refresh_last_attempt = 0

            with patch.object(md, "_disk_cache_age_seconds", return_value=None), \
                 patch.object(md, "_load_disk_cache", return_value={}), \
                 patch.object(md, "_load_bundled_snapshot", side_effect=synchronized_snapshot_load), \
                 patch.object(md, "_fetch_from_network", side_effect=blocked_fetch):
                with ThreadPoolExecutor(max_workers=callers) as pool:
                    futures = [pool.submit(fetch_models_dev) for _ in range(callers)]
                    results = [future.result(timeout=2) for future in futures]

                assert all(result == SAMPLE_REGISTRY for result in results)
                assert network_started.wait(timeout=1)
                assert network_calls == 1
        finally:
            release_network.set()
            assert network_finished.wait(timeout=2)
            deadline = time.monotonic() + 2
            while md._bg_refresh_inflight and time.monotonic() < deadline:
                time.sleep(0.001)
            md._models_dev_cache = old_cache
            md._models_dev_cache_time = old_cache_time
            md._bg_refresh_inflight = old_inflight
            md._bg_refresh_last_attempt = old_last_attempt

    def test_force_refresh_waits_for_network(self):
        """Explicit refresh remains the one synchronous network path."""
        import agent.models_dev as md

        live_registry = {"live": {"id": "live", "models": {}}}
        started = threading.Event()
        release = threading.Event()
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = live_registry

        def blocked_get(*_args, **_kwargs):
            started.set()
            assert release.wait(timeout=5)
            return response

        old_cache = md._models_dev_cache
        old_cache_time = md._models_dev_cache_time
        try:
            md._models_dev_cache = {}
            md._models_dev_cache_time = 0
            with patch.object(md.requests, "get", side_effect=blocked_get), \
                 patch.object(md, "_save_disk_cache"):
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(fetch_models_dev, True)
                    assert started.wait(timeout=1)
                    assert not future.done()
                    release.set()
                    assert future.result(timeout=2) == live_registry
        finally:
            release.set()
            md._models_dev_cache = old_cache
            md._models_dev_cache_time = old_cache_time

    def test_concurrent_bundled_reader_cannot_overwrite_live_refresh(self):
        """A late snapshot parser never replaces an already-published live result."""
        import agent.models_dev as md

        live_registry = {"live": {"id": "live", "models": {}}}
        second_loader_entered = threading.Event()
        live_published = threading.Event()
        loader_lock = threading.Lock()
        loader_calls = 0

        def ordered_snapshot_load():
            nonlocal loader_calls
            with loader_lock:
                loader_calls += 1
                call_number = loader_calls
            if call_number == 1:
                assert second_loader_entered.wait(timeout=5)
                return SAMPLE_REGISTRY
            second_loader_entered.set()
            assert live_published.wait(timeout=5)
            return SAMPLE_REGISTRY

        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = live_registry

        def record_live_write(data):
            assert data == live_registry
            live_published.set()

        old_cache = md._models_dev_cache
        old_cache_time = md._models_dev_cache_time
        old_inflight = md._bg_refresh_inflight
        old_last_attempt = md._bg_refresh_last_attempt
        try:
            md._models_dev_cache = {}
            md._models_dev_cache_time = 0
            md._bg_refresh_inflight = False
            md._bg_refresh_last_attempt = 0

            with patch.object(md, "_disk_cache_age_seconds", return_value=None), \
                 patch.object(md, "_load_disk_cache", return_value={}), \
                 patch.object(md, "_load_bundled_snapshot", side_effect=ordered_snapshot_load), \
                 patch.object(md.requests, "get", return_value=response), \
                 patch.object(md, "_save_disk_cache", side_effect=record_live_write):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [pool.submit(fetch_models_dev) for _ in range(2)]
                    results = [future.result(timeout=5) for future in futures]

                assert SAMPLE_REGISTRY in results
                assert live_registry in results
                cached, _ = md._cache_snapshot()
                assert cached == live_registry
        finally:
            deadline = time.monotonic() + 2
            while md._bg_refresh_inflight and time.monotonic() < deadline:
                time.sleep(0.001)
            md._models_dev_cache = old_cache
            md._models_dev_cache_time = old_cache_time
            md._bg_refresh_inflight = old_inflight
            md._bg_refresh_last_attempt = old_last_attempt


# ---------------------------------------------------------------------------
# get_model_capabilities — vision via modalities.input
# ---------------------------------------------------------------------------


CAPS_REGISTRY = {
    "google": {
        "id": "google",
        "models": {
            "gemma-4-31b-it": {
                "id": "gemma-4-31b-it",
                "attachment": False,
                "tool_call": True,
                "modalities": {"input": ["text", "image"]},
                "limit": {"context": 128000, "output": 8192},
            },
            "gemma-3-1b": {
                "id": "gemma-3-1b",
                "tool_call": True,
                "limit": {"context": 32000, "output": 8192},
            },
            "text-only-with-stale-attachment": {
                "id": "text-only-with-stale-attachment",
                "attachment": True,
                "tool_call": True,
                "modalities": {"input": ["text"]},
                "limit": {"context": 128000, "output": 8192},
            },
        },
    },
    "anthropic": {
        "id": "anthropic",
        "models": {
            "claude-sonnet-4": {
                "id": "claude-sonnet-4",
                "attachment": True,
                "tool_call": True,
                "limit": {"context": 200000, "output": 64000},
            },
        },
    },
}


class TestGetModelCapabilities:
    """Tests for get_model_capabilities vision detection."""

    def test_vision_from_attachment_flag(self):
        """Models with attachment=True and no modalities should report supports_vision=True."""
        with patch("agent.models_dev.fetch_models_dev", return_value=CAPS_REGISTRY):
            caps = get_model_capabilities("anthropic", "claude-sonnet-4")
        assert caps is not None
        assert caps.supports_vision is True

    def test_vision_from_modalities_input_image(self):
        """Models with 'image' in modalities.input but attachment=False should
        still report supports_vision=True (the core fix in this PR)."""
        with patch("agent.models_dev.fetch_models_dev", return_value=CAPS_REGISTRY):
            caps = get_model_capabilities("google", "gemma-4-31b-it")
        assert caps is not None
        assert caps.supports_vision is True

    def test_text_only_modalities_override_stale_attachment_flag(self):
        """Text-only modalities must win over stale attachment=True metadata."""
        with patch("agent.models_dev.fetch_models_dev", return_value=CAPS_REGISTRY):
            caps = get_model_capabilities("google", "text-only-with-stale-attachment")
        assert caps is not None
        assert caps.supports_vision is False

    def test_no_vision_without_attachment_or_modalities(self):
        """Models with neither attachment nor image modality should be non-vision."""
        with patch("agent.models_dev.fetch_models_dev", return_value=CAPS_REGISTRY):
            caps = get_model_capabilities("google", "gemma-3-1b")
        assert caps is not None
        assert caps.supports_vision is False

    def test_modalities_non_dict_handled(self):
        """Non-dict modalities field should not crash."""
        registry = {
            "google": {"id": "google", "models": {
                "weird-model": {
                    "id": "weird-model",
                    "modalities": "text",  # not a dict
                    "limit": {"context": 200000, "output": 8192},
                },
            }},
        }
        with patch("agent.models_dev.fetch_models_dev", return_value=registry):
            caps = get_model_capabilities("gemini", "weird-model")
        assert caps is not None
        assert caps.supports_vision is False

    def test_model_not_found_returns_none(self):
        """Unknown model should return None."""
        with patch("agent.models_dev.fetch_models_dev", return_value=CAPS_REGISTRY):
            caps = get_model_capabilities("anthropic", "nonexistent-model")
        assert caps is None
