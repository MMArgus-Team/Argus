"""Tests for vision-capability classification + config validation warnings.

Covers:
  * agent.vision_capability.classify_vision — built-in family table + the
    text-only/vision/unknown three-way result.
  * hermes_cli.config._validate_vision_capability — warns for vision-REQUIRED
    roles (main agent, multimodal worker/monitor/memory) on text-only models,
    the reversed supports_vision case, and the "follows main" skip.
"""

from __future__ import annotations

import pytest

from agent.vision_capability import classify_vision


class TestClassifyVision:
    @pytest.mark.parametrize("model", [
        "kimi-k2.6", "kimi-k2.7-code", "qwen3.6-flash", "qwen3-vl-plus",
        "qwen2.5-vl-72b", "qwen-vl-max", "gemini-3.5-flash", "gpt-4o",
        "glm-4v-plus", "glm-5v-turbo", "claude-sonnet-4", "o3-mini",
        "pixtral-12b", "llava-1.6", "mimo-v2.5",
        # Modern Qwen chat models are natively multimodal (text+image).
        "qwen3.5-flash", "qwen3.7-plus", "qwen-plus", "qwen-flash", "qwen-max",
        "qwen3-max",
    ])
    def test_known_vision(self, model):
        assert classify_vision("custom", model, "") is True

    @pytest.mark.parametrize("model", [
        "deepseek-chat", "deepseek-reasoner", "deepseek-v4-pro",
        "qwen2.5-72b-instruct", "gpt-oss-120b",
        "glm-4-9b", "glm-4-flash", "qwq-32b",
        "text-embedding-3-large",
    ])
    def test_known_text_only(self, model):
        assert classify_vision("custom", model, "") is False

    @pytest.mark.parametrize("model", [
        "some-random-model-xyz", "", "unknown-llm-2099",
        # GLM base chat (4.5/4.6/5) vision support is version-dependent → not
        # judged, so we neither warn nor claim vision.
        "glm-4.6", "glm-4.5",
    ])
    def test_unknown_returns_none(self, model):
        assert classify_vision("custom", model, "") is None

    def test_models_dev_fallback_via_url(self):
        # deepseek not in the built-in table? it is — but confirm url-inferred
        # provider path doesn't crash and text-only stays False.
        assert classify_vision("custom", "deepseek-chat",
                               "https://api.deepseek.com/v1") is False

    def test_never_raises(self):
        # Garbage inputs must resolve to None, not raise.
        assert classify_vision(None, None, None) is None  # type: ignore[arg-type]


class TestValidateVisionCapability:
    def _run(self, config):
        from hermes_cli.config import _validate_vision_capability
        return _validate_vision_capability(config)

    def test_all_vision_capable_no_warnings(self):
        cfg = {
            "model": {"default": "kimi-k2.6", "provider": "custom",
                      "base_url": "https://api.moonshot.cn/v1",
                      "supports_vision": True},
            "multimodal": {
                "worker_base_url": "https://api.moonshot.cn/v1",
                "worker_model": "kimi-k2.7-code",
                "monitor_base_url": "https://x/v1", "monitor_model": "qwen3.6-flash",
                "memory_provider": "gemini",
                "memory_base_url": "https://runway/v1", "memory_model": "gemini-3.5-flash",
            },
        }
        assert self._run(cfg) == []

    def test_main_text_only_with_supports_vision_true_is_reversed(self):
        cfg = {"model": {"default": "deepseek-chat", "provider": "custom",
                         "base_url": "https://api.deepseek.com/v1",
                         "supports_vision": True}}
        issues = self._run(cfg)
        assert len(issues) == 1
        assert "reversed" in issues[0].message
        assert issues[0].severity == "warning"

    def test_main_text_only_without_declaration_warns(self):
        cfg = {"model": {"default": "gpt-oss-120b", "provider": "custom"}}
        issues = self._run(cfg)
        assert len(issues) == 1
        assert "text-only" in issues[0].message
        assert issues[0].severity == "warning"

    def test_worker_text_only_dedicated_endpoint_warns(self):
        cfg = {
            "model": {"default": "kimi-k2.6", "provider": "custom",
                      "base_url": "https://api.moonshot.cn/v1"},
            "multimodal": {"worker_base_url": "https://api.deepseek.com/v1",
                           "worker_provider": "custom", "worker_model": "deepseek-chat"},
        }
        issues = self._run(cfg)
        assert any("deep-research watcher" in i.message for i in issues)

    def test_worker_follows_main_when_base_url_empty(self):
        cfg = {
            "model": {"default": "kimi-k2.6", "provider": "custom",
                      "base_url": "https://api.moonshot.cn/v1"},
            "multimodal": {"worker_base_url": "", "worker_provider": "custom",
                           "worker_model": "deepseek-chat"},
        }
        issues = self._run(cfg)
        assert not any("worker" in i.message for i in issues)

    def test_monitor_text_only_warns(self):
        # deepseek-chat is genuinely text-only → monitor (a vision role) warns.
        cfg = {
            "model": {"default": "kimi-k2.6", "provider": "custom",
                      "base_url": "https://api.moonshot.cn/v1"},
            "multimodal": {"monitor_base_url": "https://x/v1",
                           "monitor_provider": "custom", "monitor_model": "deepseek-chat"},
        }
        issues = self._run(cfg)
        assert any("monitor" in i.message for i in issues)

    def test_monitor_qwen35_flash_no_warn(self):
        # Regression: qwen3.5-flash is natively multimodal → NO false warning.
        cfg = {
            "model": {"default": "kimi-k2.6", "provider": "custom",
                      "base_url": "https://api.moonshot.cn/v1"},
            "multimodal": {"monitor_base_url": "https://x/v1",
                           "monitor_provider": "custom", "monitor_model": "qwen3.5-flash"},
        }
        issues = self._run(cfg)
        assert not any("monitor" in i.message for i in issues)

    def test_unknown_model_stays_silent(self):
        cfg = {"model": {"default": "some-future-model", "provider": "custom"}}
        assert self._run(cfg) == []

    def test_auxiliary_role_is_exempt(self):
        # auxiliary.* configured with a text-only model must NOT produce a
        # vision warning (auxiliary never sees images).
        cfg = {
            "model": {"default": "kimi-k2.6", "provider": "custom",
                      "base_url": "https://api.moonshot.cn/v1"},
            "auxiliary": {"compression": {"model": "deepseek-chat",
                                          "base_url": "https://api.deepseek.com/v1"}},
        }
        assert self._run(cfg) == []
