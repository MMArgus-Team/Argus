"""Regression tests for independent model configuration of the three roles:
main agent (untouched), multimodal deep-analysis workers, and the monitor.

Covers:
  * build_config maps multimodal.worker_* / monitor_* into the Config dataclass.
  * HermesClientFactory.worker_client(): dedicated endpoint when worker_base_url
    is set (independent model, e.g. Kimi), else fall back to main resolution.

Pure unittest (no network). worker_client's dedicated branch builds an
AsyncOpenAI pointed at the configured base_url WITHOUT making a request, so we
can assert on the client's base_url + returned model offline.
"""
import unittest


class TestBuildConfigMapping(unittest.TestCase):
    def test_legacy_flat_worker_alias_mapped(self):
        """Pre-v33 flat multimodal.worker_* still works via the watcher_* alias."""
        from agent.multimodal.hermes_glue import build_config
        cfg = build_config({"multimodal": {
            "worker_provider": "custom",
            "worker_base_url": "https://api.moonshot.cn/v1",
            "worker_api_key": "sk-worker",
            "worker_model": "kimi-k2.7-code",
            "monitor_provider": "custom",
            "monitor_base_url": "https://api.moonshot.cn/v1",
            "monitor_api_key": "sk-monitor",
            "monitor_model": "kimi-k2.7-code",
        }})
        # worker_* aliases onto the renamed watcher_* dataclass fields.
        self.assertEqual(cfg.watcher_base_url, "https://api.moonshot.cn/v1")
        self.assertEqual(cfg.watcher_api_key, "sk-worker")
        self.assertEqual(cfg.monitor_base_url, "https://api.moonshot.cn/v1")
        self.assertEqual(cfg.monitor_model, "kimi-k2.7-code")
        # worker_model flows into cfg.model + marks explicit (so engine build
        # doesn't overwrite it with the main model).
        self.assertEqual(cfg.model, "kimi-k2.7-code")
        self.assertTrue(cfg._worker_model_explicit)

    def test_empty_worker_defaults(self):
        from agent.multimodal.hermes_glue import build_config
        cfg = build_config({"multimodal": {}})
        self.assertEqual(cfg.watcher_base_url, "")
        self.assertEqual(cfg.monitor_base_url, "")

    def test_legacy_facts_capacity_migrates_only_when_new_key_is_absent(self):
        from agent.multimodal.hermes_glue import build_config

        legacy = build_config({"multimodal": {"facts_max": 20}})
        self.assertEqual(legacy.search_facts_max, 20)

        both = build_config({"multimodal": {
            "facts_max": 20,
            "search_facts_max": 7,
        }})
        self.assertEqual(both.search_facts_max, 7)


class TestNestedSchemaMapping(unittest.TestCase):
    """v32 layout: model.{monitor,watcher,memory[.recall]} + settings.* + audio.*
    flatten back onto the same flat Config fields the workers read."""

    def _nested(self):
        return {
            "model": {
                "default": "qwen3.6-flash", "provider": "custom",
                "base_url": "https://main", "api_key": "sk-main",
                "monitor": {"provider": "custom", "base_url": "https://mon",
                            "api_key": "sk-mon", "model": "qwen3.5-flash"},
                "watcher": {"provider": "custom",
                            "base_url": "https://api.moonshot.cn/v1",
                            "api_key": "sk-w", "model": "kimi-k2.7-code"},
                "memory": {"provider": "gemini", "base_url": "https://gem",
                           "api_key": "sk-gem", "model": "gemini-3.5-flash",
                           # v33: recall has NO endpoint — only behavior knobs.
                           "recall": {"topk_micro": 15}},
            },
            "settings": {"enabled": True, "monitor_tick_sec": 7,
                         "anysearch_api_key": "as_x"},
            "audio": {"asr_url": "http://a", "dashscope_api_key": "sk-dash",
                      "realtime_tts_voice": "Ethan"},
        }

    def test_model_roles_map_to_flat_fields(self):
        from agent.multimodal.hermes_glue import build_config
        cfg = build_config(self._nested())
        # watcher → watcher_* (+ worker model pins cfg.model as explicit)
        self.assertEqual(cfg.watcher_base_url, "https://api.moonshot.cn/v1")
        self.assertEqual(cfg.watcher_api_key, "sk-w")
        self.assertEqual(cfg.model, "kimi-k2.7-code")
        self.assertTrue(cfg._worker_model_explicit)
        # monitor
        self.assertEqual(cfg.monitor_base_url, "https://mon")
        self.assertEqual(cfg.monitor_model, "qwen3.5-flash")
        # memory (v33: recall has no endpoint — it uses model.memory; only the
        # recall BEHAVIOR knob flows through)
        self.assertEqual(cfg.memory_provider, "gemini")
        self.assertEqual(cfg.memory_model, "gemini-3.5-flash")
        self.assertFalse(hasattr(cfg, "recall_base_url"))
        self.assertEqual(cfg.recall_topk_micro, 15)

    def test_settings_and_audio_map(self):
        from agent.multimodal.hermes_glue import build_config, flatten_mm_config
        nested = self._nested()
        cfg = build_config(nested)
        flat = flatten_mm_config(nested)
        # audio.* → Config dataclass fields
        self.assertEqual(cfg.asr_url, "http://a")
        self.assertEqual(cfg.dashscope_api_key, "sk-dash")
        self.assertEqual(cfg.realtime_tts_voice, "Ethan")
        # settings anysearch key is a dataclass field
        self.assertEqual(cfg.anysearch_api_key, "as_x")
        # settings knob read straight from the flattened dict (monitor_engine path)
        self.assertEqual(flat.get("monitor_tick_sec"), 7)

    def test_enabled_from_settings(self):
        from agent.multimodal.hermes_glue import multimodal_enabled
        self.assertTrue(multimodal_enabled(self._nested()))
        self.assertFalse(multimodal_enabled({"settings": {"enabled": False}}))
        # legacy fallback still honored
        self.assertFalse(multimodal_enabled({"multimodal": {"enabled": False}}))

    def test_new_schema_precedence_over_legacy(self):
        # If both new nested and legacy flat exist, new wins.
        from agent.multimodal.hermes_glue import build_config
        cfg = build_config({
            "model": {"watcher": {"base_url": "https://new", "model": "new-m"}},
            "multimodal": {"worker_base_url": "https://old", "worker_model": "old-m"},
        })
        self.assertEqual(cfg.watcher_base_url, "https://new")
        self.assertEqual(cfg.model, "new-m")

    def test_reviewer_event_and_gate_switches_map_independently(self):
        from agent.multimodal.hermes_glue import build_config, flatten_mm_config

        nested = {
            "model": {
                "memory": {
                    "reviewer": {
                        "event_enabled": False,
                        "event_gate": {"enabled": True},
                    },
                },
            },
        }

        flat = flatten_mm_config(nested)
        cfg = build_config(nested)

        self.assertIs(flat["reviewer_event_enabled"], False)
        self.assertIs(flat["reviewer_event_gate_enabled"], True)
        self.assertIs(cfg.reviewer_event_enabled, False)
        self.assertIs(cfg.reviewer_event_gate_enabled, True)


class TestWorkerClient(unittest.TestCase):
    def test_dedicated_endpoint_when_configured(self):
        from agent.multimodal.hermes_glue import build_config, HermesClientFactory
        cfg = build_config({"model": {"watcher": {
            "base_url": "https://api.moonshot.cn/v1",
            "api_key": "sk-worker",
            "model": "kimi-k2.7-code",
        }}})
        client, model = HermesClientFactory(cfg).worker_client()
        self.assertEqual(model, "kimi-k2.7-code")
        # AsyncOpenAI base_url includes the configured host (no network call made).
        self.assertIn("api.moonshot.cn", str(client.base_url))

    def test_falls_back_when_no_worker_endpoint(self):
        from agent.multimodal.hermes_glue import build_config, HermesClientFactory
        cfg = build_config({"multimodal": {}})
        factory = HermesClientFactory(cfg)
        # Stub the main resolution so the test never touches the network / real
        # provider config; worker_client must delegate to it when unconfigured.
        sentinel = object()
        factory._cached = (sentinel, "main-model")
        client, model = factory.worker_client()
        self.assertIs(client, sentinel)
        self.assertEqual(model, "main-model")


if __name__ == "__main__":
    unittest.main()
