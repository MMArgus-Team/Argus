"""Regression tests for Kimi/Moonshot parameter adaptation on the multimodal
direct-LLM callers (workers + monitor).

Root cause fixed: after switching those roles to Kimi, hardcoded
`temperature=0.x` + `enable_thinking` broke every call — Kimi allows only
temperature=1 (HTTP 400 "invalid temperature: only 1 is allowed") and uses
`thinking:{"type":...}` instead of enable_thinking. This left deep-research
summaries failing and the monitor daemon silently 400-ing every tick.

kimi_fix_create_kwargs coerces those params for Moonshot models (no-op
otherwise); wrap_kimi_client applies it transparently on an owned client.
"""
import unittest

from agent.multimodal.hermes_glue import (
    kimi_fix_create_kwargs, wrap_kimi_client,
)


class TestKimiFixKwargs(unittest.TestCase):
    def test_kimi_forces_temperature_1(self):
        k = kimi_fix_create_kwargs({"model": "kimi-k2.6", "temperature": 0.2})
        self.assertEqual(k["temperature"], 1)

    def test_kimi_translates_enable_thinking_off(self):
        k = kimi_fix_create_kwargs({
            "model": "kimi-k2.6", "temperature": 0.2, "top_p": 0.8,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False},
                           "top_k": 20}})
        self.assertEqual(k["temperature"], 1)
        self.assertNotIn("top_p", k)
        self.assertEqual(k["extra_body"]["thinking"], {"type": "disabled"})
        self.assertNotIn("chat_template_kwargs", k["extra_body"])
        self.assertNotIn("top_k", k["extra_body"])

    def test_thinking_only_model_never_gets_disabled(self):
        # k2.7-code is thinking-ONLY: it rejects thinking:{type:disabled} with
        # HTTP 400. Even when the caller wants thinking off, the fixer must strip
        # the thinking key (default = enabled), not send disabled.
        k = kimi_fix_create_kwargs({
            "model": "kimi-k2.7-code", "temperature": 0.2,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}})
        self.assertEqual(k["temperature"], 1)
        eb = k.get("extra_body", {})
        self.assertNotIn("thinking", eb)  # never 'disabled'

    def test_thinking_only_strips_explicit_disabled(self):
        k = kimi_fix_create_kwargs({
            "model": "kimi-k2.7-code",
            "extra_body": {"thinking": {"type": "disabled"}}})
        self.assertNotIn("thinking", k.get("extra_body", {}))

    def test_hybrid_model_still_gets_disabled(self):
        # k2.6 is hybrid — disabling thinking is allowed, so it must still work.
        k = kimi_fix_create_kwargs({
            "model": "kimi-k2.6",
            "extra_body": {"enable_thinking": False}})
        self.assertEqual(k["extra_body"]["thinking"], {"type": "disabled"})

    def test_kimi_translates_enable_thinking_on(self):
        k = kimi_fix_create_kwargs({
            "model": "kimi-k2.6", "temperature": 0.3,
            "extra_body": {"enable_thinking": True}})
        self.assertEqual(k["extra_body"]["thinking"], {"type": "enabled"})

    def test_non_kimi_untouched(self):
        orig = {"model": "qwen3.6-flash", "temperature": 0.2,
                "extra_body": {"enable_thinking": False}}
        k = kimi_fix_create_kwargs(dict(orig))
        self.assertEqual(k["temperature"], 0.2)
        self.assertEqual(k["extra_body"], {"enable_thinking": False})

    def test_kimi_empty_extra_body_dropped(self):
        # extra_body that only held enable_thinking → becomes {thinking:...},
        # never left as an empty dict.
        k = kimi_fix_create_kwargs({
            "model": "kimi-k2.6",
            "extra_body": {"enable_thinking": False}})
        self.assertEqual(k["extra_body"], {"thinking": {"type": "disabled"}})


class _Comp:
    def __init__(self): self.last = None
    def create(self, **kw): self.last = kw; return "resp"


class _Chat:
    def __init__(self): self.completions = _Comp()


class _FakeClient:
    def __init__(self): self.chat = _Chat()


class TestWrapKimiClient(unittest.TestCase):
    def test_wrapper_coerces_and_is_idempotent(self):
        c = _FakeClient()
        wrap_kimi_client(c)
        c.chat.completions.create(model="kimi-k2.6", temperature=0.0)
        self.assertEqual(c.chat.completions.last["temperature"], 1)
        # double-wrap must not double-wrap or change behavior
        wrap_kimi_client(c)
        c.chat.completions.create(model="kimi-k2.6", temperature=0.0)
        self.assertEqual(c.chat.completions.last["temperature"], 1)

    def test_wrapper_leaves_non_kimi_calls_alone(self):
        c = _FakeClient()
        wrap_kimi_client(c)
        c.chat.completions.create(model="qwen3.6-flash", temperature=0.2)
        self.assertEqual(c.chat.completions.last["temperature"], 0.2)


if __name__ == "__main__":
    unittest.main()
