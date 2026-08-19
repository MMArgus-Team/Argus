"""The voice roles must send no sampling params at all.

History: every voice LLM call used to pin ``temperature`` (and vLLM-style
``top_p``/``top_k`` elsewhere). On gateways that manage sampling server-side
that is a hard 400 —

    invalid temperature: only 1 is allowed for this model

— and the voice path reached none of the four workarounds that existed for it
(a per-model fixed/omit table, a client wrapper that stripped the keys, an
adapter-level strip, and a reactive drop-and-retry). Every decide_speak /
decide_route / phrase / intent call failed, silently falling back to "speak
anyway" / raw passthrough / "assume the sentence ended".

Sampling params are now dropped everywhere, so there is nothing to be rejected
and nothing to retry. These tests pin that contract at the voice choke point.

Replaces test_voice_portable_params_retry.py, which tested the retry that no
longer needs to exist.

Written with ``asyncio.run`` rather than ``@pytest.mark.asyncio`` so they run
without the pytest-asyncio plugin (not installed in this checkout).
"""

import asyncio

import pytest

import agent.multimodal.voice_agent_context as C


SAMPLING_KEYS = ("temperature", "top_p", "top_k")


class _FakeCompletions:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        message = type("M", (), {"content": self.reply})()
        choice = type("Ch", (), {"message": message})()
        return type("R", (), {"choices": [choice], "usage": None})()


class _FakeClient:
    def __init__(self, reply='{"speak": true, "reason": "ok"}'):
        self.chat = type("Chat", (), {})()
        self.chat.completions = _FakeCompletions(reply)

    @property
    def calls(self):
        return self.chat.completions.calls


def _snapshot():
    return C.WorldSnapshot(
        recent_dialogue=[{"role": "user", "content": "你好，那个是。"}],
        trigger_event={"kind": "main_agent_reply", "text": "你好！", "task_id": ""},
        voice_last_2=[],
        silence_sec=999999.0,
    )


class TestNoSamplingParams:
    def _drive_all(self, model):
        """Exercise all five voice roles, return every wire kwargs dict."""
        reply = (
            '{"speak": true, "reason": "r", "route": "self", "answer": "在的",'
            ' "text": "好了", "speak_to_me": true, "is_end": true,'
            ' "addressed": true}'
        )
        client = _FakeClient(reply)
        snap = _snapshot()

        async def _run():
            await C.decide_speak(client=client, model=model, snapshot=snap)
            await C.decide_route(client=client, model=model, snapshot=snap)
            await C.phrase_utterance(client=client, model=model, snapshot=snap,
                                     source="watcher", raw_text="一段报告")
            await C.judge_addressed_to_me(client=client, model=model,
                                          user_text="喂")
            await C.judge_intent_eou_remote(client=client, model=model,
                                            user_text="呃")

        asyncio.run(_run())
        assert len(client.calls) == 5, "all five roles must have been exercised"
        return client.calls

    @pytest.mark.parametrize("model", [
        "GPT-5.6 Luna", "kimi-k3-baidu-DIBP", "qwen3.7-plus", "Kimi-K2.6",
    ])
    def test_no_role_sends_any_sampling_param(self, model):
        for kwargs in self._drive_all(model):
            for key in SAMPLING_KEYS:
                assert key not in kwargs, f"{key} leaked for model={model}"
            extra = kwargs.get("extra_body") or {}
            for key in SAMPLING_KEYS:
                assert key not in extra, f"{key} leaked via extra_body"

    def test_one_attempt_per_call_no_retry(self):
        """Nothing can be rejected, so a second attempt is pure latency."""
        client = _FakeClient()
        asyncio.run(C.decide_speak(
            client=client, model="GPT-5.6 Luna", snapshot=_snapshot()))
        assert len(client.calls) == 1

    def test_completion_budget_is_widened(self):
        """Voice max_tokens is 50-200; a reasoning model needs far more or the
        reply comes back empty because thinking shares the budget."""
        client = _FakeClient()
        asyncio.run(C.decide_speak(
            client=client, model="m", snapshot=_snapshot()))
        assert client.calls[0]["max_completion_tokens"] >= 8192
        assert "max_tokens" not in client.calls[0]

    def test_enable_thinking_switch_survives(self):
        """chat_template_kwargs is a routing switch, not a sampling knob."""
        client = _FakeClient()
        asyncio.run(C.decide_speak(
            client=client, model="m", snapshot=_snapshot()))
        ctk = client.calls[0]["extra_body"]["chat_template_kwargs"]
        assert ctk == {"enable_thinking": False}

    def test_errors_still_surface_as_a_None_decision(self):
        """Without the retry wrapper, a failure must still be swallowed by the
        caller rather than crashing the voice loop."""
        class _Boom(_FakeCompletions):
            async def create(self, **kwargs):
                self.calls.append(kwargs)
                raise RuntimeError("Rate limit exceeded")

        client = _FakeClient()
        client.chat.completions = _Boom("")
        assert asyncio.run(C.decide_speak(
            client=client, model="m", snapshot=_snapshot())) is None
        assert len(client.calls) == 1


class TestKimiK3ReasoningCap:
    """Kimi K3 at default effort reasons until the gateway's 60s deadline, while
    the voice budgets are 2s (intent/EOU) and 6s (decide/phrase)."""

    def _capture(self, model):
        client = _FakeClient()
        asyncio.run(C.decide_speak(
            client=client, model=model, snapshot=_snapshot()))
        return client.calls[-1]

    def test_kimi_k3_gets_low_reasoning_effort(self):
        assert self._capture("kimi-k3-baidu-DIBP")["reasoning_effort"] == "low"

    def test_spaced_or_cased_k3_names_still_match(self):
        assert C._model_is_kimi_k3("Kimi K3 Baidu") is True
        assert C._model_is_kimi_k3("KIMI-K3") is True

    def test_other_models_are_not_capped(self):
        """Never silently downgrade reasoning for a model that didn't need it."""
        for model in ("Kimi-K2.6", "GPT-5.6 Luna", "qwen3.7-plus"):
            assert "reasoning_effort" not in self._capture(model)
        assert C._model_is_kimi_k3("Kimi-K2.6") is False
