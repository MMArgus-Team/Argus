"""The voice chain must log the actual system/user prompts, verbatim.

``[VA_LLM] in=`` only ever carried the *structured* payload (the world-snapshot
dict). The system prompt — the thing that decides whether the assistant speaks,
how it routes, and how it words an utterance — was never logged at all, so
checking "what did the model actually see" meant reading source. And every
ordinary ``vtrace`` field goes through ``json.dumps`` and is capped at
``ARGUS_TRACE_MAXLEN`` (2000), which both escapes and silently clips a 2 KB
system prompt.

``vtrace_prompt`` therefore writes the bodies raw and uncapped. These tests pin
that: uncapped, unescaped, both roles, all five voice call sites, and still
completely silent when the trace switch is off.
"""

from __future__ import annotations

import asyncio
import logging

import pytest


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(record.getMessage())


@pytest.fixture()
def trace_home(monkeypatch):
    """Capture trace output in-process, with tracing forced ON.

    Deliberately not driven through ARGUS_HOME: ``get_hermes_home()`` resolves
    the log directory once at import time, so setting the env var from a test
    lands too late and the assertions would silently read an empty file.
    Installing a handler also satisfies ``_ensure_handler``'s early-out, so no
    real file is created.
    """
    import agent.multimodal.voice_trace as vt
    cap = _Capture()
    prev = list(vt.log.handlers)
    vt.log.handlers = [cap]
    vt.log.setLevel(logging.INFO)
    monkeypatch.setattr(vt, "_enabled_cache", True)
    yield cap, vt
    vt.log.handlers = prev


def _read(cap) -> str:
    return "\n".join(cap.lines)


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


def _snapshot(ctx):
    return ctx.WorldSnapshot(
        recent_dialogue=[{"role": "user", "content": "帮我设置一个深度调研的。"}],
        trigger_event={"kind": "main_agent_reply", "text": "好的，我已经开始了。"},
        voice_last_2=["已看到代码页面。"],
        silence_sec=3.2,
    )


class TestVerbatimPromptDump:
    def test_system_prompt_is_logged_in_full_not_truncated(self, trace_home):
        cap, _vt = trace_home
        import agent.multimodal.voice_agent_context as ctx

        asyncio.run(ctx.decide_speak(
            client=_FakeClient(), model="GPT-5.6 Luna",
            snapshot=_snapshot(ctx)))

        body = _read(cap)
        system = ctx._DECIDE_SPEAK_SYSTEM
        assert len(system) > 2000, (
            "this prompt must exceed the 2000-char vtrace cap, otherwise the "
            "no-truncation assertion below proves nothing"
        )
        # The whole prompt, first line to last, and not escaped into one line.
        assert system in body
        assert "…(+" not in body, "the prompt must not be clipped"
        assert "\\n" not in body.split("SYSTEM ---")[1][:200], (
            "bodies must be raw text, not json-escaped"
        )

    def test_user_prompt_is_the_exact_wire_string(self, trace_home):
        cap, _vt = trace_home
        import agent.multimodal.voice_agent_context as ctx

        asyncio.run(ctx.decide_speak(
            client=_FakeClient(), model="M", snapshot=_snapshot(ctx)))

        body = _read(cap)
        # The assembled "Reference:\n{json}" message, not just the query text.
        assert "Reference:" in body
        assert "帮我设置一个深度调研的。" in body
        assert '"silence_sec": 3.2' in body

    def test_header_stays_greppable(self, trace_home):
        cap, _vt = trace_home
        import agent.multimodal.voice_agent_context as ctx

        asyncio.run(ctx.decide_speak(
            client=_FakeClient(), model="GPT-5.6 Luna",
            snapshot=_snapshot(ctx)))

        head = [ln for ln in _read(cap).splitlines()
                if "decide_speak.prompt" in ln and "ARGUS|" in ln][0]
        assert 'model="GPT-5.6 Luna"' in head
        assert "system_chars=" in head and "user_chars=" in head

    @pytest.mark.parametrize("call_name", [
        "decide_speak", "decide_route", "phrase", "intent", "intent_eou",
    ])
    def test_every_voice_role_dumps_its_prompt(self, trace_home, call_name):
        cap, _vt = trace_home
        import agent.multimodal.voice_agent_context as ctx

        reply = (
            '{"speak": true, "reason": "r", "route": "self", "answer": "在的",'
            ' "text": "好了", "speak_to_me": true, "is_end": true,'
            ' "addressed": true}'
        )
        client = _FakeClient(reply)
        snap = _snapshot(ctx)

        async def _drive():
            await ctx.decide_speak(client=client, model="M", snapshot=snap)
            await ctx.decide_route(client=client, model="M", snapshot=snap)
            await ctx.phrase_utterance(client=client, model="M", snapshot=snap,
                                       source="watcher", raw_text="一段报告")
            await ctx.judge_addressed_to_me(client=client, model="M",
                                            user_text="喂")
            await ctx.judge_intent_eou_remote(client=client, model="M",
                                              user_text="呃")

        asyncio.run(_drive())
        assert f"{call_name}.prompt" in _read(cap)

    def test_prompt_is_dumped_even_when_the_call_fails(self, trace_home):
        """A failed call is exactly when you need to see what was sent."""
        cap, _vt = trace_home
        import agent.multimodal.voice_agent_context as ctx

        class _Boom(_FakeCompletions):
            async def create(self, **kwargs):
                raise RuntimeError("Rate limit exceeded")

        client = _FakeClient()
        client.chat.completions = _Boom("")
        assert asyncio.run(ctx.decide_speak(
            client=client, model="M", snapshot=_snapshot(ctx))) is None
        assert "decide_speak.prompt" in _read(cap)
        assert ctx._DECIDE_SPEAK_SYSTEM in _read(cap)


class TestDisabledByDefault:
    def test_nothing_is_written_when_trace_is_off(self, monkeypatch):
        import agent.multimodal.voice_trace as vt
        cap = _Capture()
        prev = list(vt.log.handlers)
        vt.log.handlers = [cap]
        monkeypatch.setattr(vt, "_enabled_cache", False)
        try:
            import agent.multimodal.voice_agent_context as ctx
            asyncio.run(ctx.decide_speak(
                client=_FakeClient(), model="M", snapshot=_snapshot(ctx)))
        finally:
            vt.log.handlers = prev

        assert cap.lines == [], "the switch must gate the dump entirely"
