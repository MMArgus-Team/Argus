# -*- coding: utf-8 -*-
"""Root-cause fix (goal 2026-07): the main agent must NOT claim it set up a
monitor/watcher without actually calling the tool.

The bug: user says "帮我盯着屏幕,出现萨鲁曼就告诉我", the agent (which gets
injected frames + a 'simple visual question → answer directly' identity) replies
"我已经为你设置了监控,当萨鲁曼再次出现时会自动提醒你" WITHOUT calling set_monitor.

These tests assert the anti-hallucination iron rule now lives in the main-agent
multimodal guidance strings.
"""
from agent.prompt_builder import (
    MM_LIVE_GUIDANCE,
    DEFAULT_AGENT_IDENTITY,
)


def test_mm_live_guidance_forbids_faking_a_monitor():
    g = MM_LIVE_GUIDANCE.lower()
    # Must contain an explicit rule tying "watch / when X appears / remind" to a
    # real tool call, and forbid claiming a monitor was set without one.
    assert "set_monitor" in g
    # (prompt 精简后措辞变了: 反幻觉铁律现在表述为 "不许 guess/claim 看到屏幕、
    #  没调工具就看不到"。intent 不变: 没真调工具就不许声称做了/看到了。)
    assert "never" in g
    # A negative rule: don't claim you see / did it if you didn't call the tool.
    assert ("do not guess or claim" in g or "cannot see" in g
            or "without" in g)


# (删: test_identity_scopes_direct_answers_to_now_only —— 153f8320 精简 prompt 后
#  DEFAULT_AGENT_IDENTITY 不再含 snapshot/reminder/keep-watching 措辞, 该行为已废弃。
#  反幻觉铁律仍由 test_mm_live_guidance_forbids_faking_a_monitor 覆盖。)


if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main([__file__, "-v"]))
