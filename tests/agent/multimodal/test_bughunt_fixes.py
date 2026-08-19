"""Regression guards for the integrated bug-hunt fixes.

- #1: thinking on/off must be re-read PER TURN (agent built once; toggling
      agent.reasoning_config later must take effect).
- #2: cooldown gate must apply only to the immediate-report path, so silent /
      aggregate monitors still see-and-log every event.
- #4: WatcherAgent build failure must null self._loop so None-guards fire.

Pure in-memory; no cloud, no hardware.
"""
import types


# ── #1: thinking on/off per-turn refresh ────────────────────────────────────
# Source of truth is now agent.reasoning_config (from agent.reasoning_effort
# in config.yaml via parse_reasoning_effort). Off = {"enabled": False};
# anything else (missing / {"enabled": True, "effort": ...}) = ON.
def _per_turn_extra_body(agent, session):
    """Mirror of the per-turn refresh in _run_prompt_submit.run()."""
    if str(session.get("source") or "") == "multimodal" and agent is not None:
        from tui_gateway.server import _apply_mm_deep_thinking
        _apply_mm_deep_thinking(agent, session)


def test_thinking_takes_effect_after_first_turn():
    agent = types.SimpleNamespace(
        _extra_body_additions={"enable_thinking": False},
        model="qwen3.7-plus",
        provider="custom",
        base_url="https://example.test/v1",
        reasoning_config={"enabled": False},
    )
    session = {"source": "multimodal"}

    _per_turn_extra_body(agent, session)          # turn 1: off
    assert agent._extra_body_additions["enable_thinking"] is False

    agent.reasoning_config = {"enabled": True, "effort": "medium"}
    _per_turn_extra_body(agent, session)          # turn 2: MUST now be on
    assert agent._extra_body_additions["enable_thinking"] is True

    agent.reasoning_config = {"enabled": False}
    _per_turn_extra_body(agent, session)          # turn 3: MUST be off
    assert agent._extra_body_additions["enable_thinking"] is False


def test_thinking_only_for_multimodal_source():
    agent = types.SimpleNamespace(
        _extra_body_additions={"enable_thinking": True},
        model="qwen3.7-plus",
        provider="custom",
        base_url="https://example.test/v1",
        reasoning_config={"enabled": True, "effort": "medium"},
    )
    session = {"source": "cli"}
    _per_turn_extra_body(agent, session)          # non-MM source: untouched
    assert agent._extra_body_additions == {"enable_thinking": True}


def test_gpt_compatible_endpoint_gets_no_qwen_private_kwargs():
    agent = types.SimpleNamespace(
        _extra_body_additions={"chat_template_kwargs": {"enable_thinking": True}},
        model="gpt-5.6-luna",
        provider="custom",
        base_url="https://doc.devops.beta.xiaohongshu.com/u/example",
        reasoning_config={"enabled": True, "effort": "medium"},
    )
    _per_turn_extra_body(agent, {"source": "multimodal"})
    assert agent._extra_body_additions == {}


# ── #2: cooldown gate only for immediate path ───────────────────────────────
def _is_immediate(m):
    """Mirror of the daemon's _immediate computation."""
    return not (m.get("silent", False) or (m.get("report_interval") or 0) > 0)


def test_cooldown_only_gates_immediate_monitors():
    # Immediate monitor (no silent, no T) → cooldown applies.
    assert _is_immediate({"silent": False, "report_interval": None}) is True
    # Silent monitor → not immediate → cooldown must NOT block its file logging.
    assert _is_immediate({"silent": True, "report_interval": None}) is False
    # Aggregation monitor (T>0) → not immediate → cooldown must NOT block it.
    assert _is_immediate({"silent": False, "report_interval": 60}) is False


# ── #4: WatcherAgent build failure nulls the loop ───────────────────────────
def test_router_engine_build_failure_nulls_loop():
    from agent.multimodal.watcher_engine import WatcherAgent
    # Force _build to fail without any real deps.
    eng = WatcherAgent.__new__(WatcherAgent)
    eng._loop = None
    eng._healthy = None
    import threading
    eng._ready = threading.Event()
    eng._sid = "t"
    eng.model = ""
    # ★ merge (dev_0807): WatcherAgent 新增了生命周期状态机, teardown 路径
    #   (_run 失败 → _publish_stopped) 依赖这组属性。__new__ 绕过 __init__,
    #   故手动补齐, 与 __init__ 的初始化保持一致。
    eng._state_lock = threading.RLock()
    eng._state = WatcherAgent.STATE_NEW
    eng._startup_done = threading.Event()
    eng._stopped = threading.Event()
    eng._stopped_callbacks = []
    eng._stopped_callbacks_fired = False
    eng._startup_error = None
    eng._stop = threading.Event()

    def _boom():
        return False
    eng._build = _boom  # type: ignore
    eng._run()  # runs on this thread; build fails → should close+null loop
    assert eng._loop is None, "build failure must null self._loop for None-guards"
    assert eng._healthy is False
    assert eng._ready.is_set()
