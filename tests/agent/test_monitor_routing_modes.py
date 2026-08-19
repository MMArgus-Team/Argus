from types import SimpleNamespace

import pytest

from agent.monitor_routing import (
    direct_monitor_request_text,
    infer_monitor_trigger_mode,
)
from agent.prompt_builder import MM_LIVE_GUIDANCE
from toolsets import TOOLSETS, _HERMES_CORE_TOOLS


@pytest.mark.parametrize(
    "text",
    [
        "每次看到手机都提醒我",
        "监控我的画面，每看到有什么新物品就告诉我名字",
        "每看见一个新物品就通知我",
        "每发现一个新物品就告诉我",
        "每出现一个新物品就提醒我",
        "每新增一类物品就告诉我",
        "每有一个人进入画面就提醒我",
        "每当画面出现猫就通知我",
        "持续盯着门口",
        "一直监控比分",
        "整场监控库里进球",
        "库里得分就告诉我",
        "Alert me whenever a goal is scored.",
        "Keep watching throughout the whole game.",
        "全程监控画面，每一次看到新物品都告诉我",
        "全程一小时，每一次看到猫都提醒我",
        "总计监控一小时，每一次看到新物品都告诉我",
        "总共观察画面，每一次出现人都告诉我",
        "每次点开第一个视频都提醒我",
        "Tell me every time I open the first video.",
    ],
)
def test_explicit_recurrence_infers_continuous(text):
    assert infer_monitor_trigger_mode(text) == "continuous"


@pytest.mark.parametrize(
    "text",
    [
        "帮我监控一下我点开第一个视频就告诉我这个视频的标题",
        "第一次看到手机就告诉我",
        "只通知我一次",
        "下一次库里进球时告诉我",
        "Tell me about the next goal.",
        "Alert me the first time the dialog appears.",
    ],
)
def test_explicit_one_shot_infers_once(text):
    assert infer_monitor_trigger_mode(text) == "once"


@pytest.mark.parametrize(
    "text",
    [
        "看到手机就告诉我",
        "如果画面出现手机就告诉我",
        "发现新物品就告诉我",
        "Tell me when the dialog appears.",
        "",
    ],
)
def test_missing_lifecycle_cardinality_falls_back_to_model(text):
    agent = SimpleNamespace(
        _multimodal_session=True,
        valid_tool_names={"set_monitor"},
    )

    assert infer_monitor_trigger_mode(text) is None
    assert direct_monitor_request_text(agent, text) == ""


@pytest.mark.parametrize(
    "text",
    [
        "每次看到手机都只提醒我一次",
    ],
)
def test_per_episode_once_scope_falls_back_to_model(text):
    assert infer_monitor_trigger_mode(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "监控画面，每个新物品出现时只提醒我一次",
        "Alert me once for each new object that appears on screen.",
    ],
)
def test_per_item_once_scope_falls_back_to_model(text):
    assert infer_monitor_trigger_mode(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "持续监控进球，但只通知一次",
        "每次看到手机都提醒我，整个监控总共只提醒一次",
        "Alert me whenever it appears, but notify me only once.",
        "Alert me whenever it appears, but notify me only once overall.",
        "持续监控画面，通知我后就停止",
    ],
)
def test_explicit_global_single_delivery_overrides_recurrence(text):
    assert infer_monitor_trigger_mode(text) == "once"


@pytest.mark.parametrize(
    "text",
    [
        "整场监控库里下一次进球时告诉我",
        "Watch the whole game and tell me about the next goal.",
        "持续监控我的画面，我点开第一个视频就告诉我标题",
    ],
)
def test_conflicting_lifecycle_cues_fall_back_to_model(text):
    agent = SimpleNamespace(
        _multimodal_session=True,
        valid_tool_names={"set_monitor"},
    )

    assert infer_monitor_trigger_mode(text) is None
    assert direct_monitor_request_text(agent, text) == ""


def test_clicking_first_video_is_a_fast_create_but_ambiguous_alert_is_not():
    agent = SimpleNamespace(
        _multimodal_session=True,
        valid_tool_names={"set_monitor"},
        mm_monitors={"mon_old": {"monitor_query": "看到水杯就提醒我"}},
    )
    explicit_once = "帮我监控一下我点开第一个视频就告诉我这个视频的标题"
    ambiguous = "看到手机也提醒我"

    assert direct_monitor_request_text(agent, explicit_once) == explicit_once
    assert direct_monitor_request_text(agent, ambiguous) == ""


def test_each_new_object_request_is_a_continuous_fast_create():
    agent = SimpleNamespace(
        _multimodal_session=True,
        valid_tool_names={"set_monitor"},
    )
    text = "监控我的画面，每看到有什么新物品就告诉我名字"

    assert infer_monitor_trigger_mode(text) == "continuous"
    assert direct_monitor_request_text(agent, text) == text


def test_explicit_sports_visual_monitor_is_a_continuous_fast_create():
    agent = SimpleNamespace(
        _multimodal_session=True,
        valid_tool_names={"set_monitor"},
    )
    text = "整场监控库里进球，每次进球都告诉我"

    assert direct_monitor_request_text(agent, text) == text
    assert infer_monitor_trigger_mode(text) == "continuous"


@pytest.mark.parametrize(
    "text",
    [
        "持续监控服务器，挂了告诉我",
        "整场监控 API 延迟，每次超时告诉我",
        "监控 8080 端口，每次断开都提醒我",
        "持续监控数据库，每次失败告诉我",
    ],
)
def test_non_visual_service_monitors_do_not_use_fast_create(text):
    agent = SimpleNamespace(
        _multimodal_session=True,
        valid_tool_names={"set_monitor"},
    )

    assert direct_monitor_request_text(agent, text) == ""


@pytest.mark.parametrize(
    "text",
    [
        "修改已有监控，看到手机也提醒我",
        "暂停门口监控",
        "删除手机监控",
        "给已有监控加一个手机条件，看到手机告诉我",
    ],
)
def test_explicit_existing_monitor_management_never_fast_creates(text):
    agent = SimpleNamespace(
        _multimodal_session=True,
        valid_tool_names={"set_monitor"},
        mm_monitors={"mon_old": {"monitor_query": "看到水杯就提醒我"}},
    )

    assert direct_monitor_request_text(agent, text) == ""


def test_list_monitor_is_not_model_visible_or_recommended():
    assert "set_monitor" in _HERMES_CORE_TOOLS
    assert "list_monitor" not in _HERMES_CORE_TOOLS
    assert TOOLSETS["monitor"]["tools"] == ["set_monitor"]
    assert "monitor_ref" in MM_LIVE_GUIDANCE
    assert "list_monitor" not in MM_LIVE_GUIDANCE
