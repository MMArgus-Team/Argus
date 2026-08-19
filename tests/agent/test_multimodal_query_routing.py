"""Main-agent contract for semantic one-shot multimodal routing.

These tests deliberately exercise the real request builder and registered
multimodal schemas.  The routing contract is advisory: the model sees all
relevant tools and chooses from the user's full meaning; request assembly must
never pin a tool merely because a turn contains a historically overloaded
keyword such as ``table``, ``brand``, or ``company``.
"""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.prompt_builder import MM_LIVE_GUIDANCE, MM_QUERY_CONTRACT_MARKER
from run_agent import AIAgent
from tools.computer_use.schema import COMPUTER_USE_SCHEMA
from tools.vision_tools import VIDEO_ANALYZE_SCHEMA, VISION_ANALYZE_SCHEMA


@pytest.fixture(autouse=True)
def _isolated_hermes_home(monkeypatch, tmp_path):
    """Keep real-agent construction inside the test sandbox.

    These cases intentionally instantiate the production ``AIAgent`` instead
    of mocking schema resolution.  Give each case a real temporary Hermes home
    so logging/config setup is exercised without reading or writing the user's
    active profile.
    """
    hermes_home = tmp_path / ".argus"
    hermes_home.mkdir()
    monkeypatch.setenv("ARGUS_HOME", str(hermes_home))
    # Logging is unrelated to request routing and installs process-global file
    # handlers. Avoid leaking a handler that points at this per-test temporary
    # directory while keeping every schema/config/request-building import real.
    monkeypatch.setattr("hermes_logging.setup_logging", lambda **_kwargs: None)


@pytest.fixture()
def multimodal_routing_agent():
    """A real AIAgent request builder backed by the registered MM schemas."""
    with patch("run_agent.OpenAI"):
        agent = AIAgent(
            api_key="test-key-1234567890",
            provider="custom",
            model="qwen3.7-plus",
            base_url="https://internal.example/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            enabled_toolsets=["live_watcher", "monitor"],
        )

    # Exercise the former Qwen-thinking hard-route conditions too.  These
    # provider flags may be forwarded, but they must not inject a second
    # system instruction or pin a multimodal function.
    agent._extra_body_additions = {
        "enable_thinking": True,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    return agent


def _tool_schemas_by_name(api_kwargs):
    return {
        item["function"]["name"]: item["function"]
        for item in api_kwargs.get("tools") or []
    }


def test_prompt_routes_one_shot_visual_questions_to_query_worker_entry():
    assert MM_LIVE_GUIDANCE.startswith(MM_QUERY_CONTRACT_MARKER + "\n")
    assert MM_LIVE_GUIDANCE.count(MM_QUERY_CONTRACT_MARKER) == 1
    assert "current or historical camera/screen" in MM_LIVE_GUIDANCE
    assert "call query_multimodal" in MM_LIVE_GUIDANCE
    assert "answer directly, use multimodal Recall, use Search, or combine them" in (
        MM_LIVE_GUIDANCE
    )
    assert "complete, self-contained delegation" in MM_LIVE_GUIDANCE


def test_prompt_routes_simple_visual_qa_to_direct_queryworker_ownership():
    assert "response_mode='direct' for pure visual QA" in MM_LIVE_GUIDANCE
    assert "simple visual + Recall/Search QA" in MM_LIVE_GUIDANCE
    assert "replies directly to the user" in MM_LIVE_GUIDANCE
    assert "After a direct-mode handoff" in MM_LIVE_GUIDANCE


def test_prompt_routes_visual_plus_complex_skill_through_evidence_then_main():
    for capability in (
        "PDF/document reader",
        "local file access",
        "terminal",
        "browser interaction",
        "complex Skill",
    ):
        assert capability in MM_LIVE_GUIDANCE
    assert "response_mode='evidence'" in MM_LIVE_GUIDANCE
    assert "you retain reply ownership" in MM_LIVE_GUIDANCE
    assert "continue with the required tools in later rounds" in MM_LIVE_GUIDANCE
    assert "write exactly one final answer yourself" in MM_LIVE_GUIDANCE
    assert "summarize the xxx.pdf shown on my screen" in MM_LIVE_GUIDANCE


def test_prompt_preserves_mixed_visual_and_external_fact_semantics():
    assert "Never narrow an outside-fact request" in MM_LIVE_GUIDANCE
    assert "retail price" in MM_LIVE_GUIDANCE
    assert "is not a request for its visible price label" in MM_LIVE_GUIDANCE


def test_prompt_uses_relevant_prior_qa_in_a_self_contained_handoff():
    """Main reuses relevant QA without sending its entire chat history."""
    from tools.mm_memory_tool import QUERY_MULTIMODAL_SCHEMA

    for contract in (
        "consider both the current question and prior QA",
        "information directly useful for resolving the request",
        "QueryWorker does not receive the main chat history",
        "include it and its uncertainty",
        "Leave current visual referents for QueryWorker",
        "Do not paste unrelated history",
    ):
        assert contract in MM_LIVE_GUIDANCE

    query_help = (
        QUERY_MULTIMODAL_SCHEMA["parameters"]["properties"]["query"]["description"]
    )
    for contract in (
        "current question and directly useful prior QA",
        "uncertainty of reused context",
        "current visual referents",
        "ask-time frames",
    ):
        assert contract in query_help


def test_prompt_routes_known_historical_image_retrieval_directly_to_frame_tool():
    assert (
        "explicit request to display or retrieve the real historical image of an "
        "already-known item" in MM_LIVE_GUIDANCE
    )
    assert "call show_memory_frame directly as specified below" in MM_LIVE_GUIDANCE
    assert (
        "If the user already names the item and asks to see it, call "
        "show_memory_frame directly" in MM_LIVE_GUIDANCE
    )


def test_prompt_keeps_raw_frame_monitor_and_watcher_responsibilities_distinct():
    assert "Use get_current_frame only when the user explicitly asks" in MM_LIVE_GUIDANCE
    assert "Do not use it as the normal one-shot visual-QA path" in MM_LIVE_GUIDANCE
    assert "To watch until a future condition is triggered, call set_monitor" in (
        MM_LIVE_GUIDANCE
    )
    assert "To continuously analyse the stream" in MM_LIVE_GUIDANCE
    assert "call set_live_watcher" in MM_LIVE_GUIDANCE


def test_prompt_does_not_reference_legacy_recall_entry_or_capability_split():
    assert "recall_multimodal_memory" not in MM_LIVE_GUIDANCE
    assert "vision-capable main model" not in MM_LIVE_GUIDANCE
    assert "when the main model is text-only" not in MM_LIVE_GUIDANCE


@pytest.mark.parametrize(
    ("case", "user_text"),
    [
        ("current-frame", "现在画面里的银行叫什么名字？"),
        ("historical-frame", "刚才画面中出现的第二个物品是什么？"),
        (
            "pdf-table",
            "论文 PDF 的 Table 1 中支持 Omni 的 benchmark 有哪些？",
        ),
        ("figure-page", "屏幕里这篇 paper 的 Figure 2 在第 6 页表达了什么？"),
        ("referential-price", "我之前展示的第二个物品价格多少？"),
        (
            "visual-entity-plus-external-fact",
            "识别画面中的银行，再查它 2026-07-24 的港股收盘价（港元）。",
        ),
        ("ordinary-brand", "帮我把这段品牌定位文案改得更简洁。"),
        ("ordinary-display", "请显示公司本季度的 OKR 文本。"),
        ("ordinary-company", "公司的组织架构应该怎么调整？"),
        ("ordinary-referent-company", "这家公司的融资历史怎么样？"),
    ],
    ids=lambda value: value if isinstance(value, str) and " " not in value else None,
)
def test_request_builder_never_keyword_forces_multimodal_tool(
    multimodal_routing_agent,
    case,
    user_text,
):
    """Visual and lookalike text turns both remain semantic model choices."""
    api_messages = [
        {"role": "system", "content": "stable-system-prefix"},
        {"role": "user", "content": user_text},
    ]
    original = deepcopy(api_messages)

    kwargs = multimodal_routing_agent._build_api_kwargs(api_messages)

    assert kwargs["messages"] == original, case
    assert "tool_choice" not in kwargs, case
    assert not any(
        message.get("role") == "system"
        and "SYSTEM OVERRIDE" in str(message.get("content"))
        for message in kwargs["messages"]
    ), case

    schemas = _tool_schemas_by_name(kwargs)
    assert "query_multimodal" in schemas, case
    assert "recall_multimodal_memory" not in schemas, case


def test_query_multimodal_is_available_as_an_unpinned_semantic_choice(
    multimodal_routing_agent,
):
    """The model can select the unified entry without a client-side gate."""
    kwargs = multimodal_routing_agent._build_api_kwargs([
        {
            "role": "user",
            "content": (
                "画面里这家银行在港股上市了，查一下它们 "
                "2026-07-24 的收盘价是多少港元"
            ),
        },
    ])

    schemas = _tool_schemas_by_name(kwargs)
    query_schema = schemas["query_multimodal"]
    assert query_schema["parameters"]["required"] == ["query"]
    assert "current or historical" in query_schema["description"]
    assert "external Search" in query_schema["description"]
    assert "every requested fact" in query_schema["description"]
    assert "Never replace an outside-fact request" in query_schema["description"]
    assert "tool_choice" not in kwargs


def test_real_system_prompt_publishes_one_stable_query_contract(
    multimodal_routing_agent,
):
    first = multimodal_routing_agent._build_system_prompt()
    second = multimodal_routing_agent._build_system_prompt()

    assert first == second
    assert first.count(MM_QUERY_CONTRACT_MARKER) == 1
    assert "call query_multimodal" in first
    assert "recall_multimodal_memory" not in first


def test_registered_schemas_keep_raw_query_monitor_and_watcher_roles_distinct(
    multimodal_routing_agent,
):
    kwargs = multimodal_routing_agent._build_api_kwargs([
        {"role": "user", "content": "你自己根据语义选对工具"},
    ])
    schemas = _tool_schemas_by_name(kwargs)

    raw_frame = schemas["get_current_frame"]["description"]
    one_shot = schemas["query_multimodal"]["description"]
    historical_frame = schemas["show_memory_frame"]["description"]
    monitor = schemas["set_monitor"]["description"]
    watcher = schemas["set_live_watcher"]["description"]

    assert "raw current frames" in raw_frame
    assert "call query_multimodal instead" in raw_frame
    assert "one-shot question" in one_shot
    assert "future trigger" in one_shot
    assert "real historical key-frame images" in historical_frame
    assert "already know the exact item name, call this tool directly" in historical_frame
    assert "per-event alert" in monitor
    assert "summary/report/analysis" in monitor
    assert "continuously" in watcher
    assert "summary/report" in watcher


def test_stored_media_and_computer_schemas_defer_live_visual_qa_to_query_worker():
    vision = VISION_ANALYZE_SCHEMA["description"]
    video = VIDEO_ANALYZE_SCHEMA["description"]
    computer = COMPUTER_USE_SCHEMA["description"]

    for description in (vision, video):
        assert "raw/latest current frames" in description
        assert "use get_current_frame" in description
        assert "ordinary one-shot question" in description
        assert "current or past live stream" in description
        assert "use query_multimodal" in description

    assert "ordinary visual question about the current or past live screen" in computer
    assert "use query_multimodal" in computer
    assert "raw/latest capture" in computer
    assert "use get_current_frame" in computer
    assert "Reserve computer_use for desktop INTERACTION" in computer


@pytest.mark.parametrize(
    "user_text",
    [
        "显示公司品牌手册的目录文本",
        "brand display company",
        "table 这个词在 HTML 里是什么意思？",
    ],
)
def test_non_multimodal_agent_is_unchanged_by_visual_lookalike_keywords(user_text):
    """Ordinary sessions neither gain nor get forced into an MM tool."""
    with patch("run_agent.OpenAI"):
        agent = AIAgent(
            api_key="test-key-1234567890",
            provider="custom",
            model="qwen3.7-plus",
            base_url="https://internal.example/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            enabled_toolsets=["web"],
        )

    messages = [{"role": "user", "content": user_text}]
    kwargs = agent._build_api_kwargs(messages)

    assert kwargs["messages"] == messages
    assert "tool_choice" not in kwargs
    assert "query_multimodal" not in _tool_schemas_by_name(kwargs)
    assert not any(
        message.get("role") == "system"
        for message in kwargs["messages"]
    )


@pytest.mark.parametrize(
    "user_text",
    [
        "帮我修改公司品牌的显示名称",
        "这家公司的融资历史怎么样？",
        "HTML table 如何做响应式布局？",
        "请回忆之前讨论过的品牌文案。",
    ],
)
def test_lookalike_text_in_mm_session_reaches_normal_model_turn(
    multimodal_routing_agent,
    user_text,
):
    """Having MM tools visible does not turn lexical matches into tool calls."""
    multimodal_routing_agent.client = MagicMock()
    multimodal_routing_agent.client.chat.completions.create.return_value = (
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="semantic normal answer",
                        tool_calls=None,
                        reasoning_content=None,
                        model_extra={},
                    ),
                    finish_reason="stop",
                ),
            ],
            usage=None,
        )
    )

    with (
        patch.object(multimodal_routing_agent, "_persist_session"),
        patch.object(multimodal_routing_agent, "_save_trajectory"),
        patch.object(multimodal_routing_agent, "_cleanup_task_resources"),
    ):
        result = multimodal_routing_agent.run_conversation(user_text)

    assert result["api_calls"] == 1
    assert result["final_response"] == "semantic normal answer"
    create = multimodal_routing_agent.client.chat.completions.create
    assert create.call_count == 1
    kwargs = create.call_args.kwargs
    assert "tool_choice" not in kwargs
    assert not any(
        message.get("role") == "system"
        and "SYSTEM OVERRIDE" in str(message.get("content"))
        for message in kwargs["messages"]
    )
