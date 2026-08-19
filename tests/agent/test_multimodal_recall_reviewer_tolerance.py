import asyncio
import base64
import json
from io import BytesIO
from types import SimpleNamespace

from agent.multimodal.hermes_glue import build_config, HermesClientFactory
from agent.multimodal._workers import (
    EntityReviewer,
    EventReviewer,
    RecallAgent,
    ReviewerEndpointLimiter,
)


def _tiny_jpeg_b64():
    from PIL import Image

    image = Image.new("RGB", (32, 24), "white")
    buf = BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_nested_recall_verify_backend_config_is_applied():
    cfg = build_config({
        "model": {
            "memory": {
                "recall": {
                    "verify_provider": "custom",
                    "verify_base_url": "https://verify.example.test/v1/messages",
                    "verify_api_key": "verify-key",
                    "verify_model": "GPT-5.6 Luna",
                },
            },
        },
    })

    assert cfg.recall_verify_provider == "custom"
    assert cfg.recall_verify_base_url == "https://verify.example.test/v1/messages"
    assert cfg.recall_verify_api_key == "verify-key"
    assert cfg.recall_verify_model == "GPT-5.6 Luna"


def test_recall_decision_accepts_bare_keys_tool_call():
    raw = (
        '{thought: "还需要查字幕", can_answer: false, useful_info: "", '
        'tool_calls: [{name: "search_audio", args: {query: "租车 一天 多少钱"}}]}'
    )

    parsed, repairs = RecallAgent._parse_decision_json(raw)
    parsed, norm_repairs = RecallAgent._normalize_decision_tool_calls(parsed or {})

    assert parsed["can_answer"] is False
    assert parsed["tool_calls"] == [
        {"name": "search_audio", "args": {"query": "租车 一天 多少钱"}}
    ]
    assert repairs or norm_repairs


def test_recall_decision_recovers_plain_text_tool_hint():
    parsed, repairs = RecallAgent._parse_decision_json(
        "需要检索 search_audio: 零跑A10 租车 一天 多少钱"
    )

    assert parsed is not None
    assert parsed["can_answer"] is False
    assert parsed["tool_calls"] == [
        {"name": "search_audio", "args": {"query": "零跑A10 租车 一天 多少钱"}}
    ]
    assert repairs


def test_recall_decision_recursively_unwraps_json_in_useful_info():
    inner = {
        "can_answer": False,
        "useful_info": "尚未确认车牌前缀",
        "tool_calls": [{
            "name": "search_screen_text",
            "args": {"query": "2V6J5", "limit": 10},
        }],
    }
    raw = json.dumps({
        "thought": "",
        "can_answer": True,
        "useful_info": json.dumps(inner, ensure_ascii=False),
        "tool_calls": [],
    }, ensure_ascii=False)

    parsed, repairs = RecallAgent._parse_decision_json(raw)
    parsed, norm_repairs = RecallAgent._normalize_decision_tool_calls(parsed or {})

    assert parsed["can_answer"] is False
    assert parsed["useful_info"] == "尚未确认车牌前缀"
    assert parsed["tool_calls"] == [{
        "name": "search_screen_text",
        "args": {"query": "2V6J5", "limit": 10},
    }]
    assert repairs or norm_repairs


def test_recall_decision_recursively_unwraps_top_level_json_string():
    inner = json.dumps({
        "thought": "需要查事件摘要",
        "can_answer": False,
        "useful_info": "",
        "tool_calls": [{
            "name": "search_micro",
            "args": {"query": "车牌 2V6J5"},
        }],
    }, ensure_ascii=False)

    parsed, repairs = RecallAgent._parse_decision_json(json.dumps(inner))
    parsed, _ = RecallAgent._normalize_decision_tool_calls(parsed or {})

    assert parsed["can_answer"] is False
    assert parsed["tool_calls"] == [{
        "name": "search_micro",
        "args": {"query": "车牌 2V6J5"},
    }]
    assert repairs


def test_recall_decision_tool_signal_cannot_fall_open_to_answerable():
    raw = (
        r'provider wrapper broke JSON: \"can_answer\": false, '
        r'next=\"search_screen_text\", query=\"2V6J5\"'
    )

    parsed, repairs = RecallAgent._parse_decision_json(raw)

    assert parsed is not None
    assert parsed["can_answer"] is False
    assert repairs


def test_decide_next_errors_when_retrieval_signal_has_no_recovered_calls():
    class DummyRecall(RecallAgent):
        def __init__(self):
            self.cfg = SimpleNamespace(
                recall_decide_frames=0,
                recall_max_rounds=3,
                recall_max_tokens=512,
                recall_temperature=0.0,
            )
            self.mem = SimpleNamespace(get_recent_entities=lambda *_a, **_k: [])
            self.buf = None
            self.recorder = None
            self.model = "GPT-5.6 Luna"

        async def _create_chat_completion(self, *_args, **_kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content='can_answer: false; next tool search_screen_text',
            ))])

    decision = asyncio.run(
        DummyRecall()._decide_next(
            brief="查找车牌 2V6J5",
            user_text="车牌前两个字是什么",
            ask_ts=464.0,
            clues=[],
            round_idx=0,
        )
    )

    assert decision["can_answer"] is False
    assert decision["tool_calls"] == []
    assert "executable tool_calls could not be recovered" in decision["error"]


def test_recall_verify_recursively_unwraps_stringified_json():
    inner = {
        "keep": ["f_plate"],
        "visual_correction": "车牌前缀不是粤B",
        "exact_text": "鄂A2V6J5",
        "uncertain": False,
    }
    raw = json.dumps({"content": json.dumps(inner, ensure_ascii=False)},
                     ensure_ascii=False)

    parsed, repairs = RecallAgent._parse_verify_json(raw)

    assert parsed == inner
    assert repairs


def test_exact_text_crop_uses_rapidocr_bbox_as_localizer():
    from PIL import Image

    img = Image.new("RGB", (320, 180), "black")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    jpeg_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    rec = SimpleNamespace(ocr_blocks=[SimpleNamespace(
        text="A2V6J5", bbox=[120, 80, 42, 14], confidence=0.91,
    )])

    crops = RecallAgent._exact_text_crop_b64s(jpeg_b64, rec)

    assert len(crops) == 1
    crop = Image.open(BytesIO(base64.b64decode(crops[0])))
    assert max(crop.size) >= 300


def test_exact_text_verify_retries_invalid_json_then_uses_grounded_text():
    class DummyRecall(RecallAgent):
        def __init__(self):
            self.cfg = SimpleNamespace(
                recall_verify_max_frames=2,
                recall_verify_retries=1,
                recall_verify_retry_delay_sec=0.0,
            )
            self.frame_store = SimpleNamespace(get_many=lambda _fids: [
                SimpleNamespace(
                    frame_id="f_plate", ts=12.0,
                    jpeg_b64=_tiny_jpeg_b64(),
                ),
            ])
            self.screen_text_store = None
            self.recorder = None
            self.model = "GPT-5.6 Luna"
            self.calls = 0

        async def _create_chat_completion(self, *_args, **_kwargs):
            self.calls += 1
            content = "not-json" if self.calls == 1 else json.dumps({
                "keep": ["f_plate"],
                "visual_correction": "",
                "exact_text": "鄂A2V6J5",
                "uncertain": False,
            }, ensure_ascii=False)
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=content),
            )])

    recall = DummyRecall()
    verified, correction = asyncio.run(
        recall._verify_frames_with_grounding(
            ["f_plate"], query="车牌 2V6J5 前两个字是什么",
        )
    )

    assert recall.calls == 2
    assert verified == ["f_plate"]
    assert "鄂A2V6J5" in correction


def test_exact_text_verify_uses_dedicated_client_and_usage_kind():
    class FakeClient:
        def __init__(self, text):
            self.text = text
            self.calls = []

        async def call_chat(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            return self.text

    primary = FakeClient("primary must not be called")
    verifier = FakeClient(json.dumps({
        "keep": ["f_plate"],
        "visual_correction": "",
        "exact_text": "鄂A2V6J5",
        "uncertain": False,
    }, ensure_ascii=False))
    cfg = SimpleNamespace(
        recall_verify_max_frames=2,
        recall_verify_retries=0,
        recall_verify_retry_delay_sec=0.0,
        recall_verify_model="GPT-5.6 Luna",
        model="watcher-model",
    )
    recall = RecallAgent(
        cfg,
        SimpleNamespace(),
        primary,
        SimpleNamespace(),
        frame_store=SimpleNamespace(get_many=lambda _fids: [
            SimpleNamespace(
                frame_id="f_plate", ts=12.0, jpeg_b64=_tiny_jpeg_b64(),
            ),
        ]),
        model="primary-luna",
        verify_client=verifier,
        verify_model="GPT-5.6 Luna",
    )

    verified, correction = asyncio.run(
        recall._verify_frames_with_grounding(
            ["f_plate"], query="车牌 2V6J5 前两个字是什么",
        )
    )

    assert not primary.calls
    assert len(verifier.calls) == 1
    assert verifier.calls[0][1]["usage_kind"] == "recall_verify_frames"
    assert verified == ["f_plate"]
    assert "鄂A2V6J5" in correction


def test_exact_text_verify_failure_invalidates_context_guess():
    class DummyRecall(RecallAgent):
        def __init__(self):
            self.cfg = SimpleNamespace(
                recall_verify_max_frames=2,
                recall_verify_retries=1,
                recall_verify_retry_delay_sec=0.0,
            )
            self.frame_store = SimpleNamespace(get_many=lambda _fids: [
                SimpleNamespace(
                    frame_id="f_plate", ts=12.0,
                    jpeg_b64=_tiny_jpeg_b64(),
                ),
            ])
            self.screen_text_store = None
            self.recorder = None
            self.model = "GPT-5.6 Luna"
            self.calls = 0

        async def _create_chat_completion(self, *_args, **_kwargs):
            self.calls += 1
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="粤B，因为画面位于深圳"),
            )])

    recall = DummyRecall()
    verified, correction = asyncio.run(
        recall._verify_frames_with_grounding(
            ["f_plate"], query="车牌 2V6J5 前两个字是什么",
        )
    )

    assert recall.calls == 2
    assert verified is None
    assert "不能采信" in correction
    assert "地点" in correction


def test_reviewer_retries_only_transient_overload():
    class FlakyClient:
        name = "messages"
        model = "kimi/kimi-k3"

        def __init__(self):
            self.calls = 0
            self.last_error = ""

        async def call_chat(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                self.last_error = "429 EngineOverloadedError"
                return None
            self.last_error = ""
            return '{"actions": []}'

    class DummyReviewer(EntityReviewer):
        def __init__(self):
            self.cfg = SimpleNamespace(
                reviewer_overload_retries=2,
                reviewer_retry_backoff_sec=0.0,
                dump_raw=False,
            )
            self.client = FlakyClient()
            self.recorder = None
            self.llm_semaphore = asyncio.Semaphore(1)

    reviewer = DummyReviewer()
    raw = asyncio.run(reviewer._call_llm(
        [{"role": "user", "content": "review"}],
        max_tokens=128,
        temperature=0.2,
        kind="memory_reviewer",
        extra={},
    ))

    assert raw == '{"actions": []}'
    assert reviewer.client.calls == 2


def test_reviewer_source_target_action_normalizes_to_merge_entities():
    class DummyEntityReviewer(EntityReviewer):
        def __init__(self):
            pass

    reviewer = DummyEntityReviewer()
    action, repairs = reviewer._normalize_action({
        "target_id": "ent_screen_150640d7",
        "source_id": "ent_app_022493d6",
        "merged_attrs": {"visible_text": "行车辅助"},
        "reason": "两者是同一中控设置界面",
    })

    assert action["op"] == "merge_entities"
    assert action["winner_id"] == "ent_screen_150640d7"
    assert action["loser_ids"] == ["ent_app_022493d6"]
    assert repairs
    assert reviewer._action_target_ids(action) == [
        "ent_screen_150640d7",
        "ent_app_022493d6",
    ]


def test_role_mismatch_reviewer_action_is_not_persisted_as_revision():
    class DummyEventReviewer(EventReviewer):
        def __init__(self):
            pass

    reviewer = DummyEventReviewer()

    assert reviewer._should_persist_failed_action(
        {"op": "merge_entities"},
        "op='merge_entities' 不在本 reviewer 允许集 ['merge_micros']",
    ) is False


def test_entity_reviewer_owns_prune_entity():
    assert "prune_entity" in EntityReviewer.ALLOWED_OPS
    assert "prune_entity" not in EventReviewer.ALLOWED_OPS


def test_event_reviewer_omits_entity_visuals_and_uses_own_frame_budget():
    class MinimalEventReviewer(EventReviewer):
        def __init__(self):
            self.cfg = SimpleNamespace(reviewer_event_frames=80)
            self._event_gate_macro_count = 0
            self._event_gate_seen_macros = set()
            self.FRAME_BUDGET_OVERRIDE = max(
                8, int(self.cfg.reviewer_event_frames))

    reviewer = MinimalEventReviewer()

    assert reviewer.INCLUDE_ENTITY_VISUALS is False
    assert reviewer.FRAME_BUDGET_OVERRIDE == 80


def test_entity_reviewer_keeps_original_timeline_budget_contract():
    class MinimalEntityReviewer(EntityReviewer):
        def __init__(self):
            self.cfg = SimpleNamespace(reviewer_entity_frames=12)
            self.FRAME_BUDGET_OVERRIDE = max(
                4, int(self.cfg.reviewer_entity_frames))

    reviewer = MinimalEntityReviewer()

    assert reviewer.INCLUDE_ENTITY_VISUALS is True
    assert reviewer.FRAME_BUDGET_OVERRIDE == 12


def test_nested_event_gate_config_uses_strict_thresholds():
    cfg = build_config({
        "model": {
            "memory": {
                "reviewer": {
                    "event_gate": {
                        "enabled": True,
                        "sample_every_macros": 5,
                        "min_micro_count": 15,
                        "min_entity_state_changes": 80,
                        "min_distinct_entities": 25,
                        "min_asr_cues": 12,
                        "min_asr_chars": 600,
                    }
                }
            }
        }
    })

    assert cfg.reviewer_event_gate_enabled is True
    assert cfg.reviewer_event_sample_every_macros == 5
    assert cfg.reviewer_event_min_micro_count == 15
    assert cfg.reviewer_event_min_entity_state_changes == 80
    assert cfg.reviewer_event_min_distinct_entities == 25
    assert cfg.reviewer_event_min_asr_cues == 12
    assert cfg.reviewer_event_min_asr_chars == 600


def test_nested_reviewer_resilience_config_is_applied():
    cfg = build_config({
        "model": {
            "memory": {
                "reviewer": {
                    "max_concurrency": 1,
                    "single_endpoint_interval_sec": 0.75,
                    "overload_retries": 3,
                    "retry_backoff_sec": 1.25,
                    "event_frames": 80,
                    "base_urls": [
                        "https://example.test/u/one/v1/messages",
                        "https://example.test/u/two/v1/messages",
                    ],
                },
                "recall": {
                    "verify_retries": 2,
                    "verify_retry_delay_sec": 0.75,
                },
            },
        },
    })

    assert cfg.reviewer_max_concurrency == 1
    assert cfg.reviewer_single_endpoint_interval_sec == 0.75
    assert cfg.reviewer_overload_retries == 3
    assert cfg.reviewer_retry_backoff_sec == 1.25
    assert cfg.reviewer_event_frames == 80
    assert cfg.reviewer_base_urls == [
        "https://example.test/u/one/v1/messages",
        "https://example.test/u/two/v1/messages",
    ]
    assert cfg.recall_verify_retries == 2
    assert cfg.recall_verify_retry_delay_sec == 0.75


def test_reviewer_client_pool_builds_one_client_per_url():
    cfg = build_config({
        "model": {"memory": {"reviewer": {
            "provider": "custom",
            "base_urls": [
                "https://example.test/u/one/v1/messages",
                "https://example.test/u/two/v1/messages",
            ],
            "api_key": "whatever",
            "model": "kimi/kimi-k3",
        }}},
    })

    clients = HermesClientFactory(cfg).reviewer_clients()
    try:
        assert len(clients) == 2
        assert clients[0].endpoint.endswith("/u/one/v1/messages")
        assert clients[1].endpoint.endswith("/u/two/v1/messages")
        assert clients[0] is not clients[1]
    finally:
        async def close_clients():
            await asyncio.gather(*[client.aclose() for client in clients])
        asyncio.run(close_clients())


def test_single_endpoint_limiter_serializes_and_delays_next_request():
    async def run():
        limiter = ReviewerEndpointLimiter(
            max_concurrency=1, min_start_interval_sec=0.03)
        entered = []

        async def worker():
            async with limiter.slot():
                entered.append(asyncio.get_running_loop().time())
                await asyncio.sleep(0.01)

        await asyncio.gather(worker(), worker())
        return entered

    entered = asyncio.run(run())

    assert len(entered) == 2
    assert entered[1] - entered[0] >= 0.035
