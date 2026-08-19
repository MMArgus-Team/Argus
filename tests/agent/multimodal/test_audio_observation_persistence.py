import asyncio
import time

import numpy as np

from agent.multimodal._config import Config
from agent.multimodal._embedding import encode_vector
from agent.multimodal._memory import (
    ConversationLog,
    Entity,
    Frame,
    FrameStore,
    MacroEvent,
    MemoryStore,
    MicroEvent,
    ScreenTextBlock,
    ScreenTextRecord,
    ScreenTextStore,
    Turn,
)
from agent.multimodal._workers import MemoryToolBox


def test_audio_observations_survive_conversation_trim_and_restart(tmp_path):
    db_path = tmp_path / "audio-memory.sqlite"
    cfg = Config(mem_db_path=str(db_path))
    store = MemoryStore(cfg)
    conversation = ConversationLog(
        max_chars=24,
        min_turns=1,
        max_bg_obs=20,
        audio_store=store,
    )

    async def _append():
        await conversation.append(
            "system", "alpha durable transcript", kind="audio_observation",
            rel_ts=10.0, speaker="SPEAKER_00",
        )
        await conversation.append(
            "system", "beta later transcript", kind="audio_observation",
            rel_ts=20.0, speaker="SPEAKER_01",
        )
        await conversation.append(
            "system", "gamma newest transcript", kind="audio_observation",
            rel_ts=30.0,
        )

    asyncio.run(_append())

    # The bounded prompt log has already discarded the earliest transcript.
    assert all("alpha" not in t.content for t in conversation.snapshot())

    # Reopening the session DB proves retrieval no longer depends on RAM state.
    reopened = MemoryStore(cfg)
    toolbox = MemoryToolBox(reopened, conversation=ConversationLog())
    rows = toolbox._search_audio("alpha", ask_ts=100.0, top_k=8)
    assert [t.content for t in rows] == ["alpha durable transcript"]
    assert rows[0].speaker == "SPEAKER_00"

    # Snapshot isolation still prevents future ASR from leaking into a query.
    assert toolbox._search_audio("beta", ask_ts=15.0, top_k=8) == []


def test_audio_observations_are_cleared_by_explicit_memory_reset(tmp_path):
    cfg = Config(mem_db_path=str(tmp_path / "reset.sqlite"))
    store = MemoryStore(cfg)
    conversation = ConversationLog(audio_store=store)

    asyncio.run(conversation.append(
        "system", "persistent before reset", kind="audio_observation",
        rel_ts=5.0,
    ))
    assert len(store.get_audio_observations(ask_ts=10.0)) == 1

    deleted = store.reset()
    assert deleted["audio_observations"] == 1
    assert store.get_audio_observations(ask_ts=10.0) == []


def test_writer_audio_cursor_keeps_complete_same_timestamp_batch(tmp_path):
    cfg = Config(mem_db_path=str(tmp_path / "cursor.sqlite"))
    store = MemoryStore(cfg)
    for text, ts in (("first", 10.0), ("second", 10.0), ("future", 30.0)):
        store.insert_audio_observation(Turn(
            role="system", content=text, wall_ts=time.time(),
            kind="audio_observation", rel_ts=ts,
        ))

    batch, cursor = store.get_audio_observations_after_id(0, ask_ts=20.0)
    assert [turn.content for turn in batch] == ["first", "second"]

    # A failed Writer wake does not persist this cursor. Reading from the old
    # cursor therefore retries the exact same complete interval.
    retry, retry_cursor = store.get_audio_observations_after_id(0, ask_ts=20.0)
    assert [turn.content for turn in retry] == ["first", "second"]
    assert retry_cursor == cursor

    # Once committed, only newly visible rows are consumed.
    store.set_meta("writer_asr_cursor_id", str(cursor))
    later, later_cursor = store.get_audio_observations_after_id(cursor, ask_ts=40.0)
    assert [turn.content for turn in later] == ["future"]
    assert later_cursor > cursor


def test_search_audio_returns_cross_modal_temporal_evidence(tmp_path):
    db_path = tmp_path / "evidence.sqlite"
    cfg = Config(mem_db_path=str(db_path), frame_store_max=10)
    store = MemoryStore(cfg)
    frame_store = FrameStore(cfg)
    screen_store = ScreenTextStore(cfg)

    frame_id = frame_store.maybe_store(
        Frame(ts=20.0, jpeg_b64="ZmFrZS1qcGVn", source_type="screen"),
        micro_id="micro_hat", note="路人经过画面",
    )
    assert frame_id
    store.insert_micro(MicroEvent(
        id="micro_hat", t_start=15.0, t_end=25.0,
        description="讲到保护区时，一名戴红色帽子的路人经过。",
        frame_ids=[frame_id],
    ))
    store.insert_macro(MacroEvent(
        id="macro_reserve", t_start=10.0, t_end=30.0,
        label="保护区介绍", summary="演讲人介绍美国保护区。",
    ))
    entity, _ = store.upsert_entity(Entity(
        id="ent_passerby", name="戴红帽的路人", type="PERSON",
        attributes={"hat_color": "red"}, first_seen=20.0, last_seen=20.0,
    ))
    store.link_entity_event(entity.id, "micro_hat", t_observed=20.0)
    screen_store.upsert_frame_text(ScreenTextRecord(
        frame_id=frame_id, t_observed=20.0, app="video",
        window_title="保护区讲座",
        ocr_blocks=[ScreenTextBlock(text="美国国家保护区")],
        raw_text="美国国家保护区", source="rapidocr",
    ))
    for text, ts in (("接下来介绍美国的保护区", 18.0),
                     ("保护区分布在多个州", 20.0),
                     ("这里需要长期保护", 22.0)):
        store.insert_audio_observation(Turn(
            role="system", content=text, wall_ts=time.time(),
            kind="audio_observation", rel_ts=ts,
        ))

    toolbox = MemoryToolBox(
        store, frame_store=frame_store, screen_text_store=screen_store)
    result = toolbox.call(
        "search_audio", {"query": "美国 保护区", "top_k": 4}, ask_ts=40.0)

    assert "ASR_HIT" in result
    assert "ASR_CONTEXT" in result and "长期保护" in result
    assert "MICRO_CONTEXT" in result and "micro_hat" in result
    assert "MACRO_CONTEXT" in result and "macro_reserve" in result
    assert "NEARBY_KEYFRAMES" in result and frame_id in result
    assert "OCR_NEARBY" in result and "美国国家保护区" in result
    assert "RELATED_ENTITIES" in result and "hat_color=red" in result


def test_keyframe_survives_lru_eviction_and_restart_with_real_timestamp(tmp_path):
    cfg = Config(
        mem_db_path=str(tmp_path / "frames.sqlite"), frame_store_max=1)
    frame_store = FrameStore(cfg)
    old_id = frame_store.maybe_store(
        Frame(ts=5.5, jpeg_b64="b2xkLWpwZWc=", source_type="camera"),
        micro_id="micro_old", note="old persisted frame",
    )
    frame_store.maybe_store(
        Frame(ts=2000.0, jpeg_b64="bmV3LWpwZWc=", source_type="screen"),
        micro_id="micro_new",
    )
    assert old_id not in frame_store._frames

    reopened = FrameStore(cfg)
    restored = reopened.get(old_id)
    assert restored is not None
    assert restored.ts == 5.5
    assert restored.micro_id == "micro_old"
    assert restored.source_type == "camera"
    assert reopened.nearby_index(5.5, window_sec=1.0)[0]["frame_id"] == old_id


def test_unbounded_frame_vector_search_keeps_early_history(tmp_path):
    cfg = Config(mem_db_path=str(tmp_path / "vectors.sqlite"))
    store = MemoryStore(cfg)
    store.mm_embedding_client = type("EnabledMM", (), {"enabled": True})()
    with store._lock, store._connect() as conn:
        for idx in range(650):
            vec = (np.array([1.0, 0.0], dtype=np.float32) if idx == 0
                   else np.array([0.0, 1.0], dtype=np.float32))
            conn.execute(
                """INSERT INTO frame_embeddings
                   (frame_id,t_observed,micro_id,embedding,created_at)
                   VALUES (?,?,?,?,?)""",
                (f"f_{idx:010d}", float(idx), None, encode_vector(vec), time.time()),
            )

    query = np.array([1.0, 0.0], dtype=np.float32)
    assert store.vector_search_frames(
        query, ask_ts=1000.0, top_k=1, pool_cap=600)[0]["frame_id"] != "f_0000000000"
    assert store.vector_search_frames(
        query, ask_ts=1000.0, top_k=1, pool_cap=0)[0]["frame_id"] == "f_0000000000"
