"""Regression tests for stale entity/micro leakage into multimodal Recall."""

import asyncio
import types

from agent.multimodal._config import Config
from agent.multimodal._memory import Entity, MemoryStore, MicroEvent
from agent.multimodal._workers import MemoryToolBox
from agent.multimodal._workers import RecallWorker


def _entity(eid: str, name: str, *, attrs=None, last_seen: float = 20.0) -> Entity:
    return Entity(
        id=eid,
        name=name,
        type="OBJECT",
        attributes=attrs or {},
        aliases=[],
        first_seen=10.0,
        last_seen=last_seen,
        seen_count=1,
        representative_frame_id="f_4bde6cc4bc",
        updated_at=1.0,
    )


def _micro(mid: str, description: str, *, superseded_by=None) -> MicroEvent:
    return MicroEvent(
        id=mid,
        t_start=18.0,
        t_end=20.0,
        description=description,
        subject="用户",
        object="金典牛奶纸盒",
        action="举到镜头前展示",
        macro_id=None,
        facts_keys=[],
        frame_ids=["f_4bde6cc4bc"],
        created_at=1.0,
        superseded_by=superseded_by,
    )


def test_merged_loser_never_leaks_and_direct_id_redirects_to_winner():
    store = MemoryStore(Config())
    winner_id = "ent_object_9312452e"
    loser_id = "ent_object_1a38b4fd"

    store.upsert_entity(_entity(
        winner_id,
        "金典牛奶纸盒包装",
        attrs={
            "brand": "金典(SATINE)",
            "container_type": "硬质利乐砖纸盒包装",
            "net_content": "250mL",
        },
        last_seen=30.0,
    ))
    store.upsert_entity(_entity(
        loser_id,
        "白色软管状容器",
        attrs={"capacity": "200ml", "shape": "软管"},
    ))
    store.insert_micro(_micro(
        "micro_milk",
        "画面清楚显示白色金典牛奶纸盒，带金典品牌标识和利乐砖包装。",
    ))

    # Winner and loser sharing the same primary-key links reproduced the old
    # UPDATE OR IGNORE failure: loser rows used to remain after merge.
    for eid in (winner_id, loser_id):
        store.link_entity_event(eid, "micro_milk", 20.0)
        store.link_entity_frame(eid, "f_4bde6cc4bc", "micro_milk", 19.0)

    assert store.merge_entities(
        [loser_id], winner_id, reviewer_round=1,
        reason="画面确认为硬质利乐砖牛奶盒而非软管", t_observed=20.0,
    ) is True

    canonical, chain = store.resolve_entity(loser_id)
    assert canonical is not None
    assert canonical.id == winner_id
    assert chain == [loser_id, winner_id]

    recent_ids = [e.id for e in store.get_recent_entities(ask_ts=100.0)]
    assert recent_ids == [winner_id]
    # The old description can remain an alias for reference resolution, but the
    # returned row must always be the authoritative winner.
    keyword_ids = [
        e.id for e in store.search_entity_by_keyword(
            "白色软管 200ml", ask_ts=100.0, top_k=10)
    ]
    assert loser_id not in keyword_ids
    assert winner_id in keyword_ids

    with store._connect() as c:
        loser_event_n = c.execute(
            "SELECT COUNT(*) AS n FROM entity_event WHERE entity_id=?",
            (loser_id,),
        ).fetchone()["n"]
        loser_frame_n = c.execute(
            "SELECT COUNT(*) AS n FROM entity_frame WHERE entity_id=?",
            (loser_id,),
        ).fetchone()["n"]
        winner_event_n = c.execute(
            "SELECT COUNT(*) AS n FROM entity_event WHERE entity_id=?",
            (winner_id,),
        ).fetchone()["n"]
        winner_frame_n = c.execute(
            "SELECT COUNT(*) AS n FROM entity_frame WHERE entity_id=?",
            (winner_id,),
        ).fetchone()["n"]
    assert (loser_event_n, loser_frame_n) == (0, 0)
    assert (winner_event_n, winner_frame_n) == (1, 1)

    obs = MemoryToolBox(store).call(
        "get_entity_context", {"entity_id": loser_id}, ask_ts=100.0)
    assert f"{loser_id} -> {winner_id}" in obs
    assert "canonical entity (authoritative current state)" in obs
    assert "金典牛奶纸盒包装" in obs
    assert "brand=金典(SATINE)" in obs
    assert "硬质利乐砖纸盒包装" in obs

    # A repeated Reviewer action must be a successful no-op, not inflate the
    # winner's seen_count/revision_count a second time.
    before = store.peek_entity(winner_id)
    assert store.merge_entities([loser_id], winner_id, t_observed=21.0) is True
    after = store.peek_entity(winner_id)
    assert before is not None and after is not None
    assert after.seen_count == before.seen_count
    assert after.revision_count == before.revision_count


def test_superseded_micro_is_filtered_from_all_recall_reads():
    store = MemoryStore(Config())
    store.insert_micro(_micro(
        "micro_old_tube",
        "误识别为白色软管和200ml日用品。",
        superseded_by="micro_milk_corrected",
    ))
    store.insert_micro(_micro(
        "micro_milk_corrected",
        "修订后确认是金典牛奶利乐砖纸盒。",
    ))

    by_time = store.get_micro_by_time(0.0, 100.0, ask_ts=100.0)
    by_keyword = store.search_micro_by_keyword(
        "白色软管 200ml", ask_ts=100.0, top_k=10)

    assert [m.id for m in by_time] == ["micro_milk_corrected"]
    assert by_keyword == []



def test_startup_repairs_legacy_stranded_loser_links(tmp_path):
    db_path = tmp_path / "legacy-merge.sqlite"
    cfg = Config(mem_db_path=str(db_path))
    store = MemoryStore(cfg)
    winner_id = "ent_winner"
    loser_id = "ent_loser"
    store.upsert_entity(_entity(winner_id, "金典牛奶盒", last_seen=30.0))
    store.upsert_entity(_entity(loser_id, "白色软管", last_seen=20.0))
    store.insert_micro(_micro("micro_milk", "金典牛奶盒"))

    # Simulate a DB produced by the buggy merge implementation: both links
    # survived and loser is already soft-merged. Remove the migration marker so
    # the next startup performs the compatibility repair.
    for eid in (winner_id, loser_id):
        store.link_entity_event(eid, "micro_milk", 20.0)
        store.link_entity_frame(eid, "f_4bde6cc4bc", "micro_milk", 19.0)
    with store._connect() as c:
        c.execute(
            "UPDATE entities SET merged_into=? WHERE id=?",
            (winner_id, loser_id),
        )
        c.execute("DELETE FROM meta WHERE key='entity_merge_links_v2'")

    repaired = MemoryStore(cfg)
    with repaired._connect() as c:
        assert c.execute(
            "SELECT COUNT(*) AS n FROM entity_event WHERE entity_id=?",
            (loser_id,),
        ).fetchone()["n"] == 0
        assert c.execute(
            "SELECT COUNT(*) AS n FROM entity_frame WHERE entity_id=?",
            (loser_id,),
        ).fetchone()["n"] == 0
        assert c.execute(
            "SELECT COUNT(*) AS n FROM entity_frame WHERE entity_id=?",
            (winner_id,),
        ).fetchone()["n"] == 1


def test_visual_verifier_returns_identity_correction_with_kept_frames():
    class _Frame:
        frame_id = "f_4bde6cc4bc"
        ts = 19.0
        jpeg_b64 = "ZmFrZQ=="

    class _FrameStore:
        @staticmethod
        def get_many(_fids):
            return [_Frame()]

    class _Completions:
        @staticmethod
        async def create(**_kwargs):
            msg = types.SimpleNamespace(content=(
                '{"keep":["f_4bde6cc4bc"],'
                '"visual_correction":"画面显示金典牛奶利乐砖，并非白色软管。"}'
            ))
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=msg)])

    worker = types.SimpleNamespace(
        cfg=types.SimpleNamespace(
            recall_verify_max_frames=8,
            model="stub-model",
        ),
        model="stub-model",
        frame_store=_FrameStore(),
        client=types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=_Completions())),
        recorder=None,
    )
    worker._completion_controls = lambda *, max_tokens, temperature: {
        "max_tokens": max_tokens, "temperature": temperature}

    async def _create_chat_completion(msgs, *, max_tokens=256, temperature=0.1,
                                      enable_thinking=False, channel_tag=""):
        return await worker.client.chat.completions.create(
            model=worker.model, messages=msgs, max_tokens=max_tokens,
            temperature=temperature)
    worker._create_chat_completion = _create_chat_completion
    method = RecallWorker._verify_frames_with_grounding.__get__(
        worker, type(worker))
    kept, correction = asyncio.run(method(
        ["f_4bde6cc4bc"], query="白色软管是什么"))

    assert kept == ["f_4bde6cc4bc"]
    assert "金典牛奶" in correction
    assert "并非白色软管" in correction
