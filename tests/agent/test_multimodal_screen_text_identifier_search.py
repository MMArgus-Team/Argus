from agent.multimodal._config import Config
from agent.multimodal._memory import (
    ScreenTextBlock,
    ScreenTextRecord,
    ScreenTextStore,
    mm_identifier_variants,
)


def test_identifier_variants_cover_vehicle_plate_suffixes():
    variants = mm_identifier_variants("粤B70463D 车辆 业务 品牌")

    assert variants[:4] == ["b70463d", "70463d", "70463", "b70463"]


def test_screen_text_search_matches_partial_plate_ocr(tmp_path):
    store = ScreenTextStore(
        Config(), db_path=str(tmp_path / "screen-text.sqlite"))
    store.upsert_frame_text(ScreenTextRecord(
        frame_id="f_bus",
        t_observed=89.0,
        raw_text="东部公交 70463D",
        ocr_blocks=[ScreenTextBlock(text="东部公交 70463D")],
        source="ocr:rapidocr",
    ))
    store.upsert_frame_text(ScreenTextRecord(
        frame_id="f_taxi",
        t_observed=150.0,
        raw_text="蓝白涂装出租车 车辆",
        ocr_blocks=[ScreenTextBlock(text="蓝白涂装出租车 车辆")],
        source="writer_vlm",
    ))

    hits = store.search("粤B70463D 车辆 业务 品牌", ask_ts=200.0, limit=3)

    assert [hit.frame_id for hit in hits][:1] == ["f_bus"]
