from agent.multimodal.scene_dhash import (
    SceneDhashController,
    pace_from_scene,
)


def test_scene_prompt_accepts_english_labels_for_internal_pace():
    assert pace_from_scene("meeting") == "slow"
    assert pace_from_scene("real_time_competition") == "live"
    assert pace_from_scene("driving") == "fast"


def test_scene_parse_maps_english_label_to_legacy_internal_label():
    scene, threshold = SceneDhashController._parse('{"scene": "office"}')

    assert scene == "办公"
    assert threshold is None
    assert pace_from_scene(scene) == "slow"
