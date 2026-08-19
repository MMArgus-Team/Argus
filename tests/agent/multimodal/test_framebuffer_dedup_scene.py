# -*- coding: utf-8 -*-
"""FrameBuffer entry-dedup + SceneDhashController (scene → dHash threshold).

Covers the抽帧-moved-upstream refactor:
  * FrameBuffer.push drops near-identical frames (dHash < cutoff) at entry.
  * A larger cutoff is more aggressive and therefore cannot retain more frames.
  * A valid dHash value of zero participates in dedup; only decode failure is
    fail-open.
  * set_dhash_threshold clamps to [min,max].
  * sample_uniform picks evenly across the recent window.
  * SceneDhashController maps a model-classified scene to coupled pacing/dHash
    policy; on failure/timeout it leaves the threshold unchanged.
"""
import asyncio
import base64
import json
import sqlite3
import threading
from io import BytesIO
from types import SimpleNamespace

import pytest

from agent.multimodal._memory import (
    FrameBuffer,
    Frame,
    FrameStore,
    _compute_dhash,
    _hamming,
)
from agent.multimodal._config import Config


def _grad(seed: int) -> str:
    from PIL import Image
    img = Image.new("L", (32, 32))
    px = img.load()
    for y in range(32):
        for x in range(32):
            px[x, y] = (x * 8 + y * 4 + seed * 37) % 256
    b = BytesIO()
    img.convert("RGB").save(b, format="JPEG")
    return base64.b64encode(b.getvalue()).decode()


def _solid(level: int = 255) -> str:
    """A valid image whose mathematical dHash is exactly zero."""
    from PIL import Image
    img = Image.new("L", (32, 32), color=level)
    b = BytesIO()
    img.convert("RGB").save(b, format="JPEG")
    return base64.b64encode(b.getvalue()).decode()


def _screen_with_popup(size: int) -> str:
    """Mostly-static desktop with a localized top-right notification growing.

    The sequence exercises the real weakness/intent of a global dHash cutoff:
    dynamic policy must preserve more local changes, while static policy is
    deliberately allowed to coalesce more of them.
    """
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (1024, 640), color="white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((100, 100, 700, 500), outline="gray", width=3)
    if size > 0:
        draw.rectangle(
            (1024 - size - 16, 16, 1024 - 16, 16 + size), fill="black")
    b = BytesIO()
    img.save(b, format="JPEG", quality=90)
    return base64.b64encode(b.getvalue()).decode()


def _cfg(**over):
    base = dict(buffer_seconds=60, buffer_capture_fps=2,
               framebuffer_dhash_threshold_init=6,
               framebuffer_dhash_threshold_min=2,
               framebuffer_dhash_threshold_max=20)
    base.update(over)
    return SimpleNamespace(**base)


# ── FrameBuffer entry dedup ────────────────────────────────────────────────
def test_identical_frames_deduped_at_entry():
    fb = FrameBuffer(_cfg())
    same = _grad(0)
    for i in range(10):
        fb.push(Frame(ts=float(i), jpeg_b64=same))
    assert fb.size == 1


def test_distinct_frames_all_kept():
    fb = FrameBuffer(_cfg())
    for i in range(6):
        fb.push(Frame(ts=float(i), jpeg_b64=_grad(i * 4)))  # ~26 hamming apart
    assert fb.size == 6


def test_valid_zero_dhash_is_deduped():
    # A flat, valid image genuinely hashes to integer zero. Zero is data, not an
    # error sentinel, so repeated flat frames must still be deduplicated.
    same = _solid()
    assert _compute_dhash(same) == 0
    fb = FrameBuffer(_cfg())
    fb.push(Frame(ts=0.0, jpeg_b64=same))
    fb.push(Frame(ts=1.0, jpeg_b64=same))
    assert fb.size == 1


def test_decode_failure_is_none_and_fail_open():
    # Decode failure is represented separately from the valid hash value zero.
    # Uncomparable frames are retained rather than risking false deletion.
    assert _compute_dhash("not-a-real-jpeg") is None
    fb = FrameBuffer(_cfg())
    fb.push(Frame(ts=0.0, jpeg_b64="not-a-real-jpeg"))
    fb.push(Frame(ts=1.0, jpeg_b64="also-garbage"))
    assert fb.size == 2


def test_threshold_clamped():
    fb = FrameBuffer(_cfg())
    assert fb.set_dhash_threshold(999) == 20   # clamp to max
    assert fb.set_dhash_threshold(-5) == 2     # clamp to min
    assert fb.set_dhash_threshold(9) == 9
    assert fb.dhash_threshold == 9


def test_threshold_change_affects_future_pushes():
    fb = FrameBuffer(_cfg(framebuffer_dhash_threshold_init=2))
    # With a tiny threshold, mildly-different frames are kept.
    fb.push(Frame(ts=0.0, jpeg_b64=_grad(0)))
    fb.push(Frame(ts=1.0, jpeg_b64=_grad(1)))   # ~18 hamming ≥ 2 → kept
    assert fb.size == 2


def test_higher_cutoff_never_retains_more_frames():
    def retained(cutoff: int) -> int:
        fb = FrameBuffer(_cfg(framebuffer_dhash_threshold_init=cutoff))
        for i in range(12):
            fb.push(Frame(ts=float(i), jpeg_b64=_grad(i)))
        return fb.size

    low = retained(2)
    high = retained(20)
    assert high <= low
    # Fixture sanity: this sequence has distances that straddle the cutoffs, so
    # the test proves real policy behavior rather than a vacuous equality.
    assert high < low


def test_static_cutoff_coalesces_more_local_changes_than_dynamic_cutoff():
    from agent.multimodal.scene_dhash import pace_from_threshold

    frames = [_screen_with_popup(n) for n in
              (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160)]

    def retained(cutoff: int) -> int:
        fb = FrameBuffer(_cfg(framebuffer_dhash_threshold_init=cutoff))
        for i, jpeg in enumerate(frames):
            fb.push(Frame(ts=float(i), jpeg_b64=jpeg))
        return fb.size

    # Conventional dHash cutoff contract: static scenes use a HIGH cutoff to
    # merge more minor drift; dynamic/live scenes use a LOW cutoff to preserve
    # localized motion. A local popup is therefore represented more densely in
    # dynamic mode. Static mode is not required to retain every growth step.
    static_cutoff = 20
    dynamic_cutoff = 2
    assert pace_from_threshold(static_cutoff) == "slow"
    assert pace_from_threshold(dynamic_cutoff) == "live"
    assert retained(static_cutoff) < retained(dynamic_cutoff)

    # Sanity-check that this is genuinely a small localized visual change, not
    # an accidental duplicate fixture.
    first = _compute_dhash(frames[0])
    changed = _compute_dhash(frames[-1])
    assert first is not None and changed is not None
    assert 0 < _hamming(first, changed) < static_cutoff


@pytest.mark.parametrize("cutoff", [2, 6, 20])
def test_push_and_push_live_have_identical_dedup_decisions(cutoff):
    sequence = [
        _solid(), _solid(),
        _grad(0), _grad(0),
        _grad(1), _grad(2), _grad(2),
        _screen_with_popup(160),
    ]
    direct = FrameBuffer(_cfg(framebuffer_dhash_threshold_init=cutoff))
    live = FrameBuffer(_cfg(framebuffer_dhash_threshold_init=cutoff))
    direct_stored = []
    live_stored = []

    for i, jpeg in enumerate(sequence):
        before = direct.size
        direct.push(Frame(ts=float(i), jpeg_b64=jpeg, source_type="screen"))
        direct_stored.append(direct.size > before)
        live_stored.append(bool(live.push_live(
            jpeg, source_type="screen")["stored"]))

    assert live_stored == direct_stored
    assert live.size == direct.size


def test_sample_uniform_even_and_capped():
    fb = FrameBuffer(_cfg())
    for i in range(20):
        fb.push(Frame(ts=float(i), jpeg_b64=_grad(i * 4)))
    picked = fb.sample_uniform(window_s=100, n=3)
    assert len(picked) == 3
    # first + last of the window are included, middle is roughly centered.
    assert picked[0].ts == 0.0
    assert picked[-1].ts == 19.0


def test_push_live_stamps_server_ts_and_anchors():
    # push_live owns the mono/wall epoch + last-push marker (no longer attached
    # externally by the gateway) and stamps a server-authoritative monotonic ts.
    fb = FrameBuffer(_cfg())
    assert fb.last_push_wall is None and fb.wall_epoch is None
    r1 = fb.push_live(_grad(0))
    assert r1["stored"] is True
    assert r1["ts"] >= 0.0            # first frame ts ~0 (relative to epoch)
    assert fb.wall_epoch is not None and fb.last_push_wall is not None
    r2 = fb.push_live(_grad(40))      # distinct → stored, ts strictly increasing
    assert r2["stored"] is True
    assert r2["ts"] > r1["ts"]


def test_push_live_dedups_like_push():
    fb = FrameBuffer(_cfg())
    same = _grad(0)
    first = fb.push_live(same)
    assert first["stored"] is True
    assert first["monitor_stored"] is True
    # near-identical next frame → dropped at entry (stored False), size stays 1.
    second = fb.push_live(same)
    assert second["stored"] is False
    assert second["monitor_stored"] is True
    assert fb.size == 1
    # Monitor bypasses dHash and receives both server-accepted 2fps frames.
    assert fb.monitor_size == 2
    raw = fb.monitor_all_after(-1.0)
    assert len(raw) == 2
    assert raw[-1].ts == fb.monitor_latest_ts


def test_live_frame_carries_its_own_source_across_source_switch():
    fb = FrameBuffer(_cfg())
    fb.push_live(_grad(0), source_type="camera")
    camera_generation = fb.source_generation
    fb.push_live(_grad(40), source_type="screen")
    frames = fb.latest(10)
    assert [f.source_type for f in frames] == ["camera", "screen"]
    assert fb.current_source_type == "screen"
    assert fb.source_generation > camera_generation
    # The short monitor ring is source-local; it must not evaluate stale camera
    # frames as the opening window of a new screen-share source.
    assert [f.source_type for f in fb.monitor_all_after(-1.0)] == ["screen"]


def test_frame_store_indexes_camera_and_screen_memory(tmp_path):
    db_path = tmp_path / "all-scenes.sqlite"
    cfg = Config(mem_db_path=str(db_path))
    store = FrameStore(cfg)
    camera_id = store.maybe_store(
        Frame(ts=1.0, jpeg_b64=_grad(0), source_type="camera"),
        micro_id="micro-camera", note="camera key frame")
    screen_id = store.maybe_store(
        Frame(ts=2.0, jpeg_b64=_grad(40), source_type="screen"),
        micro_id="micro-screen", note="screen key frame")

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT frame_id,source_type,micro_id FROM memory_frames "
            "ORDER BY t_observed"
        ).fetchall()
    assert rows == [
        (camera_id, "camera", "micro-camera"),
        (screen_id, "screen", "micro-screen"),
    ]


def test_ocr_frame_filter_remains_screen_only():
    from agent.multimodal._workers import ScreenOCRWorker
    worker = object.__new__(ScreenOCRWorker)
    worker.buf = FrameBuffer(_cfg())
    assert not worker._frame_is_screen(
        Frame(ts=1.0, jpeg_b64="x", source_type="camera"))
    assert worker._frame_is_screen(
        Frame(ts=2.0, jpeg_b64="x", source_type="screen"))


def test_sample_uniform_fewer_than_n():
    fb = FrameBuffer(_cfg())
    fb.push(Frame(ts=0.0, jpeg_b64=_grad(0)))
    fb.push(Frame(ts=1.0, jpeg_b64=_grad(40)))
    assert len(fb.sample_uniform(window_s=100, n=5)) == 2


def test_clear_resets_dedup_state():
    fb = FrameBuffer(_cfg())
    same = _grad(0)
    fb.push(Frame(ts=0.0, jpeg_b64=same))
    generation = fb.source_generation
    fb.clear()
    assert fb.source_generation == generation + 1
    # After clear, the same image is NOT considered a duplicate of the cleared one.
    fb.push(Frame(ts=1.0, jpeg_b64=same))
    assert fb.size == 1
    assert fb.monitor_size == 1


# ── SceneDhashController ────────────────────────────────────────────────────
class _FakeVisionClient:
    """Returns a canned assistant message content."""
    def __init__(self, content):
        self._content = content
        self.calls = 0

        class _Comp:
            async def create(_self, **kw):
                self.calls += 1
                msg = SimpleNamespace(content=self._content)
                choice = SimpleNamespace(message=msg)
                return SimpleNamespace(choices=[choice])
        self.chat = SimpleNamespace(completions=_Comp())


def _scene_cfg(**over):
    base = dict(scene_probe_interval_s=20, scene_probe_window_s=20,
               scene_probe_frames=3, scene_probe_maxside=64,
               scene_probe_quality=40, scene_probe_use_llm=True,
               scene_probe_timeout_s=5.0)
    base.update(over)
    return SimpleNamespace(**base)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_scene_controller_derives_threshold_from_scene():
    from agent.multimodal.scene_dhash import SceneDhashController
    fb = FrameBuffer(_cfg())
    for i in range(6):
        fb.push(Frame(ts=float(i), jpeg_b64=_grad(i * 4)))
    client = _FakeVisionClient('{"scene": "会议"}')
    ctrl = SceneDhashController(_scene_cfg(), fb, client, "m", threading.Event())
    _run(ctrl._probe_once())
    assert client.calls == 1
    assert fb.dhash_threshold == 11
    # scene=会议 → current slow pacing stored for set_live_watcher/loop.
    sc = fb.current_scene
    assert sc and sc["pace"] == "slow"
    assert sc["ttl_sec"] == 200 and sc["target_frames"] == 100


@pytest.mark.parametrize(("scene", "pace", "expected_threshold"), [
    ("会议", "slow", 11),
    ("影视", "medium", 7),
    ("体育", "fast", 4),
    ("直播", "live", 2),
])
def test_scene_controller_ignores_contradictory_model_threshold(
        scene, pace, expected_threshold):
    from agent.multimodal.scene_dhash import SceneDhashController
    fb = FrameBuffer(_cfg())
    fb.push(Frame(ts=0.0, jpeg_b64=_grad(0)))
    # Old deployments may still return the retired threshold field. A valid
    # scene is authoritative so e.g. live+11 cannot silently starve consumers.
    client = _FakeVisionClient(
        json.dumps({"scene": scene, "dhash_threshold": 11}, ensure_ascii=False))
    ctrl = SceneDhashController(_scene_cfg(), fb, client, "m", threading.Event())
    _run(ctrl._probe_once())
    assert fb.dhash_threshold == expected_threshold
    assert fb.current_scene["pace"] == pace


def test_scene_controller_legacy_threshold_fallback_is_clamped():
    from agent.multimodal.scene_dhash import SceneDhashController
    fb = FrameBuffer(_cfg())
    fb.push(Frame(ts=0.0, jpeg_b64=_grad(0)))
    client = _FakeVisionClient('{"dhash_threshold": 999}')
    ctrl = SceneDhashController(_scene_cfg(), fb, client, "m", threading.Event())
    _run(ctrl._probe_once())
    assert fb.dhash_threshold == 20
    assert fb.current_scene["pace"] == "slow"


def test_scene_controller_keeps_threshold_on_bad_output():
    from agent.multimodal.scene_dhash import SceneDhashController
    fb = FrameBuffer(_cfg(framebuffer_dhash_threshold_init=7))
    fb.push(Frame(ts=0.0, jpeg_b64=_grad(0)))
    client = _FakeVisionClient("这不是JSON随便说点啥")
    ctrl = SceneDhashController(_scene_cfg(), fb, client, "m", threading.Event())
    _run(ctrl._probe_once())
    assert fb.dhash_threshold == 7  # unchanged on parse failure


def test_scene_controller_parses_fenced_json():
    from agent.multimodal.scene_dhash import SceneDhashController
    scene, thr = SceneDhashController._parse(
        '```json\n{"scene":"编程","dhash_threshold":12}\n```')
    assert scene == "编程"
    assert thr == 12


def test_scene_controller_no_frames_noop():
    from agent.multimodal.scene_dhash import SceneDhashController
    fb = FrameBuffer(_cfg())  # empty buffer
    client = _FakeVisionClient('{"scene":"编程","dhash_threshold":12}')
    ctrl = SceneDhashController(_scene_cfg(), fb, client, "m", threading.Event())
    _run(ctrl._probe_once())
    assert client.calls == 0  # never called the model — nothing to probe


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
