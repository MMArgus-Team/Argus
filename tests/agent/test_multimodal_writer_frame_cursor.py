import base64
import io
from types import SimpleNamespace

from PIL import Image

from agent.multimodal._memory import Frame, FrameBuffer
from agent.multimodal._workers import MemoryWriter


class _MetaStore:
    def __init__(self):
        self.values = {}

    def set_meta(self, key, value):
        self.values[key] = value
        return True


class _PendingBuffer:
    def __init__(self, frames):
        self.frames = list(frames)

    def writer_all_after(self, cursor):
        return [frame for frame in self.frames if frame.ts > cursor]

    @property
    def latest_ts(self):
        return self.frames[-1].ts if self.frames else None


def _writer_for_cursor_test(frames, *, cursor=-1.0, max_frames=20):
    writer = MemoryWriter.__new__(MemoryWriter)
    writer.buf = _PendingBuffer(frames)
    writer.cfg = SimpleNamespace(writer_recent_frames=max_frames)
    writer.mem = _MetaStore()
    writer._frame_cursor_meta_key = "writer_frame_cursor_ts"
    writer._frame_cursor_ts = cursor
    return writer


def test_frame_buffer_writer_snapshot_merges_raw_and_sparse_without_duplicates():
    cfg = SimpleNamespace(
        buffer_seconds=1800,
        buffer_capture_fps=2,
        framebuffer_dhash_threshold_init=6,
    )
    buf = FrameBuffer(cfg)
    # Invalid image bytes deliberately bypass visual dedup in this unit test.
    for ts in (1.0, 2.0, 3.0):
        buf.push(Frame(ts=ts, jpeg_b64="not-an-image", source_type="screen"))

    frames = buf.writer_all_after(1.0)

    assert [frame.ts for frame in frames] == [2.0, 3.0]


def test_query_snapshot_reads_last_raw_frames_before_anchor_not_sparse_dedup():
    cfg = SimpleNamespace(
        buffer_seconds=1800,
        buffer_capture_fps=2,
        framebuffer_dhash_threshold_init=6,
    )
    buf = FrameBuffer(cfg)
    image = Image.new("RGB", (32, 32), color=(20, 80, 120))
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG", quality=80)
    jpeg_b64 = base64.b64encode(encoded.getvalue()).decode("ascii")

    for ts in (1.0, 2.0, 3.0, 4.0):
        buf.push(Frame(ts=ts, jpeg_b64=jpeg_b64, source_type="camera"))

    # The long-term ring intentionally keeps only a representative for four
    # identical images, while one-shot visual QA must retain the exact 2fps
    # captures before the user's ask-time boundary.
    assert [frame.ts for frame in buf.all_le(3.5)] == [1.0]
    assert [frame.ts for frame in buf.raw_all_le(3.5, 3)] == [1.0, 2.0, 3.0]


def test_writer_uniformly_covers_backlog_and_advances_exclusive_cursor():
    frames = [
        Frame(ts=float(i), jpeg_b64="", source_type="screen")
        for i in range(1, 71)
    ]
    writer = _writer_for_cursor_test(frames, max_frames=20)

    sampled, snapshot_end, pending_count = writer._select_unprocessed_frames()

    assert pending_count == 70
    assert len(sampled) == 20
    assert sampled[0].ts == 1.0
    assert sampled[-1].ts == 70.0
    assert all(a.ts < b.ts for a, b in zip(sampled, sampled[1:]))

    writer._commit_frame_cursor(snapshot_end)
    replay, next_end, next_count = writer._select_unprocessed_frames()

    assert writer._frame_cursor_ts == 70.0
    assert writer.mem.values["writer_frame_cursor_ts"] == "70.0"
    assert replay == []
    assert next_end is None
    assert next_count == 0


def test_writer_cursor_does_not_advance_until_commit():
    frames = [
        Frame(ts=float(i), jpeg_b64="", source_type="camera")
        for i in range(1, 31)
    ]
    writer = _writer_for_cursor_test(frames, cursor=10.0, max_frames=8)

    first_sample, first_end, first_count = writer._select_unprocessed_frames()
    retry_sample, retry_end, retry_count = writer._select_unprocessed_frames()

    assert first_count == retry_count == 20
    assert first_end == retry_end == 30.0
    assert [frame.ts for frame in first_sample] == [
        frame.ts for frame in retry_sample
    ]
    assert writer._frame_cursor_ts == 10.0


def test_writer_resets_stale_cursor_when_frame_timeline_restarts():
    frames = [
        Frame(ts=float(i), jpeg_b64="", source_type="screen")
        for i in range(1, 6)
    ]
    writer = _writer_for_cursor_test(frames, cursor=464.0, max_frames=20)

    sampled, snapshot_end, pending_count = writer._select_unprocessed_frames()

    assert [frame.ts for frame in sampled] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert snapshot_end == 5.0
    assert pending_count == 5
    assert writer._frame_cursor_ts == -1.0
