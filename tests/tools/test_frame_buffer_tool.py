"""Focused tests for frame-buffer reader keyframe selection."""

from types import SimpleNamespace

from tools.frame_buffer_tool import _dhash_pick


def test_dhash_pick_falls_back_to_even_sampling_for_bad_images():
    frames = [
        SimpleNamespace(ts=float(i), jpeg_b64="not-a-jpeg")
        for i in range(6)
    ]

    picked = _dhash_pick(frames, 3)

    # A decode failure is ``None``, not a numeric hash. The picker must avoid
    # feeding it to Hamming distance and retain deterministic temporal coverage
    # across the whole gap, including both endpoints.
    assert [frame.ts for frame in picked] == [0.0, 2.0, 5.0]
