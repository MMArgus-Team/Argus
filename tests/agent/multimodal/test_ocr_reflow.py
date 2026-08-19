"""Tests for the shared ocr_reflow module — verifies the pure-image OCR
enhancement helpers (garbage filter, resize, reflow, top-N) behave as intended.

Imported by path to avoid the agent.multimodal package init pulling openai
(which isn't required for these pure-Python helpers)."""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest


_ROOT = pathlib.Path(__file__).resolve().parents[3]
_MOD_PATH = _ROOT / "agent" / "multimodal" / "ocr_reflow.py"


@pytest.fixture(scope="module")
def reflow():
    spec = importlib.util.spec_from_file_location("ocr_reflow", _MOD_PATH)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m


# ── is_readable_text ────────────────────────────────────────────────────────
def test_is_readable_normal_text(reflow):
    assert reflow.is_readable_text("hello 你好 世界") is True
    assert reflow.is_readable_text("Error: connection refused (code 42)") is True


def test_is_readable_rejects_replacement_char(reflow):
    assert reflow.is_readable_text("hel�lo") is False


def test_is_readable_rejects_high_control_ratio(reflow):
    # >15% C0 control chars → garbage
    txt = "abc" + "\x01" * 5
    assert reflow.is_readable_text(txt) is False


def test_is_readable_empty(reflow):
    assert reflow.is_readable_text("") is False


# ── resize_for_ocr ──────────────────────────────────────────────────────────
def _fake_rgb(w: int, h: int):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_resize_upscales_small(reflow):
    rgb = _fake_rgb(1024, 600)
    out, w, h = reflow.resize_for_ocr(rgb, 1024, 600)
    # longest side clamped up to 1280
    assert max(w, h) == 1280


def test_resize_downscales_large(reflow):
    rgb = _fake_rgb(3000, 2000)
    out, w, h = reflow.resize_for_ocr(rgb, 3000, 2000)
    assert max(w, h) == 2048


def test_resize_leaves_midrange_alone(reflow):
    rgb = _fake_rgb(1500, 900)
    out, w, h = reflow.resize_for_ocr(rgb, 1500, 900)
    assert (w, h) == (1500, 900)
    assert out is rgb  # returned unchanged reference


def test_resize_respects_upper_cap(reflow):
    rgb = _fake_rgb(3000, 2000)
    _out, w, _h = reflow.resize_for_ocr(rgb, 3000, 2000, max_side=1600)
    assert max(w, _h) == 1600


# ── reflow_ocr_lines: two-column layout preserves left→right ────────────────
def test_reflow_two_column_reading_order(reflow):
    # Left column (x 0-200) has two lines; right column (x 600-800) has one.
    # Gutter of 400px is well above the 3% threshold on span ~800.
    frags = [
        ("LEFT top", (0, 0, 200, 30)),
        ("LEFT bottom", (0, 100, 200, 130)),
        ("RIGHT only", (600, 0, 800, 30)),
    ]
    paras = reflow.reflow_ocr_lines(frags)
    texts = [p[0] for p in paras]
    # left column paragraphs appear before right
    left_idx = next(i for i, t in enumerate(texts) if "LEFT" in t)
    right_idx = next(i for i, t in enumerate(texts) if "RIGHT" in t)
    assert left_idx < right_idx


def test_reflow_same_line_horizontal_join(reflow):
    # Two fragments on the same y within a single column (small gap so the
    # gutter detector doesn't split them). Should be joined into one line.
    frags = [
        ("hello", (0, 0, 50, 20)),
        ("world", (60, 0, 110, 20)),  # gap 10 << gutter_thr → same column
    ]
    paras = reflow.reflow_ocr_lines(frags)
    assert len(paras) == 1
    assert "hello" in paras[0][0] and "world" in paras[0][0]


# ── frags_to_paragraph_blocks: top-N filtering + normalization ─────────────
def test_frags_to_blocks_keeps_top_n_in_reading_order(reflow):
    # 20 paragraphs stacked vertically, varying widths → area varies. Only
    # top-N=5 by area should survive; those that do must remain in reading
    # order (y ascending).
    frags = []
    for i in range(20):
        # width alternates: even i gets big blocks, odd i small — keeps 10 big ones
        w = 500 if i % 2 == 0 else 30
        y = i * 100
        frags.append((f"P{i}", (0, y, w, y + 40)))
    raw, blocks = reflow.frags_to_paragraph_blocks(
        frags, img_w=800, img_h=2100, top_n=5)
    assert len(blocks) == 5
    # kept blocks should be from the "big" (even-i) set
    kept_texts = [b["text"] for b in blocks]
    for t in kept_texts:
        idx = int(t[1:])
        assert idx % 2 == 0, f"kept small block {t}"
    # reading order preserved (y ascending in image = later in list)
    indices = [int(t[1:]) for t in kept_texts]
    assert indices == sorted(indices)


def test_frags_to_blocks_bbox_normalized_and_region_type(reflow):
    frags = [("only", (0, 0, 100, 50))]
    raw, blocks = reflow.frags_to_paragraph_blocks(frags, img_w=200, img_h=100)
    assert raw == "only"
    assert len(blocks) == 1
    b = blocks[0]
    assert b["region_type"] == "ocr_paragraph"
    assert b["confidence"] == 1.0
    # bbox in [0,1], origin bottom-left
    assert all(0.0 <= v <= 1.0 for v in b["bbox"])


def test_frags_to_blocks_double_newline_separator(reflow):
    # Two paragraphs vertically separated → raw_text joined with blank line
    frags = [
        ("first", (0, 0, 100, 20)),
        ("second", (0, 200, 100, 220)),  # big vertical gap → new paragraph
    ]
    raw, blocks = reflow.frags_to_paragraph_blocks(frags, img_w=200, img_h=300)
    assert len(blocks) == 2
    assert "\n\n" in raw


def test_frags_to_blocks_handles_missing_bbox(reflow):
    # unboxed fragments are appended verbatim, not dropped
    frags = [("boxed", (0, 0, 100, 20)), ("nobox", None)]
    raw, blocks = reflow.frags_to_paragraph_blocks(frags, img_w=200, img_h=200)
    texts = [b["text"] for b in blocks]
    assert "boxed" in texts and "nobox" in texts
