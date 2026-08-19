"""Tests for window_text_bridge (Part 3).

Loaded by path to avoid triggering agent.multimodal package init (which pulls
openai + a big multimodal graph). We inject a fake `window_text` module into
sys.modules so the bridge's lazy import returns our stub — no real AX/UIA is
touched."""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

import pytest


_ROOT = pathlib.Path(__file__).resolve().parents[3]
_MOD_PATH = _ROOT / "agent" / "multimodal" / "window_text_bridge.py"


def _load_bridge():
    spec = importlib.util.spec_from_file_location(
        "window_text_bridge_test_load", _MOD_PATH)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m


class _StubUIElement:
    """Duck-typed replacement for window_text.UIElement (only the fields the
    bridge reads: text / x / y / w / h / semantic)."""

    def __init__(self, text, x=0.0, y=0.0, w=0.0, h=0.0, semantic="body_text"):
        self.text = text
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.semantic = semantic


def _install_fake_window_text(
    *, ax_blocks=None, app="访达", title="项目 - main.py",
    api_ok=True, nav=None, capture_raises=False,
):
    """Register a fake `window_text` module before importing the bridge, so its
    lazy import picks up this stub. Each test calls this then reloads bridge."""
    fake = types.ModuleType("window_text")
    fake.api_status = lambda: bool(api_ok)  # type: ignore[attr-defined]

    def _capture(pid, window_number, bounds=None):
        if capture_raises:
            raise RuntimeError("AX unavailable (test stub)")
        return list(ax_blocks or []), app, title

    fake.capture_window_text = _capture  # type: ignore[attr-defined]
    fake.extract_nav_anchor = (  # type: ignore[attr-defined]
        lambda number, pid, ttl: nav or {"kind": "", "value": ""})
    sys.modules["window_text"] = fake
    return fake


@pytest.fixture(autouse=True)
def _reset_state():
    """Each test starts with fresh bridge state (cache/import re-triggered)."""
    sys.modules.pop("window_text", None)
    sys.modules.pop("window_text_bridge_test_load", None)
    yield
    sys.modules.pop("window_text", None)
    sys.modules.pop("window_text_bridge_test_load", None)


# ── parse_source_id ────────────────────────────────────────────────────────
def test_parse_valid_window_source():
    b = _load_bridge()
    assert b.parse_source_id("window:12345:0") == ("window", 12345)


def test_parse_valid_screen_source():
    b = _load_bridge()
    assert b.parse_source_id("screen:0:0") == ("screen", 0)


def test_parse_valid_no_trailing():
    b = _load_bridge()
    assert b.parse_source_id("window:99") == ("window", 99)


def test_parse_invalid_returns_none():
    b = _load_bridge()
    for bad in ["", "bogus", "window:abc:0", "screen:", "tab:1:0", None]:
        assert b.parse_source_id(bad or "") is None


# ── extract_for_frame_source: skip cases ──────────────────────────────────
def test_screen_source_skipped():
    """screen:* → 全屏共享无法定位单窗口, bridge 应放弃 (None), caller 走 OCR。"""
    _install_fake_window_text(ax_blocks=[_StubUIElement("x" * 30)])
    b = _load_bridge()
    assert b.extract_for_frame_source("screen:0:0") is None


def test_empty_source_id_skipped():
    b = _load_bridge()
    assert b.extract_for_frame_source("") is None


def test_invalid_source_id_skipped():
    b = _load_bridge()
    assert b.extract_for_frame_source("no-colons") is None


def test_window_text_unavailable_returns_none():
    """window_text 无法 import (缺依赖 / 平台不支持) → 直接放弃, 不抛。
    通过 sys.meta_path 注入一个 finder, 让 'window_text' 的 import 显式失败。"""
    sys.modules.pop("window_text", None)

    class _BrokenFinder:
        def find_spec(self, name, path=None, target=None):
            if name == "window_text":
                raise ImportError("simulated: window_text unavailable")
            return None

    finder = _BrokenFinder()
    sys.meta_path.insert(0, finder)
    try:
        b = _load_bridge()
        assert b.extract_for_frame_source("window:1:0") is None
    finally:
        sys.meta_path.remove(finder)


def test_api_status_false_skips():
    """macOS 无 AX 授权时 api_status()=False → bridge 应放弃, 让 OCR 兜底。"""
    _install_fake_window_text(api_ok=False, ax_blocks=[_StubUIElement("A" * 30)])
    b = _load_bridge()
    assert b.extract_for_frame_source("window:1:0") is None


def test_capture_raises_returns_none():
    """AX 调用异常 → 不抛, 返回 None 走 OCR。"""
    _install_fake_window_text(capture_raises=True)
    b = _load_bridge()
    assert b.extract_for_frame_source("window:1:0") is None


def test_ax_empty_and_no_nav_returns_none():
    """AX 抓到但全是碎片 <20 字, 也没 URL/路径 → 视为无正文, 让 OCR 接手。"""
    _install_fake_window_text(ax_blocks=[_StubUIElement("hi")])  # 2 chars < 20
    b = _load_bridge()
    assert b.extract_for_frame_source("window:1:0") is None


# ── extract_for_frame_source: success paths ───────────────────────────────
def test_ax_success_populates_contract():
    """AX 抓到成段正文 → 返回符合 OCR worker 契约的 dict, 含 app/window_title/
    raw_text/ocr_blocks, source_tag='winax'。"""
    _install_fake_window_text(ax_blocks=[
        _StubUIElement("这是主编辑区里的一大段正文文字, 应该保留下来做 raw_text",
                       x=10, y=20, w=800, h=400, semantic="code_editor"),
        _StubUIElement("这是第二块正文, 也超过 20 字所以保留",
                       x=10, y=430, w=800, h=200),
    ])
    b = _load_bridge()
    r = b.extract_for_frame_source("window:12345:0")
    assert r is not None
    assert r["app"] == "访达"
    assert r["window_title"].startswith("项目")
    assert "主编辑区" in r["raw_text"]
    assert "第二块正文" in r["raw_text"]
    assert r["source_tag"] == "winax"
    # blocks: 两个 AX 块; region_type 透传自 semantic
    assert len(r["ocr_blocks"]) == 2
    assert r["ocr_blocks"][0]["region_type"] == "code_editor"
    assert r["ocr_blocks"][0]["confidence"] == 1.0
    assert r["ocr_blocks"][0]["bbox"] == [10.0, 20.0, 800.0, 400.0]


def test_nav_anchor_prepended_as_first_block():
    """浏览器 URL / 文件路径应作为第一个 block, 前缀'导航地址：'。"""
    _install_fake_window_text(
        ax_blocks=[_StubUIElement("a" * 30, x=0, y=0, w=800, h=400)],
        nav={"kind": "url", "value": "https://example.com/x"},
    )
    b = _load_bridge()
    r = b.extract_for_frame_source("window:1:0")
    assert r is not None
    assert r["ocr_blocks"][0]["text"].startswith("导航地址：https://example.com/x")
    assert r["ocr_blocks"][0]["region_type"] == "nav_url"
    # raw_text 首段就是导航地址 (双换行拼)
    assert r["raw_text"].split("\n\n")[0].startswith("导航地址：")


def test_nav_only_without_ax_still_works():
    """AX 空但有 nav (explorer 显示文件夹, 无正文只有路径) 也应返回结果。"""
    _install_fake_window_text(
        ax_blocks=[],  # no AX blocks
        nav={"kind": "path", "value": "/Users/me/proj"},
    )
    b = _load_bridge()
    r = b.extract_for_frame_source("window:2:0")
    assert r is not None
    assert r["ocr_blocks"][0]["text"] == "导航地址：/Users/me/proj"
    assert r["ocr_blocks"][0]["region_type"] == "nav_path"
    assert r["raw_text"] == "导航地址：/Users/me/proj"
