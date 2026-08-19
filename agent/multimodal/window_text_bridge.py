"""window_text_bridge.py — Part 3: 视频流 ↔ window_text 桥接层。

只在 desktop 屏幕共享 + 用户选了具体窗口 (Frame.source_id 形如 'window:12345:0')
的场景生效。用 window_text.py 的 AX/UIA 阶梯 + URL/路径锚点抓结构化正文 (又快又准),
拿不到就返回 None, 让 ScreenOCRWorker 回落到 RapidOCR (Part 1 增强路径)。

★ 刻意不用 window_text 的 OCR 兜底 —— 它会自己截屏 + rapidocr, 与视频流 OCR
   worker 重复; 让 worker 走 Part 1 的增强 RapidOCR 就够了 (且用的是共享帧数据)。

★ 平台 & 权限:
   * macOS: 需 辅助功能 (AX 文字) 权限; 未授权 -> api_status()=False -> 返回 None。
     浏览器 URL / Finder 路径需自动化权限, 首次调用会弹授权框, 拒绝后拿不到锚点。
   * Windows: 一般无需权限 (UIA 读普通窗口 OK)。
   * 其它平台: window_text._IS_MAC/_IS_WIN 都是 False -> 直接返回 None。

★ 依赖:
   * pyobjc (mac) / uiautomation + pywin32 (win) —— 缺失则 window_text 抛
     RuntimeError, 我们 catch 后返回 None, 不影响 OCR 主链路。
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("hermes.multimodal.window_text_bridge")


# ── source_id 解析 ─────────────────────────────────────────────────────────
_SOURCE_ID_RE = re.compile(r"^(window|screen):(\d+)(?::\d+)?$")


def parse_source_id(source_id: str) -> Optional[Tuple[str, int]]:
    """Electron desktopCapturer source id -> (kind, number)。
    形如 'window:12345:0' -> ('window', 12345); 'screen:0:0' -> ('screen', 0)。
    非法/空 -> None (caller 视为"不走窗口抓取")。"""
    if not source_id:
        return None
    m = _SOURCE_ID_RE.match(source_id.strip())
    if not m:
        return None
    kind = m.group(1)
    try:
        number = int(m.group(2))
    except ValueError:
        return None
    return kind, number


# ── window_text lazy import ────────────────────────────────────────────────
_WT_MOD: Any = None
_WT_TRIED = False
_WT_ERR = ""


def _load_window_text() -> Any:
    """惰性 import window_text.py (仓库根)。缺依赖 (pyobjc/uiautomation) 或非
    mac/win 平台 -> 返回 None; 结果缓存, 每次调用不重复付出 import 代价。"""
    global _WT_MOD, _WT_TRIED, _WT_ERR
    if _WT_TRIED:
        return _WT_MOD
    _WT_TRIED = True
    try:
        import window_text as _wt  # type: ignore[import-untyped]
        _WT_MOD = _wt
    except Exception as e:  # noqa: BLE001
        _WT_ERR = f"{type(e).__name__}: {e}"
        log.info("[window_text_bridge] window_text 不可用: %s (走 OCR 兜底)", _WT_ERR)
        _WT_MOD = None
    return _WT_MOD


# ── 主入口 ────────────────────────────────────────────────────────────────
def extract_for_frame_source(
    source_id: str,
    *,
    min_body_chars: int = 20,
) -> Optional[Dict[str, Any]]:
    """给一帧的 source_id 抓结构化窗口正文, 供 ScreenOCRWorker 直接持久化。

    返回 dict (与 OCR client extract() 单帧结果契约一致):
      {"app": str, "window_title": str, "raw_text": str,
       "ocr_blocks": [{text, bbox, confidence, region_type}], "source_tag": str}
    * source_tag: "winax" (AX 系统 API) 或 "" (未走此路径, caller 应回落)。
    * bbox 用像素坐标 [x, y, w, h] (窗口局部, 非归一化) —— 与 ScreenTextBlock
      契约兼容 (它对 bbox 只做 float 化, 不强制归一化)。

    返回 None 表示"这一帧不适合走窗口抓取, 请回落 OCR":
      * source_id 非 'window:*' (screen: 全屏无从定位单窗口)
      * window_text 不可用 (缺依赖 / 非 mac/win / 权限被拒)
      * 该窗口 AX 树读不到成段正文 (canvas/Electron 渲染层等)

    只调 window_text.capture_window_text (AX/UIA 结构化原文) + extract_nav_anchor
    (URL/路径), 刻意不触发 window_text 的 OCR 兜底 —— 那与视频流 OCR worker 重复。
    """
    parsed = parse_source_id(source_id)
    if parsed is None:
        return None
    kind, number = parsed
    if kind != "window":
        # 全屏共享 (screen:*) 对应"整个显示器", 里面可能有几十个窗口, 没有单一
        # AX 目标。回落 OCR (RapidOCR 直接吃这一帧的 JPEG) 是最合理的。
        return None

    wt = _load_window_text()
    if wt is None:
        return None

    # 权限门: mac 未授 AX -> 直接放弃; win 恒 True。
    try:
        if not wt.api_status():
            return None
    except Exception as e:  # noqa: BLE001
        log.info("[window_text_bridge] api_status raised: %s", e)
        return None

    # AX/UIA 抓正文
    try:
        ax_blocks, app, title = wt.capture_window_text(pid=0, window_number=number)
    except Exception as e:  # noqa: BLE001
        log.info("[window_text_bridge] capture_window_text(%s) failed: %s", number, e)
        return None

    # 导航锚点 (URL / 文件路径): 与正文独立, 拿不到不阻塞
    nav_kind, nav_value = "", ""
    try:
        nav = wt.extract_nav_anchor(number, 0, title or "")
        if isinstance(nav, dict):
            nav_kind = str(nav.get("kind") or "").strip()
            nav_value = str(nav.get("value") or "").strip()
    except Exception as e:  # noqa: BLE001
        log.info("[window_text_bridge] extract_nav_anchor(%s) failed: %s", number, e)

    # 逐块过滤 <20 字碎片 (与 window_text extract_window 保持一致)
    kept = [
        b for b in (ax_blocks or [])
        if len((getattr(b, "text", "") or "").strip()) >= max(1, int(min_body_chars))
    ]
    if not kept and not nav_value:
        # AX 读不到正文, 也没锚点 -> 这窗口就是纯 canvas/图标, 让 OCR 兜底。
        return None

    # 拼段落级 blocks: window_text 的 UIElement 已经是"块级"(每块一段控件的文字),
    # 直接映射到 ScreenTextBlock 契约。bbox 保留窗口局部像素坐标。
    blocks: List[Dict[str, Any]] = []
    if nav_value:
        blocks.append({
            "text": f"导航地址：{nav_value}",
            "bbox": [],
            "confidence": 1.0,
            "region_type": f"nav_{nav_kind}" if nav_kind else "nav",
        })
    for b in kept:
        text = (getattr(b, "text", "") or "").strip()
        if not text:
            continue
        try:
            bbox = [float(getattr(b, "x", 0.0) or 0.0),
                    float(getattr(b, "y", 0.0) or 0.0),
                    float(getattr(b, "w", 0.0) or 0.0),
                    float(getattr(b, "h", 0.0) or 0.0)]
        except Exception:
            bbox = []
        # semantic 已由 window_text 归纳为 body_text/code_editor/text_area/... 直接透传
        region = str(getattr(b, "semantic", "") or "ax_block").strip()
        blocks.append({
            "text": text,
            "bbox": bbox,
            "confidence": 1.0,
            "region_type": region,
        })

    raw_text = "\n\n".join(b["text"] for b in blocks if b.get("text"))
    if not raw_text:
        return None

    return {
        "app": str(app or ""),
        "window_title": str(title or ""),
        "raw_text": raw_text,
        "ocr_blocks": blocks,
        "source_tag": "winax",
    }
