"""window_text.py — 网页版「点哪个窗口，抓哪个窗口的文字」(macOS + Windows, 单文件).

启动:
    python window_text.py [--port 8765] [--prompt]

浏览器打开 http://127.0.0.1:8765 :
  * 左边列出当前所有窗口(app名 + 标题 + 缩略图)
  * 点某个窗口 -> 抓它的文字块 -> 右边分块卡片展示(来源 API / OCR、坐标、文字)

抓取阶梯(best-effort, 两平台一致):
  1) 系统无障碍 API 读结构化原文 —— 原生 app, 准且快
       macOS: Accessibility (AX);  Windows: UI Automation (UIA)
  2) 读不到/过少(Electron / 浏览器 / canvas) -> 截图 + RapidOCR(onnx, 跨平台)

截图后端(macOS, 版本自适应):
  * macOS 14+ (Sonoma) : ScreenCaptureKit / SCScreenshotManager (苹果主推, 优先走)
  * 更老 或 SCK 不可用 : CGWindowListCreateImage (Sonoma 起 deprecated, 自动回退)
  两条路径统一返回 numpy RGB; SCK 本次失败也会回退旧 API, 不会因新 API 就截不到。

依赖:
  两平台共用:  pip install numpy pillow rapidocr onnxruntime
  macOS 额外:  pip install pyobjc-framework-ApplicationServices \
                            pyobjc-framework-Cocoa pyobjc-framework-Quartz \
                            pyobjc-framework-ScreenCaptureKit   # 14+ 主路径, 缺则回退旧 API
  Windows 额外: pip install uiautomation pywin32

权限:
  macOS  : 系统设置 -> 隐私与安全性 -> 辅助功能(AX) + 屏幕录制(截图/OCR),
           授给运行本脚本的 app(你的终端), 勾选后彻底退出重开(TCC 按进程生效)。
  Windows: 读普通窗口无需特殊权限; 读「以管理员运行的窗口」需本脚本也以管理员运行。
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

_IS_MAC = sys.platform == "darwin"
_IS_WIN = sys.platform.startswith("win")

# ── 共享 OCR 增强模块 (与 agent/multimodal/_workers.py 的视频流 OCR 复用) ──────
# 若在仓库内跑 -> 用共享实现, 单一权威; 若脱离仓库跑 (mm_modules 独立场景) ->
# 回退到本文件下方的本地副本, 保持独立运行能力。
try:
    from agent.multimodal import ocr_reflow as _shared_reflow  # type: ignore
except Exception:
    _shared_reflow = None


# =========================================================================== #
# 数据结构 + 过滤
# =========================================================================== #
@dataclass
class UIElement:
    role: str = ""
    role_desc: str = ""
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0
    text: str = ""
    semantic: str = ""
    app_name: str = ""
    window_id: int = 0
    pid: int = 0


# 承载文字的角色 / 要排除的界面杂项 / 正文容器
_TEXT_ROLES = {"AXStaticText", "AXTextField", "AXTextArea", "AXText"}
_EXCLUDE_ROLES = {
    "AXMenuBar", "AXMenuBarItem", "AXMenu", "AXMenuItem", "AXButton",
    "AXToolbar", "AXTabGroup", "AXTab", "AXPopUpButton", "AXCheckBox",
    "AXRadioButton", "AXTitleUIElement",
}
_BODY_CONTAINER_ROLES = {"AXTextArea", "AXTextField"}
# 菜单栏/工具栏区域: 进入后整棵子树都不算正文(与 Windows 的
# _WIN_CHROME_CONTAINERS 对齐)。关键: 里面的 AXStaticText 本身角色不在
# _EXCLUDE_ROLES 里, 只靠 _EXCLUDE_ROLES 逐控件过滤挡不住它们 —— 必须整片剪掉。
# ★ 不含 AXTabGroup: 标签组在部分 app 里也套着标签**内容**(同 Windows TabControl
#   的教训), 整片剪会误杀正文; 它已在 _EXCLUDE_ROLES 里按单控件排除, 够了。
_MAC_CHROME_CONTAINERS = {"AXMenuBar", "AXToolbar"}
_BODY_MIN_AREA = 2000.0      # px^2, 过滤图标/快捷键小标签
_BODY_MIN_LEN = 10           # 字, 过滤 "OK"/"10"/单字
_MAX_CTRL_FRAC = 0.15        # 垃圾字符(控制/私用/代理/替换符)占比 >15% -> 丢弃


def _is_garbage_char(ch):
    """疑似二进制解码残渣的字符: C0/C1 控制区、替换符、私用区、代理区、
    以及不含常见符号的杂散区块。正常正文(汉字/假名/韩文/字母数字/常见标点/常见
    符号)不算。用于按占比判乱码, 单个字符判不准, 看整段比例。"""
    o = ord(ch)
    if ch in "\t\n\r":
        return False
    if o < 0x20 or o == 0x7f:            # C0 控制
        return True
    if 0x80 <= o <= 0x9f:                # C1 控制
        return True
    if ch == "�":                        # Unicode 替换符 = 解码失败铁证
        return True
    if 0xE000 <= o <= 0xF8FF:            # 私用区(乱码常落这)
        return True
    if 0xD800 <= o <= 0xDFFF:            # 代理区(孤立代理=坏数据)
        return True
    return False


def _is_readable_text(txt: str) -> bool:
    """剔除二进制/乱码:
      * 出现 Unicode 替换符 '�' -> 解码失败, 直接判乱码(正常文本几乎不含它);
      * 垃圾字符(控制区/私用区/代理区/替换符)占比 > 阈值 -> 乱码。
    注意: 纯 ASCII 字母碎片(如 'QET QET')靠字符类型分不出, 不强判, 避免误杀
    正常终端/代码内容; 主要拦住带替换符/控制残渣的那类(即实际见到的 iTerm 乱码)。"""
    if not txt:
        return False
    if "�" in txt:                       # 有替换符 = 铁定解码失败
        return False
    garbage = sum(1 for ch in txt if _is_garbage_char(ch))
    return (garbage / len(txt)) <= _MAX_CTRL_FRAC


def _is_body_block(el: UIElement) -> bool:
    """严格: 只留正文块, 不要 chrome/碎屑。"""
    if el.role in _EXCLUDE_ROLES:
        return False
    txt = (el.text or "").strip()
    if not txt or not _is_readable_text(txt):
        return False
    if el.role in _BODY_CONTAINER_ROLES:
        return len(txt) >= 1
    area = float(el.w) * float(el.h)
    return area >= _BODY_MIN_AREA and len(txt) >= _BODY_MIN_LEN


def filter_body_blocks(elements):
    return [e for e in elements if _is_body_block(e)]


def _is_body_block_win(el: UIElement) -> bool:
    """Windows(UIA)专用过滤, 比 mac 宽松:
    UIA 已按角色分好类(_win_walk 只收文字承载控件, 交互杂项已排除), 且列表项/
    文件名/短链接常只有几个字、BoundingRectangle 可能不可靠 -> 不套 mac 的
    area/min_len 阈值, 只做可读性 + 去纯符号/纯空白。去重已在 _win_walk 做过。"""
    txt = (el.text or "").strip()
    if not txt or not _is_readable_text(txt):
        return False
    # 丢纯符号/纯标点碎屑(图标字形常是单个 PUA 字符, _is_readable_text 已挡大部分)
    if len(txt) == 1 and not txt.isalnum():
        return False
    return True


def filter_body_blocks_win(elements):
    return [e for e in elements if _is_body_block_win(e)]


# =========================================================================== #
# 权限 (平台分派)
#   macOS  : TCC 辅助功能 + 屏幕录制, 有查询/请求 API
#   Windows: 读普通窗口无需权限, 恒为 True
# =========================================================================== #
def _mac_ax_status() -> bool:
    try:
        from ApplicationServices import AXIsProcessTrusted
        return bool(AXIsProcessTrusted())
    except Exception:
        return False


def _mac_screen_status() -> bool:
    try:
        from Quartz import CGPreflightScreenCaptureAccess
        return bool(CGPreflightScreenCaptureAccess())
    except Exception:
        return False


def _mac_request_ax(prompt: bool = True) -> bool:
    try:
        from ApplicationServices import (AXIsProcessTrustedWithOptions,
                                         kAXTrustedCheckOptionPrompt)
        return bool(AXIsProcessTrustedWithOptions(
            {kAXTrustedCheckOptionPrompt: bool(prompt)}))
    except Exception:
        return _mac_ax_status()


def _mac_request_screen() -> bool:
    try:
        from Quartz import CGRequestScreenCaptureAccess
        CGRequestScreenCaptureAccess()
    except Exception:
        pass
    return _mac_screen_status()


def api_status() -> bool:
    """系统API(读文字)是否可用。mac=AX授权; win=恒True(普通窗口无需权限)。"""
    if _IS_MAC:
        return _mac_ax_status()
    return True


def screen_status() -> bool:
    """截图/OCR 是否可用。mac=屏幕录制授权; win=恒True。"""
    if _IS_MAC:
        return _mac_screen_status()
    return True


def request_ax(prompt: bool = True) -> bool:
    return _mac_request_ax(prompt) if _IS_MAC else True


def request_screen() -> bool:
    return _mac_request_screen() if _IS_MAC else True


def perm_report() -> str:
    if _IS_MAC:
        ax, sc = api_status(), screen_status()
        lines = [
            "macOS 权限状态:",
            f"  辅助功能 (AX 文字)   : {'已授权' if ax else '未授权'}",
            f"  屏幕录制 (截图/OCR)  : {'已授权' if sc else '未授权'}",
        ]
        if ax and sc:
            lines.append("  已就绪 — 可正常抓取。")
            return "\n".join(lines)
        lines += [
            "",
            "请授权给「运行本脚本的 app」(你的终端):",
            "  系统设置 -> 隐私与安全性 -> 辅助功能   -> 勾选该终端",
            "  系统设置 -> 隐私与安全性 -> 屏幕录制   -> 勾选该终端",
            "然后彻底退出并重开该终端(授权按进程生效)。",
        ]
        return "\n".join(lines)
    if _IS_WIN:
        return ("Windows: 无需特殊权限即可抓取普通窗口。\n"
                "(读取「以管理员运行的窗口」需本脚本也以管理员身份运行。)")
    return f"不支持的平台: {sys.platform}"


# =========================================================================== #
# macOS 系统 API 后端 — Accessibility (AX) 读结构化原文
# =========================================================================== #
def _require_ax():
    try:
        import ApplicationServices  # noqa: F401
        import AppKit  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "pyobjc 未安装. 运行: pip install pyobjc-framework-"
            "ApplicationServices pyobjc-framework-Cocoa pyobjc-framework-Quartz"
        ) from e
    from ApplicationServices import AXIsProcessTrusted
    if not AXIsProcessTrusted():
        raise RuntimeError("辅助功能权限未授予 (系统设置 -> 隐私与���全性 -> 辅助功能)。")


def _attr(el, name):
    from ApplicationServices import AXUIElementCopyAttributeValue
    err, val = AXUIElementCopyAttributeValue(el, name, None)
    return val if err == 0 else None


def _children(el):
    kids = _attr(el, "AXChildren")
    return list(kids) if kids else []


_kAXValueCGPointType = 1
_kAXValueCGSizeType = 2


def _unwrap_point_size(el):
    """(x,y,w,h): 解包 AXPosition/AXSize 的 AXValueRef -> CGPoint/CGSize。"""
    from ApplicationServices import AXValueGetValue
    x = y = w = h = 0.0
    pos = _attr(el, "AXPosition")
    size = _attr(el, "AXSize")
    try:
        if pos is not None:
            ok, p = AXValueGetValue(pos, _kAXValueCGPointType, None)
            if ok:
                x, y = float(p.x), float(p.y)
    except Exception:
        pass
    try:
        if size is not None:
            ok, s = AXValueGetValue(size, _kAXValueCGSizeType, None)
            if ok:
                w, h = float(s.width), float(s.height)
    except Exception:
        pass
    return x, y, w, h


def _is_junk_char(ch):
    """图标/占位符字形, 非真文字 -> 去掉:
      U+FFFC 对象替换符(网页图片/图标位)、U+FFFD 替换符、
      PUA 私用区(E000-F8FF, 图标字体如 Segoe MDL2 的按钮图标)。"""
    o = ord(ch)
    return ch in ("￼", "�") or (0xE000 <= o <= 0xF8FF)


def _clean_ax_text(s):
    """清掉 NUL / 杂散控制字符 / 图标占位符。iTerm2 等终端会在宽字符间夹 \\x00,
    网页/工具栏会夹 U+FFFC 和 PUA 图标字形; 留着会顶爆坏字符比例或输出乱码。
    保留 tab/newline/CR, 去掉 C0 控制符、DEL、对象替换符、PUA 图标。"""
    if not s:
        return s
    return "".join(
        ch for ch in s
        if ch in "\t\n\r" or (ord(ch) >= 0x20 and ord(ch) != 0x7f
                              and not _is_junk_char(ch)))


def _role_to_semantic(role, ax_id):
    aid = (ax_id or "").lower()
    if "editor" in aid or "code" in aid:
        return "code_editor"
    if "status" in aid:
        return "status_bar"
    if role == "AXTextArea":
        return "text_area"
    if role == "AXTextField":
        return "text_field"
    return "body_text"


def _walk(el, pid, win_id, app_name, out, max_depth=25):
    """递归遍历窗口 UI 树, 收集承载文字的节点 -> out(list[UIElement])。"""
    if max_depth <= 0:
        return
    role = _attr(el, "AXRole") or ""
    # 标题栏/菜单栏/工具栏 -> 整棵子树跳过(里面的 StaticText 不算正文)
    if role in _MAC_CHROME_CONTAINERS:
        return
    if role in _TEXT_ROLES:
        text = (_attr(el, "AXValue") or _attr(el, "AXTitle")
                or _attr(el, "AXDescription") or "")
        text = _clean_ax_text(str(text)) if text is not None else ""
        if text.strip():
            x, y, w, h = _unwrap_point_size(el)
            ax_id = str(_attr(el, "AXIdentifier") or "")
            out.append(UIElement(
                role=role, role_desc=str(_attr(el, "AXRoleDescription") or ""),
                x=x, y=y, w=w, h=h, text=text, app_name=app_name,
                window_id=win_id, pid=pid,
                semantic=_role_to_semantic(role, ax_id)))
    for kid in _children(el):
        _walk(kid, pid, win_id, app_name, out, max_depth - 1)


def _window_key(win, idx):
    wn = _attr(win, "AXWindowNumber")
    try:
        return int(wn) if wn is not None else idx
    except Exception:
        return idx


def _scan_window(win, pid, win_id, app_name):
    out = []
    _walk(win, pid, win_id, app_name, out)
    return filter_body_blocks(out)


def _mac_capture_window(pid, window_number, bounds=None):
    """抓用户点的那一个窗口(只抓它, 绝不猜同 app 的别的窗口)。
    匹配顺序: AXWindowNumber 精确 -> bounds 几何最近 -> 单窗口app 唯一。
    返回 (blocks, app_name, title)。AX 未授权则 _require_ax 抛错。"""
    _require_ax()
    from AppKit import NSWorkspace
    from ApplicationServices import AXUIElementCreateApplication

    app_name = ""
    for app in NSWorkspace.sharedWorkspace().runningApplications():
        if int(app.processIdentifier()) == int(pid):
            app_name = str(app.localizedName() or "")
            break
    ax_app = AXUIElementCreateApplication(int(pid))
    wins = _attr(ax_app, "AXWindows") or []
    if not wins:
        return [], app_name, ""

    target = None
    for i, w in enumerate(wins):
        if _window_key(w, i) == int(window_number):
            target = w
            break
    if target is None and bounds:
        bx, by = float(bounds.get("x", 0)), float(bounds.get("y", 0))
        bw, bh = float(bounds.get("w", 0)), float(bounds.get("h", 0))
        best, best_d = None, 1e18
        for w in wins:
            x, y, ww, hh = _unwrap_point_size(w)
            d = abs(x - bx) + abs(y - by) + abs(ww - bw) + abs(hh - bh)
            if d < best_d:
                best, best_d = w, d
        if best is not None and best_d <= max(40.0, 0.05 * (bw + bh)):
            target = best
    if target is None and len(wins) == 1:
        target = wins[0]
    if target is None:
        return [], app_name, ""

    wid = _window_key(target, 0)
    blocks = _scan_window(target, int(pid), wid, app_name)
    return blocks, app_name, str(_attr(target, "AXTitle") or "")


# =========================================================================== #
# Windows 系统 API 后端 — UI Automation (UIA) 读"成段正文"
# =========================================================================== #
# 唯一规则: 遍历 UIA 树, 任何能读到 TextPattern 文本的控件 = 一块成段正文,
#   取它、且不再深入其子树(TextPattern 已按阅读顺序聚合了整个区域的文字)。
# 为什么这一条就够 (真机数据支撑):
#   · 浏览器正文/编辑器/终端/文档 都实现 TextPattern, 一次性给出整块正文;
#   · 菜单/工具栏/按钮/标签/文件名条目 都不实现 TextPattern -> 天然被排除, 不用
#     列 chrome 容器、不用剪枝、不用面积阈值;
#   · 父容器的 TextPattern 已含子孙文字(实测 DocumentControl 2277 字覆盖 195/239
#     子 TextControl) -> "取父不钻子" 消除重复, 这是之前一堆去重/剪枝 if 的根因。
# 没有任何 TextPattern 大块的窗口(纯文件夹/列表) = 无成段正文, 返回空(上层提示)。
_WIN_TEXT_TO_ROLE = {"TextControl": "AXStaticText", "EditControl": "AXTextField",
                     "DocumentControl": "AXTextArea"}


def _require_uia():
    try:
        import uiautomation  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "uiautomation 未安装. 运行: pip install uiautomation pywin32") from e


def _win_control_from_handle(hwnd):
    import uiautomation as auto
    try:
        return auto.ControlFromHandle(int(hwnd))
    except Exception:
        return None


def _win_textpattern_text(ctrl):
    """读一个控件的 TextPattern 正文 (终端/文档正文的正确入口, 等价于 macOS AX
    把 iTerm2 正文暴露成 AXTextArea)。优先只取可见区(GetVisibleRanges), 避免把
    终端上百万字的历史滚动缓冲全拉出来; 拿不到可见区再退回整段。无 TextPattern
    返回 ''。"""
    try:
        import uiautomation as auto
        pat = ctrl.GetPattern(auto.PatternId.TextPattern)
    except Exception:
        return ""
    if pat is None:
        return ""
    # 1) 只取可见区 (跟 OCR / 用户实际看到的一致)。
    # ★ GetVisibleRanges 把每一行(可见片段)切成一个独立 range, 单个 range 内不带
    #   换行 -> 必须用换行拼接, 否则整块文字全黏成一行(346...False347... 那种)。
    try:
        ranges = pat.GetVisibleRanges()
        if ranges:
            lines = []
            for r in ranges:
                try:
                    seg = r.GetText(-1) or ""
                except Exception:
                    continue
                # range 自己可能已带尾换行(跨软换行的段), 去掉再统一按行拼
                lines.append(seg.rstrip("\r\n"))
            vis = _clean_ax_text("\n".join(lines)).strip()
            if vis:
                return vis
    except Exception:
        pass
    # 2) 退回整段 (某些控件不实现 GetVisibleRanges); DocumentRange 本身带换行
    try:
        return _clean_ax_text(pat.DocumentRange.GetText(-1) or "").strip()
    except Exception:
        return ""


_WIN_MIN_BODY_CHARS = 40      # 一块正文至少这么长, 才算"成段"(过滤地址栏/搜索框残字)

# 标题栏/菜单栏/工具栏: 这些是窗口外框, 不是正文。落进这些区域的文字一律不抓
# (整棵子树剪掉)。标准软件里它们有明确类型标记, 这条排除是必要的、可靠的。
_WIN_CHROME_CONTAINERS = {
    "TitleBarControl", "MenuBarControl", "ToolBarControl", "AppBarControl",
}


def _win_walk(ctrl, hwnd, app_name, out, max_depth=25):
    """递归遍历 UIA 树, 收"成段正文" -> out(list[UIElement])。
    规则:
      1) 落在标题栏/菜单栏/工具栏区域的 -> 整棵子树跳过(那是窗口外框, 不是正文);
      2) 节点能读到 TextPattern 文本(且够长)-> 收这一整块, 不再深入子树
         (TextPattern 已聚合整个区域文字, 钻子树只会拿到重复碎片);
      3) 否则继续往下找。"""
    if max_depth <= 0:
        return
    try:
        ctype = ctrl.ControlTypeName
    except Exception:
        ctype = ""

    # 1) 标题栏/菜单栏/工具栏 -> 整棵子树不抓
    if ctype in _WIN_CHROME_CONTAINERS:
        return

    text = _win_textpattern_text(ctrl)
    if text and len(text.strip()) >= _WIN_MIN_BODY_CHARS:
        x = y = w = h = 0.0
        try:
            r = ctrl.BoundingRectangle
            x, y = float(r.left), float(r.top)
            w, h = float(r.right - r.left), float(r.bottom - r.top)
        except Exception:
            pass
        aid = ""
        try:
            aid = ctrl.AutomationId or ""
        except Exception:
            aid = ""
        role = _WIN_TEXT_TO_ROLE.get(ctype, "AXTextArea")
        out.append(UIElement(
            role=role, role_desc=ctype, x=x, y=y, w=w, h=h, text=text,
            app_name=app_name, window_id=int(hwnd), pid=0,
            semantic=_role_to_semantic(role, aid)))
        return                    # ★ 取到整块正文 -> 不钻子树(避免重复)

    try:
        kids = ctrl.GetChildren()
    except Exception:
        kids = []
    for kid in kids:
        _win_walk(kid, hwnd, app_name, out, max_depth - 1)


def _win_window_rect(hwnd):
    """HWND -> (l, t, r, b) 窗口矩形, 失败返回 None。"""
    try:
        import ctypes
        from ctypes import wintypes
        _win32_declare_sigs()
        rect = wintypes.RECT()
        if ctypes.windll.user32.GetWindowRect(int(hwnd), ctypes.byref(rect)):
            return (rect.left, rect.top, rect.right, rect.bottom)
    except Exception:
        pass
    return None


def _clip_to_window(blocks, rect, tol=8.0):
    """丢掉块中心落在窗口矩形外的块(防 ControlFromHandle 越界到别的窗口/嵌入内容)。
    拿不到坐标(w==h==0, 如终端 TextPattern 块)的保留。"""
    if not rect:
        return blocks
    l, t, r, b = rect
    kept = []
    for el in blocks:
        if el.w <= 0 and el.h <= 0:          # 无坐标 -> 无从判断, 保留
            kept.append(el)
            continue
        cx = el.x + el.w / 2.0
        cy = el.y + el.h / 2.0
        if (l - tol) <= cx <= (r + tol) and (t - tol) <= cy <= (b + tol):
            kept.append(el)
    return kept


def _win_capture_window(pid, window_number, bounds=None):
    """抓一个窗口(window_number 即 Win32 HWND) -> (blocks, app_name, title)。
    UIA 读普通窗口无需权限。"""
    _require_uia()
    win = _win_control_from_handle(int(window_number))
    if win is None:
        return [], "", ""
    try:
        title = win.Name or ""
    except Exception:
        title = ""
    app_name = _win_process_name(pid) or (title.split(" - ")[-1] if " - " in title else title)
    out = []
    _win_walk(win, int(window_number), app_name, out)
    out = _clip_to_window(out, _win_window_rect(window_number))
    return filter_body_blocks_win(out), app_name, title


def capture_window_text(pid, window_number, bounds=None):
    """平台分派: 用系统 API 抓指定窗口的文字块 -> (blocks, app_name, title)。"""
    if _IS_MAC:
        return _mac_capture_window(pid, window_number, bounds=bounds)
    if _IS_WIN:
        return _win_capture_window(pid, window_number, bounds=bounds)
    return [], "", ""


# =========================================================================== #
# 导航锚点 (URL / 文件路径) —— 独立于正文的一路, 走 API 精确取, 不受正文阈值限制。
# 浏览器地址栏在工具栏区(正文 walk 已排除工具栏), 文件路径地址栏默认是面包屑,
# 所以这两样必须单独精确提取, 不能靠正文/OCR。
# =========================================================================== #
_SHELL_APP = None
_SHELL_TRIED = False


def _win_shell_windows():
    """Shell.Application().Windows() —— explorer/IE 窗口集合, 拿真实 LocationURL。
    COM 对象缓存; comtypes 缺失或 COM 失败返回 None(容错, 不影响其它)。"""
    global _SHELL_APP, _SHELL_TRIED
    if _SHELL_APP is None and not _SHELL_TRIED:
        _SHELL_TRIED = True
        try:
            import comtypes.client
            _SHELL_APP = comtypes.client.CreateObject("Shell.Application")
        except Exception:
            _SHELL_APP = None
    if _SHELL_APP is None:
        return None
    try:
        return _SHELL_APP.Windows()
    except Exception:
        return None


def _fileurl_to_path(url):
    """file:///D:/A/B -> D:\\A\\B; 非 file:// 原样返回。"""
    if not url:
        return ""
    if url.lower().startswith("file:///"):
        try:
            from urllib.parse import unquote
            p = unquote(url[len("file:///"):])
            return p.replace("/", "\\")
        except Exception:
            return url
    return url


def _win_shell_location(hwnd):
    """按 HWND 从 Shell 窗口集合拿真实文件系统路径(如 D:\\MyProjects\\mm_hermes)。
    虚拟位置(主文件夹/此电脑)LocationURL 为空 -> 返回 ''。COM 全程容错。"""
    wins = _win_shell_windows()
    if wins is None:
        return ""
    try:
        count = wins.Count
    except Exception:
        return ""
    for i in range(count):
        try:
            it = wins.Item(i)
            if it is None:
                continue
            if int(it.HWND) != int(hwnd):
                continue
            url = ""
            try:
                url = it.LocationURL or ""
            except Exception:
                url = ""
            return _fileurl_to_path(url)
        except Exception:
            continue
    return ""


def _win_breadcrumb_path(hwnd):
    """explorer 地址栏面包屑(SplitButtonControl 序列)拼路径, 作 Shell 拿不到时的兜底。
    '此电脑>新加卷 (D:)>MyProjects>mm_hermes' -> 'D:\\MyProjects\\mm_hermes';
    拿不到盘符(虚拟位置)-> '此电脑 > 主文件夹' 这种可读位置。"""
    import re
    win = _win_control_from_handle(int(hwnd))
    if win is None:
        return ""
    best = []

    def kids(c):
        try:
            return c.GetChildren()
        except Exception:
            return []

    def walk(c, d=0):
        nonlocal best
        if d > 28:
            return
        segs = []
        for k in kids(c):
            try:
                if k.ControlTypeName == "SplitButtonControl":
                    nm = (k.Name or "").strip()
                    if nm and nm != "更多":       # '更多' 是面包屑溢出按钮, 跳过
                        segs.append(nm)
            except Exception:
                pass
        if len(segs) >= 2 and len(segs) > len(best):
            best = segs
        for k in kids(c):
            walk(k, d + 1)
    walk(win)
    if not best:
        return ""
    parts, drive = [], None
    _VIRTUAL = ("此电脑", "桌面", "主文件夹", "快速访问", "网络", "OneDrive")
    for s in best:
        m = re.search(r"\(([A-Za-z]:)\)", s)          # '新加卷 (D:)' -> D:
        if m:
            drive = m.group(1)
            parts = []
            continue
        if s in _VIRTUAL:
            continue
        parts.append(s)
    if drive:
        return drive + "\\" + "\\".join(parts) if parts else drive + "\\"
    return " > ".join(best)                            # 虚拟位置: 可读面包屑


def _win_class_name(hwnd):
    """HWND -> 窗口类名, 失败返回 ''。用来判窗口种类(explorer=CabinetWClass)。"""
    try:
        import ctypes
        from ctypes import wintypes
        u = ctypes.windll.user32
        u.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        u.GetClassNameW.restype = ctypes.c_int
        buf = ctypes.create_unicode_buffer(256)
        u.GetClassNameW(int(hwnd), buf, 256)
        return buf.value or ""
    except Exception:
        return ""


_ADDR_HINT = ("地址", "address", "url", "location")


def _looks_like_url(v):
    """v 像不像一个网址(用于判浏览器地址栏值, 排除路径/普通文本)。"""
    v = (v or "").strip()
    if "://" in v:
        return True
    # 无协议头的域名式: 含点且首段像域名/IP, 不含反斜杠(路径特征)
    if "\\" in v or " " in v:
        return False
    head = v.split("/")[0]
    return "." in head and len(head) >= 3


def _win_browser_url(hwnd):
    """找浏览器地址栏 EditControl 的 URL(ValuePattern)。
    判据: EditControl 且 Value 确实像网址(有协议头或域名式), 不认路径/普通文本。"""
    import uiautomation as auto
    win = _win_control_from_handle(int(hwnd))
    if win is None:
        return ""

    def kids(c):
        try:
            return c.GetChildren()
        except Exception:
            return []

    found = [""]

    def walk(c, d=0):
        if d > 28 or found[0]:
            return
        try:
            tn = c.ControlTypeName
        except Exception:
            return
        if tn == "EditControl":
            nm = aid = val = ""
            try:
                nm = (c.Name or "").lower()
            except Exception:
                pass
            try:
                aid = (c.AutomationId or "").lower()
            except Exception:
                pass
            try:
                vp = c.GetPattern(auto.PatternId.ValuePattern)
                val = (vp.Value or "") if vp else ""
            except Exception:
                val = ""
            # 必须值本身像网址(排除文件路径/搜索框残字); Name 命中只作加权
            name_hit = any(h in nm or h in aid for h in _ADDR_HINT)
            if val.strip() and _looks_like_url(val) and (name_hit or "://" in val):
                found[0] = val.strip()
                return
        for k in kids(c):
            walk(k, d + 1)
    walk(win)
    return found[0]


def _win_nav_anchor(pid, hwnd, title):
    """Windows 导航锚点 -> {"kind": "url"|"path"|"", "value": str}。
    explorer 类窗口(CabinetWClass): 走 Shell 真实路径 -> 面包屑兜底;
    其它窗口: 找浏览器地址栏 URL。按窗口种类分, 不会把 VS 工具栏当面包屑。"""
    is_explorer = _win_class_name(hwnd) == "CabinetWClass"

    if is_explorer:
        try:
            path = _win_shell_location(hwnd)      # 真实路径 D:\...
        except Exception:
            path = ""
        if not path:
            try:
                path = _win_breadcrumb_path(hwnd)  # 虚拟位置 -> 面包屑兜底
            except Exception:
                path = ""
        if path:
            return {"kind": "path", "value": path}
        return {"kind": "", "value": ""}

    # 非 explorer: 只找浏览器地址栏 URL(值必须像网址)
    try:
        url = _win_browser_url(hwnd)
    except Exception:
        url = ""
    if url:
        return {"kind": "url", "value": url}
    return {"kind": "", "value": ""}


def _osascript(script, timeout=2.0):
    """跑一段 AppleScript, 返回 stdout(strip) 或 ''。超时/出错返回 ''。"""
    import subprocess
    try:
        r = subprocess.run(["osascript", "-e", script],
                           capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


# 已知浏览器(bundle 名) -> 都实现同一套 AppleScript 词汇: front window active tab URL
_MAC_BROWSERS = {
    "Google Chrome": "chrome", "Google Chrome Canary": "chrome",
    "Microsoft Edge": "chrome", "Brave Browser": "chrome",
    "Arc": "chrome", "Chromium": "chrome",
    "Safari": "safari", "Safari Technology Preview": "safari",
}


def _mac_app_name_for_pid(pid):
    """pid -> 前台 app 本地化名(NSRunningApplication)。失败返回 ''。"""
    try:
        from AppKit import NSRunningApplication
        app = NSRunningApplication.runningApplicationWithProcessIdentifier_(int(pid))
        return str(app.localizedName() or "") if app else ""
    except Exception:
        return ""


def _mac_nav_anchor(pid, title=""):
    """macOS 导航锚点: 浏览器地址栏 URL / Finder 当前文件夹路径。
    用 AppleScript(osascript, 系统自带, 无额外依赖):
      * 浏览器: 问 front window 的 active tab URL (Chrome 系) 或 current tab URL (Safari);
      * 访达 Finder: 问 front window target 的 POSIX path。
    需自动化权限(首次弹窗授权 osascript 控制该 app)。拿不到返回空。"""
    app = _mac_app_name_for_pid(pid) if pid else ""

    # --- 浏览器 URL ---
    kind_map = _MAC_BROWSERS.get(app)
    if kind_map == "chrome":
        url = _osascript(
            f'tell application "{app}" to get URL of active tab of front window')
        if url:
            return {"kind": "url", "value": url}
    elif kind_map == "safari":
        url = _osascript(
            f'tell application "{app}" to get URL of current tab of front window')
        if url:
            return {"kind": "url", "value": url}

    # --- Finder 文件夹路径 ---
    if app in ("Finder", "访达"):
        path = _osascript(
            'tell application "Finder" to get POSIX path of '
            '(target of front window as alias)')
        if path:
            return {"kind": "path", "value": path}

    return {"kind": "", "value": ""}


def extract_nav_anchor(number, pid=0, title=""):
    """平台分派: 抓窗口的导航锚点(URL / 文件路径)。"""
    try:
        if _IS_WIN:
            return _win_nav_anchor(pid, int(number), title)
        if _IS_MAC:
            return _mac_nav_anchor(pid, title)
    except Exception:
        pass
    return {"kind": "", "value": ""}


# =========================================================================== #
# 截图 (平台分派) -> 统一返回 (numpy RGB (H,W,3) uint8, w, h) 或 None
#   macOS  : ScreenCaptureKit (macOS 12.3+, 苹果主推) 优先,
#            回退 CGWindowListCreateImage (Sonoma 起 deprecated, 老系统仍可用)
#   Windows: Win32 PrintWindow (按 HWND, 能截被遮挡/后台窗口, 无需权限)
# =========================================================================== #
def _cgimage_to_rgb(img):
    """CGImage -> (numpy RGB (H,W,3) uint8, w, h) 或 None。两条截图路径共用。"""
    import numpy as np
    import Quartz
    from Quartz import (CGImageGetDataProvider, CGDataProviderCopyData,
                        CGImageGetBytesPerRow)
    if img is None:
        return None
    w = Quartz.CGImageGetWidth(img)
    h = Quartz.CGImageGetHeight(img)
    if w == 0 or h == 0:
        return None
    provider = CGImageGetDataProvider(img)
    data = CGDataProviderCopyData(provider)
    bpr = CGImageGetBytesPerRow(img)
    buf = np.frombuffer(data, dtype=np.uint8)[:h * bpr].reshape(h, bpr // 4, 4)
    rgb = buf[:, :w, :3][:, :, ::-1]     # BGRA -> RGB
    return np.ascontiguousarray(rgb), w, h


# ScreenCaptureKit 能力探测缓存: None=未探测, True/False=探测结果
_SCK_AVAILABLE = None


def _sck_available():
    """ScreenCaptureKit + SCScreenshotManager 是否可用。
    需 macOS 12.3+ 且 pyobjc 带 ScreenCaptureKit 绑定; SCScreenshotManager 需 14+。
    探测结果缓存。"""
    global _SCK_AVAILABLE
    if _SCK_AVAILABLE is not None:
        return _SCK_AVAILABLE
    _SCK_AVAILABLE = False
    if not _IS_MAC:
        return False
    try:
        import ScreenCaptureKit  # noqa: F401
        from ScreenCaptureKit import (SCShareableContent,  # noqa: F401
                                       SCScreenshotManager,  # noqa: F401
                                       SCContentFilter,      # noqa: F401
                                       SCStreamConfiguration)  # noqa: F401
        # SCScreenshotManager.captureImageWithFilter... 是 14.0 才有的一步式截图
        _SCK_AVAILABLE = bool(
            hasattr(SCScreenshotManager,
                    "captureImageWithFilter_configuration_completionHandler_"))
    except Exception:
        _SCK_AVAILABLE = False
    return _SCK_AVAILABLE


def _pump_until(done_flag, timeout=5.0):
    """在当前线程 run loop 上抽取事件, 等异步 completion handler 回来。
    done_flag: 单元素 list 当可变布尔用。超时返回。"""
    from Foundation import NSRunLoop, NSDate
    loop = NSRunLoop.currentRunLoop()
    deadline = time.time() + timeout
    while not done_flag[0] and time.time() < deadline:
        loop.runMode_beforeDate_(
            "kCFRunLoopDefaultMode",
            NSDate.dateWithTimeIntervalSinceNow_(0.02))


def _mac_capture_window_rgb_sck(window_number):
    """ScreenCaptureKit 按窗口号截图 (macOS 14+ 主推路径)。
    SCShareableContent 拿窗口列表 -> 匹配 windowID -> SCScreenshotManager 一步截。
    两个 API 都是异步 completion handler, 用 run loop 泵成同步。"""
    try:
        from ScreenCaptureKit import (SCShareableContent, SCScreenshotManager,
                                       SCContentFilter, SCStreamConfiguration)
    except Exception:
        return None

    # --- 1) 取可共享窗口列表 (异步) ---
    content_box = [None]
    done = [False]

    def _on_content(content, error):
        if error is None:
            content_box[0] = content
        done[0] = True

    try:
        SCShareableContent.\
            getShareableContentExcludingDesktopWindows_onScreenWindowsOnly_completionHandler_(
                False, False, _on_content)
    except Exception:
        return None
    _pump_until(done)
    content = content_box[0]
    if content is None:
        return None

    # --- 2) 按 windowID 匹配目标 SCWindow ---
    target = None
    try:
        for win in content.windows():
            if int(win.windowID()) == int(window_number):
                target = win
                break
    except Exception:
        return None
    if target is None:
        return None

    # --- 3) 构造 filter + config, 一步式截 CGImage (异步) ---
    try:
        flt = SCContentFilter.alloc().initWithDesktopIndependentWindow_(target)
        cfg = SCStreamConfiguration.alloc().init()
        try:                       # 按窗口物理尺寸设分辨率, 拿全清晰度
            fr = target.frame()
            scale = 2.0            # Retina; 多截无妨, 反正后面 OCR 会缩
            cfg.setWidth_(int(fr.size.width * scale))
            cfg.setHeight_(int(fr.size.height * scale))
        except Exception:
            pass
    except Exception:
        return None

    img_box = [None]
    done2 = [False]

    def _on_img(cgimage, error):
        if error is None:
            img_box[0] = cgimage
        done2[0] = True

    try:
        SCScreenshotManager.\
            captureImageWithFilter_configuration_completionHandler_(
                flt, cfg, _on_img)
    except Exception:
        return None
    _pump_until(done2)
    if img_box[0] is None:
        return None
    return _cgimage_to_rgb(img_box[0])


def _mac_capture_window_rgb_legacy(window_number):
    """CGWindowListCreateImage 按窗口号截图 (Sonoma 起 deprecated, 老系统回退)。"""
    from Quartz import (CGRectNull, CGWindowListCreateImage,
                        kCGWindowImageBoundsIgnoreFraming,
                        kCGWindowListOptionIncludingWindow)
    try:
        img = CGWindowListCreateImage(
            CGRectNull, kCGWindowListOptionIncludingWindow,
            int(window_number), kCGWindowImageBoundsIgnoreFraming)
    except Exception:
        return None
    return _cgimage_to_rgb(img)


# SCK 一旦成功过就置 True; 探测通过但从没成功过 -> 打一次回退警告(便于真机排查)
_SCK_EVER_OK = False
_SCK_WARNED = False


def _mac_capture_window_rgb(window_number):
    """版本自适应: macOS 14+ 且 SCK 可用走 ScreenCaptureKit, 否则回退旧 API。
    SCK 路径失败(权限/窗口消失/绑定异常)也回退, 保证不因新 API 挂掉就截不到。"""
    global _SCK_EVER_OK, _SCK_WARNED
    if _sck_available():
        rgb = _mac_capture_window_rgb_sck(window_number)
        if rgb is not None:
            _SCK_EVER_OK = True
            return rgb
        # SCK 探测通过但本次失败 -> 回退旧 API 兜底; 若从没成功过, 提示一次
        if not _SCK_EVER_OK and not _SCK_WARNED:
            _SCK_WARNED = True
            print("[window_text] ScreenCaptureKit 可用但截图失败, 已回退旧 API "
                  "(CGWindowListCreateImage)。多为屏幕录制权限未授或选择器不匹配。")
    return _mac_capture_window_rgb_legacy(window_number)


_WIN32_SIG_DONE = False


def _win32_declare_sigs():
    """给会用到的 Win32 函数声明 argtypes/restype (只做一次)。
    ★ 关键: 不声明时 ctypes 默认把所有参数/返回值当 C int(32 位)。64 位 Windows 上
    HWND/HDC/HBITMAP 都是 64 位指针 -> 传大句柄会 'OverflowError: int too long to
    convert', 或返回值被截断成错句柄 -> 截到别的东西(比如整个桌面)。这就是
    chrome 抓成整桌面 + too long 报错的根因。"""
    global _WIN32_SIG_DONE
    if _WIN32_SIG_DONE:
        return
    import ctypes
    from ctypes import wintypes
    u = ctypes.windll.user32
    g = ctypes.windll.gdi32
    HWND, HDC, HBMP = wintypes.HWND, wintypes.HDC, wintypes.HBITMAP
    HGDIOBJ = wintypes.HGDIOBJ

    u.GetWindowRect.argtypes = [HWND, ctypes.POINTER(wintypes.RECT)]
    u.GetWindowRect.restype = wintypes.BOOL
    u.GetWindowDC.argtypes = [HWND]
    u.GetWindowDC.restype = HDC
    u.ReleaseDC.argtypes = [HWND, HDC]
    u.ReleaseDC.restype = ctypes.c_int
    u.PrintWindow.argtypes = [HWND, HDC, wintypes.UINT]
    u.PrintWindow.restype = wintypes.BOOL

    g.CreateCompatibleDC.argtypes = [HDC]
    g.CreateCompatibleDC.restype = HDC
    g.CreateCompatibleBitmap.argtypes = [HDC, ctypes.c_int, ctypes.c_int]
    g.CreateCompatibleBitmap.restype = HBMP
    g.SelectObject.argtypes = [HDC, HGDIOBJ]
    g.SelectObject.restype = HGDIOBJ
    g.DeleteObject.argtypes = [HGDIOBJ]
    g.DeleteObject.restype = wintypes.BOOL
    g.DeleteDC.argtypes = [HDC]
    g.DeleteDC.restype = wintypes.BOOL
    g.GetDIBits.argtypes = [HDC, HBMP, wintypes.UINT, wintypes.UINT,
                            ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT]
    g.GetDIBits.restype = ctypes.c_int

    # --- 窗口枚举 / 标题 / pid 用到的 user32(句柄同样是 64 位, 必须声明) ---
    u.IsWindowVisible.argtypes = [HWND]
    u.IsWindowVisible.restype = wintypes.BOOL
    u.GetWindowTextLengthW.argtypes = [HWND]
    u.GetWindowTextLengthW.restype = ctypes.c_int
    u.GetWindowTextW.argtypes = [HWND, wintypes.LPWSTR, ctypes.c_int]
    u.GetWindowTextW.restype = ctypes.c_int
    u.GetWindowThreadProcessId.argtypes = [HWND, ctypes.POINTER(wintypes.DWORD)]
    u.GetWindowThreadProcessId.restype = wintypes.DWORD
    # GetWindowLongPtrW: 64 位下取 ex-style 的正确入口(GetWindowLongW 会截断)
    _glp = getattr(u, "GetWindowLongPtrW", None) or u.GetWindowLongW
    _glp.argtypes = [HWND, ctypes.c_int]
    _glp.restype = ctypes.c_ssize_t

    # --- kernel32: OpenProcess 返回 HANDLE(64 位), 不声明会截断句柄 ---
    k = ctypes.windll.kernel32
    k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k.OpenProcess.restype = wintypes.HANDLE
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    k.CloseHandle.restype = wintypes.BOOL
    k.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD)]
    k.QueryFullProcessImageNameW.restype = wintypes.BOOL
    _WIN32_SIG_DONE = True


def _win_capture_window_rgb(hwnd):
    """PrintWindow 抓单个窗口(含被遮挡的) -> (numpy RGB, w, h) 或 None。"""
    import numpy as np
    import ctypes
    from ctypes import wintypes
    _win32_declare_sigs()
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    hwnd = int(hwnd)

    rect = wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None
    w, h = rect.right - rect.left, rect.bottom - rect.top
    if w <= 0 or h <= 0:
        return None

    hwindc = user32.GetWindowDC(hwnd)
    if not hwindc:
        return None
    memdc = gdi32.CreateCompatibleDC(hwindc)
    bmp = gdi32.CreateCompatibleBitmap(hwindc, w, h)
    old = gdi32.SelectObject(memdc, bmp)
    # PW_RENDERFULLCONTENT=2 让 Chromium/UWP 等也能被正确渲染进位图
    ok = user32.PrintWindow(hwnd, memdc, 2)
    if not ok:
        ok = user32.PrintWindow(hwnd, memdc, 0)

    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
                    ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
                    ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                    ("biSizeImage", wintypes.DWORD),
                    ("biXPelsPerMeter", wintypes.LONG),
                    ("biYPelsPerMeter", wintypes.LONG),
                    ("biClrUsed", wintypes.DWORD), ("biClrImportant", wintypes.DWORD)]

    bmi = BITMAPINFOHEADER()
    bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.biWidth = w
    bmi.biHeight = -h                      # top-down
    bmi.biPlanes = 1
    bmi.biBitCount = 32
    bmi.biCompression = 0                  # BI_RGB
    buf = ctypes.create_string_buffer(w * h * 4)
    got = gdi32.GetDIBits(memdc, bmp, 0, h, buf, ctypes.byref(bmi), 0)
    if old:
        gdi32.SelectObject(memdc, old)     # 还原, 再删 bmp(避免删到正在选中的对象)
    gdi32.DeleteObject(bmp)
    gdi32.DeleteDC(memdc)
    user32.ReleaseDC(hwnd, hwindc)
    if not got or not ok:
        return None
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    rgb = arr[:, :, :3][:, :, ::-1]        # BGRA -> RGB
    return np.ascontiguousarray(rgb), w, h


def _capture_window_rgb(window_number):
    if _IS_MAC:
        return _mac_capture_window_rgb(window_number)
    if _IS_WIN:
        return _win_capture_window_rgb(window_number)
    return None


_RAPID = None


def _rapid_engine():
    """惰性初始化 RapidOCR, 兼容新旧包名, 关闭方向分类(use_cls=False)省时间。
    屏幕文字几乎不会旋转 180°, cls 这步纯浪费; 关掉每帧省一次推理。没装则报错。"""
    global _RAPID
    if _RAPID is not None:
        return _RAPID
    # 旧版 rapidocr_onnxruntime: RapidOCR(use_cls=False)
    try:
        from rapidocr_onnxruntime import RapidOCR
        try:
            _RAPID = RapidOCR(use_cls=False)
        except Exception:
            _RAPID = RapidOCR()
        return _RAPID
    except Exception:
        pass
    # 新版 rapidocr(3.x): RapidOCR(params={"Global.use_cls": False})
    try:
        from rapidocr import RapidOCR
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "OCR 引擎 rapidocr 未安装. 运行:\n"
            "  pip install rapidocr onnxruntime") from e
    try:
        _RAPID = RapidOCR(params={"Global.use_cls": False})
    except Exception:
        _RAPID = RapidOCR()
    return _RAPID


def _rapid_call(engine, rgb):
    """兼容新旧返回格式 -> list of (box_pts, text, conf)。
    旧版: [[box,text,conf],...]; 新版 3.x: 对象(.boxes/.txts/.scores)。"""
    out = engine(rgb)
    res = out[0] if isinstance(out, tuple) else out
    if hasattr(res, "boxes") and hasattr(res, "txts"):
        boxes = res.boxes if res.boxes is not None else []
        txts = res.txts if res.txts is not None else []
        scores = res.scores if getattr(res, "scores", None) is not None else []
        triples = []
        for i, t in enumerate(txts):
            b = boxes[i] if i < len(boxes) else None
            c = scores[i] if i < len(scores) else 1.0
            triples.append((b, t, c))
        return triples
    if not res:
        return []
    return [(item[0], item[1], item[2]) for item in res]


def _box_to_xywh(box_pts, img_w, img_h):
    """rapidocr 4 点框 -> 归一化 [x,y,w,h](原点左下, 与阅读排序约定一致)。
    新版 box 是 numpy 数组, 用 len() 判空(数组真值有歧义)。"""
    if box_pts is None or len(box_pts) == 0:
        return None
    try:
        xs = [float(p[0]) for p in box_pts]
        ys = [float(p[1]) for p in box_pts]
    except Exception:
        return None
    if img_w <= 0 or img_h <= 0:
        return None
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return [round(x0 / img_w, 4), round(1.0 - y1 / img_h, 4),
            round((x1 - x0) / img_w, 4), round((y1 - y0) / img_h, 4)]


def _box_to_pixels(box_pts):
    """rapidocr 4 点框 -> 像素 (x0, y0, x1, y1)(原点左上, y 向下)。
    用于版面重排的几何聚类。取不到框返回 None。"""
    if box_pts is None or len(box_pts) == 0:
        return None
    try:
        xs = [float(p[0]) for p in box_pts]
        ys = [float(p[1]) for p in box_pts]
    except Exception:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _norm_xywh_from_pixels(px, img_w, img_h):
    """像素框 (x0,y0,x1,y1) -> 归一化 [x,y,w,h](原点左下, 同 _box_to_xywh)。
    段落合并框拿去前端展示 geo 标签用。"""
    if not px or img_w <= 0 or img_h <= 0:
        return None
    x0, y0, x1, y1 = px
    return [round(x0 / img_w, 4), round(1.0 - y1 / img_h, 4),
            round((x1 - x0) / img_w, 4), round((y1 - y0) / img_h, 4)]


def _split_columns(boxed):
    """把带框碎片按【列】切分, 解决多栏布局(侧边栏+正文/IDE/聊天)阅读顺序错乱。

    用【投影空白检测】找栏间竖直空白带(gutter), 比"按碎片宽度聚类"稳 —— 后者会被
    正文里的宽行桥接两栏。做法:
      1) 把每个碎片的 [x0,x1] 投影到 x 轴, 用扫描线求出所有【被文字覆盖的区间】;
      2) 相邻覆盖区间之间的空隙 = 候选 gutter; 空隙宽度 > gutter_thr 的才算真栏界;
      3) 用这些栏界把 x 轴切成若干段, 每个碎片按其中心 x 归入所在段 = 一列。
    gutter_thr 取窗口宽度的 3%(自适应, 且与字号/碎片宽无关), 下限 25px。

    返回: [列碎片列表, ...], 按列左边界升序(左栏在前)。
    单栏 / 无明显 gutter -> 只有 1 列, 行为与原先一致(无回归)。"""
    if not boxed:
        return []
    xs = [b[0] for _, b in boxed] + [b[2] for _, b in boxed]
    span = max(xs) - min(xs)
    if span <= 0:
        return [boxed]
    gutter_thr = max(25.0, span * 0.03)

    # 1) 扫描线: 合并所有碎片的 x 覆盖区间
    ivals = sorted((b[0], b[2]) for _, b in boxed)
    merged = []                          # [(cx0, cx1), ...] 被文字覆盖的区间
    cx0, cx1 = ivals[0]
    for a, bb in ivals[1:]:
        if a <= cx1:                     # 重叠/相接 -> 合并
            cx1 = max(cx1, bb)
        else:
            merged.append((cx0, cx1))
            cx0, cx1 = a, bb
    merged.append((cx0, cx1))

    # 2) 覆盖区间之间足够宽的空隙 = 栏界; 收集切分边界(gutter 中点)
    bounds = []
    for i in range(len(merged) - 1):
        gap = merged[i + 1][0] - merged[i][1]
        if gap > gutter_thr:
            bounds.append((merged[i][1] + merged[i + 1][0]) / 2.0)
    if not bounds:                       # 无栏界 -> 单栏
        return [boxed]

    # 3) 按中心 x 把碎片分到各栏(bounds 已升序)
    ncols = len(bounds) + 1
    cols = [[] for _ in range(ncols)]
    for t, b in boxed:
        cx = (b[0] + b[2]) / 2.0
        idx = 0
        while idx < len(bounds) and cx >= bounds[idx]:
            idx += 1
        cols[idx].append((t, b))
    return [c for c in cols if c]        # 左 -> 右, 丢空栏


def _reflow_single_column(boxed):
    """单栏内: 行聚类 -> 行内横向拼接 -> 纵向分段。
    返回 [(段落文本, (x0,y0,x1,y1)), ...], 按栏内阅读顺序(上 -> 下)。"""
    # --- 1) 行聚类: 按 y 中心排序, 垂直重叠判同行 -----------------------------
    boxed = sorted(boxed, key=lambda it: (it[1][1] + it[1][3]) / 2.0)
    lines = []                       # 每行: {"frags":[(t,box)...], "y0","y1"}
    for t, b in boxed:
        x0, y0, x1, y1 = b
        placed = False
        for ln in lines:
            oy0, oy1 = ln["y0"], ln["y1"]
            overlap = min(y1, oy1) - max(y0, oy0)
            min_h = max(1.0, min(y1 - y0, oy1 - oy0))
            if overlap / min_h > 0.5:
                ln["frags"].append((t, b))
                ln["y0"] = min(oy0, y0)
                ln["y1"] = max(oy1, y1)
                placed = True
                break
        if not placed:
            lines.append({"frags": [(t, b)], "y0": y0, "y1": y1})

    # --- 2) 行内排序 + 横向拼接 ------------------------------------------------
    built = []                       # 每行: (text, x0, y0, x1, y1)
    for ln in lines:
        frags = sorted(ln["frags"], key=lambda it: it[1][0])
        heights = [fb[3] - fb[1] for _, fb in frags]
        heights.sort()
        char_h = heights[len(heights) // 2] if heights else 0.0
        gap_thr = char_h * 0.5
        parts, prev_x1 = [], None
        for t, fb in frags:
            fx0, _, fx1, _ = fb
            if prev_x1 is not None and (fx0 - prev_x1) > gap_thr:
                parts.append(" ")
            parts.append(t)
            prev_x1 = fx1
        text = "".join(parts).strip()
        xs0 = min(fb[0] for _, fb in frags)
        ys0 = min(fb[1] for _, fb in frags)
        xs1 = max(fb[2] for _, fb in frags)
        ys1 = max(fb[3] for _, fb in frags)
        if text:
            built.append((text, xs0, ys0, xs1, ys1))

    if not built:
        return []

    # --- 3) 纵向分段: 相邻行间距超阈值 -> 新段落 ------------------------------
    built.sort(key=lambda r: r[2])                      # 按 y0 升序
    row_h = sorted(r[4] - r[2] for r in built)
    med_h = row_h[len(row_h) // 2] if row_h else 0.0
    para_gap = med_h * 1.4

    paras = []
    cur = None
    prev_bottom = None
    for text, x0, y0, x1, y1 in built:
        if cur is not None and prev_bottom is not None and \
                (y0 - prev_bottom) > para_gap:
            paras.append(cur)
            cur = None
        if cur is None:
            cur = {"lines": [], "x0": x0, "y0": y0, "x1": x1, "y1": y1}
        cur["lines"].append(text)
        cur["x0"] = min(cur["x0"], x0)
        cur["y0"] = min(cur["y0"], y0)
        cur["x1"] = max(cur["x1"], x1)
        cur["y1"] = max(cur["y1"], y1)
        prev_bottom = y1
    if cur is not None:
        paras.append(cur)

    return [("\n".join(p["lines"]), (p["x0"], p["y0"], p["x1"], p["y1"]))
            for p in paras]


def _reflow_ocr_lines(items):
    """把凌乱的 OCR 碎片按几何版面整合成通顺段落。
    输入 items: [(text, (x0,y0,x1,y1) 像素 或 None), ...]。
    输出: [(段落文本(可多行), 合并像素框 或 None), ...], 按阅读顺序。

    版面重排(纯几何, 无 LLM):
      0) 列切分 : 先按 x 分栏 —— 多栏界面(侧边栏+正文/IDE/聊天)左右栏不再穿插
      1) 行聚类 : 栏内垂直重叠比 > 0.5 判同一行(抗���号差异)
      2) 行���拼: 按 x 升序, 间距 > 半个字高插空格(救被切断的词)
      3) 分段落: 栏内相邻行垂直间距 > 中位行高 * 1.4 -> 起新段
    输出顺序: 列左->右, 栏内段落上->下。
    无 bbox 的碎片(异常)按原顺序追加末尾, 不丢。"""
    boxed = [(t, b) for (t, b) in items if b is not None]
    unboxed = [(t, b) for (t, b) in items if b is None]
    if not boxed:
        return [(t, None) for t, _ in items]

    out = []
    for col in _split_columns(boxed):        # 列左 -> 右
        out.extend(_reflow_single_column(col))   # 栏内上 -> 下
    if not out:
        out = [(t, b) for t, b in unboxed] if unboxed else \
              [(t, None) for t, _ in items]
        return out
    out.extend((t, None) for t, _ in unboxed)    # 无框碎片追加末尾, 不丢
    return out


# ── 若共享模块可用, 用它覆盖上面的本地实现 (单一权威, 防漂移) ────────────────
# 本地副本保留是给"脱离仓库独立跑"的兜底 (mm_modules 场景, 见记忆 window-text-tcc-blocker)。
if _shared_reflow is not None:
    _is_readable_text = _shared_reflow.is_readable_text  # type: ignore[assignment]
    _box_to_pixels = _shared_reflow.box_to_pixels        # type: ignore[assignment]
    _norm_xywh_from_pixels = _shared_reflow.norm_xywh_from_pixels  # type: ignore[assignment]
    _split_columns = _shared_reflow._split_columns       # type: ignore[assignment]
    _reflow_single_column = _shared_reflow._reflow_single_column  # type: ignore[assignment]
    _reflow_ocr_lines = _shared_reflow.reflow_ocr_lines  # type: ignore[assignment]


# OCR 前把截图的【最长边】规整到 [512, 1024] 区间, 兼顾精度与速度:
#   最长边 < 1280  -> 放大到 1280 (小字/小对话框太小 OCR 掉字, 实测 notepad++ 需放大)
#   最长边 > 2048  -> 缩到  2048 (再大提精度有限却费算力; 3200 过头, 1024 又丢小字)
#   1280 ~ 2048    -> 原样不动
_OCR_MIN_SIDE = 1280
_OCR_MAX_SIDE = 2048
_OCR_MIN_CONF = 0.5             # 置信度阈值: 0.3 太松会收错字, 提到 0.5
_OCR_TOP_N = 15                 # OCR 段落只保留面积最大的前 N 块(正文主体), 其余舍


def _resize_for_ocr(rgb, w, h):
    """把最长边规整到 [1280,2048]: 太小放大, 太大缩小, 区间内不动。
    返回 (resize 后 rgb, new_w, new_h)。缩放用 LANCZOS(放大清晰/缩小抗锯齿)。"""
    side = max(w, h)
    if side <= 0:
        return rgb, w, h
    if side < _OCR_MIN_SIDE:
        scale = _OCR_MIN_SIDE / float(side)
    elif side > _OCR_MAX_SIDE:
        scale = _OCR_MAX_SIDE / float(side)
    else:
        return rgb, w, h          # 已在区间内, 不动
    try:
        from PIL import Image
        import numpy as np
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        img = Image.fromarray(rgb).resize((nw, nh), Image.Resampling.LANCZOS)
        return np.ascontiguousarray(np.asarray(img)), nw, nh
    except Exception:
        return rgb, w, h


if _shared_reflow is not None:
    def _resize_for_ocr(rgb, w, h):  # type: ignore[no-redef]
        """代理到共享模块 (行为一致: 最长边规整到 [1280, 2048])。"""
        return _shared_reflow.resize_for_ocr(rgb, w, h)


def ocr_window_blocks(window_number, app_name="", title="", pid=0,
                      min_conf=_OCR_MIN_CONF):
    """截一个窗口 + RapidOCR -> list[UIElement], 版面整合成段落(几何重排)。
    识别前按需放大提精度; 置信度低于阈值的丢弃。
    bbox 用放大图坐标, 归一化时用放大后的 w/h -> 坐标天然还原到 0..1, 无需换算。"""
    cap = _capture_window_rgb(int(window_number))
    if cap is None:
        return []
    engine = _rapid_engine()
    rgb, w, h = cap
    rgb, w, h = _resize_for_ocr(rgb, w, h)      # 最长边规整到 [512,1024]
    frags = []
    for box_pts, text, conf in _rapid_call(engine, rgb):
        t = str(text).strip()
        if t and float(conf) >= min_conf and _is_readable_text(t):
            frags.append((t, _box_to_pixels(box_pts)))

    # 重排成段落(已是阅读顺序), 再按面积只留最大的 top-N 段落 —— 大块=正文主体,
    # 小块(零星标签/图标字)被舍。选出 top-N 后【仍按阅读顺��】输出, 保证通顺。
    paras = list(_reflow_ocr_lines(frags))       # [(text, px 或 None), ...]
    if len(paras) > _OCR_TOP_N:
        def _area(px):
            if not px:
                return 0.0
            x0, y0, x1, y1 = px
            return max(0.0, x1 - x0) * max(0.0, y1 - y0)
        idx = sorted(range(len(paras)),
                     key=lambda i: _area(paras[i][1]), reverse=True)[:_OCR_TOP_N]
        keep = sorted(idx)                       # 还原阅读顺序
        paras = [paras[i] for i in keep]

    out = []
    for text, px in paras:
        box = _norm_xywh_from_pixels(px, w, h) if px else None
        el = UIElement(role="OCRText", semantic="ocr_paragraph", text=text,
                       app_name=app_name, window_id=int(window_number),
                       pid=int(pid))
        if box:
            el.x, el.y, el.w, el.h = box
        out.append(el)
    return out


# =========================================================================== #
# 窗口枚举 + 缩略图 (给前端列表)
# =========================================================================== #
def _clean_title(t):
    """CGWindowList 偶尔把二进制当窗口名; 去控制字符, 乱码则返回空。"""
    if not t:
        return ""
    kept = [c for c in t if c in "\t " or (ord(c) >= 0x20 and c != "�")]
    s = "".join(kept).strip()
    if not s or len(s) < len(t) * 0.5:
        return ""
    return s


def _window_thumbnail(window_number, max_side=320):
    """窗口截图(numpy RGB) -> 缩小的 base64 PNG(data-URI) 或 ''。跨平台(PIL)。"""
    cap = _capture_window_rgb(int(window_number))
    if cap is None:
        return ""
    rgb, w, h = cap
    try:
        import base64
        import io
        from PIL import Image
        im = Image.fromarray(rgb)
        scale = min(1.0, float(max_side) / float(max(w, h) or 1))
        if scale < 1.0:
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        b = io.BytesIO()
        im.save(b, format="PNG")
        return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()
    except Exception:
        return ""


def _win_process_name(pid):
    """Windows: pid -> 进程名(不含 .exe), 失败返回 ''。"""
    try:
        import ctypes
        from ctypes import wintypes
        _win32_declare_sigs()
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not h:
            return ""
        buf = ctypes.create_unicode_buffer(260)
        size = wintypes.DWORD(260)
        ok = k32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size))
        k32.CloseHandle(h)
        if not ok:
            return ""
        name = buf.value.replace("\\", "/").split("/")[-1]
        return name[:-4] if name.lower().endswith(".exe") else name
    except Exception:
        return ""


def _mac_list_windows(include_thumbnail):
    import Quartz
    info = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly
        | Quartz.kCGWindowListExcludeDesktopElements,
        Quartz.kCGNullWindowID)
    if not info:
        return []
    out = []
    for w in info:
        if int(w.get("kCGWindowLayer", 1)) != 0:      # 只要普通窗口
            continue
        b = w.get("kCGWindowBounds") or {}
        width, height = int(b.get("Width", 0)), int(b.get("Height", 0))
        if width < 40 or height < 40:                 # 跳过极小辅助窗口
            continue
        num = int(w.get("kCGWindowNumber"))
        out.append({
            "number": num,
            "pid": int(w.get("kCGWindowOwnerPID", 0)),
            "app": str(w.get("kCGWindowOwnerName", "") or ""),
            "title": _clean_title(str(w.get("kCGWindowName", "") or "")),
            "x": int(b.get("X", 0)), "y": int(b.get("Y", 0)),
            "w": width, "h": height,
            "thumb": _window_thumbnail(num) if include_thumbnail else "",
        })
    return out


def _win_list_windows(include_thumbnail):
    """EnumWindows 枚举可见顶层窗口 -> 同 mac 结构。number == HWND。"""
    import ctypes
    from ctypes import wintypes
    _win32_declare_sigs()
    user32 = ctypes.windll.user32
    _get_long = getattr(user32, "GetWindowLongPtrW", None) or user32.GetWindowLongW

    WNDENUMPROC = ctypes.WINFUNCTYPE(
        wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    hwnds = []

    def _cb(hwnd, lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:                              # 无标题的多为辅助窗口
            return True
        # 跳过工具窗口(WS_EX_TOOLWINDOW=0x80)
        GWL_EXSTYLE = -20
        ex = _get_long(hwnd, GWL_EXSTYLE)
        if ex & 0x00000080:
            return True
        hwnds.append(int(hwnd))
        return True

    user32.EnumWindows(WNDENUMPROC(_cb), 0)
    out = []
    for hwnd in hwnds:
        rect = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            continue
        width, height = rect.right - rect.left, rect.bottom - rect.top
        if width < 40 or height < 40:
            continue
        n = user32.GetWindowTextLengthW(hwnd)
        tbuf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, tbuf, n + 1)
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        out.append({
            "number": int(hwnd),
            "pid": int(pid.value),
            "app": _win_process_name(pid.value),
            "title": _clean_title(tbuf.value),
            "x": int(rect.left), "y": int(rect.top),
            "w": int(width), "h": int(height),
            "thumb": _window_thumbnail(hwnd) if include_thumbnail else "",
        })
    return out


def list_windows(include_thumbnail=True):
    """枚举当前所有正常窗口 -> [{number,pid,app,title,x,y,w,h,thumb}]。
    number: mac=CGWindowNumber, win=HWND。"""
    if _IS_MAC:
        return _mac_list_windows(include_thumbnail)
    if _IS_WIN:
        return _win_list_windows(include_thumbnail)
    return []


# =========================================================================== #
# 抓取编排: 只要"成段正文", UIA(API) 优先, 读不到才 OCR, 都没有则明说无正文
# =========================================================================== #
# 单块过滤: 一块文字 < 这个字数就丢(独立判断, 不算整窗总和)。
#   < 20 字的碎块(单个图标标签/按钮字/散字)信息量太低, 逐块丢掉;
#   >= 20 的短句正文(聊天消息/待办项/表格行)保留。
_MIN_BLOCK_CHARS = 20


def _keep_blocks(blocks):
    """逐块过滤: 只留 strip 后字数 >= _MIN_BLOCK_CHARS 的块。"""
    return [b for b in blocks
            if len((b.text or "").strip()) >= _MIN_BLOCK_CHARS]


def extract_window(number, pid=0, bounds=None) -> dict:
    """抓指定窗口正文 -> {text, blocks, source('ax'|'ocr'|'none'), ...}。
    统一规则:
      1) UIA/AX 系统 API 读文字, 逐块丢 <20 字碎块, 还有内容 -> 用(ax);
      2) 读不到(canvas/Scintilla 等) -> 截图 OCR + 几何重排, 同样逐块丢 <20(ocr);
      3) 两路都没有 -> source=none, 明说无正文。"""
    app = title = ""
    ax_blocks = []
    if api_status():
        try:
            ax_blocks, app, title = capture_window_text(pid, number, bounds=bounds)
        except Exception:
            ax_blocks = []

    # 导航锚点(URL/文件路径): 独立一路, 与正文/OCR 无关, 不受阈值限制。
    try:
        nav = extract_nav_anchor(number, pid, title)
    except Exception:
        nav = {"kind": "", "value": ""}

    def _result(text, blocks, source, note=""):
        # 后端直接把多块合并成【一个完整文本块】返回: 导航锚点(URL/路径)若有放最前,
        # 各块之间双换行连接。text 字段与这唯一的块内容一致; blocks 只含这一个块,
        # 调用方(网页/其它)拿到的就是一整块, 无需自己再拼。
        nv = (nav or {}).get("value", "").strip()
        if nv:
            nv = "导航地址：" + nv        # 加前缀, 读者一眼看懂这是当前 URL/路径
            text = nv + "\n\n" + text if text else nv
        merged = []
        if text:
            el = UIElement(role="MergedText", semantic="merged",
                           text=text, app_name=app,
                           window_id=int(number), pid=int(pid))
            # 合并块的包围盒 = 各原始块的并集(有坐标的话), 方便下游定位
            xs0 = [b.x for b in blocks if (b.w or b.h)]
            ys0 = [b.y for b in blocks if (b.w or b.h)]
            xs1 = [b.x + b.w for b in blocks if (b.w or b.h)]
            ys1 = [b.y + b.h for b in blocks if (b.w or b.h)]
            if xs0:
                el.x, el.y = min(xs0), min(ys0)
                el.w, el.h = max(xs1) - el.x, max(ys1) - el.y
            merged = [el]
        return {"text": text, "blocks": merged, "source": source,
                "app": app, "title": title, "note": note, "nav": nav}

    ax_kept = _keep_blocks(ax_blocks)         # 逐块丢 <20 字碎块
    if ax_kept:                               # API 还有正文 -> 直接用
        return _result(_blocks_to_text(ax_kept), ax_kept, "ax")

    # API 无正文 -> OCR 兜底; 几何重排成段后同样逐块丢 <20 字。
    if screen_status():
        try:
            ocr_blocks = ocr_window_blocks(number, app, title, pid)
        except Exception as e:  # noqa: BLE001  (rapidocr 没装 -> 让上层可见)
            return _result("", [], "none", note=str(e))
        ocr_kept = _keep_blocks(ocr_blocks)
        if ocr_kept:
            return _result(_blocks_to_text(ocr_kept), ocr_kept, "ocr")

    # 无成段正文 —— 但导航锚点可能仍有(explorer 文件夹: 无正文, 有路径)
    note = "此窗口无成段正文(可能是纯图标/列表界面, 或正文渲染成图无法读取)。"
    if not screen_status() or not api_status():
        note = perm_report()
    return _result("", [], "none", note=note)


def _blocks_to_text(blocks) -> str:
    """把多个文字块连成【一个完整文本区】: 块之间用双换行分隔(空行), 段内原样。
    去掉相邻重复块与空块。"""
    parts, prev = [], None
    for b in blocks:
        t = (b.text or "").strip()
        if not t or t == prev:
            continue
        parts.append(t)
        prev = t
    return "\n\n".join(parts)


# =========================================================================== #
# HTTP 服务 + 内嵌前端
# =========================================================================== #
def _perm_json():
    return {"ax": api_status(), "screen": screen_status(), "note": perm_report()}


def _pack(r):
    return {
        "source": r["source"], "app": r["app"], "title": r["title"],
        "note": r["note"], "text": r["text"],
        "nav": r.get("nav") or {"kind": "", "value": ""},
        "blocks": [{"role": b.role, "semantic": b.semantic,
                    "x": b.x, "y": b.y, "w": b.w, "h": b.h, "text": b.text}
                   for b in r["blocks"]],
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, html):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        try:
            if u.path in ("/", "/index.html"):
                return self._html(INDEX_HTML)
            if u.path == "/api/windows":
                return self._json({"windows": list_windows(),
                                   "perms": _perm_json()})
            if u.path == "/api/window":
                q = parse_qs(u.query)
                num = int((q.get("number") or ["0"])[0])
                pid = int((q.get("pid") or ["0"])[0])

                def _f(k):
                    try:
                        return float((q.get(k) or ["0"])[0])
                    except ValueError:
                        return 0.0
                bounds = {"x": _f("x"), "y": _f("y"), "w": _f("w"), "h": _f("h")}
                return self._json(_pack(extract_window(num, pid, bounds)))
        except Exception as e:  # noqa: BLE001
            return self._json({"error": str(e)}, code=500)
        self._json({"error": "not found"}, code=404)


INDEX_HTML = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>窗口文字提取器</title>
<style>
  :root{--bg:#0e0f13;--panel:#171922;--edge:#262a38;--fg:#e6e8ef;--mut:#8b90a3;
        --acc:#5b8cff;--ax:#3ecf8e;--ocr:#f5a524}
  *{box-sizing:border-box}
  body{margin:0;font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
       background:var(--bg);color:var(--fg);height:100vh;display:flex;flex-direction:column}
  header{padding:12px 16px;border-bottom:1px solid var(--edge);display:flex;
         align-items:center;gap:12px;flex:0 0 auto}
  header h1{font-size:15px;margin:0;font-weight:600}
  header .sp{flex:1}
  button{background:var(--panel);color:var(--fg);border:1px solid var(--edge);
          border-radius:8px;padding:6px 12px;cursor:pointer;font-size:13px}
  button:hover{border-color:var(--acc)}
  .banner{background:#2a1e12;color:var(--ocr);border:1px solid #4a3419;
          padding:8px 12px;margin:10px 16px;border-radius:8px;white-space:pre-wrap;
          font-size:12px;display:none}
  main{flex:1;display:flex;min-height:0}
  .col{display:flex;flex-direction:column;min-height:0}
  .left{flex:0 0 340px;border-right:1px solid var(--edge);overflow:auto;padding:10px}
  .right{flex:1;overflow:auto;padding:14px}
  .win{display:flex;gap:10px;padding:8px;border:1px solid var(--edge);
       border-radius:10px;margin-bottom:8px;cursor:pointer;background:var(--panel)}
  .win:hover{border-color:var(--acc)}
  .win.sel{border-color:var(--acc);box-shadow:0 0 0 1px var(--acc) inset}
  .win img{width:96px;height:60px;object-fit:cover;border-radius:6px;background:#000;
           flex:0 0 auto}
  .win .meta{min-width:0}
  .win .app{font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .win .ttl{color:var(--mut);font-size:12px;white-space:nowrap;overflow:hidden;
            text-overflow:ellipsis}
  .win .dim{color:var(--mut);font-size:11px;margin-top:2px}
  .hd{display:flex;align-items:center;gap:10px;margin-bottom:12px;flex-wrap:wrap}
  .hd .src{font-size:11px;padding:2px 8px;border-radius:20px;border:1px solid var(--edge)}
  .src.ax{color:var(--ax);border-color:#1e4d38}
  .src.ocr{color:var(--ocr);border-color:#4a3419}
  .src.none{color:var(--mut)}
  .blk{border:1px solid var(--edge);border-radius:10px;padding:10px 12px;
       margin-bottom:8px;background:var(--panel)}
  .blk .tags{font-size:11px;color:var(--mut);margin-bottom:6px;display:flex;gap:8px;
             flex-wrap:wrap}
  .blk .tags b{color:var(--fg);font-weight:600}
  .blk pre{margin:0;white-space:pre-wrap;word-break:break-word;font:13px/1.55
           ui-monospace,SFMono-Regular,Menlo,monospace}
  .empty{color:var(--mut);padding:24px;text-align:center}
  .count{color:var(--mut);font-size:12px}
  .nav{background:#12233a;border:1px solid #1e3a5f;color:#9cc4ff;border-radius:8px;
       padding:8px 12px;margin-bottom:12px;font-size:13px;word-break:break-all}
  .nav span{color:#e6e8ef;font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
</style></head>
<body>
<header>
  <h1>窗口文字提取器</h1>
  <span class="count" id="cnt"></span>
  <span class="sp"></span>
  <button id="refresh">刷新窗口</button>
</header>
<div class="banner" id="banner"></div>
<main>
  <div class="col left" id="list"><div class="empty">加载窗口中…</div></div>
  <div class="col right" id="detail"><div class="empty">← 左边点一个窗口，抓它的文字</div></div>
</main>
<script>
const $=s=>document.querySelector(s);
function esc(s){return (s||"").replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}

async function loadWindows(){
  $("#list").innerHTML='<div class="empty">加载窗口中…</div>';
  let r;
  try{ r=await (await fetch("/api/windows")).json(); }
  catch(e){ $("#list").innerHTML='<div class="empty">后端未响应</div>'; return; }
  const b=$("#banner");
  if(r.perms && (r.perms.ax===false || r.perms.screen===false)){ b.style.display="block"; b.textContent=r.perms.note; }
  else b.style.display="none";
  const ws=r.windows||[];
  $("#cnt").textContent=ws.length+" 个窗口";
  if(!ws.length){ $("#list").innerHTML='<div class="empty">没有窗口。<br>多半是屏幕录制权限未授权（见上方提示）。</div>'; return; }
  $("#list").innerHTML="";
  for(const w of ws){
    const el=document.createElement("div");
    el.className="win";
    el.innerHTML=`${w.thumb?`<img src="${w.thumb}">`:`<img>`}
      <div class="meta">
        <div class="app">${esc(w.app)||"(无名)"}</div>
        <div class="ttl">${esc(w.title)||"—"}</div>
        <div class="dim">#${w.number} · ${w.w}×${w.h}</div>
      </div>`;
    el.onclick=()=>{ document.querySelectorAll(".win").forEach(x=>x.classList.remove("sel"));
      el.classList.add("sel"); pick(w); };
    $("#list").appendChild(el);
  }
}

async function pick(w){
  $("#detail").innerHTML='<div class="empty">抓取中…（首次 OCR 需加载模型，稍候）</div>';
  const qs=`number=${w.number}&pid=${w.pid}&x=${w.x}&y=${w.y}&w=${w.w}&h=${w.h}`;
  let r;
  try{ r=await (await fetch(`/api/window?${qs}`)).json(); }
  catch(e){ $("#detail").innerHTML='<div class="empty">抓取失败</div>'; return; }
  if(r.error){ $("#detail").innerHTML=`<div class="empty">错误：${esc(r.error)}</div>`; return; }
  const text=r.text||"";
  let html=`<div class="hd">
      <span class="src ${r.source}">${(r.source||'none').toUpperCase()}</span>
      <b>${esc(r.app)}</b><span class="count">${esc(r.title)}</span>
      <span class="sp"></span><span class="count">${(r.blocks||[]).length} 块 · ${text.length} 字</span>
    </div>`;
  if(!text){
    html+=`<div class="empty">没抓到文字。${r.note?"<br><pre style='text-align:left;white-space:pre-wrap;color:var(--ocr)'>"+esc(r.note)+"</pre>":""}</div>`;
  }else{
    // 呈现为一个完整文本区: 后端已合并成单块(导航地址若有在最前)
    html+=`<div class="blk"><pre>${esc(text)}</pre></div>`;
  }
  $("#detail").innerHTML=html;
}

$("#refresh").onclick=loadWindows;
loadWindows();
</script>
</body></html>"""


def _win_set_dpi_aware():
    """让本进程 per-monitor DPI 感知。否则高分屏上 GetWindowRect 返回逻辑坐标
    (被系统缩放), 而 PrintWindow 出的位图是物理像素 -> 尺寸对不上, 截出来错位/
    只截到窗口一角(看起来像'抓多了/抓错了')。启动时调一次。"""
    if not _IS_WIN:
        return
    import ctypes
    try:                     # Win10 1703+: per-monitor v2, 最佳
        ctypes.windll.user32.SetProcessDpiAwarenessContext(-4)
        return
    except Exception:
        pass
    try:                     # Win8.1+: PROCESS_PER_MONITOR_DPI_AWARE=2
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except Exception:
        pass
    try:                     # Vista+: 系统级 DPI 感知(兜底)
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def main():
    import argparse
    import webbrowser
    _win_set_dpi_aware()
    ap = argparse.ArgumentParser(description="网页版窗口文字提取器 (macOS + Windows)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--prompt", action="store_true",
                    help="macOS: 启动时弹出权限授权框 (Windows 无需)")
    ap.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    args = ap.parse_args()

    if args.prompt and _IS_MAC:
        request_ax(prompt=True)
        request_screen()

    url = f"http://{args.host}:{args.port}"
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[window_text] {url}  (Ctrl-C 退出)")
    if _IS_MAC and not (api_status() and screen_status()):
        print("[window_text] 权限未齐；网页会显示授权指引。")
    if not args.no_open:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[window_text] bye")


if __name__ == "__main__":
    main()
