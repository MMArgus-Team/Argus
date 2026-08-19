"""ocr_reflow.py — 纯图像 OCR 增强 (无窗口/平台依赖, 供视频流 OCR 与 window_text 共用)。

抽自 window_text.py 的"给定一张图就能跑"的那部分, 保持算法与中文注释逐字一致,
供两处复用, 消除重复、防漂移:
  * 乱码过滤   : is_readable_text / _is_garbage_char
  * 尺寸规整   : resize_for_ocr —— OCR 前把最长边规整到 [1280,2048], 小图放大救小字
  * 几何工具   : box_to_pixels / norm_xywh_from_pixels
  * 版面重排   : reflow_ocr_lines (+ _split_columns / _reflow_single_column)
  * 便捷入口   : frags_to_paragraph_blocks —— 碎片 → 段落 → top-N → 归一化 blocks

只依赖 numpy / PIL (PIL 仅 resize_for_ocr 内按需 import, 缺失则原样返回)。
不 import http.server / pyobjc / ctypes —— 纯 CPU、可在任何进程/线程里跑。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# 垃圾字符(控制/私用/代理/替换符)占比 >15% -> 丢弃
_MAX_CTRL_FRAC = 0.15

# OCR 前把截图的【最长边】规整到 [1280, 2048] 区间, 兼顾精度与速度:
#   最长边 < 1280  -> 放大到 1280 (小字/小对话框太小 OCR 掉字)
#   最长边 > 2048  -> 缩到  2048 (再大提精度有限却费算力)
#   1280 ~ 2048    -> 原样不动
_OCR_MIN_SIDE = 1280
_OCR_MAX_SIDE = 2048
_OCR_MIN_CONF = 0.5             # 置信度阈值: 0.3 太松会收错字, 提到 0.5
_OCR_TOP_N = 15                 # OCR 段落只保留面积最大的前 N 块(正文主体), 其余舍


# =========================================================================== #
# 乱码过滤
# =========================================================================== #
def _is_garbage_char(ch) -> bool:
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
    if ch == "�":                   # Unicode 替换符 = 解码失败铁证
        return True
    if 0xE000 <= o <= 0xF8FF:            # 私用区(乱码常落这)
        return True
    if 0xD800 <= o <= 0xDFFF:            # 代理区(孤立代理=坏数据)
        return True
    return False


def is_readable_text(txt: str) -> bool:
    """剔除二进制/乱码:
      * 出现 Unicode 替换符 '�' -> 解码失败, 直接判乱码(正常文本几乎不含它);
      * 垃圾字符(控制区/私用区/代理区/替换符)占比 > 阈值 -> 乱码。
    注意: 纯 ASCII 字母碎片(如 'QET QET')靠字符类型分不出, 不强判, 避免误杀
    正常终端/代码内容; 主要拦住带替换符/控制残渣的那类(即实际见到的 iTerm 乱码)。"""
    if not txt:
        return False
    if "�" in txt:                  # 有替换符 = 铁定解码失败
        return False
    garbage = sum(1 for ch in txt if _is_garbage_char(ch))
    return (garbage / len(txt)) <= _MAX_CTRL_FRAC


# =========================================================================== #
# 尺寸规整
# =========================================================================== #
def resize_for_ocr(rgb, w: int, h: int, *, max_side: int = _OCR_MAX_SIDE):
    """把最长边规整到 [1280, max_side]: 太小放大, 太大缩小, 区间内不动。
    返回 (resize 后 rgb, new_w, new_h)。缩放用 LANCZOS(放大清晰/缩小抗锯齿)。
    max_side 允许调用方收紧上限(默认 2048); 下限恒为 1280(救小字)。
    PIL 缺失或异常 -> 原样返回, 不崩。"""
    side = max(w, h)
    if side <= 0:
        return rgb, w, h
    upper = max(_OCR_MIN_SIDE, int(max_side or _OCR_MAX_SIDE))
    if side < _OCR_MIN_SIDE:
        scale = _OCR_MIN_SIDE / float(side)
    elif side > upper:
        scale = upper / float(side)
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


# =========================================================================== #
# 几何工具
# =========================================================================== #
def box_to_pixels(box_pts):
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


def norm_xywh_from_pixels(px, img_w: int, img_h: int):
    """像素框 (x0,y0,x1,y1) -> 归一化 [x,y,w,h](原点左下)。
    段落合并框拿去前端展示 geo 标签用。"""
    if not px or img_w <= 0 or img_h <= 0:
        return None
    x0, y0, x1, y1 = px
    return [round(x0 / img_w, 4), round(1.0 - y1 / img_h, 4),
            round((x1 - x0) / img_w, 4), round((y1 - y0) / img_h, 4)]


# =========================================================================== #
# 版面重排
# =========================================================================== #
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


def reflow_ocr_lines(items):
    """把凌乱的 OCR 碎片按几何版面整合成通顺段落。
    输入 items: [(text, (x0,y0,x1,y1) 像素 或 None), ...]。
    输出: [(段落文本(可多行), 合并像素框 或 None), ...], 按阅读顺序。

    版面重排(纯几何, 无 LLM):
      0) 列切分 : 先按 x 分栏 —— 多栏界面(侧边栏+正文/IDE/聊天)左右栏不再穿插
      1) 行聚类 : 栏内垂直重叠比 > 0.5 判同一行(抗字号差异)
      2) 行内拼: 按 x 升序, 间距 > 半个字高插空格(救被切断的词)
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


# =========================================================================== #
# 便捷入口: 碎片 → 段落 → top-N → 归一化 blocks (视频流 OCR 主路径)
# =========================================================================== #
def frags_to_paragraph_blocks(
    frags: List[Tuple[str, Optional[Tuple[float, float, float, float]]]],
    img_w: int,
    img_h: int,
    *,
    top_n: int = _OCR_TOP_N,
) -> Tuple[str, List[Dict[str, Any]]]:
    """把 OCR 碎片整流成"段落级 blocks + 通顺全文"。

    frags: [(text, px_box 或 None), ...]  (px_box = box_to_pixels 的结果)。
    返回 (raw_text, blocks):
      * blocks[i] = {text, bbox:[x,y,w,h](归一化,原点左下), confidence:1.0,
                     region_type:'ocr_paragraph'}
      * raw_text = 各段落文本按阅读顺序、段间双换行拼接。

    步骤: reflow_ocr_lines 版面重排 -> 按面积保留 top-N(仍按阅读顺序输出)
          -> norm_xywh_from_pixels 归一化 bbox -> 拼 raw_text。
    confidence 段落级统一 1.0(段是多碎片合并的, 原碎片置信度已在过滤阶段用掉)。"""
    paras = list(reflow_ocr_lines(frags))       # [(text, px 或 None), ...]
    if top_n and len(paras) > top_n:
        def _area(px):
            if not px:
                return 0.0
            x0, y0, x1, y1 = px
            return max(0.0, x1 - x0) * max(0.0, y1 - y0)
        idx = sorted(range(len(paras)),
                     key=lambda i: _area(paras[i][1]), reverse=True)[:top_n]
        keep = sorted(idx)                       # 还原阅读顺序
        paras = [paras[i] for i in keep]

    blocks: List[Dict[str, Any]] = []
    texts: List[str] = []
    for text, px in paras:
        t = (text or "").strip()
        if not t:
            continue
        bbox = norm_xywh_from_pixels(px, img_w, img_h) if px else []
        blocks.append({
            "text": t,
            "bbox": bbox or [],
            "confidence": 1.0,
            "region_type": "ocr_paragraph",
        })
        texts.append(t)
    raw_text = "\n\n".join(texts)
    return raw_text, blocks
