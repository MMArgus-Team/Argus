# -*- coding: utf-8 -*-
"""`hermes mm-memory-eval` 子命令: 离线视频 → 构多模态记忆 → 跑 QA 召回评测。

只负责 argparse; 实际逻辑在 hermes_cli/mm_memory_eval.py (编排) +
hermes_cli/mm_eval_io.py (文件读写/校验)。
"""
from __future__ import annotations

from typing import Callable


def build_mm_memory_eval_parser(subparsers, *, cmd_mm_memory_eval: Callable) -> None:
    p = subparsers.add_parser(
        "mm-memory-eval",
        help="离线读一个视频构建多模态记忆, 逐条跑 QA 并写回 answer_predict",
        description=(
            "读取 <video> 与 <json>{title, qa_list}, 校验 title 与视频文件名一致, "
            "把视频快速(非实时)喂进多模态记忆流水线(逻辑与在线看视频流一致, 不涉及 "
            "monitor/watcher), 然后逐条 query 经记忆召回作答, 给每个 qa 追加 "
            "answer_predict 写回 JSON。"),
    )
    p.add_argument(
        "video",
        help="视频文件路径, 或视频【文件夹】。传文件夹 → 评 JSON 里所有视频 "
             "(开跑前遍历检查, 要求每个视频都在文件夹内, 缺则报错不开跑)。")
    p.add_argument("json", help="QA JSON: 单视频 {title, qa_list} 或全量数组 [{video_url, qa_list}]")
    p.add_argument(
        "--mode", choices=["tool", "agent"], default="tool",
        help=("回答方式: tool=直接记忆召回(默认); agent=先显式召回, "
              "再由主 agent 结合证据与提问时刻最近帧综合作答"))
    p.add_argument(
        "--source", choices=["camera", "screen"], default="camera",
        help="离线视频模拟的线上输入源: camera=摄像头, screen=屏幕共享/OCR/table 路径")
    p.add_argument(
        "--answer-timing", choices=["before", "after"], default="after",
        help="时序题到达 time 时的答题时机: before=推入该帧前答; after=推入并跑 OCR/wake 后答(默认, 更贴近线上用户看见后提问)")
    p.add_argument(
        "--query-types", default=None,
        help="只评指定 query_type, 逗号分隔, 如 b 或 a,b,d,e。默认不筛选, 评 qa_list 全部题")
    p.add_argument(
        "--asr-vtt", default="auto",
        help=("给静音视频补充已有 ASR 字幕, 作为 streaming audio_observation 注入记忆。"
              "默认 auto 自动找同目录 <视频名>.auto.asr.vtt / .asr.vtt / .vtt; "
              "可传 none 关闭, 或传具体 VTT 文件/目录"))
    try:
        import argparse as _argparse
        _bool_action = _argparse.BooleanOptionalAction
    except Exception:  # pragma: no cover
        _bool_action = None
    if _bool_action is not None:
        p.add_argument(
            "--scene-probe", action=_bool_action, default=True,
            help="离线评测中是否按视频时间驱动 scene/dHash controller (默认开启; 可用 --no-scene-probe 关闭)")
    else:  # pragma: no cover
        p.add_argument("--scene-probe", action="store_true", default=True)
    p.add_argument("--out", default=None,
                   help="写回路径 (默认原地覆盖输入 JSON)")
    p.add_argument("--trace-out", default=None,
                   help="每题 recall trace sidecar 路径; 不指定则自动写到结果 JSON 旁边")
    p.add_argument(
        "--log-dir", default=None,
        help=("本次评测的独立日志目录; 默认自动写到 "
              "~/.argus/logs/mm-memory-eval/<run_id>/eval.log。并发跑多个评测时建议保持默认"))
    p.add_argument("--capture-fps", type=float, default=None,
                   help="解码采样帧率 (默认取 config 的 buffer_capture_fps=2.0, 即屏幕共享帧率)")
    p.add_argument("--max-side", type=int, default=None,
                   help="帧最长边缩放上限像素 (默认: screen=1536/ocr_max_side, camera=720)")
    p.add_argument("--jpeg-quality", type=int, default=80,
                   help="帧 JPEG 质量 (默认 80)")
    p.add_argument("--timeout", type=float, default=120.0,
                   help="单条 recall 超时秒数 (默认 120)")
    p.set_defaults(func=cmd_mm_memory_eval)
