#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把项目本地权重放到 weights/ (不入 git, 见 weights/README.md)。

目前只需一套本地权重:

  Qwen2.5-0.5B-Instruct  → weights/qwen2.5-0.5b-instruct/
    (HF 下载: Qwen/Qwen2.5-0.5B-Instruct)
    供 voice_intent_local (语音意图 / 分诊 / 语义 EOU 的本地推理)。

注: OCR 不在此下载。RapidOCR 的 PP-OCR onnx 已随 rapidocr wheel 自带 (无参
    构造 RapidOCR() 会用包内默认模型), 无需放进 weights/ 或显式指定路径。
    只需 pip install -U "rapidocr" onnxruntime 即可。

用法:
    python download_weights.py                # 下 Qwen2.5-0.5B-Instruct (已存在跳过)
    python download_weights.py --hf-mirror    # 走 hf-mirror.com (国内更快)

依赖:
    pip install huggingface_hub
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_WEIGHTS = _ROOT / "weights"

# ── Qwen2.5-0.5B-Instruct (HF model repo) ────────────────────────────────
_QWEN_REPO = "Qwen/Qwen2.5-0.5B-Instruct"
_QWEN_DIR = _WEIGHTS / "qwen2.5-0.5b-instruct"


def _ensure_hf_hub() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("ERROR: 需要 huggingface_hub:\n"
              "    pip install huggingface_hub", file=sys.stderr)
        sys.exit(2)


def _hf_snapshot(repo_id: str, local_dir: Path) -> None:
    from huggingface_hub import snapshot_download
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[hf] {repo_id} → {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        # 文件真实落地 (非 HF cache 软链), 方便再软链到 HERMES_HOME + 跨机器拷贝。
        local_dir_use_symlinks=False,
    )
    print(f"[ok] {repo_id}")


def do_qwen() -> None:
    _ensure_hf_hub()
    _hf_snapshot(_QWEN_REPO, _QWEN_DIR)
    if not (_QWEN_DIR / "config.json").is_file():
        print(f"[warn] {_QWEN_DIR} 缺 config.json — 下载可能不完整", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description="把项目本地权重放到 weights/")
    ap.add_argument("--hf-mirror", action="store_true",
                    help="走 https://hf-mirror.com 镜像 (国内更快)")
    args = ap.parse_args()

    if args.hf_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("[cfg] HF_ENDPOINT=https://hf-mirror.com")

    do_qwen()

    print("\n[done] 权重已就位。启动 dashboard 时 weights/ 会软链到 HERMES_HOME,\n"
          "       config.yaml 里以相对路径引用 (见 weights/README.md)。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
