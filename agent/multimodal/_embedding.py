# -*- coding: utf-8 -*-
"""Text embedding client for the multimodal memory system.

一期只做文本 embedding（micro description / entity name+attrs），供 MemoryStore
的向量检索路径使用；frame image embedding 放到二期。

设计要点
--------
1. **OpenAI-compatible /embeddings 协议**: DashScope 通过 compatible-mode 端点
   （``https://dashscope.aliyuncs.com/compatible-mode/v1``）暴露标准协议，
   text-embedding-v3 支持 ``dimensions`` 参数（256/512/768/1024）。同一份客户端
   也能对接 OpenAI / 自建 vLLM。
2. **sync + async 双入口**: Writer 是 async（``embed_texts()`` / 后台任务），
   MemoryToolBox.call 是 sync 但跑在 asyncio.to_thread 池里（可以放心用同步 HTTP）。
3. **失败即降级**: 单次调用失败返回 None（不抛异常），调用方判空后继续走关键词
   兜底，不阻塞主流程。
4. **归一化输出**: 输出向量已 L2 归一化，后续存 float16 BLOB + 点积 = 余弦相似度。

配置来源
--------
四件套 + 数值参数，与 model.memory / model.monitor 等对齐：
- ``model.embedding.provider``  ("openai" / "custom" / ""; ""→关闭 embedding)
- ``model.embedding.base_url``  (空 → 关闭)
- ``model.embedding.api_key``
- ``model.embedding.model``     (默认 "text-embedding-v3")
- ``model.embedding.dimensions`` (默认 1024)
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional, Sequence

import numpy as np

log = logging.getLogger("hermes.multimodal.embedding")

# 底层 HTTP：httpx 是主 agent 已有的依赖，同步/异步 API 一致。
try:
    import httpx  # type: ignore
except Exception as exc:  # pragma: no cover - handled at runtime
    httpx = None  # type: ignore
    log.warning("[embedding] httpx unavailable (%s); embedding disabled", exc)


class EmbeddingClient:
    """Thin wrapper around OpenAI-compatible ``POST /embeddings``.

    Not tied to a specific vendor: any endpoint that speaks the OpenAI protocol
    works (DashScope compatible-mode, self-hosted vLLM, OpenAI proper). Failure
    modes are logged and swallowed — callers get ``None`` and fall back.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str = "text-embedding-v3",
        dimensions: int = 1024,
        timeout_sec: float = 8.0,
        max_input_chars: int = 2000,
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.model = model or "text-embedding-v3"
        self.dimensions = int(dimensions) if dimensions else 1024
        self.timeout_sec = float(timeout_sec) if timeout_sec else 8.0
        # 单条文本上限（token≈char/2, 2000 char ≈ 1000 token, dashscope 单条 8192 上限内绰绰有余）
        self.max_input_chars = int(max_input_chars) if max_input_chars else 2000

    # ------------------------------------------------------------------ #
    # Availability
    # ------------------------------------------------------------------ #
    @property
    def enabled(self) -> bool:
        return bool(httpx is not None and self.base_url and self.api_key)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _prep_inputs(self, texts: Sequence[str]) -> List[str]:
        """Trim/clean input strings to the per-call char cap."""
        out: List[str] = []
        for t in texts:
            s = (t or "").strip()
            if not s:
                # Some providers reject empty strings; substitute a placeholder so
                # the batch index alignment stays intact. Callers should filter
                # before calling if empties matter.
                s = " "
            if len(s) > self.max_input_chars:
                s = s[: self.max_input_chars]
            out.append(s)
        return out

    def _endpoint(self) -> str:
        return f"{self.base_url}/embeddings"

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _body(self, texts: Sequence[str]) -> dict:
        # DashScope compatible-mode 支持 dimensions；OpenAI-compat 其他实现若不认
        # 该字段一般会被忽略（不会 400）。encoding_format=float 是 OpenAI 默认。
        return {
            "model": self.model,
            "input": list(texts),
            "dimensions": self.dimensions,
            "encoding_format": "float",
        }

    @staticmethod
    def _parse_response(data: dict, n_expected: int) -> Optional[np.ndarray]:
        try:
            arr = data.get("data") or []
            if len(arr) != n_expected:
                log.warning(
                    "[embedding] response length mismatch: got %d, expected %d",
                    len(arr), n_expected)
                return None
            # OpenAI/DashScope 保证按 index 返回，稳一手排序
            arr = sorted(arr, key=lambda x: int(x.get("index", 0)))
            vecs = np.asarray(
                [x["embedding"] for x in arr], dtype=np.float32)
            # L2 归一化，后续点积 = 余弦
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms < 1e-8, 1.0, norms)
            return (vecs / norms).astype(np.float32)
        except Exception as e:
            log.warning("[embedding] parse response failed: %s", e)
            return None

    # ------------------------------------------------------------------ #
    # Sync API (used by MemoryToolBox.call which runs inside asyncio.to_thread)
    # ------------------------------------------------------------------ #
    def embed_texts_sync(self, texts: Sequence[str]) -> Optional[np.ndarray]:
        if not self.enabled or not texts:
            return None
        prepped = self._prep_inputs(texts)
        t0 = time.time()
        try:
            with httpx.Client(timeout=self.timeout_sec) as client:
                resp = client.post(
                    self._endpoint(), headers=self._headers(),
                    json=self._body(prepped))
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            log.warning("[embedding] sync request failed: %s", e)
            return None
        vecs = self._parse_response(data, len(prepped))
        log.debug("[embedding] sync %d texts in %.2fs",
                  len(prepped), time.time() - t0)
        return vecs

    def embed_text_sync(self, text: str) -> Optional[np.ndarray]:
        """Convenience wrapper: returns a single (D,) vector or None."""
        vecs = self.embed_texts_sync([text])
        if vecs is None or vecs.shape[0] == 0:
            return None
        return vecs[0]

    # ------------------------------------------------------------------ #
    # Async API (used by MemoryWriter / MemoryReviewer background tasks)
    # ------------------------------------------------------------------ #
    async def embed_texts(self, texts: Sequence[str]) -> Optional[np.ndarray]:
        if not self.enabled or not texts:
            return None
        prepped = self._prep_inputs(texts)
        t0 = time.time()
        try:
            async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
                resp = await client.post(
                    self._endpoint(), headers=self._headers(),
                    json=self._body(prepped))
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            log.warning("[embedding] async request failed: %s", e)
            return None
        vecs = self._parse_response(data, len(prepped))
        log.debug("[embedding] async %d texts in %.2fs",
                  len(prepped), time.time() - t0)
        return vecs

    async def embed_text(self, text: str) -> Optional[np.ndarray]:
        vecs = await self.embed_texts([text])
        if vecs is None or vecs.shape[0] == 0:
            return None
        return vecs[0]


# --------------------------------------------------------------------------- #
# DashScope multimodal embedding (二期: frame image embedding, T→I 跨模态检索)
# --------------------------------------------------------------------------- #
class MultimodalEmbeddingClient:
    """DashScope 多模态 embedding 客户端 (multimodal-embedding-v1).

    与 EmbeddingClient (OpenAI 协议, text-embedding-v3) 的差异:
      - 端点: DashScope 原生 /api/v1/services/embeddings/multimodal-embedding/multimodal-embedding
      - 协议: {"input": {"contents": [{"text":...}/{"image":"data:image/jpeg;base64,..."}]}}
      - 响应: output.embeddings[].embedding (按 index 对齐输入)
      - 空间: text 与 image 向量在同一语义空间 → query 文本向量可直接与帧图像
        向量做余弦, 实现 T→I 跨模态检索 ("那个奇怪的交通工具" → 历史关键帧).

    ★ 与一期文本向量 (text-embedding-v3) 是**两个独立空间**, 不可混用:
      micro/entity 文本向量仍走 text-embedding-v3; frame 图像向量与
      search_frames_by_text 的 query 向量走本客户端.
    """

    DEFAULT_ENDPOINT = (
        "https://dashscope.aliyuncs.com/api/v1/services/"
        "embeddings/multimodal-embedding/multimodal-embedding"
    )
    # multimodal-embedding-v1 单图 ≤3MB (原始字节). base64 膨胀 ~4/3,
    # 留余量按原始 2.8MB 检查.
    MAX_IMAGE_BYTES = 2_800_000

    def __init__(
        self,
        *,
        base_url: str = "",
        api_key: str = "",
        model: str = "multimodal-embedding-v1",
        timeout_sec: float = 6.0,
        max_input_chars: int = 500,
        dimensions: int = 0,
        res_level: int = -1,
    ) -> None:
        self.base_url = (base_url or "").rstrip("/") or self.DEFAULT_ENDPOINT
        self.api_key = api_key or ""
        self.model = model or "multimodal-embedding-v1"
        self.timeout_sec = float(timeout_sec) if timeout_sec else 6.0
        # v1 文本上限 512 token; 中文 ~1字/token, 留余量截 500 字.
        self.max_input_chars = int(max_input_chars) if max_input_chars else 500
        # ★ tongyi-embedding-vision-{plus,flash}-2026-03-06 专属参数:
        #   dimensions > 0 → 传 parameters.dimension (64~1152/768);
        #   res_level in 0..3 → 传 parameters.res_level (单图 127/402/578/1026 token,
        #   高档位对"小物体在大场景"的帧检索 +5~10% 效果).
        #   默认 0 / -1 = 不传 (multimodal-embedding-v1 等老模型不支持, 保持兼容).
        self.dimensions = int(dimensions) if dimensions else 0
        self.res_level = int(res_level) if res_level is not None else -1

    @property
    def enabled(self) -> bool:
        # base_url 有默认值, 所以是否启用只看 api_key + model
        return bool(httpx is not None and self.api_key and self.model)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _body(self, contents: list) -> dict:
        body: dict = {"model": self.model, "input": {"contents": contents}}
        # 仅新版快照模型 (tongyi-embedding-vision-*-2026-03-06) 支持这两个
        # 参数; 老模型 (multimodal-embedding-v1 / tongyi-embedding-vision-plus)
        # 传了会被 400 拒绝, 所以只在显式配置时才带上.
        params: dict = {}
        if self.dimensions > 0:
            params["dimension"] = self.dimensions
        if 0 <= self.res_level <= 3:
            params["res_level"] = self.res_level
        if params:
            body["parameters"] = params
        return body

    def _parse(self, data: dict, n_expected: int) -> Optional[np.ndarray]:
        try:
            arr = ((data.get("output") or {}).get("embeddings")) or []
            if len(arr) != n_expected:
                log.warning(
                    "[mm-embedding] response length mismatch: got %d, expected %d",
                    len(arr), n_expected)
                return None
            arr = sorted(arr, key=lambda x: int(x.get("index", 0)))
            vecs = np.asarray([x["embedding"] for x in arr], dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms < 1e-8, 1.0, norms)
            return (vecs / norms).astype(np.float32)
        except Exception as e:
            log.warning("[mm-embedding] parse response failed: %s", e)
            return None

    @staticmethod
    def _image_content(jpeg_b64: str) -> Optional[dict]:
        """Build the image content item; None if the payload is oversized."""
        b64 = (jpeg_b64 or "").strip()
        if not b64:
            return None
        if b64.startswith("data:"):
            comma = b64.find(",")
            if comma >= 0:
                b64 = b64[comma + 1:]
        # base64 长度 * 3/4 ≈ 原始字节数
        approx_bytes = len(b64) * 3 // 4
        if approx_bytes > MultimodalEmbeddingClient.MAX_IMAGE_BYTES:
            log.warning(
                "[mm-embedding] image too large (~%d bytes > %d), skipped",
                approx_bytes, MultimodalEmbeddingClient.MAX_IMAGE_BYTES)
            return None
        return {"image": f"data:image/jpeg;base64,{b64}"}

    def _text_content(self, text: str) -> dict:
        s = (text or "").strip() or " "
        if len(s) > self.max_input_chars:
            s = s[: self.max_input_chars]
        return {"text": s}

    # ------------------------------------------------------------------ #
    # Async API (Writer 后台任务)
    # ------------------------------------------------------------------ #
    async def _embed_contents(self, contents: list) -> Optional[np.ndarray]:
        if not self.enabled or not contents:
            return None
        try:
            async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
                resp = await client.post(
                    self.base_url, headers=self._headers(),
                    json=self._body(contents))
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            log.warning("[mm-embedding] async request failed: %s", e)
            return None
        return self._parse(data, len(contents))

    async def embed_image(self, jpeg_b64: str) -> Optional[np.ndarray]:
        """单张图像 → (D,) 归一化向量. 超限/失败返回 None."""
        item = self._image_content(jpeg_b64)
        if item is None:
            return None
        vecs = await self._embed_contents([item])
        return None if vecs is None or vecs.shape[0] == 0 else vecs[0]

    async def embed_text(self, text: str) -> Optional[np.ndarray]:
        """query 文本 → 与图像同空间的 (D,) 向量 (T→I 检索的 query 侧)."""
        vecs = await self._embed_contents([self._text_content(text)])
        return None if vecs is None or vecs.shape[0] == 0 else vecs[0]

    # ------------------------------------------------------------------ #
    # Sync API (MemoryToolBox, 跑在 asyncio.to_thread 池里)
    # ------------------------------------------------------------------ #
    def _embed_contents_sync(self, contents: list) -> Optional[np.ndarray]:
        if not self.enabled or not contents:
            return None
        try:
            with httpx.Client(timeout=self.timeout_sec) as client:
                resp = client.post(
                    self.base_url, headers=self._headers(),
                    json=self._body(contents))
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            log.warning("[mm-embedding] sync request failed: %s", e)
            return None
        return self._parse(data, len(contents))

    def embed_image_sync(self, jpeg_b64: str) -> Optional[np.ndarray]:
        item = self._image_content(jpeg_b64)
        if item is None:
            return None
        vecs = self._embed_contents_sync([item])
        return None if vecs is None or vecs.shape[0] == 0 else vecs[0]

    def embed_text_sync(self, text: str) -> Optional[np.ndarray]:
        vecs = self._embed_contents_sync([self._text_content(text)])
        return None if vecs is None or vecs.shape[0] == 0 else vecs[0]


# --------------------------------------------------------------------------- #
# BLOB serialization helpers (used by MemoryStore)
# --------------------------------------------------------------------------- #
def encode_vector(vec: Optional[np.ndarray]) -> Optional[bytes]:
    """Serialize a (D,) float32/64 vector to a float16 BLOB (halves storage).

    float16 gives ~3 significant decimal digits — plenty for cosine-sim ranking.
    A 1024-dim vector → 2048 bytes/row (vs 4096 bytes if we kept float32).
    """
    if vec is None:
        return None
    arr = np.asarray(vec).astype(np.float16, copy=False)
    return arr.tobytes()


def decode_vector(blob: Optional[bytes]) -> Optional[np.ndarray]:
    """Deserialize a float16 BLOB back to a float32 (D,) vector (for math).

    Returns None on empty/malformed input so callers can skip that row.
    """
    if not blob:
        return None
    try:
        return np.frombuffer(blob, dtype=np.float16).astype(np.float32)
    except Exception as e:
        log.debug("[embedding] decode blob failed: %s", e)
        return None


def decode_matrix(blobs: Sequence[Optional[bytes]]) -> Optional[np.ndarray]:
    """Deserialize a list of BLOBs into a stacked (N, D) float32 matrix.

    Rows with malformed / dimension-mismatched blobs are silently skipped;
    returns None when nothing survives. The caller MUST use the same "kept"
    filter to align row ids with matrix rows.
    """
    vecs: List[np.ndarray] = []
    for b in blobs:
        v = decode_vector(b)
        if v is not None and v.size > 0:
            vecs.append(v)
    if not vecs:
        return None
    # 允许不同 dim 的历史 blob 共存时按最常见 dim 过滤（防止 stack 报错）
    dims = [v.size for v in vecs]
    d = max(set(dims), key=dims.count)
    kept = [v for v in vecs if v.size == d]
    if not kept:
        return None
    return np.stack(kept, axis=0).astype(np.float32)
