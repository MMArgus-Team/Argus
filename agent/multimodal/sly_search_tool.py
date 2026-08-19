import os
import time
import json
import asyncio
import base64
import logging
import random
from io import BytesIO
from typing import Union, List

import httpx
from PIL import Image

# 配置日志
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# --- 配置部分 ---
# ★ 整合到 Hermes 时的改动: 原文件在 import 时直接写 os.environ["http_proxy"],
#   这会把整个 dashboard 进程的所有 HTTP (qwen/gemini 等) 都强制走内网代理而崩掉.
#   改为: 不污染全局 env, 代理只作用于本工具内部的 httpx client (per-client proxy).
#   代理 / RAG 地址都允许用环境变量覆盖 (内网地址, 换环境时改这俩即可).
PROXY_URL = os.environ.get("MM_SEARCH_PROXY_URL", "http://10.7.4.2:3128") or None
RAG_API_URL = os.environ.get(
    "MM_SEARCH_RAG_API_URL",
    "http://allin-super-liguankai.devops.sl.beta.xiaohongshu.com/ser/in")

HEADERS = {"Content-Type": "application/json"}
TIMEOUT_CONFIG = httpx.Timeout(300.0, connect=4.0)


def _new_client() -> httpx.AsyncClient:
    """A dedicated httpx client for this tool: the proxy applies only here and
    never pollutes the process-wide HTTP config."""
    kwargs = {"timeout": TIMEOUT_CONFIG}
    if PROXY_URL:
        kwargs["proxy"] = PROXY_URL
    return httpx.AsyncClient(**kwargs)

def compress_and_b64(image_input: Union[str, Image.Image], max_size=1024) -> str:
    """Compress an image (path or PIL.Image) to <= max_size on the long edge and
    return it as a base64-encoded JPEG."""
    try:
        if isinstance(image_input, str):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            # 修复：创建副本，防止修改原始对象
            img = image_input.copy().convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")

        # 保持比例缩放
        img.thumbnail((max_size, max_size), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=95) # 95 quality is usually sufficient
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        logging.error(f"Image processing failed: {e}")
        raise

async def _send_rag_request(client: httpx.AsyncClient, payload: dict, task_name: str) -> List[dict]:
    """POST payload to the RAG API with up to 3 attempts (backoff between retries).
    Returns the response's ``data`` list, or [] on repeated failure."""
    for attempt in range(3):
        start_req = time.perf_counter()
        try:
            response = await client.post(
                url=RAG_API_URL,
                headers=HEADERS,
                json=payload
            )
            req_cost_ms = (time.perf_counter() - start_req)
            logging.info(
                f"[{task_name}] "
                f"Status: {response.status_code} "
                f"Cost: {req_cost_ms:.2f}s "
                f"(Attempt {attempt + 1})"
            )
            response.raise_for_status()
            resp_json = response.json()

            if isinstance(resp_json.get("data"), list):
               return resp_json["data"]
            
            logging.warning(f"[{task_name}] Unexpected response type: {type(resp_json)}")
            return []
            

        except (httpx.RequestError, httpx.HTTPStatusError, json.JSONDecodeError) as e:
            logging.warning(f"[{task_name}] Retry {attempt+1}/3 failed: {e}")
            # C22: 只有还有下一轮时才退避等待, 最后一轮失败直接结束, 避免无谓延迟
            if attempt < 2:
                await asyncio.sleep(random.uniform(0.5 + attempt * 0.5, 3.0))
        
        except Exception as e:
            logging.error(f"[{task_name}] Unexpected error: {e}")
            break
            
    logging.error(f"[{task_name}] All retries failed.")
    return []

async def unified_image_search(
        image_input: Union[str, Image.Image], 
        keys: list = ['xhs', 'google'], 
        threshold: float = 0.6, 
        extra_params: dict = None
) -> List[dict]:
    
        img_base64 = compress_and_b64(image_input)

        # 公共基础参数（版本二格式）
        base_params = {
            "messages": [],
            "query_text": "",
            "query_image_base64": f"data:image/jpeg;base64,{img_base64}",
            "use_note_modality": "one_image",
            "reranker_compute_type": ["image2text"],
        }

        async with _new_client() as client:
            tasks = []

            # --- Google Task ---
            # 策略：不走 threshold 过滤，直接取前3，不做 rerank
            if 'google' in keys:
                google_params = base_params.copy()
                google_params.update({
                        "activeXhsImage": False,
                        "activeGoogleImage": True,
                        "disable_search_note_image": True,
                        "disable_search_google_image": False,
                        "num_search_google_image": 3,
                        "disable_rerank": True,
                })
                if extra_params:
                    google_params.update(extra_params)

                async def google_worker():
                        res = await _send_rag_request(client, google_params, "GoogleImg")
                        return res[:3]  # 不过 threshold，直接截断前3

                tasks.append(google_worker())

            # --- XHS Task ---
            # 策略：走 threshold 过滤，做 rerank
            if 'xhs' in keys:
                xhs_params = base_params.copy()
                xhs_params.update({
                        "activeXhsImage": True,
                        "activeGoogleImage": False,
                        "disable_search_note_image": False,
                        "disable_search_google_image": True,
                        "num_search_note_image": 10,
                        "disable_rerank": False,
                })
                if extra_params:
                    xhs_params.update(extra_params)

                async def xhs_worker():
                    res = await _send_rag_request(client, xhs_params, "XhsImg")
                    return [item for item in res if item.get('score', 0) >= threshold]

                tasks.append(xhs_worker())

            results = await asyncio.gather(*tasks)

        # 合并并按 score 降序排列
        merged_data = []
        for result in results:
            merged_data.extend(result)
        merged_data.sort(key=lambda x: x.get('score', 0.8), reverse=True)
        return merged_data

async def unified_text_search(
            query: str, 
            keys: list = ['xhs'], 
            threshold: float = 0.6, 
            thresh_topk: int = 10,
            extra_params: dict = None) -> List[dict]:
    
    request_json = {
        "activeXhsText": "xhs" in keys,
        "activeGoogleText": "google" in keys,
        "messages": [{"content": query}],
        "disable_search_note_text": "xhs" not in keys,
        "disable_search_google_text": "google" not in keys,
        "disable_video": True,
        "num_search_note_text": 10,
        "num_search_google_text": 10,
        "use_note_modality": "text",
        "query_text": query,
        "reranker_compute_type": ["text2text"],
        "disable_rerank": False,
        "fetch_detail": False
    }

    if extra_params:
        request_json.update(extra_params)

    async with _new_client() as client:
        data = await _send_rag_request(client, request_json, "TextSearch")
    
    filtered_data = [item for item in data if item.get('score', 0) >= threshold]
    filtered_data = filtered_data[:thresh_topk]
    filtered_data.sort(key=lambda x: x.get('score', 0.8), reverse=True)

    return filtered_data

async def text_search_observation(query: str, text_search_keys=['google', 'xhs']):
    
    obs_list = await unified_text_search(query, keys=text_search_keys)
    if not obs_list: return "无相关信息返回"
    
    txt_list = []
    for i, obs in enumerate(obs_list):

        score = obs.get('score', 0)
        title = obs.get('title')
        content = obs.get('content', '')
        txt_list.append(f"网页{i+1}. 相关性: {score:.2f} 标题: {title} 内容: {content}")
    
    return "\n\n".join(txt_list)

async def image_search_observation(image_input, image_search_keys=['google', 'xhs']):
    data = await unified_image_search(image_input, keys=image_search_keys)
    if not data: return "无相关信息返回", []
    
    tool_str_list = []
    for i, item in enumerate(data):
        title = item.get("title", "")
        content = item.get("content", "")
        score = item.get("score", 0.8)
        tool_str_list.append(
            f"网页 {i+1}. 相关性: {score:.2f} 标题: {title}\n 内容: {content}\n"
        )
    
    return "\n".join([f"{i+1}. {s}" for i, s in enumerate(tool_str_list)]), []


if __name__ == "__main__":
    
    print("#" * 200)
    #  image_path = r"/mnt/tidalfs-bdsz01/dataset/llm_dataset/video_agent_data/video_agent_sft_dataset_1219/Encyclopedic-VQA/iNaturalist/train_mini/03928_Animalia_Chordata_Aves_Passeriformes_Meliphagidae_Entomyzon_cyanotis/6fd177af-3c7f-4449-bff2-0bde4d88e684.jpg"
    # image_path = r"/mnt/tidalfs-bdsz01/dataset/llm_dataset/video_agent_data/video_agent_sft_dataset_1219/Encyclopedic-VQA/google-landmarks/images/0/5/5/055018e3f799b9c9.jpg"
    # image_path = r"/mnt/tidalfs-bdsz01/dataset/llm_dataset/video_agent_data/FVQA/FVQA-ZH/images/fvqa_test_188.png"
    # image_path = r"/mnt/tidalfs-bdsz01/dataset/llm_dataset/video_agent_data/FVQA/FVQA-ZH/images/fvqa_test_495.png"
    #  image_path = r"/mnt/tidalfs-bdsz01/dataset/llm_dataset/video_agent_data/FVQA/FVQA-ZH/images/fvqa_test_611.png"
    image_path = r"/mnt/tidalfs-bdsz01/usr/chenjiabin/dataset/single-turn-with-search-anno-image/1028-19.webp"

    # extra_params = {
    #     "fetch_detail": True,
    # }
    extra_params = None
#    data = asyncio.run(unified_image_search(image_path, ['google', 'xhs'], 0.6, extra_params))
#    print(json.dumps(data, indent = 2, ensure_ascii=False))

    data = asyncio.run(unified_text_search("2026 年 5 月 28 日 腾讯 收盘价", ['google', 'xhs'], 0.8,  extra_params = extra_params))
    print(json.dumps(data, indent = 2, ensure_ascii=False))
