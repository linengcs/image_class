"""
LLM Client — 封装 OpenAI Vision API 调用
- 图片压缩至 512px 短边，降低 token 成本
- 自动 base64 编码
- 重试机制
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from typing import Any

from volcenginesdkarkruntime import Ark
from PIL import Image

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
MAX_SHORT_SIDE = 512  # 图片压缩目标短边
JPEG_QUALITY = 75
MAX_RETRIES = 3
MODEL_NAME = os.getenv("VISION_MODEL", "doubao-seed-2-0-mini-260215")

client: Ark | None = None


def _get_client(api_key: str | None = None) -> Ark:
    if api_key:
        return Ark(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=api_key,
        )
    global client
    if client is None:
        env_key = os.getenv("ARK_API_KEY")
        if not env_key:
            raise RuntimeError("请传入或设置环境变量 ARK_API_KEY")
        client = Ark(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=env_key,
        )
    return client


# ---------------------------------------------------------------------------
# 图片处理
# ---------------------------------------------------------------------------

def compress_image(image_bytes: bytes) -> bytes:
    """压缩图片至 MAX_SHORT_SIDE，返回 JPEG bytes."""
    img = Image.open(io.BytesIO(image_bytes))

    # 处理 EXIF 旋转
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    # 计算缩放比例
    w, h = img.size
    short_side = min(w, h)
    if short_side > MAX_SHORT_SIDE:
        scale = MAX_SHORT_SIDE / short_side
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    # 转 RGB（去掉 alpha 通道）
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def image_to_base64(image_bytes: bytes) -> str:
    """将图片 bytes 编码为 data URI."""
    compressed = compress_image(image_bytes)
    b64 = base64.b64encode(compressed).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# ---------------------------------------------------------------------------
# LLM 调用
# ---------------------------------------------------------------------------

def call_vision(
    images_b64: list[str],
    prompt: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    api_key: str | None = None,
) -> str:
    """
    调用 Vision 模型，返回文本响应。

    Parameters
    ----------
    images_b64 : 已编码的 base64 data-URI 列表
    prompt : 文本提示
    temperature : 生成温度
    max_tokens : 最大输出 token
    """
    # 构造 content 数组
    content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    for img_uri in images_b64:
        content.append({
            "type": "input_image",
            "image_url": img_uri,
        })

    messages = [{"role": "user", "content": content}]

    for attempt in range(MAX_RETRIES):
        try:
            resp = _get_client(api_key).responses.create(
                model=MODEL_NAME,
                input=messages,
                temperature=temperature,
            )
            
            if hasattr(resp, "output") and isinstance(resp.output, list):
                for item in resp.output:
                    if getattr(item, "type", "") == "message":
                        content_list = getattr(item, "content", [])
                        for c in content_list:
                            if getattr(c, "type", "") == "output_text":
                                return getattr(c, "text", "")
            return ""
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt
                print(f"[LLM] 重试 {attempt + 1}/{MAX_RETRIES}，等待 {wait}s ... 错误: {e}")
                time.sleep(wait)
            else:
                raise
    
    return ""

def call_vision_json(
    images_b64: list[str],
    prompt: str,
    **kwargs,
) -> Any:
    """调用 Vision 模型并解析 JSON 响应."""
    raw = call_vision(images_b64, prompt, **kwargs)

    # 尝试从 markdown code block 中提取 JSON
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0]
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0]

    return json.loads(raw.strip())
