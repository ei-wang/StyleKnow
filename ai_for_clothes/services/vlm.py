"""
统一的 VLM（视觉大模型）能力封装：
- 图片类型判断（全身照 vs 单件衣物）
- 全身照多件衣物解析
- 小红书穿搭公式提取

upload API 与 agent 节点共用，避免重复实现。
"""

from __future__ import annotations

import base64
import json
import re
from typing import Any, Dict, List, Optional, Literal

from openai import AsyncOpenAI, OpenAI
import httpx

from storage.config_store import load_config
from domain.clothes import ClothesSemantics
from services.openai_compatible import extract_json_from_response


ImageType = Literal["full_body", "single_item"]


def _get_openai_client() -> OpenAI:
    """获取配置好的 OpenAI 客户端（同步）"""
    config = load_config()
    if not config.api_key:
        raise ValueError("缺少 API Key，请在配置中设置")
    
    # 兼容处理 base_url
    base_url = (config.api_base or "").rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1"
    
    return OpenAI(
        api_key=config.api_key,
        base_url=base_url
    )


def _get_async_openai_client() -> AsyncOpenAI:
    """获取配置好的 AsyncOpenAI 客户端"""
    config = load_config()
    if not config.api_key:
        raise ValueError("缺少 API Key，请在配置中设置")
    
    # 兼容处理 base_url
    base_url = (config.api_base or "").rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1"
    
    return AsyncOpenAI(
        api_key=config.api_key,
        base_url=base_url
    )


def _image_to_data_url(image_bytes: bytes, mime: str = "image/png") -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def detect_image_type_sync(image_bytes: bytes) -> ImageType:
    """
    判断图片类型：全身照（多件）/单件衣物（单品）。
    """
    config = load_config()
    if not config.api_key:
        return "single_item"

    prompt = (
        "你是一个图像识别专家。请分析这张图片，判断它是：\n"
        "1. 全身照（包含多件衣物的人像照，如上衣+裤子+鞋子等完整穿搭）\n"
        "2. 单件衣物（只有一件衣物的产品图或平铺图）\n\n"
        "只需要回答全身照或单件衣物，不要其他内容。"
    )

    try:
        client = _get_openai_client()
        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": _image_to_data_url(image_bytes)}},
                    ],
                }
            ],
            max_tokens=50,
        )
        content = response.choices[0].message.content.strip()
        return "full_body" if "全身" in content else "single_item"
    except Exception:
        return "single_item"


async def detect_image_type(image_bytes: bytes) -> ImageType:
    """
    detect_image_type_sync 的 async 版本。
    """
    config = load_config()
    if not config.api_key:
        return "single_item"

    prompt = (
        "你是一个图像识别专家。请分析这张图片，判断它是：\n"
        "1. 全身照（包含多件衣物的人像照，如上衣+裤子+鞋子等完整穿搭）\n"
        "2. 单件衣物（只有一件衣物的产品图或平铺图）\n\n"
        "只需要回答全身照或单件衣物，不要其他内容。"
    )

    try:
        client = _get_async_openai_client()
        response = await client.chat.completions.create(
            model=config.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": _image_to_data_url(image_bytes)}},
                    ],
                }
            ],
            max_tokens=50,
        )
        content = response.choices[0].message.content.strip()
        return "full_body" if "全身" in content else "single_item"
    except Exception:
        return "single_item"


def analyze_full_body_items_sync(image_bytes: bytes) -> List[ClothesSemantics]:
    """
    分析全身照，输出其中所有可见衣物（多件）。
    """
    config = load_config()
    if not config.api_key:
        return []

    prompt = (
        "你是一个专业的衣物识别助手。请分析这张全身穿搭照，识别出所有可见的衣物。\n\n"
        "请按以下 JSON 数组格式输出（每件衣物一个对象）：\n"
        "[\n"
        "  {\n"
        "    \"category\": \"上衣/裤子/裙子/鞋子/配饰/外套\",\n"
        "    \"item\": \"具体款式，如T恤、衬衫、牛仔裤、运动鞋等\",\n"
        "    \"style_semantics\": [\"风格标签列表\"],\n"
        "    \"season_semantics\": [\"季节标签\"],\n"
        "    \"usage_semantics\": [\"使用场景标签\"],\n"
        "    \"color_semantics\": [\"颜色标签\"],\n"
        "    \"description\": \"一段描述\",\n"
        "    \"material\": \"材质\",\n"
        "    \"pattern\": \"图案\",\n"
        "    \"fit\": \"版型\"\n"
        "  }\n"
        "]\n\n"
        "请尽可能识别出所有衣物，包括：内搭、外套、裤子/裙子、鞋子、配饰等。"
        "如果不确定某件衣物的属性，用空字符串或空数组表示。"
        "只输出 JSON 数组，不要其他内容。"
    )

    try:
        client = _get_openai_client()
        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": _image_to_data_url(image_bytes)}},
                    ],
                }
            ],
            max_tokens=2000,
        )
        content = response.choices[0].message.content
        
        # 尽量兼容 markdown 包裹
        arr_match = re.search(r"\[[\s\S]*\]", content)
        if not arr_match:
            return []
        items_data = json.loads(arr_match.group())
        out: List[ClothesSemantics] = []
        for item in items_data:
            try:
                out.append(ClothesSemantics(**item))
            except Exception:
                continue
        return out
    except Exception:
        return []


def extract_outfit_formula_sync(xhs_image_url: str) -> Optional[Dict[str, Any]]:
    """
    从一张穿搭图提取穿搭公式（整体风格/配色/单品/技巧）。
    """
    config = load_config()
    if not config.api_key:
        return None

    try:
        if xhs_image_url.startswith("http"):
            img_resp = httpx.get(xhs_image_url, timeout=30.0)
            img_resp.raise_for_status()
            image_bytes = img_resp.content
        else:
            with open(xhs_image_url, "rb") as f:
                image_bytes = f.read()
    except Exception:
        return None

    prompt = (
        "你是一个专业的时尚搭配师。请分析这张穿搭图片，提取详细的穿搭公式。\n\n"
        "请按以下 JSON 格式输出：\n"
        "{\n"
        "  \"overall_style\": \"整体风格，如简约、复古、街头\",\n"
        "  \"color_scheme\": \"主色调，如黑白配、同色系\",\n"
        "  \"items\": [\n"
        "    {\n"
        "      \"category\": \"上衣/裤子/裙子/鞋子/配饰/外套\",\n"
        "      \"description\": \"具体描述，如白色基础款T恤\",\n"
        "      \"key_features\": [\"特点1\", \"特点2\"]\n"
        "    }\n"
        "  ],\n"
        "  \"matching_tips\": [\"搭配技巧1\", \"搭配技巧2\"]\n"
        "}\n\n"
        "请尽可能详细地识别图中所有可见的衣物配饰。只输出 JSON，不要其他内容。"
    )

    try:
        client = _get_openai_client()
        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": _image_to_data_url(image_bytes, mime="image/jpeg")}},
                    ],
                }
            ],
            max_tokens=1500,
        )
        content = response.choices[0].message.content
        return extract_json_from_response(content)
    except Exception:
        return None
