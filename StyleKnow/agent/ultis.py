# -*- coding: utf-8 -*-
"""
工具函数集合

包含：
- LLM 配置和初始化（同步 + 异步）
- 数据库初始化
- 偏好更新函数
- JSON 解析工具
"""

import os
import json
import re
import asyncio
import numpy as np
from typing import List, Dict, Optional, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


# ========== LLM 配置 ==========
# 从配置中加载
from storage.config_store import load_config as _load_config

_config = _load_config()
OPENAI_API_KEY: str = _config.api_key or os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL: str = _config.model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_BASE: str = _config.api_base or os.environ.get("OPENAI_API_BASE", "")

# 同步 LLM 实例（用于 embedding 等）
llm_instance: Optional[ChatOpenAI] = None

# 异步 LLM 实例（用于对话和 Agent）
async_llm_instance: Optional[ChatOpenAI] = None


def get_llm() -> Optional[ChatOpenAI]:
    """获取或初始化同步 LLM 实例（用于 embedding 等）"""
    global llm_instance
    if llm_instance is None:
        if not OPENAI_API_KEY:
            print("[WARNING] 未设置 OPENAI_API_KEY，将使用 Mock 模式")
            return None

        # 构建 API 基础 URL
        base_url = OPENAI_API_BASE.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = base_url + "/v1"

        llm_instance = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0.7,
            api_key=OPENAI_API_KEY,
            base_url=base_url
        )
    return llm_instance


def get_async_llm() -> Optional[ChatOpenAI]:
    """获取或初始化异步 LLM 实例（用于对话和 Agent）

    使用 ChatOpenAI 的 ainvoke 方法实现异步调用。
    httpx 客户端会复用连接，提升调用速度。
    """
    global async_llm_instance
    if async_llm_instance is None:
        if not OPENAI_API_KEY:
            print("[WARNING] 未设置 OPENAI_API_KEY，将使用 Mock 模式")
            return None

        # 构建 API 基础 URL
        base_url = OPENAI_API_BASE.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = base_url + "/v1"

        # 使用 ChatOpenAI，支持 ainvoke 异步调用
        # 内部使用 httpx.AsyncClient，复用连接
        async_llm_instance = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0.7,
            api_key=OPENAI_API_KEY,
            base_url=base_url,
        )
    return async_llm_instance


# ========== 数据库初始化 ==========
from storage.db import InMemoryWardrobeDB, RecommendedClothingDB

wardrobe_db: Optional[InMemoryWardrobeDB] = None
recommended_db: Optional[RecommendedClothingDB] = None


def init_wardrobe_db() -> InMemoryWardrobeDB:
    """初始化数据库实例"""
    global wardrobe_db
    if wardrobe_db is None:
        wardrobe_db = InMemoryWardrobeDB()
        # wardrobe_db.mock()
    return wardrobe_db


def get_wardrobe_db() -> InMemoryWardrobeDB:
    """获取数据库实例（如果未初始化则初始化）"""
    global wardrobe_db
    if wardrobe_db is None:
        wardrobe_db = InMemoryWardrobeDB()
    return wardrobe_db


def init_recommended_db(default_ttl_days: int = 7) -> RecommendedClothingDB:
    """初始化推荐数据库实例"""
    global recommended_db
    if recommended_db is None:
        recommended_db = RecommendedClothingDB(default_ttl_days=default_ttl_days)
    return recommended_db


def get_recommended_db() -> RecommendedClothingDB:
    """获取推荐数据库实例（如果未初始化则初始化）"""
    global recommended_db
    if recommended_db is None:
        recommended_db = RecommendedClothingDB()
    return recommended_db


# ========== 用户偏好更新 ==========
def update_user_preference(
    user_id: str,
    prefs: Dict[str, Any],
    scene: str = None
) -> bool:
    """
    更新用户偏好（支持场景化）

    Args:
        user_id: 用户 ID
        prefs: 偏好字典，如 {"style": "简约", "color": "蓝色"}
        scene: 可选，指定场景。如果不指定，更新默认场景的偏好。

    Returns:
        是否更新成功
    """
    db = get_wardrobe_db()
    return db.update_user_preference(user_id, prefs, scene=scene)


# ========== JSON 解析工具 ==========
def parse_llm_json_response(response: str) -> str:
    """
    健壮地解析 LLM 的 JSON 输出
    
    处理各种边缘情况：Markdown 代码块、多余文本、格式错误等
    
    Args:
        response: LLM 返回的原始响应
    
    Returns:
        格式化的反馈字符串 "[PASS] 理由" 或 "[REJECT] 理由"
    """
    # 去除 markdown 代码块
    response = response.strip()
    if response.startswith("```"):
        lines = response.split("\n")
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        response = "\n".join(lines)
    
    # 尝试提取 JSON 对象
    json_pattern = r'\{[^{}]*\}'
    match = re.search(json_pattern, response)
    
    if match:
        try:
            data = json.loads(match.group())
            if "pass" in data and "reason" in data:
                pass_value = data["pass"]
                reason = data["reason"]
                if pass_value:
                    return f"[PASS] {reason}"
                else:
                    return f"[REJECT] {reason}"
        except json.JSONDecodeError:
            pass
    
    # 降级解析
    response_lower = response.lower()
    if '"pass": true' in response_lower or 'pass": true' in response_lower:
        reason_match = re.search(r'"reason":\s*"([^"]*)"', response)
        reason = reason_match.group(1) if reason_match else "搭配方案审核通过"
        return f"[PASS] {reason}"
    else:
        reason_match = re.search(r'"reason":\s*"([^"]*)"', response)
        reason = reason_match.group(1) if reason_match else "搭配存在问题"
        return f"[REJECT] {reason}"


def clean_xhs_response(raw_json: dict) -> list[dict]:
    """清洗小红书原始响应，剔除无用废话，提取核心特征"""
    cleaned_notes = []
    items = raw_json.get("data", {}).get("data", {}).get("items", [])
    
    for item in items:
        # 跳过非笔记类型的内容（如广告或纯直播流）
        if item.get("model_type") != "note":
            continue
            
        note = item.get("note", {})
        
        # 1. 提取基础文本
        title = note.get("title", "")
        desc = note.get("desc", "")
        
        # 2. 提取互动数据（可用于后续 Agent 的置信度评估）
        liked_count = note.get("liked_count", 0)
        
        # 3. 提取高品质图片 URL（通常取前 1-2 张即可，多图浪费 VLM 算力）
        # 优先取 url_size_large 保证清晰度
        images = []
        for img in note.get("images_list", [])[:2]: 
            img_url = img.get("url_size_large") or img.get("url")
            if img_url:
                images.append(img_url)
                
        # 4. 组装为最简结构
        cleaned_notes.append({
            "note_id": note.get("id"),
            "title": title,
            "text_content": desc,
            "likes": liked_count,
            "image_urls": images
        })
        
    return cleaned_notes