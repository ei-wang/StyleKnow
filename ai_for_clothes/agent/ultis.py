# -*- coding: utf-8 -*-
"""
工具函数集合

包含：
- LLM 配置和初始化
- 数据库初始化
- 偏好更新函数
- JSON 解析工具
"""

import os
import json
import re
import numpy as np
from typing import List, Dict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


# ========== LLM 配置 ==========
# 从配置中加载
from storage.config_store import load_config as _load_config

_config = _load_config()
OPENAI_API_KEY: str = _config.api_key or os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL: str = _config.model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_BASE: str = _config.api_base or os.environ.get("OPENAI_API_BASE", "")

llm_instance: Optional[ChatOpenAI] = None


def get_llm() -> Optional[ChatOpenAI]:
    """获取或初始化 LLM 实例"""
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


# ========== 偏好更新函数 ==========
def update_user_preference(
    scene: str,
    outfit_items: List[dict],
    is_like: bool
) -> bool:
    """
    用户偏好更新接口（由用户点赞/踩触发）
    
    公式: new_vector = 0.9 * old_vector + 0.1 * current_outfit_vector * feedback_sign
    
    Args:
        scene: 场景名称 (commute/vacation/casual/sports)
        outfit_items: 用户最终选择的搭配物品列表
        is_like: 用户是否喜欢这套搭配
    
    Returns:
        bool: 更新是否成功
    """
    global wardrobe_db
    if wardrobe_db is None:
        wardrobe_db = init_wardrobe_db()
    
    # 检查场景是否存在偏好向量
    if scene not in wardrobe_db.preference_vectors:
        print(f"[WARN] 场景 '{scene}' 无偏好向量，初始化...")
        wardrobe_db.preference_vectors[scene] = np.random.randn(128).astype(np.float32)
        wardrobe_db.preference_vectors[scene] = (
            wardrobe_db.preference_vectors[scene] / 
            np.linalg.norm(wardrobe_db.preference_vectors[scene])
        )
    
    # 获取旧向量
    old_vector = wardrobe_db.preference_vectors[scene].copy()
    
    # 生成当前搭配的向量（简化：用物品向量的平均）
    if outfit_items:
        item_vectors = []
        for item in outfit_items:
            item_id = item.get("item_id")
            # 使用 self.items 而不是 items_collection
            item_data = wardrobe_db.items.get(item_id)
            if item_data:
                item_vec = item_data.get("vector_embedding")
                if item_vec is not None:
                    item_vectors.append(np.array(item_vec))
        
        if item_vectors:
            current_outfit_vector = np.mean(item_vectors, axis=0)
        else:
            current_outfit_vector = np.random.randn(128).astype(np.float32)
    else:
        current_outfit_vector = np.random.randn(128).astype(np.float32)
    
    # 归一化
    if np.linalg.norm(current_outfit_vector) > 0:
        current_outfit_vector = current_outfit_vector / np.linalg.norm(current_outfit_vector)
    
    # 反馈符号
    feedback_sign = 1.0 if is_like else -1.0
    
    # 应用更新公式
    new_vector = 0.9 * old_vector + 0.1 * current_outfit_vector * feedback_sign
    
    # 归一化
    if np.linalg.norm(new_vector) > 0:
        new_vector = new_vector / np.linalg.norm(new_vector)
    
    # 保存
    wardrobe_db.preference_vectors[scene] = new_vector
    
    print(f"\n[Preference] 偏好已更新: 场景={scene}, is_like={is_like}, feedback_sign={feedback_sign}")
    print(f"[Preference] 旧向量范数: {np.linalg.norm(old_vector):.4f}, 新向量范数: {np.linalg.norm(new_vector):.4f}")
    
    return True


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