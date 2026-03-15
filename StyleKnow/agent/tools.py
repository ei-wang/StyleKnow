# -*- coding: utf-8 -*-
"""
Agent 工具链 - LangChain Tool 封装

所有业务功能都封装为 @tool，供 LLM 动态调用：
- search_xhs_tool: 检索小红书灵感
- search_wardrobe_tool: 检索用户衣柜
- update_preference_tool: 记录用户偏好
- get_weather_tool: 获取天气信息（见 weather_tools.py）

核心原则：
- 每个工具都有详细的 Google Style Docstring
- 工具内部直接调用 storage/db.py 或外部 API
- 返回结果格式化为紧凑字符串，节省 Token
"""

import os
import sys
from typing import List, Optional

# 确保可以导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.tools import tool

from storage.db import get_wardrobe_db
from tools.xhs_search import search_xhs_notes_by_task


# ========== 小红书检索工具 ==========
@tool
def search_xhs_tool(query: str, category: Optional[str] = None) -> str:
    """
    检索小红书穿搭灵感。

    当需要获取当前流行穿搭趋势、时尚搭配参考时使用此工具。
    特别适用于：不知道穿什么时获取灵感、查看某场景的热门穿搭。

    Args:
        query: 搜索关键词。建议包含场景+风格，如"通勤穿搭 简约干练"、"海边度假 休闲"等。
        category: 可选，指定品类筛选。如"上衣"、"裤子"、"裙子"等。不指定则返回整体穿搭。

    Returns:
        格式化的小红书穿搭灵感列表，包含标题、描述、点赞数。
        如果无结果，返回提示信息。

    Examples:
        >>> search_xhs_tool("通勤穿搭 简约")
        >>> search_xhs_tool("海边度假 连衣裙")
    """
    # 调用底层搜索函数
    result = search_xhs_notes_by_task(
        search_query=query,
        scene=category or "general",
        top_k=5
    )
    
    notes = result.get("notes", [])
    
    if not notes:
        return f"未找到与'{query}'相关的小红书穿搭灵感"
    
    # 格式化为紧凑字符串
    lines = [f"小红书穿搭灵感 ({len(notes)}条):"]
    for i, note in enumerate(notes[:5], 1):
        title = note.get("title", "")[:30]
        desc = note.get("desc", "")[:50]
        likes = note.get("liked_count", 0)
        lines.append(f"{i}. {title}")
        if desc:
            lines.append(f"   {desc}...")
        lines.append(f"   👍 {likes}")
    
    return "\n".join(lines)


# ========== 衣柜检索工具 ==========
@tool
def search_wardrobe_tool(
    keywords: List[str],
    category: Optional[str] = None,
    top_k: int = 5
) -> str:
    """
    检索用户衣柜中的衣物。

    当需要从用户的个人衣物收藏中检索符合特定条件的衣物时使用。
    支持按品类、关键词、场景进行检索。

    Args:
        keywords: 检索关键词列表。建议包含：颜色、风格、材质、场景等。
                 如 ["通勤", "简约", "黑色", "西装"]。
        category: 可选，指定品类筛选。常用值：上衣、裤子、裙子、鞋子、配饰、外套。
                 不指定则搜索所有品类。
        top_k: 可选，返回结果数量，默认5件。

    Returns:
        格式化的衣物列表，包含名称、品类、标签。
        如果无结果，返回提示信息和补充建议。

    Examples:
        >>> search_wardrobe_tool(["通勤", "简约", "白色"])
        >>> search_wardrobe_tool(["度假", "连衣裙"], category="裙子", top_k=3)
    """
    # 获取数据库实例
    db = get_wardrobe_db()
    
    try:
        # 使用混合检索
        query_text = " ".join(keywords)
        results = db.search_hybrid(
            query_text=query_text,
            keywords=keywords,
            category=category,
            top_k=top_k
        )
        
        if not results:
            # 尝试场景检索作为兜底
            scene_keywords = ["commute", "vacation", "casual", "sports"]
            scene = next((k for k in scene_keywords if k in keywords), "commute")
            results = db.search_by_scene(scene=scene, top_k=top_k, category=category)
        
        if not results:
            keywords_str = "、".join(keywords)
            category_str = f" {category}" if category else ""
            return f"衣柜中未找到匹配'{keywords_str}'的{category_str}衣物。建议：可尝试扩大搜索范围或添加新衣物。"
        
        # 格式化为紧凑字符串
        lines = [f"找到{len(results)}件匹配衣物:"]
        for i, r in enumerate(results, 1):
            item = r["item"]
            basic = item.get("basic_info", {})
            name = basic.get("name", "未命名")
            cat = basic.get("category", "未知")
            tags = item.get("semantic_tags", [])[:5]
            tags_str = "、".join(tags) if tags else "无"
            sim = r.get("similarity", 0)
            lines.append(f"{i}. {name} ({cat})")
            lines.append(f"   标签: {tags_str}")
            lines.append(f"   匹配度: {sim:.2f}")
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"检索衣柜失败: {str(e)}"


# ========== 用户偏好更新工具 ==========
@tool
def update_preference_tool(key: str, value: str) -> str:
    """
    记录用户的长期偏好。

    当用户在对话中明确表达喜好或厌恶时使用。
    系统会将此偏好持久化，用于后续的个性化推荐。
    此工具不会直接回复用户，而是默默记录偏好。

    Args:
        key: 偏好键名。常用值：
             - style: 风格偏好（如"简约"、"复古"、"暗黑"）
             - color: 颜色偏好（如"喜欢蓝色"、"不要黑色"）
             - brand: 品牌偏好
             - fit: 版型偏好（如"宽松"、"修身"）
             - season: 季节偏好
             - avoid: 厌恶项（如"不要过于正式"）
        value: 偏好值。如"简约"、"蓝色"、"XL"、"秋冬"等。

    Returns:
        确认消息，包含已记录的偏好内容。

    Examples:
        >>> update_preference_tool("style", "简约")
        >>> update_preference_tool("color", "喜欢蓝色")
        >>> update_preference_tool("avoid", "不要过于正式")
    """
    # 获取数据库实例
    db = get_wardrobe_db()
    
    # 默认用户ID（实际项目中应从上下文获取）
    user_id = "default_user"
    
    try:
        # 构建偏好字典
        prefs = {key: value}
        
        # 更新用户偏好
        success = db.update_user_preference(user_id, prefs)
        
        if success:
            return f"已记录偏好: {key} = {value}"
        else:
            return f"记录偏好失败: {key} = {value}"
            
    except Exception as e:
        return f"更新偏好失败: {str(e)}"


# ========== 用户偏好查询工具 ==========
@tool
def get_user_preference_tool() -> str:
    """
    查询用户的长期偏好。

    当需要了解用户的风格偏好、颜色偏好等信息来进行个性化推荐时使用。
    返回用户之前通过 update_preference_tool 记录的所有偏好。

    Returns:
        格式化的用户偏好列表。如果无记录，返回提示信息。

    Examples:
        >>> get_user_preference_tool()
    """
    db = get_wardrobe_db()
    user_id = "default_user"
    
    try:
        prefs = db.get_user_preference(user_id)
        
        if not prefs:
            return "暂无用户偏好记录"
        
        lines = ["用户偏好:"]
        for key, value in prefs.items():
            lines.append(f"- {key}: {value}")
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"查询偏好失败: {str(e)}"


# ========== 工具列表导出 ==========
# 供 LLM 绑定使用
AGENT_TOOLS = [
    search_xhs_tool,
    search_wardrobe_tool,
    update_preference_tool,
    get_user_preference_tool,
]
