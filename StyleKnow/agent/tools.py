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
from typing import List, Optional, Dict

# 确保可以导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from storage.db import get_wardrobe_db, get_recommended_db
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

    # 自动保存到推荐数据库
    recommended_db = get_recommended_db()
    saved_count = 0

    for note in notes:
        # 构建推荐衣物数据
        # 注意：note.get("images") 返回的是字符串列表，不是字典列表
        images = note.get("images", [])
        first_image = images[0] if images else ""
        
        xhs_item = {
            "title": note.get("title", ""),
            "desc": note.get("desc", ""),
            "image_url": first_image,
            "category": _infer_category_from_query(query) if not category else category,
            "semantic_tags": _extract_tags_from_note(note, query),
            "scene": category or _infer_scene_from_query(query),
            "reason": f"小红书热门笔记: {note.get('title', '')[:50]}",
        }

        # 生成向量（如果可用）
        try:
            from services.embedding import embed_text
            tags_text = " ".join(xhs_item.get("semantic_tags", []))
            vector = embed_text(tags_text)
            xhs_item["vector_embedding"] = vector
        except Exception:
            pass

        # 保存到推荐数据库
        recommended_db.add_from_xhs(xhs_item, ttl_days=7)
        saved_count += 1

    # 格式化为紧凑字符串
    lines = [f"小红书穿搭灵感 ({len(notes)}条，已自动保存到推荐库):"]
    for i, note in enumerate(notes[:5], 1):
        title = note.get("title", "")[:30]
        desc = note.get("desc", "")[:50]
        likes = note.get("liked_count", 0)
        lines.append(f"{i}. {title}")
        if desc:
            lines.append(f"   {desc}...")
        lines.append(f"   👍 {likes}")

    return "\n".join(lines)


def _infer_category_from_query(query: str) -> str:
    """从查询词推断品类"""
    query_lower = query.lower()

    category_keywords = {
        "上衣": ["上衣", "T恤", "衬衫", "卫衣", "毛衣", "针织", "polo"],
        "裤子": ["裤子", "裤", "牛仔", "阔腿", "休闲裤"],
        "裙子": ["裙子", "裙", "连衣裙", "半身裙"],
        "外套": ["外套", "夹克", "大衣", "风衣", "西装", "羽绒服"],
        "鞋子": ["鞋子", "鞋", "运动鞋", "皮鞋", "靴子", "凉鞋"],
        "配饰": ["配饰", "帽子", "围巾", "包", "项链", "耳环"],
    }

    for category, keywords in category_keywords.items():
        if any(kw in query_lower for kw in keywords):
            return category

    return "未知"


def _infer_scene_from_query(query: str) -> str:
    """从查询词推断场景"""
    query_lower = query.lower()

    scene_keywords = {
        "通勤": ["通勤", "上班", "职业", "商务"],
        "休闲": ["休闲", "日常", "居家", "周末"],
        "约会": ["约会", "浪漫", "甜蜜"],
        "度假": ["度假", "海边", "旅游", "旅行"],
        "运动": ["运动", "健身", "跑步", "瑜伽"],
        "聚会": ["聚会", "派对", "社交"],
    }

    for scene, keywords in scene_keywords.items():
        if any(kw in query_lower for kw in keywords):
            return scene

    return "日常"


def _extract_tags_from_note(note: Dict, query: str) -> List[str]:
    """从笔记中提取语义标签"""
    tags = set()

    # 从标题和描述中提取
    text = f"{note.get('title', '')} {note.get('desc', '')}"

    # 风格关键词
    style_keywords = ["简约", "干练", "甜美", "浪漫", "复古", "潮流", "时尚", "休闲", "通勤", "运动", "优雅", "高级"]
    for style in style_keywords:
        if style in text:
            tags.add(style)

    # 颜色关键词
    color_keywords = ["黑色", "白色", "灰色", "蓝色", "绿色", "红色", "粉色", "黄色", "棕色", "卡其", "米色"]
    for color in color_keywords:
        if color in text:
            tags.add(color)

    # 品类关键词
    category_keywords = ["T恤", "衬衫", "卫衣", "毛衣", "连衣裙", "半身裙", "裤子", "牛仔裤", "外套", "大衣", "夹克"]
    for cat in category_keywords:
        if cat in text:
            tags.add(cat)

    # 如果没有提取到标签，使用查询词
    if not tags and query:
        tags.add(query)

    return list(tags)


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
                 如 ["通勤", "简约", "黑色", "西装"]。如果不需要限定品类，可以不指定 category。
        category: 可选，指定品类筛选。常用值：上衣、裤子、裙子、鞋子、配饰、外套。
                 不指定则搜索所有品类。建议：除非用户明确指定某品类，否则不传此参数。
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


# ========== 批量检索工具（任务拆解 + 并行执行）==========
@tool
def search_wardrobe_batch_tool(
    requests: List[Dict],
    scene: str = None,
    styles: List[str] = None,
    config: RunnableConfig = None
) -> str:
    """
    批量检索用户衣柜中的衣物（并行执行）。

    当需要为用户搭配完整穿搭时使用此工具。
    一次调用即可并行检索多个品类，比串行调用快很多。

    Args:
        requests: 检索请求列表，每个元素包含：
            - category: 品类（如"上衣"、"裤子"、"鞋子"）
            - keywords: 关键词列表（如["通勤", "简约"]）
            - top_k: 返回数量（可选，默认3）
        scene: 场景名称（可选）。如指定，会结合用户偏好进行过滤和排序。
               常用场景：通勤、商务、休闲、约会、度假、运动、聚会、日常
        styles: 风格关键词列表（可选）。由 LLM 根据用户查询和上下文动态判断。
               如 ["简约", "干练", "商务"]。优先于 scene 使用。

    Returns:
        格式化的多品类检索结果，包含每个品类的衣物列表。

    Examples:
        >>> search_wardrobe_batch_tool([
        >>>     {"category": "上衣", "keywords": ["通勤", "简约"], "top_k": 3},
        >>>     {"category": "裤子", "keywords": ["通勤", "简约"], "top_k": 2},
        >>>     {"category": "鞋子", "keywords": ["通勤", "简约"], "top_k": 2},
        >>> ])
        >>> search_wardrobe_batch_tool([
        >>>     {"category": "上衣", "keywords": ["约会", "甜美"]},
        >>>     {"category": "裙子", "keywords": ["约会", "甜美"]},
        >>> ], styles=["浪漫", "精致"])
    """
    # 获取用户ID
    user_id = "default_user"
    if config:
        user_id = config.get("configurable", {}).get("user_id", "default_user")

    # 获取数据库实例
    db = get_wardrobe_db()

    try:
        # 构建搜索请求
        search_requests = []
        for req in requests:
            category = req.get("category", "")
            keywords = req.get("keywords", [])
            top_k = req.get("top_k", 3)

            if not category:
                continue

            search_requests.append({
                "category": category,
                "keywords": keywords,
                "top_k": top_k,
                "scene": scene,
                "styles": styles
            })

        if not search_requests:
            return "未提供有效的检索请求"

        # 调用并行批量检索
        print(f"[Tools] 批量检索: {len(search_requests)} 个品类, scene={scene}, styles={styles}")
        results = db.search_batch(search_requests, user_id=user_id)

        # 格式化为返回字符串
        lines = ["【衣柜检索结果】"]
        lines.append(f"场景: {scene if scene else '无'}, 风格: {styles if styles else '无'}\n")

        total_items = 0
        for category, category_results in results.items():
            if isinstance(category_results, dict) and "error" in category_results:
                lines.append(f"【{category}】检索失败: {category_results['error']}")
                continue

            count = len(category_results)
            total_items += count

            lines.append(f"【{category}】({count}件):")

            if not category_results:
                lines.append("  无匹配衣物")
                continue

            for i, r in enumerate(category_results[:top_k], 1):
                item = r["item"]
                basic = item.get("basic_info", {})
                name = basic.get("name", "未命名")
                tags = item.get("semantic_tags", [])[:3]
                tags_str = "、".join(tags) if tags else "无"
                sim = r.get("similarity", 0)
                lines.append(f"  {i}. {name}")
                lines.append(f"     标签: {tags_str} | 匹配度: {sim:.2f}")

            lines.append("")

        lines.insert(2, f"共检索到 {total_items} 件衣物\n")

        return "\n".join(lines)

    except Exception as e:
        return f"批量检索失败: {str(e)}"


# ========== 用户偏好更新工具 ==========
@tool
def update_preference_tool(
    key: str,
    value: str,
    scene: str = None,
    config: RunnableConfig = None
) -> str:
    """
    记录用户的长期偏好。

    当用户在对话中明确表达喜好或厌恶时使用。
    系统会将此偏好持久化，用于后续的个性化推荐。
    此工具不会直接回复用户，而是默默记录偏好。

    偏好结构（按场景区分）：
    {
        "default_scene": "通勤",
        "scenes": {
            "通勤": {"style": "简约", "color": "蓝色"},
            "约会": {"style": "甜美", "color": "粉色"},
            ...
        }
    }

    Args:
        key: 偏好键名。常用值：
             - style: 风格偏好（如"简约"、"复古"、"暗黑"）
             - color: 颜色偏好（如"蓝色"、"红色"）
             - brand: 品牌偏好
             - fit: 版型偏好（如"宽松"、"修身"）
             - season: 季节偏好
             - avoid: 厌恶项（如"过于正式"、"黑色"）
        value: 偏好值。可以是单个值，也可以是逗号分隔的多个值。
               如"简约"、"蓝色,黑色"、"不要过于正式"。
        scene: 可选，指定场景。如"通勤"、"约会"、"度假"等。
               如果不指定，默认更新当前场景的偏好。

    Returns:
        确认消息，包含已记录的偏好内容。

    Examples:
        >>> update_preference_tool("style", "简约")
        >>> update_preference_tool("color", "蓝色,白色")
        >>> update_preference_tool("avoid", "过于正式", scene="通勤")
    """
    # 从 config 中动态获取 user_id（支持多租户隔离）
    user_id = "default_user"
    if config:
        user_id = config.get("configurable", {}).get("user_id", "default_user")

    # 获取数据库实例
    db = get_wardrobe_db()

    try:
        # 构建偏好字典
        prefs = {key: value}

        # 更新用户偏好（支持场景）
        success = db.update_user_preference(user_id, prefs, scene=scene)

        if success:
            scene_msg = f"（场景: {scene}）" if scene else "（默认场景）"
            return f"已记录偏好: {key} = {value} {scene_msg}"
        else:
            return f"记录偏好失败: {key} = {value}"

    except Exception as e:
        return f"更新偏好失败: {str(e)}"


# ========== 用户偏好查询工具 ==========
@tool
def get_user_preference_tool(scene: str = None, config: RunnableConfig = None) -> str:
    """
    查询用户的长期偏好。

    当需要了解用户的风格偏好、颜色偏好等信息来进行个性化推荐时使用。
    返回用户之前通过 update_preference_tool 记录的所有偏好。

    Args:
        scene: 可选，指定查询的场景偏好。如"通勤"、"约会"等。
               如果不指定，返回所有场景的偏好。
        config: LangChain 运行时配置，包含 user_id 信息。

    Returns:
        格式化的用户偏好列表。如果无记录，返回提示信息。

    Examples:
        >>> get_user_preference_tool()
        >>> get_user_preference_tool(scene="通勤")
    """
    # 从 config 中动态获取 user_id（支持多租户隔离）
    user_id = "default_user"
    if config:
        user_id = config.get("configurable", {}).get("user_id", "default_user")

    db = get_wardrobe_db()

    try:
        if scene:
            # 查询特定场景的偏好
            prefs = db.get_scene_preference(user_id, scene)
            if not prefs:
                return f"暂无场景'{scene}'的偏好记录"
            lines = [f"用户偏好（场景: {scene}）:"]
            for key, value in prefs.items():
                lines.append(f"- {key}: {value}")
        else:
            # 查询所有偏好
            prefs = db.get_user_preference(user_id)
            if not prefs or not prefs.get("scenes"):
                return "暂无用户偏好记录"

            default_scene = prefs.get("default_scene", "日常")
            lines = [f"用户偏好（默认场景: {default_scene}）:"]

            for scene_name, scene_prefs in prefs.get("scenes", {}).items():
                lines.append(f"\n【{scene_name}】")
                if not scene_prefs:
                    lines.append("  无偏好记录")
                for key, value in scene_prefs.items():
                    lines.append(f"  - {key}: {value}")

        return "\n".join(lines)

    except Exception as e:
        return f"查询偏好失败: {str(e)}"


# ========== 工具列表导出 ==========
# 供 LLM 绑定使用
AGENT_TOOLS = [
    search_xhs_tool,
    search_wardrobe_tool,
    search_wardrobe_batch_tool,  # 新增：批量并行检索
    update_preference_tool,
    get_user_preference_tool,
]
