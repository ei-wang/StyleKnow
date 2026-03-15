# -*- coding: utf-8 -*-
"""
LLM Prompt 模板集合

包含：
- 意图识别 Prompt
- 搜索任务生成 Prompt (衣柜/小红书)
- 生成搭配的 prompt
- 批评者评估的 prompt
"""

from typing import List, Dict
import re
import json


def format_items_info(retrieved_items: List[dict]) -> str:
    """格式化衣物信息"""
    items_info = ""
    for item in retrieved_items:
        tags = item.get("tags", item.get("semantic_tags", []))
        items_info += f"- {item['name']} ({item['category']}), 标签: {', '.join(tags)}\n"
    return items_info


# ========== 意图识别 Prompt ==========
def build_intent_recognition_prompt(
    user_query: str,
    has_image: bool = False,
    image_semantics: List[Dict] = None,
    conversation_history: List[Dict[str, str]] = None
) -> str:
    """
    构建意图识别的 Prompt
    
    Args:
        user_query: 用户原始问题
        has_image: 是否有上传图片
        image_semantics: 图片分析结果列表
        conversation_history: 对话历史 [{"role": "user/assistant", "content": "..."}]
    
    Returns:
        完整的 prompt 字符串
    """
    # 构建对话历史上下文
    history_context = ""
    if conversation_history:
        history_lines = []
        for msg in conversation_history:
            role = "用户" if msg.get("role") == "user" else "助手"
            content = msg.get("content", "")[:200]  # 截断避免过长
            history_lines.append(f"{role}: {content}")
        history_context = "\n\n【对话历史】:\n" + "\n".join(history_lines)
    
    base = f"""你是一个穿搭助手。请分析用户的输入，识别用户的意图。
{history_context}

用户输入：{user_query}
"""
    
    if has_image and image_semantics:
        # 有图片的情况，输出完整的语义信息（包括 description）
        images_info = "\n".join([
            f"""- 图片{i+1}: 
  品类: {s.get('category', '未知')}
  具体款式: {s.get('item', '未知')}
  颜色: {', '.join(s.get('color', [])) if isinstance(s.get('color'), list) else s.get('color', '未知')}
  风格: {', '.join(s.get('style_semantics', [])) if isinstance(s.get('style_semantics'), list) else s.get('style_semantics', '未知')}
  季节: {', '.join(s.get('season_semantics', [])) if isinstance(s.get('season_semantics'), list) else s.get('season_semantics', '未知')}
  使用场景: {', '.join(s.get('usage_semantics', [])) if isinstance(s.get('usage_semantics'), list) else s.get('usage_semantics', '未知')}
  材质: {s.get('material', '未知')}
  版型: {s.get('fit', '未知')}
  描述: {s.get('description', '无')}"""
            for i, s in enumerate(image_semantics)
        ])
        
        prompt = base + f"""
用户还上传了衣物图片：
{images_info}

请结合图片信息和文字问题，识别用户的意图。
"""
    else:
        prompt = base + "\n请识别用户的意图。"
    
    prompt += """

【重要提示】如果对话历史中已有场景信息，当前问题可能是追问或修改：
- 如果用户只是说"年轻一点"、"更正式"等，但没有明确场景，应该沿用历史场景
- 如果用户明确提到新场景（如"周末出去玩"），则使用新场景

请按以下 JSON 格式输出：
{
    "intent": "recommend/wardrobe_add/query",
    "has_image": true/false,
    "scene": "场景关键词，如通勤、度假、休闲、运动、日常等，如果没有则为空",
    "user_query": "用户的原始问题",
    "rewritten_query": "如果用户上传了图片，重写后的问题；否则等于user_query",
    "reason": "识别理由"
}

intent 说明：
- recommend: 用户想要穿搭推荐（如"今天穿什么"、"搭配什么"）
- wardrobe_add: 用户想要添加衣物到衣柜
- query: 用户只是询问问题""" + """
请直接输出 JSON，不要其他内容。"""
    
    return prompt


def parse_intent_response(response: str) -> Dict:
    """
    解析意图识别响应
    
    Args:
        response: LLM 响应文本
    
    Returns:
        意图字典
    """
    # 尝试提取 JSON
    json_match = re.search(r'\{[\s\S]*\}', response)
    
    if json_match:
        try:
            result = json.loads(json_match.group())
            return {
                "intent": result.get("intent", "recommend"),
                "has_image": result.get("has_image", False),
                "image_type": result.get("image_type", "none"),
                "scene": result.get("scene", ""),
                "date_info": result.get("date_info", {}),
                "user_query": result.get("user_query", ""),
                "rewritten_query": result.get("rewritten_query", result.get("user_query", "")),
                "needed_categories": result.get("needed_categories", []),
                "reason": result.get("reason", "")
            }
        except json.JSONDecodeError:
            pass
    
    # 降级处理
    return {
        "intent": "recommend",
        "has_image": False,
        "image_type": "none",
        "scene": "",
        "date_info": {},
        "user_query": "",
        "rewritten_query": "",
        "needed_categories": [],
        "reason": "解析失败，使用默认"
    }


# ========== 搜索任务生成 Prompt ==========
def build_search_tasks_prompt(
    user_query: str,
    scene: str,
    categories: List[str] = None
) -> str:
    """
    构建搜索任务生成的 Prompt
    
    根据用户查询和场景，为每个品类生成搜索任务
    
    Args:
        user_query: 用户原始查询
        scene: 场景名称
        categories: 需要的品类列表
    
    Returns:
        完整的 prompt 字符串
    """
    if categories is None:
        categories = ["上衣", "裤子", "鞋子", "配饰"]
    
    categories_str = "、".join(categories)
    
    prompt = f"""你是一位专业的时尚搭配师。用户需要根据以下需求完成穿搭搭配。

用户需求: {user_query}
场景: {scene}

请为以下每个品类生成搜索任务。每个任务需要包含：
1. category: 品类名称 (从以下选择: {categories_str})
2. keywords: 5-8个语义标签关键词，用于在用户衣柜中检索相似衣物
3. search_query: 适合在小红书搜索的关键词，用于获取穿搭灵感
4. reason: 为什么选择这个品类和这些关键词

关键要求：
- keywords 应该尽可能全面地反映衣物特征，包括：颜色、风格、材质、版型、适用场景等
- 如果用户已有特定衣物（如深灰色抓绒衣），搜索其他衣物时应考虑搭配协调性（如可以搭配黑色/蓝色牛仔裤）
- 如果是内搭品类（如打底衫），注意与外套的搭配可能性
- keywords 越多越好，确保检索覆盖面

请按以下 JSON 格式输出：
[
  {{
    "category": "上衣",
    "keywords": ["简约", "通勤", "白色", "衬衫", "聚酯纤维", "修身"],
    "search_query": "通勤穿搭 简约干练",
    "reason": "需要一件适合通勤的上衣"
  }},
  ...
]"""
    
    return prompt


def build_wardrobe_search_prompt(
    category: str,
    keywords: List[str],
    scene: str
) -> str:
    """
    构建衣柜检索的 Prompt (用于理解检索逻辑)
    
    Args:
        category: 品类
        keywords: 检索关键词
        scene: 场景
    
    Returns:
        Prompt 字符串
    """
    keywords_str = "、".join(keywords)
    
    prompt = f"""请从用户衣柜中检索符合条件的 {category}。

场景: {scene}
检索关键词: {keywords_str}

请根据语义标签进行匹配检索，返回最符合的衣物。
"""
    return prompt


def build_xhs_search_prompt(
    search_query: str,
    category: str = None
) -> str:
    """
    构建小红书搜索的 Prompt
    
    Args:
        search_query: 搜索关键词
        category: 品类 (可选)
    
    Returns:
        搜索关键词字符串
    """
    if category:
        return f"{search_query} {category}"
    return search_query


# ========== 搭配生成 Prompt ==========
def build_generate_prompt(
    retrieved_items: List[dict],
    scene: str,
    iterations: int = 0,
    critic_feedback: str = "",
    web_inspirations: List[dict] = None,
    conversation_history: List[Dict[str, str]] = None
) -> str:
    """
    构建生成搭配的 Prompt
    
    Args:
        retrieved_items: 检索到的衣物列表
        scene: 场景名称
        iterations: 当前迭代次数（>0 表示是修正轮）
        critic_feedback: 批评者反馈（如果有）
        web_inspirations: 小红书灵感（可选）
        conversation_history: 对话历史（可选）
    
    Returns:
        完整的 prompt 字符串
    """
    # 构建对话历史上下文
    history_context = ""
    if conversation_history and len(conversation_history) > 0:
        history_lines = []
        for msg in conversation_history:
            role = "用户" if msg.get("role") == "user" else "助手"
            content = msg.get("content", "")[:300]  # 截断避免过长
            history_lines.append(f"{role}: {content}")
        history_context = "\n\n【历史对话】:\n" + "\n".join(history_lines)
    
    items_info = format_items_info(retrieved_items)
    
    # 构建小红书灵感部分
    inspiration_section = ""
    if web_inspirations:
        inspiration_items = []
        for i, ins in enumerate(web_inspirations[:3]):
            inspiration_items.append(
                f"- 热门搭配{i+1}: {ins.get('title', '')} - {ins.get('desc', '')}"
            )
        inspiration_section = "\n\n热门穿搭参考:\n" + "\n".join(inspiration_items)
    
    base_prompt = f"""你是一位专业的时尚造型师。请根据以下衣物清单，为用户搭配一套衣服。
{history_context}

场景: {scene}
衣物清单:
{items_info}
{inspiration_section}

请生成一段搭配推荐文案，包含：
1. 推荐的搭配方案
2. 搭配理由

文案要求：
- 语气专业但亲切
- 突出每件衣服的特点
- 长度控制在 100-200 字"""
    
    # 如果有批评反馈，追加修正要求
    if critic_feedback and iterations > 0:
        correction_prompt = f"""

【重要修正要求】
上一轮的搭配被批评了，请根据以下反馈进行修正：
{critic_feedback}

请重新生成搭配方案，确保解决上述问题。"""
        prompt = base_prompt + correction_prompt
    else:
        prompt = base_prompt
    
    return prompt


# ========== 批评者 Prompt ==========
def build_critic_system_prompt() -> str:
    """
    构建批评者的 System Prompt（傲娇毒舌造型师）
    
    Returns:
        System prompt 字符串
    """
    return """你是一位"傲娇毒舌"风格的时尚造型师。你说话刻薄但有道理，喜欢用讽刺的语气评价穿搭。

你的任务是检查用户搭配方案是否存在问题，特别关注：
1. 颜色搭配是否协调（避免红配绿等车祸现场）
2. 季节是否匹配（夏天穿羽绒服会被人笑话）
3. 风格是否统一（正式场合穿拖鞋肯定不行）
4. 是否有明显的搭配漏洞

输出要求：
- 必须返回严格的 JSON 格式
- JSON 格式：{"pass": true/false, "reason": "你的毒舌评价理由"}
- 如果有问题，reason 要具体说明哪里不好，并附带讽刺
- 如果通过，reason 可以稍微表扬一下（但不要太过分）"""


def build_critic_human_prompt(
    draft_outfit: str,
    retrieved_items: List[dict],
    scene: str
) -> str:
    """
    构建批评者的人类 Prompt
    
    Args:
        draft_outfit: 待评估的搭配文案
        retrieved_items: 衣物详情
        scene: 场景名称
    
    Returns:
        Human prompt 字符串
    """
    items_info = format_items_info(retrieved_items)
    
    return f"""请评估以下搭配方案：

场景: {scene}

搭配方案:
{draft_outfit}

衣物详情:
{items_info}

请返回 JSON 格式的评估结果。"""


# ========== 解析 LLM 响应 ==========
def parse_search_tasks_response(response: str) -> List[Dict]:
    """
    解析 LLM 返回的搜索任务列表
    
    Args:
        response: LLM 响应文本
    
    Returns:
        搜索任务列表
    """
    import re
    import json
    
    # 尝试提取 JSON 数组
    json_match = re.search(r'\[[\s\S]*\]', response)
    
    if json_match:
        try:
            tasks = json.loads(json_match.group())
            return tasks
        except json.JSONDecodeError:
            pass
    
    # 降级：返回默认任务
    return [
        {"category": "上衣", "keywords": ["百搭"], "search_query": "穿搭", "reason": "通用"},
        {"category": "裤子", "keywords": ["百搭"], "search_query": "穿搭", "reason": "通用"},
        {"category": "鞋子", "keywords": ["百搭"], "search_query": "穿搭", "reason": "通用"},
    ]
