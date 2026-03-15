# -*- coding: utf-8 -*-
"""
LangGraph 节点函数集合

包含 7 个核心节点：
- recognize_intent_node: 意图识别 (LLM 判断)
- generate_search_tasks_node: 生成搜索任务 (LLM)
- retrieve_web_node: 检索小红书 (整体穿搭) - 先执行
- retrieve_wardrobe_node: 检索衣柜 (按品类+关键词) - 后执行
- generate_outfit_node: 生成搭配文案
- critic_evaluate_node: 批评者评估

新流程：意图识别 -> 搜索任务 -> 小红书 -> 衣柜 -> 生成 -> 评估
"""

import random
import json
import asyncio
from typing import List, Dict, Any, Optional

from agent.state import GraphState, SearchTask, WardrobeSearchResult, XHSFeedItem
from agent.ultis import get_llm, get_wardrobe_db, parse_llm_json_response
from agent.prompts import (
    build_intent_recognition_prompt,
    build_search_tasks_prompt,
    build_generate_prompt,
    build_critic_system_prompt,
    build_critic_human_prompt,
    parse_search_tasks_response,
    parse_intent_response
)
from langchain_core.messages import HumanMessage, SystemMessage


# ========== 节点1: 意图识别 ==========
def recognize_intent_node(state: GraphState) -> GraphState:
    """
    节点1: 意图识别
    
    使用 LLM 识别用户意图：
    - 纯文字询问
    - 图片+文字询问：处理图片 -> 存储 -> 重写问题
    - 如果需要天气/时间/城市信息，LLM 会自动调用工具获取
    
    流程：
    1. 如果用户上传了图片，先识别图片中的衣物并存储到数据库
    2. LLM 识别用户意图（LLM 自行判断是否需要调用天气/时间/城市工具）
    3. 根据意图类型决定后续流程
    
    Returns:
        更新后的 GraphState (包含 intent + 可选的 weather_info/trip_days)
    """
    user_query = state["user_query"]
    user_images = state.get("user_images", [])
    conversation_history = state.get("conversation_history", [])
    
    print(f"\n[Node: recognize_intent] 识别意图: '{user_query}'")
    print(f"[Node: recognize_intent] 图片数量: {len(user_images)}")
    print(f"[Node: recognize_intent] 对话历史: {len(conversation_history)} 条")
    
    # 处理用户上传的图片（如果有）
    uploaded_images = state.get("uploaded_images", [])
    
    # 注意：图片解析与入库已迁移到 upload API。
    # agent 节点仅消费 uploaded_images（由 API 返回并写入 state）。
    if user_images and not uploaded_images:
        print("[WARN] 收到 user_images 但未提供 uploaded_images；已跳过图片解析（请先调用 /upload 入库）。")
    
    # 调用 LLM 进行意图识别
    llm = get_llm()
    
    # 用于存储工具调用结果
    tool_results = {}
    
    if llm is None:
        # Mock 模式
        intent = _mock_intent_recognition(user_query, len(user_images) > 0)
    else:
        try:
            # 绑定天气工具到 LLM
            from langchain_openai import ChatOpenAI
            from agent.weather_tools import AGENT_WEATHER_TOOLS
            
            # 创建带工具的 LLM
            llm_with_tools = llm.bind_tools(AGENT_WEATHER_TOOLS)
            
            # 构建意图识别 Prompt
            image_semantics = None
            if uploaded_images:
                image_semantics = [
                    {
                        "image_type": img.get("image_type", "clothing"),
                        "category": img.get("semantics", {}).get("category", ""),
                        "item": img.get("semantics", {}).get("item", ""),
                        "color": img.get("semantics", {}).get("color", []),
                        "style_semantics": img.get("semantics", {}).get("style_semantics", []),
                        "season_semantics": img.get("semantics", {}).get("season_semantics", []),
                        "usage_semantics": img.get("semantics", {}).get("usage_semantics", []),
                        "material": img.get("semantics", {}).get("material", ""),
                        "fit": img.get("semantics", {}).get("fit", ""),
                        "description": img.get("semantics", {}).get("description", "")
                    }
                    for img in uploaded_images
                ]
            
            prompt = build_intent_recognition_prompt(
                user_query=user_query,
                has_image=len(user_images) > 0 or len(uploaded_images) > 0,
                image_semantics=image_semantics,
                conversation_history=conversation_history
            )
            
            # 第一次调用：LLM 可能决定调用工具
            response = llm_with_tools.invoke(prompt)
            
            # 处理工具调用
            if hasattr(response, "tool_calls") and response.tool_calls:
                print(f"[Node: recognize_intent] LLM 决定调用工具: {[tc['name'] for tc in response.tool_calls]}")
                
                # 执行工具调用
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call.get("args", {})
                    
                    try:
                        # 找到对应的工具函数并调用
                        tool_func = next((t for t in AGENT_WEATHER_TOOLS if t.name == tool_name), None)
                        if tool_func:
                            result = tool_func.invoke(tool_args)
                            tool_results[tool_name] = result
                            print(f"[Node: recognize_intent] 工具 {tool_name} 返回: {result[:100] if isinstance(result, str) else result}...")
                        else:
                            print(f"[Node: recognize_intent] 未找到工具: {tool_name}")
                    except Exception as e:
                        print(f"[Node: recognize_intent] 工具调用失败 {tool_name}: {e}")
                
                # 如果有工具调用，再次调用 LLM 获取最终意图
                if tool_results:
                    # 将工具结果添加到 Prompt 中
                    tool_context = "\n\n[已获取的信息]:\n"
                    for name, result in tool_results.items():
                        tool_context += f"- {name}: {result}\n"
                    
                    response2 = llm.invoke(prompt + tool_context)
                    intent = parse_intent_response(response2.content)
                else:
                    intent = parse_intent_response(response.content)
            else:
                # 没有工具调用，直接解析意图
                intent = parse_intent_response(response.content)
            
            print(f"[Node: recognize_intent] 识别结果: intent={intent['intent']}, scene={intent['scene']}")
            print(f"[Node: recognize_intent] 重写问题: {intent.get('rewritten_query', '')}")
            
            # 从工具结果中提取天气信息
            weather_info = _extract_weather_from_tool_results(tool_results, user_query)
            if weather_info:
                print(f"[Node: recognize_intent] 获取到天气信息: {weather_info['location']} - {weather_info['weather']['temperature']}°C")
                state["weather_info"] = weather_info
                
                # 尝试解析出行天数
                import re
                match = re.search(r"(\d+)天", user_query)
                state["trip_days"] = int(match.group(1)) if match else 1
            
        except Exception as e:
            print(f"[ERROR] 意图识别失败: {e}")
            intent = _mock_intent_recognition(user_query, len(user_images) > 0)
    
    # 更新状态
    return {
        **state,
        "uploaded_images": uploaded_images,
        "intent": intent,
        "parsed_intent": {
            "scene": intent.get("scene", "commute"),
            "original_query": user_query,
            "rewritten_query": intent.get("rewritten_query", user_query),
            "needed_categories": intent.get("needed_categories", []),
            "top_k": 8
        }
    }


def _extract_weather_from_tool_results(tool_results: dict, user_query: str) -> Optional[dict]:
    """
    从工具调用结果中提取天气信息
    
    Args:
        tool_results: 工具调用结果字典
        user_query: 用户原始查询（用于解析天数）
    
    Returns:
        格式化的天气信息字典
    """
    import re
    from datetime import datetime, timedelta
    
    # 从 get_weather_with_suggestion 或 get_weather_info 结果中解析
    weather_text = tool_results.get("get_weather_with_suggestion") or tool_results.get("get_weather_info", "")
    
    if not weather_text:
        return None
    
    try:
        # 解析城市名称
        location = "未知"
        lines = weather_text.split("\n")
        if "天气预报" in lines[0]:
            location = lines[0].replace("天气预报:", "").strip()
        elif "天气" in lines[0]:
            location = lines[0].replace("天气:", "").strip()
        
        # 解析天气数据
        weather_data = {}
        for line in lines[1:]:
            if "温度:" in line:
                temp = re.search(r"温度:\s*([\d.]+)", line)
                if temp:
                    weather_data["temperature"] = float(temp.group(1))
            elif "体感:" in line:
                feels = re.search(r"体感:\s*([\d.]+)", line)
                if feels:
                    weather_data["feelsLike"] = float(feels.group(1))
            elif "天气:" in line:
                weather_data["condition"] = re.search(r"天气:\s*(.+)", line).group(1).strip()
            elif "湿度:" in line:
                humid = re.search(r"湿度:\s*([\d.]+)", line)
                if humid:
                    weather_data["humidity"] = float(humid.group(1))
            elif "风力:" in line:
                wind = re.search(r"风力:\s*(.+)", line)
                if wind:
                    weather_data["windScale"] = wind.group(1).strip()
        
        # 解析穿衣建议
        clothing_suggestion = ""
        if "💡 穿衣建议:" in weather_text:
            clothing_suggestion = weather_text.split("💡 穿衣建议:")[1].strip()
        
        # 解析出行日期
        today = datetime.now()
        match = re.search(r"(\d+)天", user_query)
        days = int(match.group(1)) if match else 1
        
        # 解析"后天"等相对时间
        if "后天" in user_query:
            days = 2
            start_date = today + timedelta(days=2)
        elif "大后天" in user_query:
            days = 3
            start_date = today + timedelta(days=3)
        elif "明天" in user_query:
            days = 1
            start_date = today + timedelta(days=1)
        else:
            start_date = today
        
        dates = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]
        
        return {
            "location": location,
            "weather": weather_data,
            "dates": dates,
            "trip_days": days,
            "clothing_suggestion": clothing_suggestion,
            "retrieved_at": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        
    except Exception as e:
        print(f"[ERROR] 解析天气信息失败: {e}")
        return None


def _process_uploaded_images(user_images: List[str]) -> List[Dict]:
    """
    处理用户上传的图片

    流程：
    1. 对每张图片判断是全身照还是单件衣物
    2. 如果是单件衣物 -> 移除背景 -> VLM 分析 -> 存储
    3. 如果是全身照 -> VLM 分析多件衣物 -> 逐个存储

    Args:
        user_images: 用户上传的图片路径列表

    Returns:
        处理后的图片信息列表
    """
    from services.segment import remove_background
    from services.openai_compatible import analyze_clothes_openai
    from storage.db import get_wardrobe_db
    from domain.clothes import ClothesSemantics

    results = []
    db = get_wardrobe_db()

    for img_path in user_images:
        try:
            # 读取图片
            import requests
            from io import BytesIO

            # 尝试读取本地文件或 URL
            if img_path.startswith("http"):
                response = requests.get(img_path)
                image_bytes = response.content
            else:
                with open(img_path, "rb") as f:
                    image_bytes = f.read()

            # 首先判断图片类型：全身照还是单件衣物
            image_type = _detect_image_type(image_bytes)

            if image_type == "full_body":
                # 全身照：识别多件衣物
                print(f"[Node] 检测到全身照，开始识别多件衣物...")
                items = _analyze_full_body_image(image_bytes)

                for item_data in items:
                    # 存储每件衣物到数据库
                    storage_data = item_data["semantics"].to_storage_format()
                    item_id = db.add_item(storage_data)

                    results.append({
                        "image_url": img_path,
                        "item_id": item_id,
                        "semantics": storage_data.get("dynamic_metadata", {}),
                        "image_type": "clothing",
                        "detected_category": item_data["semantics"].category,
                        "is_from_full_body": True
                    })
                    print(f"[Node] 已存储: {storage_data['basic_info']['name']} ({item_data['semantics'].category})")

            else:
                # 单件衣物：原有流程
                # 移除背景
                processed_bytes = remove_background(image_bytes)

                # VLM 分析
                semantics: ClothesSemantics = analyze_clothes_openai(processed_bytes)

                # 转换为存储格式
                storage_data = semantics.to_storage_format()

                # 存储到数据库
                item_id = db.add_item(storage_data)

                results.append({
                    "image_url": img_path,
                    "item_id": item_id,
                    "semantics": storage_data.get("dynamic_metadata", {}),
                    "image_type": "clothing",
                    "detected_category": semantics.category,
                    "is_from_full_body": False
                })

                print(f"[Node] 已存储: {storage_data['basic_info']['name']}")

        except Exception as e:
            print(f"[ERROR] 处理图片失败: {img_path}, {e}")

    return results


def _detect_image_type(image_bytes: bytes) -> str:
    """
    使用 VLM 判断图片类型：全身照还是单件衣物

    Args:
        image_bytes: 图片字节数据

    Returns:
        "full_body" 或 "single_item"
    """
    import base64
    import json
    import httpx
    from storage.config_store import load_config

    config = load_config()
    if not config.api_key:
        # 无 API 时默认单件处理
        return "single_item"

    # 确保 api_base 格式正确
    api_base = config.api_base.rstrip("/")
    if not api_base.endswith("/v1"):
        api_base = api_base + "/v1"

    url = f"{api_base}/chat/completions"

    # 将图片转换为 base64
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    # 判断图片类型的 prompt
    detect_prompt = """你是一个图像识别专家。请分析这张图片，判断它是：
1. 全身照（包含多件衣物的人像照，如上衣+裤子+鞋子等完整穿搭）
2. 单件衣物（只有一件衣物的产品图或平铺图）

只需要回答"全身照"或"单件衣物"，不要其他内容。"""

    payload = {
        "model": config.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": detect_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                ]
            }
        ],
        "max_tokens": 50
    }

    try:
        response = httpx.post(url, headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }, json=payload, timeout=30.0)

        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()

            if "全身" in content:
                return "full_body"
            else:
                return "single_item"
    except Exception as e:
        print(f"[WARN] 图片类型检测失败: {e}")

    return "single_item"


def _analyze_full_body_image(image_bytes: bytes) -> List[Dict]:
    """
    分析全身照，识别其中的多件衣物

    Args:
        image_bytes: 图片字节数据

    Returns:
        识别出的衣物语义信息列表
    """
    import base64
    import json
    import httpx
    from storage.config_store import load_config
    from domain.clothes import ClothesSemantics

    config = load_config()
    if not config.api_key:
        raise ValueError("需要 API Key 来分析全身照")

    # 确保 api_base 格式正确
    api_base = config.api_base.rstrip("/")
    if not api_base.endswith("/v1"):
        api_base = api_base + "/v1"

    url = f"{api_base}/chat/completions"

    # 将图片转换为 base64
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    # 分析全身照的 prompt
    analyze_prompt = """你是一个专业的衣物识别助手。请分析这张全身穿搭照，识别出所有可见的衣物。

请按以下 JSON 数组格式输出（每件衣物一个对象）：
[
    {
        "category": "上衣/裤子/裙子/鞋子/配饰/外套",
        "item": "具体款式，如T恤、衬衫、牛仔裤、运动鞋等",
        "style_semantics": ["风格标签列表"],
        "season_semantics": ["季节标签"],
        "usage_semantics": ["使用场景标签"],
        "color_semantics": ["颜色标签"],
        "description": "一段描述",
        "material": "材质",
        "pattern": "图案",
        "fit": "版型"
    },
    ...
]

请尽可能识别出所有衣物，包括：内搭、外套、裤子/裙子、鞋子、配饰等。
如果不确定某件衣物的属性，用空字符串或空数组表示。
只输出 JSON 数组，不要其他内容。"""

    payload = {
        "model": config.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": analyze_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                ]
            }
        ],
        "max_tokens": 2000
    }

    try:
        response = httpx.post(url, headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }, json=payload, timeout=60.0)

        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"]

            # 提取 JSON 数组
            import re
            json_match = re.search(r'\[[\s\S]*\]', content)

            if json_match:
                items_data = json.loads(json_match.group())
                results = []

                for item_data in items_data:
                    try:
                        semantics = ClothesSemantics(**item_data)
                        results.append({"semantics": semantics})
                    except Exception as e:
                        print(f"[WARN] 解析衣物数据失败: {e}")

                return results

    except Exception as e:
        print(f"[ERROR] 全身照分析失败: {e}")

    return []


def _mock_intent_recognition(user_query: str, has_image: bool) -> Dict:
    """Mock 意图识别"""
    # 简单的关键词匹配作为 fallback
    scene = ""
    if "通勤" in user_query or "上班" in user_query:
        scene = "commute"
    elif "度假" in user_query or "旅游" in user_query:
        scene = "vacation"
    elif "运动" in user_query or "健身" in user_query:
        scene = "sports"
    elif "休闲" in user_query:
        scene = "casual"
    
    return {
        "intent": "recommend",
        "has_image": has_image,
        "scene": scene,
        "user_query": user_query,
        "rewritten_query": user_query,
        "reason": "Mock 意图识别"
    }


# ========== 节点2: 生成搜索任务 ==========
def generate_search_tasks_node(state: GraphState) -> GraphState:
    """
    节点2: 生成搜索任务 (LLM)
    
    根据用户查询和场景，为衣柜和小红书生成搜索任务
    
    Returns:
        更新后的 GraphState (包含 search_tasks)
    """
    # 使用重写后的问题（如果有的话）
    user_query = state["parsed_intent"].get("rewritten_query", state["user_query"])
    scene = state["parsed_intent"].get("scene", "commute")
    
    # 如果有上传的图片信息，也可以结合
    uploaded_images = state.get("uploaded_images", [])
    if uploaded_images:
        # 将图片的完整语义信息添加到 user_query
        img_info_parts = []
        for img in uploaded_images:
            semantics = img.get("semantics", {})
            
            # 品类信息
            category = semantics.get("category", "")
            item = semantics.get("item", "")  # 具体款式，如"抓绒衣"
            
            # 构建完整描述
            parts = []
            if item:
                parts.append(item)
            elif category:
                parts.append(category)
            
            # 颜色（可能是列表）
            colors = semantics.get("color", [])
            if colors:
                colors_str = "/".join(colors) if isinstance(colors, list) else colors
                parts.append(colors_str)
            
            # 风格（可能是列表）
            styles = semantics.get("style_semantics", [])
            if styles:
                styles_str = "/".join(styles) if isinstance(styles, list) else styles
                parts.append(styles_str)
            
            # 季节（可能是列表）
            seasons = semantics.get("season_semantics", [])
            if seasons:
                seasons_str = "/".join(seasons) if isinstance(seasons, list) else seasons
                parts.append(seasons_str)
            
            # 材质
            material = semantics.get("material", "")
            if material:
                parts.append(material)
            
            # 版型
            fit = semantics.get("fit", "")
            if fit:
                parts.append(fit)
            
            # 使用场景
            usages = semantics.get("usage_semantics", [])
            if usages:
                usages_str = "/".join(usages) if isinstance(usages, list) else usages
                parts.append(usages_str)
            
            # VLM 生成的描述（最重要的语义信息）
            description = semantics.get("description", "")
            if description:
                parts.append(f"描述:{description}")
            
            if parts:
                img_info_parts.append("、".join(parts))
        
        if img_info_parts:
            user_query = f"{user_query} (用户已有的衣物: {'; '.join(img_info_parts)})"
    
    print(f"\n[Node: generate_search_tasks] 生成搜索任务...")
    print(f"[Node: generate_search_tasks] Query: {user_query}, Scene: {scene}")
    
    # 定义需要的品类
    categories = ["上衣", "裤子", "鞋子", "配饰"]
    
    # 构建 Prompt
    prompt = build_search_tasks_prompt(
        user_query=user_query,
        scene=scene,
        categories=categories
    )
    
    # 调用 LLM
    llm = get_llm()
    
    if llm is None:
        # Mock 模式
        search_tasks = _mock_generate_search_tasks(scene)
    else:
        try:
            response = llm.invoke(prompt)
            search_tasks = parse_search_tasks_response(response.content)
            print(f"[Node: generate_search_tasks] 生成了 {len(search_tasks)} 个搜索任务")
        except Exception as e:
            print(f"[ERROR] LLM 生成搜索任务失败: {e}")
            search_tasks = _mock_generate_search_tasks(scene)
    
    return {
        **state,
        "search_tasks": search_tasks
    }


def _mock_generate_search_tasks(scene: str) -> List[SearchTask]:
    """Mock 模式的搜索任务生成"""
    return [
        {
            "category": "上衣",
            "keywords": ["简约", "通勤", "白色", "衬衫"],
            "search_query": f"{scene}穿搭 简约干练",
            "reason": "需要一件适合通勤的上衣"
        },
        {
            "category": "裤子",
            "keywords": ["通勤", "黑色", "西裤"],
            "search_query": f"{scene}穿搭 西裤",
            "reason": "需要一条通勤裤子"
        },
        {
            "category": "鞋子",
            "keywords": ["通勤", "舒适", "皮鞋"],
            "search_query": f"{scene}穿搭 皮鞋",
            "reason": "需要一双通勤鞋子"
        },
        {
            "category": "配饰",
            "keywords": ["简约", "百搭"],
            "search_query": f"{scene}穿搭 配饰",
            "reason": "需要简约配饰"
        }
    ]


# ========== 节点3: 检索小红书 (先执行) ==========
def retrieve_web_node(state: GraphState) -> GraphState:
    """
    节点3: 检索小红书 (整体穿搭) - 先执行
    
    小红书搜索是整体穿搭搜索，获取穿搭灵感
    将小红书返回的搭配与上下文结合，再去衣柜检索
    
    Returns:
        更新后的 GraphState (包含 web_formulas)
    """
    search_tasks = state.get("search_tasks", [])
    scene = state["parsed_intent"].get("scene", "commute")
    
    print(f"\n[Node: retrieve_web] 开始检索小红书...")
    
    # 从搜索任务中获取小红书搜索关键词
    xhs_search_query = _combine_xhs_search_queries(search_tasks)
    
    print(f"  - 小红书搜索关键词: {xhs_search_query}")
    
    # 执行搜索 (获取更多结果用于后续筛选)
    from tools.xhs_search import search_xhs_notes_by_task
    xhs_result = search_xhs_notes_by_task(
        search_query=xhs_search_query,
        scene=scene,
        top_k=10  # 先获取更多结果
    )
    
    # 转换为 XHSFeedItem 格式
    web_formulas: List[XHSFeedItem] = []
    for note in xhs_result.get("notes", []):
        web_formulas.append({
            "note_id": note.get("id", ""),
            "title": note.get("title", ""),
            "desc": note.get("desc", ""),
            "liked_count": note.get("liked_count", 0),
            "images": note.get("images", []),
            "formula": "",  # 后续用 VLM 提取
            "matched_score": 0.0
        })
    
    print(f"[Node: retrieve_web] 检索到 {len(web_formulas)} 篇小红书笔记")
    
    return {
        **state,
        "web_formulas": web_formulas,
        "xhs_extracted_outfits": []  # 初始化，后续节点填充
    }


# ========== 节点3.5: 处理小红书结果 (重排序 + VLM提取) ==========
def process_xhs_results_node(state: GraphState) -> GraphState:
    """
    节点3.5: 处理小红书结果
    
    流程：
    1. top-10 → 重排序 → top-3
    2. 对 top-3 使用 VLM 提取穿搭公式
    3. 将提取结果存储，供后续衣柜检索使用
    
    Returns:
        更新后的 GraphState (包含 xhs_extracted_outfits)
    """
    web_formulas = state.get("web_formulas", [])
    scene = state["parsed_intent"].get("scene", "commute")
    user_query = state.get("user_query", "")
    
    print(f"\n[Node: process_xhs_results] 处理小红书结果...")
    
    if not web_formulas:
        print("[Node: process_xhs_results] 无小红书结果，跳过")
        return {**state, "xhs_extracted_outfits": []}
    
    # 步骤1: 重排序 (top-10 → top-3)
    # 结合热度、相关度和用户需求
    reranked = _rerank_xhs_results(web_formulas, user_query, scene)
    top3 = reranked[:3]
    
    print(f"[Node: process_xhs_results] 重排序后 top-3:")
    for i, item in enumerate(top3):
        try:
            print(f"  {i+1}. {item.get('title', '')} (匹配分: {item.get('matched_score', 0):.2f})")
        except:
            print(f"  {i+1}. (标题包含特殊字符)")
    
    # 步骤2: VLM 提取穿搭公式
    extracted_outfits = []
    
    for item in top3:
        # 尝试使用 VLM 提取（统一服务）
        from services.vlm import extract_outfit_formula_sync

        images = item.get("images", []) or []
        extracted_json = extract_outfit_formula_sync(images[0]) if images else None
        extracted = None
        if extracted_json:
            extracted = {
                "source_note_id": item.get("note_id", ""),
                "source_title": item.get("title", ""),
                "source_image": images[0] if images else "",
                "overall_style": extracted_json.get("overall_style", ""),
                "color_scheme": extracted_json.get("color_scheme", ""),
                "items": extracted_json.get("items", []),
                "matching_tips": extracted_json.get("matching_tips", []),
                "overall_score": item.get("matched_score", 0),
            }
        
        if extracted:
            extracted_outfits.append(extracted)
        else:
            # VLM 失败时使用 mock
            print(f"[WARN] VLM 提取失败，使用 mock: {item.get('title', '')}")
            extracted = _mock_vlm_extract(item, scene)
            if extracted:
                extracted_outfits.append(extracted)
    
    print(f"[Node: process_xhs_results] 提取了 {len(extracted_outfits)} 个穿搭方案")
    
    return {
        **state,
        "web_formulas": top3,  # 更新为 top-3
        "xhs_extracted_outfits": extracted_outfits
    }


## NOTE: 小红书穿搭公式提取已迁移到 services.vlm.extract_outfit_formula_sync


def _rerank_xhs_results(
    results: List[Dict], 
    user_query: str, 
    scene: str
) -> List[Dict]:
    """
    重排序小红书结果
    
    结合：
    1. 热度 (liked_count)
    2. 与用户需求的匹配度
    3. 与场景的匹配度
    """
    # 提取用户需求关键词
    scene_keywords = ["通勤", "度假", "休闲", "运动", "海边", "登山", "日常"]
    query_keywords = [scene] if scene else []
    for kw in scene_keywords:
        if kw in user_query:
            query_keywords.append(kw)
    
    reranked = []
    for item in results:
        score = 0.0
        
        # 热度分数 (0-0.3)
        liked = item.get("liked_count", 0)
        hot_score = min(liked / 10000, 1.0) * 0.3
        score += hot_score
        
        # 标题匹配分数 (0-0.4)
        title = item.get("title", "")
        desc = item.get("desc", "")
        text = title + " " + desc
        
        match_count = sum(1 for kw in query_keywords if kw in text)
        match_score = min(match_count / max(len(query_keywords), 1), 1.0) * 0.4
        score += match_score
        
        # 场景匹配分数 (0-0.3)
        scene_score = 0.3 if scene and scene in text else 0.0
        score += scene_score
        
        reranked.append({
            **item,
            "matched_score": round(score, 3)
        })
    
    # 按分数排序
    reranked.sort(key=lambda x: x.get("matched_score", 0), reverse=True)
    return reranked


def _mock_vlm_extract(xhs_item: Dict, scene: str) -> Optional[Dict]:
    """
    模拟 VLM 提取穿搭公式
    
    实际实现需要调用 VLM API 分析图片
    
    这里返回模拟结果
    """
    # 简化处理：从标题和描述中提取品类信息
    title = xhs_item.get("title", "")
    desc = xhs_item.get("desc", "")
    text = title + " " + desc
    
    items = []
    
    # 提取上衣
    if any(w in text for w in ["T恤", "衬衫", "上衣", "卫衣", "针织"]):
        items.append({
            "category": "上衣",
            "description": "从图片中提取的上衣特征",
            "image_url": xhs_item.get("images", [""])[0] if xhs_item.get("images") else ""
        })
    
    # 提取裤子
    if any(w in text for w in ["裤子", "牛仔裤", "休闲裤"]):
        items.append({
            "category": "裤子",
            "description": "从图片中提取的裤子特征",
            "image_url": xhs_item.get("images", [""])[1] if len(xhs_item.get("images", [])) > 1 else ""
        })
    
    # 提取鞋子
    if any(w in text for w in ["鞋", "靴", "运动鞋"]):
        items.append({
            "category": "鞋子",
            "description": "从图片中提取的鞋子特征",
            "image_url": xhs_item.get("images", [""])[2] if len(xhs_item.get("images", [])) > 2 else ""
        })
    
    if not items:
        return None
    
    return {
        "source_note_id": xhs_item.get("note_id", ""),
        "source_title": xhs_item.get("title", ""),
        "items": items,
        "overall_score": xhs_item.get("matched_score", 0)
    }


def _combine_xhs_search_queries(search_tasks: List[SearchTask]) -> str:
    """组合小红书搜索关键词"""
    if not search_tasks:
        return "穿搭"
    
    # 使用第一个任务的搜索查询作为主查询
    first_query = search_tasks[0].get("search_query", "穿搭")
    
    # 可以添加更多关键词
    keywords = set()
    for task in search_tasks:
        keywords.update(task.get("keywords", []))
    
    # 组合
    if keywords:
        return f"{first_query} {' '.join(list(keywords)[:3])}"
    
    return first_query


# ========== 节点4: 检索衣柜 (后执行) ==========
def retrieve_wardrobe_node(state: GraphState) -> GraphState:
    """
    节点4: 检索衣柜 (按品类+关键词) - 后执行
    
    结合小红书的搜索结果，在衣柜中检索匹配的衣物
    
    兜底策略：
    1. 如果某品类无检索结果 -> 使用更宽泛的关键词重试
    2. 如果仍无结果 -> 使用场景偏好向量检索
    3. 如果仍无结果 -> 跳过该品类
    
    Returns:
        更新后的 GraphState (包含 wardrobe_results)
    """
    search_tasks = state.get("search_tasks", [])
    web_formulas = state.get("web_formulas", [])
    scene = state["parsed_intent"].get("scene", "commute")
    
    print(f"\n[Node: retrieve_wardrobe] 开始检索衣柜，共 {len(search_tasks)} 个任务...")
    
    wardrobe_results: List[WardrobeSearchResult] = []
    
    # 从小红书结果中提取额外的关键词（如果有）
    xhs_keywords = _extract_keywords_from_xhs(web_formulas)
    
    for task in search_tasks:
        category = task.get("category", "")
        keywords = task.get("keywords", [])
        
        if not category:
            continue
        
        # 第一轮：按品类+关键词检索
        combined_keywords = keywords.copy()
        for kw in xhs_keywords:
            if kw not in combined_keywords:
                combined_keywords.append(kw)
        
        search_result = _search_wardrobe_by_task(
            category=category,
            keywords=combined_keywords,
            top_k=5,
            scene=scene
        )
        
        items = search_result.get("items", [])
        search_method = "keyword"
        
        # 兜底策略1：如果无结果，使用更宽泛的关键词
        if not items:
            print(f"  [兜底] {category} 第一次无结果，尝试更宽泛关键词...")
            broader_keywords = _broader_keywords(keywords)
            search_result = _search_wardrobe_by_task(
                category=category,
                keywords=broader_keywords,
                top_k=5,
                scene=scene
            )
            items = search_result.get("items", [])
            search_method = "keyword_broad"
        
        # 兜底策略2：使用场景偏好向量检索
        if not items:
            print(f"  [兜底] {category} 关键词无结果，尝试场景偏好向量...")
            items = _search_by_scene_fallback(category, scene)
            search_method = "scene_vector"
        
        # 兜底策略3：返回任意品类衣物作为参考
        if not items:
            print(f"  [兜底] {category} 完全无结果，返回通用推荐...")
            items = _get_any_category_items(category)
            search_method = "any_category"
        
        wardrobe_results.append({
            "category": category,
            "search_task": {**task, "keywords": combined_keywords},
            "items": items,
            "search_method": search_method
        })
        
        print(f"  - {category}: 检索到 {len(items)} 件 (方法: {search_method})")
    
    print(f"[Node: retrieve_wardrobe] 共检索到 {sum(len(r['items']) for r in wardrobe_results)} 件衣物")
    
    return {
        **state,
        "wardrobe_results": wardrobe_results
    }


def _extract_keywords_from_xhs(web_formulas: List[Dict]) -> List[str]:
    """从小红书结果中提取关键词"""
    keywords = []
    style_words = ["简约", "复古", "商务", "休闲", "运动", "甜美", "酷", "街头", "海边", "度假"]
    
    for note in web_formulas[:3]:
        title = note.get("title", "")
        desc = note.get("desc", "")
        text = title + " " + desc
        
        for word in style_words:
            if word in text and word not in keywords:
                keywords.append(word)
    
    return keywords


def _broader_keywords(keywords: List[str]) -> List[str]:
    """生成更宽泛的关键词"""
    broader_map = {
        "春夏": ["春夏", "四季"],
        "秋冬": ["秋冬", "四季"],
        "通勤": ["通勤", "日常"],
        "商务": ["商务", "正式"],
        "度假": ["度假", "休闲"],
    }
    
    result = list(keywords)
    for kw, broader in broader_map.items():
        if kw in keywords:
            result.extend(broader)
    
    # 添加通用词
    result.extend(["百搭", "基础款"])
    return list(set(result))


def _search_by_scene_fallback(category: str, scene: str) -> List[Dict]:
    """使用场景偏好向量检索（兜底）"""
    from tools.db_search import search_wardrobe
    result = search_wardrobe(scene=scene, top_k=5, category=category)
    return result.get("items", [])


def _get_any_category_items(category: str) -> List[Dict]:
    """获取任意品类衣物作为参考（最兜底）"""
    from tools.db_search import list_wardrobe_items
    result = list_wardrobe_items(limit=3)
    
    # 转换格式以匹配其他检索结果
    items = []
    for item in result.get("items", [])[:3]:
        basic = item.get("basic_info", {})
        items.append({
            "item_id": item.get("item_id", ""),
            "name": basic.get("name", ""),
            "category": basic.get("category", ""),
            "image_url": basic.get("image_url", ""),
            "tags": item.get("semantic_tags", []),
            "metadata": item.get("dynamic_metadata", {}),
            "matched_tags": [],
            "similarity": 0.0
        })
    
    return items


def _generate_fallback_recommendation(
    category: str, 
    scene: str, 
    keywords: List[str]
) -> Dict:
    """
    生成兜底推荐

    当衣柜无结果时：
    1. 返回小红书推荐
    2. 提醒用户上传该品类的衣服

    Returns:
        包含推荐和提醒的字典
    """
    # 基于场景生成推荐文案
    scene_descriptions = {
        "commute": "通勤",
        "vacation": "度假",
        "casual": "休闲",
        "sports": "运动",
        "海边": "海边度假",
        "三亚": "热带海岛",
        "春季": "春季",
        "夏季": "夏季",
        "秋季": "秋季",
        "冬季": "冬季"
    }

    scene_desc = scene_descriptions.get(scene, scene)

    return {
        "category": category,
        "scene": scene,
        "recommendation": f"建议您上传 {category} 的图片来扩充衣柜，我会根据您的衣物提供个性化搭配建议。",
        "suggested_keywords": keywords,
        "fallback_reason": f"当前衣柜中没有找到适合{scene_desc}的{category}，小红书热门搭配可作为参考。"
    }


def _build_fallback_outfit(
    fallback_recommendations: List[Dict],
    web_formulas: List[Dict],
    scene: str
) -> str:
    """
    构建兜底的搭配文案

    Args:
        fallback_recommendations: 各品类的兜底推荐列表
        web_formulas: 小红书搜索结果
        scene: 场景

    Returns:
        兜底的搭配文案
    """
    # 场景描述
    scene_descriptions = {
        "commute": "通勤",
        "vacation": "度假",
        "casual": "休闲",
        "sports": "运动",
        "海边": "海边度假",
        "三亚": "热带海岛"
    }
    scene_desc = scene_descriptions.get(scene, scene)

    lines = []
    lines.append(f"=== {scene_desc}穿搭推荐 ===")
    lines.append("")
    lines.append("📦 衣柜现有衣物搭配：")

    # 添加现有衣物
    has_any_items = False
    for rec in fallback_recommendations:
        category = rec.get("category", "")
        reason = rec.get("fallback_reason", "")

        if "上传" in rec.get("recommendation", ""):
            lines.append(f"  • {category}: {rec.get('fallback_reason', '')}")
        else:
            lines.append(f"  • {category}: 已为您搭配现有衣物")
            has_any_items = True

    lines.append("")
    lines.append("💡 补充建议：")

    # 添加缺失品类的提醒
    for rec in fallback_recommendations:
        category = rec.get("category", "")
        recommendation = rec.get("recommendation", "")
        lines.append(f"  • {recommendation}")

    # 添加小红书参考
    if web_formulas:
        lines.append("")
        lines.append("📱 小红书热门搭配参考：")
        for i, note in enumerate(web_formulas[:3]):
            title = note.get("title", "")
            liked = note.get("liked_count", 0)
            if title:
                lines.append(f"  {i+1}. {title} (👍 {liked})")

    lines.append("")
    lines.append("💡 温馨提示：为了获得更精准的个性化搭配建议，建议您上传更多衣物图片来扩充衣柜哦！")

    return "\n".join(lines)


def _search_wardrobe_by_task(
    category: str,
    keywords: List[str],
    scene: str,
    top_k: int = 5
) -> Dict:
    """按品类+关键词检索衣柜"""
    from tools.db_search import search_wardrobe_by_task as _func
    return _func(category=category, keywords=keywords, top_k=top_k, scene=scene)


# ========== 节点5: 生成搭配 ==========
def generate_outfit_node(state: GraphState) -> GraphState:
    """
    节点5: 生成搭配文案
    
    使用 LLM 根据检索到的衣物（结合小红书灵感）生成搭配建议
    
    Returns:
        更新后的 GraphState
    """
    wardrobe_results = state.get("wardrobe_results", [])
    web_formulas = state.get("web_formulas", [])
    iterations = state.get("iterations", 0)
    critic_feedback = state.get("critic_feedback", "")
    scene = state["parsed_intent"].get("scene", "commute")
    conversation_history = state.get("conversation_history", [])
    
    # 日志
    if iterations > 0:
        print(f"\n[Node: generate_outfit] LLM 重新生成 (迭代 {iterations})")
    else:
        print(f"\n[Node: generate_outfit] LLM 首次生成")
        print(f"[Node: generate_outfit] 对话历史: {len(conversation_history)} 条")
    
    # 合并所有检索到的衣物
    all_items = []
    missing_categories = []  # 记录缺失的品类
    
    for result in wardrobe_results:
        items = result.get("items", [])
        all_items.extend(items)
        
        # 检查是否有缺失品类
        if not items:
            missing_categories.append(result.get("category", ""))
    
    # 初始化缺失推荐变量
    missing_recommendations = []
    add_missing_tip = False
    
    # 无检索结果 - 使用兜底策略
    if not all_items:
        print(f"[Node: generate_outfit] 衣柜无结果，使用兜底策略...")
        
        # 收集所有品类的兜底推荐
        fallback_recommendations = []
        search_tasks = state.get("search_tasks", [])
        
        for task in search_tasks:
            category = task.get("category", "")
            keywords = task.get("keywords", [])
            
            fallback = _generate_fallback_recommendation(category, scene, keywords)
            fallback_recommendations.append(fallback)
        
        # 构建兜底文案
        draft_outfit = _build_fallback_outfit(fallback_recommendations, web_formulas, scene)
        
        return {
            **state,
            "draft_outfit": draft_outfit,
            "fallback_recommendations": fallback_recommendations,
            "iterations": iterations
        }
    
    # 部分品类缺失 - 给出补充建议
    if missing_categories and all_items:
        print(f"[Node: generate_outfit] 部分品类缺失: {missing_categories}")
        
        # 为缺失品类生成兜底推荐
        search_tasks = state.get("search_tasks", [])
        missing_recommendations = []
        for category in missing_categories:
            keywords = []
            for task in search_tasks:
                if task.get("category") == category:
                    keywords = task.get("keywords", [])
                    break
            fallback = _generate_fallback_recommendation(category, scene, keywords)
            missing_recommendations.append(fallback)
        
        # 将缺失推荐添加到 state 中
        state["missing_recommendations"] = missing_recommendations
        
        # 在生成的搭配文案末尾添加缺失品类的建议
        add_missing_tip = True
    
    # 构建 Prompt（可以加入小红书灵感）
    prompt = build_generate_prompt(
        retrieved_items=all_items,
        scene=scene,
        iterations=iterations,
        critic_feedback=critic_feedback,
        web_inspirations=web_formulas,  # 新增：传入小红书灵感
        conversation_history=conversation_history  # 新增：传入对话历史
    )
    
    # 调用 LLM
    llm = get_llm()
    
    # 获取当前迭代次数（如果是从 critic 返回的重新生成）
    current_iterations = state.get("iterations", 0)
    
    if llm is None:
        # Mock 模式
        draft_outfit = _generate_outfit_mock(wardrobe_results, web_formulas, scene, current_iterations)
    else:
        try:
            response = llm.invoke(prompt)
            draft_outfit = response.content
            
            # 添加标题
            if current_iterations > 0:
                draft_outfit = f"=== 搭配推荐 (第 {current_iterations} 次修正) ===\n\n" + draft_outfit
            else:
                draft_outfit = "=== 今日搭配推荐 ===\n\n" + draft_outfit
                
        except Exception as e:
            print(f"[ERROR] LLM 调用失败: {e}")
            draft_outfit = _generate_outfit_mock(wardrobe_results, web_formulas, scene, current_iterations)
    
    print(f"\n[Node: generate_outfit] 生成搭配: {len(draft_outfit)} 字符")
    
    # 如果有缺失品类，在文案末尾添加建议
    if missing_categories and all_items and add_missing_tip:
        draft_outfit += "\n\n💡 补充建议："
        for rec in missing_recommendations:
            draft_outfit += f"\n• {rec.get('recommendation', '')}"
    
    # 返回更新后的状态（包含迭代计数）
    return {
        **state,
        "draft_outfit": draft_outfit,
        "iterations": current_iterations + 1 if critic_feedback else current_iterations,
        "missing_recommendations": missing_recommendations if missing_categories else []
    }


def _generate_outfit_mock(
    wardrobe_results: List[Dict], 
    web_formulas: List[Dict],
    scene: str, 
    iterations: int
) -> str:
    """Mock 模式的搭配生成"""
    # 按品类分组
    categories = {}
    for result in wardrobe_results:
        category = result.get("category", "未知")
        items = result.get("items", [])
        if items:
            categories[category] = [item["name"] for item in items[:2]]
    
    outfit_parts = [f"{cat}: {', '.join(names)}" for cat, names in categories.items()]
    
    if iterations > 0:
        draft_outfit = f"=== 搭配推荐 (第 {iterations} 次) ===\n\n" + "\n".join(outfit_parts)
    else:
        draft_outfit = "=== 今日搭配推荐 ===\n\n" + "\n".join(outfit_parts)
    
    # 添加小红书灵感参考
    if web_formulas:
        inspiration = f"\n\n参考小红书热门搭配: {web_formulas[0].get('title', '')}"
        draft_outfit += inspiration
    
    draft_outfit += f"\n\n说明：基于场景【{scene}】偏好智能推荐。"
    return draft_outfit


# ========== 节点6: 批评者评估 ==========
def critic_evaluate_node(state: GraphState) -> GraphState:
    """
    节点6: 批评者评估
    
    使用 LLM（傲娇毒舌造型师）评估搭配方案
    
    Returns:
        更新后的 GraphState
    """
    draft_outfit = state["draft_outfit"]
    wardrobe_results = state.get("wardrobe_results", [])
    scene = state["parsed_intent"].get("scene", "commute")
    iterations = state.get("iterations", 0)
    
    print(f"\n[Node: critic_evaluate] LLM 评估中 (第 {iterations + 1} 次)...")
    
    # 合并所有衣物
    all_items = []
    for result in wardrobe_results:
        all_items.extend(result.get("items", []))
    
    # 构建 Prompt
    system_prompt = build_critic_system_prompt()
    human_prompt = build_critic_human_prompt(draft_outfit, all_items, scene)
    
    # 调用 LLM
    llm = get_llm()
    
    if llm is None:
        # Mock 模式
        critic_feedback, is_approved = _mock_critic_evaluate(iterations)
    else:
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            response = llm.invoke(messages)
            
            # 解析 JSON
            critic_feedback = parse_llm_json_response(response.content)
            is_approved = "[PASS]" in critic_feedback
            
            if is_approved:
                print(f"\n[Node: critic_evaluate] 第 {iterations + 1} 次评估: 通过")
            else:
                print(f"\n[Node: critic_evaluate] 第 {iterations + 1} 次评估: 需修改")
                
        except Exception as e:
            print(f"[ERROR] LLM 调用失败: {e}，降级到 Mock")
            critic_feedback, is_approved = _mock_critic_evaluate(iterations)
    
    return {
        **state,
        "critic_feedback": critic_feedback,
        "iterations": iterations  # 确保迭代次数在状态中保持一致
    }


def _mock_critic_evaluate(iterations: int) -> tuple:
    """Mock 模式的批评者评估"""
    is_approved = random.random() < 0.7
    
    if is_approved:
        critic_feedback = "[PASS] 搭配方案审核通过！"
        print(f"\n[Node: critic_evaluate] 第 {iterations + 1} 次评估: 通过 (Mock)")
    else:
        feedback_options = [
            "[REJECT] 颜色搭配不够协调",
            "[REJECT] 风格不够统一",
            "[REJECT] 缺少点睛之笔",
            "[REJECT] 场合适配度一般"
        ]
        critic_feedback = random.choice(feedback_options)
        print(f"\n[Node: critic_evaluate] 第 {iterations + 1} 次评估: 需修改 - {critic_feedback} (Mock)")
    
    return critic_feedback, is_approved
