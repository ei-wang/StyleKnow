# -*- coding: utf-8 -*-
"""
CRAG (双路检索) 工具链

包含：
- 小红书数据清洗函数
- 小红书搜索 API（真实调用）
- VLM 穿搭公式提取（真实调用）
- GraphState 状态定义

作者：AI 穿搭助手
"""

import os
import json
import requests
import re
import asyncio
from typing import TypedDict, List, Dict, Any, Optional
from urllib.parse import urlencode


# ========== GraphState 状态定义 ==========
class GraphState(TypedDict):
    """
    LangGraph 工作流状态定义（CRAG 双路检索版本）
    
    字段说明：
    - user_query: 用户原始查询
    - parsed_intent: 解析后的用户意图
    - local_items: 本地衣柜召回的衣物
    - web_formulas: 小红书提取的穿搭公式
    - draft_outfit: 生成的搭配文案
    - critic_feedback: 批评者的反馈
    - iterations: 迭代次数
    """
    user_query: str                      # 用户原始查询
    parsed_intent: Dict[str, Any]        # 解析后的用户意图
    local_items: List[dict]             # 本地衣柜召回的衣物
    web_formulas: List[dict]            # 小红书提取的穿搭公式
    draft_outfit: str                   # 生成的搭配文案
    critic_feedback: str                 # 批评者的反馈
    iterations: int                     # 迭代次数


# ========== 配置 ==========
# VLM 配置
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


# ========== 小红书数据清洗函数 ==========
def clean_xhs_response(raw_json: dict) -> List[dict]:
    """
    清洗小红书 API 返回的原始 JSON 数据
    
    提取规则：
    1. 从 data.data.items 列表中提取每个 note
    2. 只保留字段：id, title, desc, liked_count
    3. 从 images_list 中提取前两张高清图片 URL
    
    Args:
        raw_json: 小红书 API 返回的原始 JSON 字典
    
    Returns:
        清洗后的笔记列表，每条包含：
        - id: 笔记 ID
        - title: 标题
        - desc: 正文描述
        - liked_count: 点赞数
        - images: 前两张高清图片 URL 列表
    """
    cleaned_items = []
    
    try:
        items = raw_json.get("data", {}).get("data", {}).get("items", [])
        
        for item in items:
            note = item.get("note", {})
            if not note:
                continue
            
            note_id = note.get("id", "")
            title = note.get("title", "")
            desc = note.get("desc", "")
            liked_count = note.get("liked_count", 0)
            
            # 提取前两张高清图片
            images_list = note.get("images_list", [])
            images = []
            
            for idx, img in enumerate(images_list[:2]):
                img_url = img.get("url") or img.get("url_size_large", "")
                if img_url:
                    images.append(img_url)
            
            cleaned_note = {
                "id": note_id,
                "title": title,
                "desc": desc,
                "liked_count": liked_count,
                "images": images
            }
            
            cleaned_items.append(cleaned_note)
    
    except Exception as e:
        print(f"[ERROR] 清洗小红书数据失败: {e}")
    
    print(f"[INFO] 清洗完成，共提取 {len(cleaned_items)} 篇笔记")
    return cleaned_items


# ========== 小红书 302.ai API ==========
def search_xhs_notes(
    keyword: str,
    page: int = 1,
    sort_type: str = "general",
    filter_note_time: str = "不限",
    filter_note_type: str = "不限",
    search_id: str = None,
    session_id: str = None,
    xsec_token: str = None,
    xsec_source: str = "pc_feed"
) -> dict:
    """
    使用 302.ai API 调用小红书搜索
    
    Args:
        keyword: 搜索关键词
        page: 页码 (默认 1)
        sort_type: 排序方式 (general/popularity_descending/time_descending)
        filter_note_time: 时间筛选 (不限/一天内/一周内/半年内)
        filter_note_type: 类型筛选 (不限/视频笔记/普通笔记)
        search_id: 搜索ID，翻页时需要
        session_id: 会话ID，翻页时需要
    
    Returns:
        小红书 API 响应的原始 JSON
    """
    from storage.config_store import load_config
    config = load_config()
    
    # 使用配置中的 API
    api_url = config.xhs_api_url
    api_key = config.xhs_api_key
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
        "Accept": "*/*",
        "Cache-Control": "no-cache"
    }
    
    # 构建查询参数
    params = {
        "keyword": keyword,
        "sort_type": sort_type,
        "page": page,
        "filter_note_type": filter_note_type,
        "filter_note_time": filter_note_time
    }
    
    # 添加可选参数
    if search_id:
        params["search_id"] = search_id
    if session_id:
        params["session_id"] = session_id
    
    try:
        response = requests.get(api_url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        
        print(f"[INFO] 小红书搜索成功: keyword='{keyword}', page={page}")
        return result
        
    except requests.exceptions.Timeout:
        print(f"[ERROR] 小红书 API 请求超时")
        return {"code": -1, "msg": "请求超时", "data": {"items": []}}
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] 小红书 API 请求失败: {e}")
        return {"code": -1, "msg": str(e), "data": {"items": []}}


def mock_search_xhs(query: str) -> dict:
    """
    模拟小红书搜索 API（用于开发测试）
    
    Args:
        query: 搜索关键词
    
    Returns:
        Mock 数据
    """
    # 这个函数保留作为开发调试使用
    mock_data = {
        "code": 200,
        "data": {
            "code": 0,
            "data": {
                "items": [
                    {
                        "model_type": "note",
                        "note": {
                            "id": "67e15a0e000000000602b807",
                            "title": "女生必学的穿搭原则",
                            "desc": "#学会穿搭人生开挂 #夏日穿搭 #夏季穿搭 #温柔穿搭 #微胖穿搭 #休闲穿搭",
                            "liked_count": 1318,
                            "images_list": [
                                {
                                    "url": "https://example.com/image1.jpg",
                                    "url_size_large": "https://example.com/image1_large.jpg"
                                },
                                {
                                    "url": "https://example.com/image2.jpg",
                                    "url_size_large": "https://example.com/image2_large.jpg"
                                }
                            ]
                        }
                    },
                    {
                        "model_type": "note",
                        "note": {
                            "id": "67e25b1f000000000703c908",
                            "title": "通勤西装万能搭配公式",
                            "desc": "#通勤穿搭 #西装外套 #职场穿搭 #干练风格",
                            "liked_count": 2567,
                            "images_list": [
                                {
                                    "url": "https://example.com/image3.jpg",
                                    "url_size_large": "https://example.com/image3_large.jpg"
                                },
                                {
                                    "url": "https://example.com/image4.jpg",
                                    "url_size_large": "https://example.com/image4_large.jpg"
                                }
                            ]
                        }
                    }
                ]
            }
        }
    }
    
    print(f"[MOCK] 模拟小红书搜索: keyword='{query}'")
    return mock_data


# ========== VLM 穿搭公式提取 ==========
async def vlm_extract_formula_async(image_urls: List[str], title: str) -> str:
    """
    异步调用 VLM 提取穿搭公式
    
    使用 OpenAI Vision API (GPT-4V) 提取图片中的穿搭信息
    
    Args:
        image_urls: 图片 URL 列表
        title: 笔记标题
    
    Returns:
        结构化穿搭公式文本
    """
    if not OPENAI_API_KEY:
        print("[WARN] 未设置 OPENAI_API_KEY，使用模拟模式")
        return _mock_formula(title)
    
    if not image_urls:
        return "无法提取公式：没有图片"
    
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        # 构建消息
        user_content = f"""请分析以下穿搭图片，提取穿搭公式。

标题: {title}

请用以下格式描述：
- 上装: （颜色+款式）
- 下装: （颜色+款式）
- 鞋子: （颜色+款式）
- 风格亮点: （1-2个关键词）

只输出格式化的内容，不要其他描述。"""
        
        # 添加图片
        image_contents = [{"type": "image_url", "image_url": {"url": url}} for url in image_urls[:2]]
        
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_content},
                        *image_contents
                    ]
                }
            ],
            max_tokens=300,
            temperature=0.3
        )
        
        formula = response.choices[0].message.content
        print(f"[INFO] VLM 提取成功: {formula[:50]}...")
        return formula
        
    except Exception as e:
        print(f"[ERROR] VLM 调用失败: {e}")
        return _mock_formula(title)


def vlm_extract_formula(image_urls: List[str], title: str) -> str:
    """
    同步调用 VLM 提取穿搭公式
    
    Args:
        image_urls: 图片 URL 列表
        title: 笔记标题
    
    Returns:
        结构化穿搭公式文本
    """
    # 直接调用异步版本
    return asyncio.run(vlm_extract_formula_async(image_urls, title))


def _mock_formula(title: str) -> str:
    """Mock 穿搭公式（当 VLM 不可用时使用）"""
    formulas = [
        "上装: 浅卡其色夹克\n下装: 水洗蓝高腰直筒牛仔裤\n鞋子: 德训鞋\n风格亮点: 美式复古",
        "上装: 黑色廓形西装外套\n下装: 白色阔腿西裤\n鞋子: 黑色乐福鞋\n风格亮点: 干练职场",
        "上装: 条纹衬衫\n下装: 卡其色休闲长裤\n鞋子: 白色运动鞋\n风格亮点: 清爽休闲",
    ]
    
    if "通勤" in title or "职场" in title:
        return formulas[1]
    elif "夏日" in title or "夏季" in title:
        return formulas[2]
    return formulas[0]


# ========== 串联测试 ==========
if __name__ == "__main__":
    print("=" * 70)
    print("CRAG 工具链串联测试")
    print("=" * 70)
    
    # Step 1: 搜索小红书
    print("\n[Step 1] 搜索小红书笔记")
    query = "通勤穿搭"
    
    # 优先尝试真实 API，如果失败则使用 Mock
    raw_response = search_xhs_notes(query, sort_type="popularity_descending")
    
    if raw_response.get("code") != 200:
        print("[INFO] 使用 Mock 数据")
        raw_response = mock_search_xhs(query)
    
    # Step 2: 清洗数据
    print("\n[Step 2] 清洗数据")
    cleaned_notes = clean_xhs_response(raw_response)
    
    for i, note in enumerate(cleaned_notes, 1):
        print(f"\n--- 笔记 {i} ---")
        print(f"ID: {note['id']}")
        print(f"标题: {note['title']}")
        print(f"点赞: {note['liked_count']}")
        print(f"图片数: {len(note['images'])}")
    
    # Step 3: VLM 提取公式
    print("\n[Step 3] VLM 提取穿搭公式")
    web_formulas = []
    
    for note in cleaned_notes:
        formula = vlm_extract_formula(note['images'], note['title'])
        web_formulas.append({
            "note_id": note['id'],
            "title": note['title'],
            "formula": formula,
            "images": note['images']
        })
        print(f"\n公式: {formula}")
    
    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)
