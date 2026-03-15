# -*- coding: utf-8 -*-
"""
穿搭推荐 Agent 工具集合

使用 LangChain @tool 装饰器封装的所有 Agent 可调用工具
"""

from langchain.tools import tool
from typing import List, Dict, Optional
import sys
import os

# 确保可以导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入各工具模块
from tools.db_search import (
    search_wardrobe as _search_wardrobe,
    search_wardrobe_by_task as _search_wardrobe_by_task,
    add_clothing_item as _add_clothing_item,
    get_wardrobe_stats as _get_wardrobe_stats
)
from tools.weather import (
    get_weather_info as _get_weather_info,
    get_weather_with_suggestion as _get_weather_with_suggestion,
    search_city_info as _search_city_info
)
from tools.xhs_search import (
    search_xhs_notes as _search_xhs_notes,
    search_xhs_notes_by_task as _search_xhs_notes_by_task,
    clean_xhs_response as _clean_xhs_response
)


# ========== 衣柜工具 ==========
@tool
def search_wardrobe(
    scene: str = "commute",
    top_k: int = 8,
    category: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> str:
    """
    根据场景检索衣柜中的衣物
    
    参数:
        scene: 场景关键词，如 "通勤"、"度假"、"休闲"、"运动" 等
        top_k: 返回的衣物数量，默认 8
        category: 可选，按品类筛选 (上衣/裤子/裙子/鞋子/配饰)
        tags: 可选，按标签筛选列表
    
    返回:
        检索到的衣物列表JSON字符串
    
    示例:
        search_wardrobe(scene="通勤")
        search_wardrobe(scene="度假", top_k=10)
    """
    result = _search_wardrobe(scene=scene, top_k=top_k, category=category, tags=tags)
    import json
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def search_wardrobe_by_task(
    category: str,
    keywords: str,
    top_k: int = 5
) -> str:
    """
    按品类+关键词检索衣柜（新版本检索节点使用）
    
    参数:
        category: 品类名称，如 "上衣"、"裤子"、"鞋子"、"配饰"
        keywords: LLM生成的关键词，逗号分隔，如 "简约,通勤,白色"
        top_k: 返回数量，默认 5
    
    返回:
        检索结果JSON字符串
    
    示例:
        search_wardrobe_by_task(category="上衣", keywords="简约,通勤,白色")
    """
    # 将逗号分隔的关键词转换为列表
    keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
    result = _search_wardrobe_by_task(category=category, keywords=keywords_list, top_k=top_k)
    import json
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def add_clothing_item(
    name: str,
    category: str,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict] = None
) -> str:
    """
    向衣柜添加新衣物
    
    参数:
        name: 衣物名称
        category: 品类 (上衣/裤子/裙子/鞋子/配饰)
        tags: 标签列表，可选
        metadata: 额外元数据字典，可选
    
    返回:
        添加结果的JSON字符串
    
    示例:
        add_clothing_item(name="红色连衣裙", category="裙子", tags=["度假", "复古"])
    """
    result = _add_clothing_item(name=name, category=category, tags=tags, metadata=metadata)
    import json
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def get_wardrobe_stats() -> str:
    """
    获取衣柜统计信息
    
    返回:
        包含总数、品类分布、可用场景的JSON字符串
    
    示例:
        get_wardrobe_stats()
    """
    result = _get_wardrobe_stats()
    import json
    return json.dumps(result, ensure_ascii=False, indent=2)


# ========== 天气工具 ==========
@tool
def get_weather_info(
    location: str = "101020100",
    city_name: Optional[str] = None
) -> str:
    """
    获取指定城市的天气信息
    
    参数:
        location: 位置标识，LocationID (如 101020100=上海) 或经纬度坐标，默认上海
        city_name: 城市名称，可选
    
    返回:
        天气信息的JSON字符串
    
    示例:
        get_weather_info(location="101020100")
        get_weather_info(city_name="北京")
    """
    result = _get_weather_info(location=location, city_name=city_name)
    import json
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def get_weather_with_suggestion(
    location: str = "101020100",
    city_name: Optional[str] = None
) -> str:
    """
    获取天气信息及穿搭建议
    
    参数:
        location: 位置标识，LocationID 或经纬度坐标，默认上海
        city_name: 城市名称，可选
    
    返回:
        天气和穿搭建议的JSON字符串
    
    示例:
        get_weather_with_suggestion(location="101020100")
    """
    result = _get_weather_with_suggestion(location=location, city_name=city_name)
    import json
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def search_city_info(query: str, limit: int = 5) -> str:
    """
    搜索城市信息
    
    参数:
        query: 城市名称关键词，支持中文、拼音
        limit: 返回结果数量，默认5
    
    返回:
        城市列表的JSON字符串
    
    示例:
        search_city_info(query="北京")
    """
    result = _search_city_info(query=query, limit=limit)
    import json
    return json.dumps(result, ensure_ascii=False, indent=2)



# ========== 小红书工具 ==========
@tool
def search_xhs_notes(
    keyword: str,
    page: int = 1,
    sort_type: str = "general",
    filter_note_time: str = "不限",
    filter_note_type: str = "不限"
) -> str:
    """
    搜索小红书穿搭笔记
    
    参数:
        keyword: 搜索关键词
        page: 页码，默认1
        sort_type: 排序方式，默认 general (按热度)
        filter_note_time: 时间筛选 (不限/一天内/一周内/半年内)
        filter_note_type: 类型筛选 (不限/视频笔记/普通笔记)
    
    返回:
        笔记列表的JSON字符串
    
    示例:
        search_xhs_notes(keyword="通勤穿搭")
        search_xhs_notes(keyword="夏日穿搭", filter_note_time="一周内")
    """
    result = _search_xhs_notes(
        keyword=keyword,
        page=page,
        sort_type=sort_type,
        filter_note_time=filter_note_time,
        filter_note_type=filter_note_type
    )
    cleaned_result = _clean_xhs_response(result)
    import json
    return json.dumps(cleaned_result, ensure_ascii=False, indent=2)


@tool
def search_xhs_notes_by_task(
    search_query: str,
    top_k: int = 5
) -> str:
    """
    按搜索任务搜索小红书（新版本使用）
    
    小红书搜索通常是整体穿搭搜索，结合场景和风格
    
    参数:
        search_query: LLM生成的搜索关键词，如 "通勤穿搭 简约干练"
        top_k: 返回数量，默认 5
    
    返回:
        笔记列表的JSON字符串
    
    示例:
        search_xhs_notes_by_task(search_query="通勤穿搭 简约干练")
    """
    result = _search_xhs_notes_by_task(search_query=search_query, top_k=top_k)
    import json
    return json.dumps(result, ensure_ascii=False, indent=2)


# ========== 获取所有工具 ==========
def get_all_tools():
    """
    获取所有可用的 Agent 工具列表
    
    返回:
        LangChain Tool 列表
    """
    return [
        search_wardrobe,
        search_wardrobe_by_task,
        add_clothing_item,
        get_wardrobe_stats,
        get_weather_info,
        get_weather_with_suggestion,
        search_city_info,
        search_xhs_notes,
        search_xhs_notes_by_task
    ]


# ========== 单元测试 ==========
if __name__ == "__main__":
    print("=" * 60)
    print("测试 LangChain @tool 工具")
    print("=" * 60)
    
    # 测试衣柜检索
    print("\n[测试1] 搜索衣柜工具")
    result = search_wardrobe.invoke({"scene": "通勤", "top_k": 5})
    print(result[:200] if len(result) > 200 else result)
    
    # 测试天气查询
    print("\n[测试2] 天气查询工具")
    result = get_weather_info.invoke({"location": "101020100"})
    print(result[:200] if len(result) > 200 else result)
    
    # 测试小红书搜索
    print("\n[测试3] 小红书搜索工具")
    result = search_xhs_notes.invoke({"keyword": "穿搭"})
    print(result[:200] if len(result) > 200 else result)
    
    # 列出所有工具
    print("\n[测试4] 所有工具列表")
    tools = get_all_tools()
    for t in tools:
        print(f"  - {t.name}: {t.description[:60]}...")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
