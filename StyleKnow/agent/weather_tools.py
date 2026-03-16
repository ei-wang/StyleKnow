# -*- coding: utf-8 -*-
"""
Agent 天气工具 - LangChain Tool 封装

用于 Agent 意图识别阶段，由 LLM 自行判断是否调用：
- get_current_time: 获取当前时间
- search_city: 搜索城市信息
- get_weather_info: 获取指定城市天气
- get_weather_with_suggestion: 获取天气+穿搭建议
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional
from langchain_core.tools import tool

from tools.weather import (
    get_weather_info as _get_weather_info,
    get_weather_with_suggestion as _get_weather_with_suggestion,
    search_city_info as _search_city_info,
    get_current_time as _get_current_time,
    calculate_future_date as _calculate_future_date
)


@tool
def get_current_time() -> str:
    """
    获取当前时间。当用户询问涉及具体日期、时间、星期几时必须调用此工具。
    
    适用场景：
    - 用户说"明天"、"后天"、"下周一"等相对时间
    - 用户想知道今天几号、星期几
    - 需要计算某个日期是星期几
    
    返回当前时间信息，包括年、月、日、时、分、秒、星期。
    """
    result = _get_current_time()
    return result.get("message", f"当前时间: {result['datetime']} {result['weekday']}")


@tool
def calculate_future_date(days_ahead: int) -> str:
    """
    计算未来某一天的日期。当用户说"后天去"、"3天后"等需要计算具体日期时调用。
    
    Args:
        days_ahead: 距离今天的天数（正数表示未来，负数表示过去）
        
    适用场景：
    - 用户说"后天去三亚" -> days_ahead=2
    - 用户说"一周后" -> days_ahead=7
    - 用户说"下周五" -> 需要先获取今天星期几，再计算
    
    返回计算后的日期和星期。
    """
    result = _calculate_future_date(days_ahead)
    return result.get("message", f"距离今天{abs(days_ahead)}天后是 {result['date']} {result['weekday']}")


@tool
def search_city(query: str) -> str:
    """
    搜索城市信息。当用户提到某个地点/城市，需要获取该城市的详细信息（用于后续天气查询）时调用。

    Args:
        query: 城市名称（支持中文、拼音、英文）

    适用场景：
    - 用户说"去三亚玩"
    - 用户说"北京天气怎么样"
    - 用户说"上海穿搭"

    返回匹配的城市列表，包含城市ID、经纬度坐标（用于天气查询）、所属省份、国家等信息。
    """
    result = _search_city_info(query, limit=3)
    if result.get("success"):
        cities = result.get("cities", [])
        if cities:
            lines = [f"找到以下城市:"]
            for c in cities:
                lines.append(f"- {c['name']} ({c.get('province', '')}/{c.get('country', '')}) ID: {c['id']} 坐标: {c['lat']},{c['lon']}")
            return "\n".join(lines)
    return f"未找到城市: {query}"


@tool
def get_weather_info(location: str, city_name: Optional[str] = None) -> str:
    """
    获取指定城市的天气信息。当用户询问某地的天气、需要根据天气给出穿搭建议时调用。

    Args:
        location: 城市ID（如 101280101=三亚，101010100=北京，101020100=上海）
                  可以使用 search_city 工具先获取城市ID
        city_name: 城市名称（可选，用于显示）

    适用场景：
    - 用户说"三亚天气怎么样"
    - 用户说"去三亚需要带什么衣服"
    - 用户说"北京冷吗"

    返回详细的天气信息，包括温度、体感温度、湿度、风力、天气状况等。
    """
    result = _get_weather_info(location, city_name)
    if result.get("success"):
        return (f"{result['location']}天气:\n"
                f"- 温度: {result.get('temperature', '?')}°C\n"
                f"- 体感: {result.get('feels_like', '?')}°C\n"
                f"- 天气: {result.get('condition', '?')}\n"
                f"- 湿度: {result.get('humidity', '?')}%\n"
                f"- 风力: {result.get('wind_scale', '?')}级 {result.get('wind_dir', '')}")
    return f"获取天气失败: {result.get('message', '未知错误')}"


@tool
def get_weather_with_suggestion(location: str, city_name: Optional[str] = None) -> str:
    """
    获取天气信息及穿搭建议。当用户需要根据天气情况获取穿衣建议时调用。

    这是最常用的工具，可以同时获取天气数据供 LLM 生成穿搭建议。

    Args:
        location: 城市ID（如 101280101=三亚，101010100=北京）或经纬度（如 "116.40,39.90"）
        city_name: 城市名称（可选）

    适用场景：
    - 用户说"去三亚玩3天穿什么"
    - 用户说"下周去北京，需要带什么衣服"
    - 用户说"明天上班怎么穿"

    返回详细的天气信息（包含温度、体感、湿度、风力、天气状况等），由 LLM 根据这些数据生成穿搭建议。
    """
    result = _get_weather_with_suggestion(location, city_name)
    if result.get("success"):
        w = result.get("weather", {})
        return (f"📍 {result['location']} 天气预报:\n"
                f"- 天气状况: {w.get('condition', '?')}\n"
                f"- 温度: {w.get('temperature', '?')}°C\n"
                f"- 体感温度: {w.get('feels_like', '?')}°C\n"
                f"- 相对湿度: {w.get('humidity', '?')}%\n"
                f"- 风力: {w.get('wind_scale', '?')}级 {w.get('wind_dir', '')}\n"
                f"\n💡 请根据以上天气信息为用户推荐合适的穿搭建议")
    return f"获取天气失败: {result.get('message', '未知错误')}"


@tool
def get_weather_forecast(location: str, days: int = 7) -> str:
    """
    获取未来天气预报。当用户询问"明天天气怎么样"、"下周天气如何"或需要了解未来几天的天气趋势时调用。

    Args:
        location: 城市ID（如 101280101=三亚，101010100=北京，101020100=上海）
                  可以使用 search_city 工具先获取城市ID
        days: 预报天数，可选 3/7/10/15/30 天，默认 7 天

    适用场景：
    - 用户说"明天天气怎么样"
    - 用户说"下周去北京需要带什么衣服"
    - 用户说"三亚未来7天天气"

    返回未来几天的天气预报，包括日期、最高/最低温度、天气状况等。
    """
    from services.weather import get_weather_forecast as _get_forecast
    import asyncio

    forecast = asyncio.run(_get_forecast(location, days))
    if forecast:
        lines = [f"天气预报 ({days}天):"]
        for day in forecast[:days]:
            lines.append(f"- {day.fxDate}: {day.tempMin}~{day.tempMax}°C {day.textDay}")
        return "\n".join(lines)
    return f"获取天气预报失败，请检查城市ID是否正确"


# 工具列表（用于绑定到 LLM）
AGENT_WEATHER_TOOLS = [
    get_current_time,
    calculate_future_date,
    search_city,
    get_weather_info,
    get_weather_with_suggestion,
    get_weather_forecast
]
