# -*- coding: utf-8 -*-
"""
天气查询工具

Agent 可直接调用的天气工具：
- get_weather_info: 获取指定城市的天气信息
- get_clothing_suggestion: 根据天气获取穿搭建议
- search_cities: 搜索城市信息

依赖: services.weather 模块
"""

import asyncio
from typing import Dict, List, Optional
import sys
import os

# 确保可以导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.weather import (
    get_weather as _get_weather,
    get_clothing_suggestion as _get_clothing_suggestion,
    get_season_from_weather as _get_season_from_weather,
    search_city as _search_city,
    WeatherInfo
)


def get_weather_info(
    location: str = "101020100",
    city_name: Optional[str] = None
) -> Dict:
    """
    获取指定城市/位置的天气信息
    
    Args:
        location: 位置标识
                 - 可以是 LocationID (如 101020100=上海, 101010100=北京)
                 - 或经纬度坐标 (如 "116.41,39.92")
                 - 默认: 101020100 (上海)
        city_name: 城市名称（可选，用于显示）
    
    Returns:
        包含天气信息的字典:
        {
            "success": bool,
            "location": str,
            "temperature": float,
            "feels_like": float,
            "condition": str,
            "humidity": float,
            "wind_dir": str,
            "wind_scale": str,
            "message": str
        }
    
    Examples:
        >>> get_weather_info("101020100")
        >>> get_weather_info("北京")
        >>> get_weather_info(city_name="上海")
    """
    # 如果传入的是城市名称，尝试查找 LocationID
    if city_name and not location:
        # 尝试搜索城市
        cities = asyncio.run(_search_city(city_name, limit=1))
        if cities:
            location = cities[0].id
        else:
            return {
                "success": False,
                "message": f"未找到城市: {city_name}"
            }
    
    # 运行异步获取天气
    weather = asyncio.run(_get_weather(location))
    
    if weather is None:
        return {
            "success": False,
            "location": location,
            "message": "获取天气信息失败，请检查 API 配置"
        }
    
    return {
        "success": True,
        "location": location,
        "temperature": weather.temperature,
        "feels_like": weather.feelsLike,
        "condition": weather.condition,
        "humidity": weather.humidity,
        "wind_dir": weather.windDir,
        "wind_scale": weather.windScale,
        "obs_time": weather.obsTime,
        "message": f"当前{weather.condition}，温度{weather.temperature}°C (体感{weather.feelsLike}°C)"
    }


def get_weather_with_suggestion(
    location: str = "101020100",
    city_name: Optional[str] = None
) -> Dict:
    """
    获取天气信息及穿搭建议
    
    Args:
        location: 位置标识 (同 get_weather_info)
        city_name: 城市名称（可选）
    
    Returns:
        包含天气和穿搭建议的字典:
        {
            "success": bool,
            "weather": {...},
            "seasons": List[str],
            "suggestion": str,
            "message": str
        }
    
    Examples:
        >>> get_weather_with_suggestion("上海")
    """
    # 获取天气
    weather_result = get_weather_info(location, city_name)
    
    if not weather_result.get("success"):
        return weather_result
    
    # 构建 WeatherInfo 对象用于穿搭建议
    weather_info = WeatherInfo(
        temperature=weather_result["temperature"],
        feelsLike=weather_result["feels_like"],
        condition=weather_result["condition"],
        icon="",
        humidity=weather_result["humidity"],
        windDir=weather_result["wind_dir"],
        windScale=weather_result["wind_scale"],
        location=weather_result["location"],
        obsTime=weather_result.get("obs_time", "")
    )
    
    # 获取穿搭建议
    suggestion = _get_clothing_suggestion(weather_info)
    
    # 获取适合季节
    seasons = _get_season_from_weather(weather_info)
    
    return {
        "success": True,
        "weather": {
            "temperature": weather_result["temperature"],
            "feels_like": weather_result["feels_like"],
            "condition": weather_result["condition"],
            "humidity": weather_result["humidity"],
            "wind_dir": weather_result["wind_dir"],
            "wind_scale": weather_result["wind_scale"]
        },
        "seasons": seasons,
        "suggestion": suggestion,
        "message": f"{weather_result['message']}\n{suggestion}"
    }


def search_city_info(query: str, limit: int = 5) -> Dict:
    """
    搜索城市信息
    
    Args:
        query: 城市名称关键词（支持中文、拼音）
        limit: 返回结果数量，默认 5
    
    Returns:
        包含城市列表的字典:
        {
            "success": bool,
            "query": str,
            "cities": List[dict],
            "message": str
        }
    
    Examples:
        >>> search_city_info("北京")
        >>> search_city_info("shanghai")
    """
    cities = asyncio.run(_search_city(query, limit))
    
    if not cities:
        return {
            "success": False,
            "query": query,
            "cities": [],
            "message": f"未找到匹配'{query}'的城市"
        }
    
    city_list = [
        {
            "name": city.name,
            "id": city.id,
            "province": city.adm1,
            "city": city.adm2,
            "country": city.country
        }
        for city in cities
    ]
    
    return {
        "success": True,
        "query": query,
        "cities": city_list,
        "message": f"找到 {len(city_list)} 个匹配'{query}'的城市"
    }


# ========== 常用城市便捷函数 ==========
def get_beijing_weather() -> Dict:
    """获取北京天气"""
    return get_weather_info("101010100", "北京")


def get_shanghai_weather() -> Dict:
    """获取上海天气"""
    return get_weather_info("101020100", "上海")


def get_guangzhou_weather() -> Dict:
    """获取广州天气"""
    return get_weather_info("101280101", "广州")


def get_shenzhen_weather() -> Dict:
    """获取深圳天气"""
    return get_weather_info("101280601", "深圳")


def get_hangzhou_weather() -> Dict:
    """获取杭州天气"""
    return get_weather_info("101210101", "杭州")


# ========== 时间工具 ==========
def get_current_time() -> Dict:
    """
    获取当前时间信息
    
    Returns:
        包含当前时间的字典:
        {
            "success": bool,
            "timestamp": int,        # Unix 时间戳
            "datetime": str,         # 格式化日期时间 "2024-01-15 10:30:00"
            "date": str,             # 日期 "2024-01-15"
            "time": str,             # 时间 "10:30:00"
            "weekday": str,          # 星期几 "星期一"
            "year": int,
            "month": int,
            "day": int,
            "hour": int,
            "message": str
        }
    """
    from datetime import datetime
    
    now = datetime.now()
    weekday_names = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    
    return {
        "success": True,
        "timestamp": int(now.timestamp()),
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "weekday": weekday_names[now.weekday()],
        "year": now.year,
        "month": now.month,
        "day": now.day,
        "hour": now.hour,
        "message": f"当前时间是 {now.strftime('%Y年%m月%d日 %H:%M')} {weekday_names[now.weekday()]}"
    }


def calculate_future_date(days_ahead: int) -> Dict:
    """
    计算未来某天的日期
    
    Args:
        days_ahead: 距离今天的天数（正数表示未来，负数表示过去）
    
    Returns:
        包含计算结果的字典:
        {
            "success": bool,
            "days_ahead": int,
            "date": str,             # 日期 "2024-01-18"
            "weekday": str,          # 星期几
            "message": str
        }
    """
    from datetime import datetime, timedelta
    
    now = datetime.now()
    future = now + timedelta(days=days_ahead)
    weekday_names = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    
    return {
        "success": True,
        "days_ahead": days_ahead,
        "date": future.strftime("%Y-%m-%d"),
        "weekday": weekday_names[future.weekday()],
        "message": f"距离今天{abs(days_ahead)}天后的日期是 {future.strftime('%Y年%m月%d日')} {weekday_names[future.weekday()]}"
    }


# ========== 单元测试 ==========
if __name__ == "__main__":
    print("=" * 60)
    print("测试天气查询工具")
    print("=" * 60)
    
    # 测试获取天气
    print("\n[测试1] 获取上海天气")
    result = get_weather_info("101020100")
    print(f"成功: {result.get('success')}")
    if result.get('success'):
        print(f"天气: {result['condition']}")
        print(f"温度: {result['temperature']}°C")
        print(f"体感: {result['feels_like']}°C")
    
    # 测试穿搭建议
    print("\n[测试2] 获取天气及穿搭建议")
    result = get_weather_with_suggestion("101020100")
    print(f"成功: {result.get('success')}")
    if result.get('success'):
        print(f"建议: {result['suggestion']}")
        print(f"适合季节: {result['seasons']}")
    
    # 测试城市搜索
    print("\n[测试3] 搜索城市")
    result = search_city_info("北京")
    print(f"成功: {result.get('success')}")
    if result.get('success'):
        for city in result['cities'][:3]:
            print(f"  - {city['name']} ({city['province']}): {city['id']}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
