# -*- coding: utf-8 -*-
"""测试 Agent 天气工具调用"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from agent.weather_tools import AGENT_WEATHER_TOOLS

print("=== 已注册的工具列表 ===")
for tool in AGENT_WEATHER_TOOLS:
    print(f"\n【{tool.name}】")
    print(f"  描述: {tool.description[:80]}...")

# 测试直接调用各个工具
print("\n\n=== 测试工具调用 ===")

# 1. 获取当前时间
print("\n【get_current_time】")
result = AGENT_WEATHER_TOOLS[0].invoke({})
print(result)

# 2. 计算未来日期
print("\n【calculate_future_date】")
result = AGENT_WEATHER_TOOLS[1].invoke({"days_ahead": 3})
print(result)

# 3. 搜索城市
print("\n【search_city】")
result = AGENT_WEATHER_TOOLS[2].invoke({"query": "北京"})
print(result)

# 4. 获取天气
print("\n【get_weather_info】")
result = AGENT_WEATHER_TOOLS[3].invoke({"location": "101010100", "city_name": "北京"})
print(result)

# 5. 获取天气+穿搭建议
print("\n【get_weather_with_suggestion】")
result = AGENT_WEATHER_TOOLS[4].invoke({"location": "101020100", "city_name": "上海"})
print(result)

# 6. 获取天气预报
print("\n【get_weather_forecast】")
result = AGENT_WEATHER_TOOLS[5].invoke({"location": "101010100", "days": 7})
print(result)
