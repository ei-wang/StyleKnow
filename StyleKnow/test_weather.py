"""
获取用户位置的多种方案测试

方案1: 前端获取经纬度 → 后端查询天气
方案2: 第三方 IP 定位 API
方案3: 城市搜索 + 用户确认
"""
import asyncio
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from services.weather import search_city, get_weather
from tools.weather import get_current_time


async def test_location_ways():
    print("=" * 60)
    print("方案 1: 前端 Geolocation API (推荐)")
    print("=" * 60)
    print("""
前端代码示例:
```javascript
navigator.geolocation.getCurrentPosition(
  (position) => {
    const lat = position.coords.latitude;
    const lon = position.coords.longitude;
    // 发送给后端: /api/weather?location=lat,lon
  },
  (error) => console.error('定位失败:', error)
);
```""")

    print()
    print("=" * 60)
    print("方案 2: 使用 ip-api.com 获取 IP 位置 (免费)")
    print("=" * 50)
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get("http://ip-api.com/json/?fields=status,country,city,lat,lon")
            data = resp.json()
            if data.get("status") == "success":
                print(f"国家: {data.get('country')}")
                print(f"城市: {data.get('city')}")
                print(f"经纬度: {data.get('lat')}, {data.get('lon')}")
                
                # 用经纬度查天气
                location = f"{data.get('lon')},{data.get('lat')}"
                weather = await get_weather(location)
                if weather:
                    print(f"温度: {weather.temperature}°C")
                    print(f"天气: {weather.condition}")
    except Exception as e:
        print(f"失败: {e}")

    print()
    print("=" * 60)
    print("方案 3: 城市搜索 + 用户确认")
    print("=" * 50)
    # 模拟用户输入城市名
    test_cities = ["北京", "上海", "广州", "深圳", "杭州"]
    for city_name in test_cities:
        cities = await search_city(city_name, limit=1)
        if cities:
            city = cities[0]
            weather = await get_weather(city.id)
            if weather:
                print(f"{city.name}: {weather.temperature}°C {weather.condition}")

    print()
    print("=" * 60)
    print("方案 4: 经纬度直接查询天气")
    print("=" * 50)
    # 和风天气支持经纬度格式: lon,lat
    location = "121.47,31.23"  # 上海
    weather = await get_weather(location)
    if weather:
        print(f"上海 (经纬度): {weather.temperature}°C {weather.condition}")
    
    location = "116.41,39.92"  # 北京
    weather = await get_weather(location)
    if weather:
        print(f"北京 (经纬度): {weather.temperature}°C {weather.condition}")

    print()
    print("=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_location_ways())
