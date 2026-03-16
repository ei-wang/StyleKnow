"""
天气服务 - 和风天气 API 集成
文档: https://dev.qweather.com/docs/api/weather/weather-now/
GeoAPI: https://dev.qweather.com/docs/api/geoapi/city-lookup/
"""
import httpx
import os
from typing import Optional, List
from pydantic import BaseModel


class CityInfo(BaseModel):
    """城市信息"""
    name: str  # 城市名称
    id: str  # LocationID
    adm1: str  # 省份
    adm2: str  # 市
    country: str  # 国家
    lat: str  # 纬度
    lon: str  # 经度


class WeatherNow(BaseModel):
    """实时天气数据"""
    obsTime: str  # 数据观测时间
    temp: str  # 温度，默认单位：摄氏度
    feelsLike: str  # 体感温度
    icon: str  # 天气状况图标代码
    text: str  # 天气状况的文字描述
    wind360: str  # 风向360角度
    windDir: str  # 风向
    windScale: str  # 风力等级
    windSpeed: str  # 风速，公里/小时
    humidity: str  # 相对湿度，百分比数值
    precip: str  # 过去1小时降水量，默认单位：毫米
    pressure: str  # 大气压强，默认单位：百帕
    vis: str  # 能见度，默认单位：公里
    cloud: Optional[str] = None  # 云量，百分比数值
    dew: Optional[str] = None  # 露点温度


class WeatherResponse(BaseModel):
    """和风天气 API 响应"""
    code: str  # 状态码
    updateTime: str  # API最近更新时间
    fxLink: str  # 响应式页面链接
    now: WeatherNow  # 实时天气数据


class WeatherInfo(BaseModel):
    """简化的天气信息（用于应用）"""
    temperature: float  # 温度
    feelsLike: float  # 体感温度
    condition: str  # 天气状况文字
    icon: str  # 天气图标代码
    humidity: float  # 湿度
    windDir: str  # 风向
    windScale: str  # 风力等级
    location: str  # 位置
    obsTime: str  # 观测时间


class DailyForecast(BaseModel):
    """每日天气预报"""
    fxDate: str  # 预报日期
    tempMax: str  # 最高温度
    tempMin: str  # 最低温度
    textDay: str  # 白天天气文字
    textNight: str  # 夜间天气文字
    iconDay: str  # 白天天气图标
    iconNight: str  # 夜间天气图标
    windDirDay: str  # 白天风向
    windScaleDay: str  # 白天风力等级
    humidity: str  # 相对湿度
    precip: str  # 降水量
    uvIndex: str  # 紫外线指数


class ForecastResponse(BaseModel):
    """和风天气预报 API 响应"""
    code: str
    daily: List[DailyForecast]


# 常用城市列表（包含经纬度）- 降级方案
COMMON_CITIES = [
    # 一线城市
    {"name": "北京", "adm1": "北京市", "country": "中国", "id": "101010100", "lat": "39.90", "lon": "116.40", "keywords": ["beijing", "北京", "bj", "capital"]},
    {"name": "上海", "adm1": "上海市", "country": "中国", "id": "101020100", "lat": "31.23", "lon": "121.47", "keywords": ["shanghai", "上海", "sh"]},
    {"name": "广州", "adm1": "广东省", "country": "中国", "id": "101280101", "lat": "23.12", "lon": "113.26", "keywords": ["guangzhou", "广州", "gz"]},
    {"name": "深圳", "adm1": "广东省", "country": "中国", "id": "101280601", "lat": "22.54", "lon": "114.06", "keywords": ["shenzhen", "深圳", "sz"]},
    
    # 新一线城市
    {"name": "杭州", "adm1": "浙江省", "country": "中国", "id": "101210101", "lat": "30.27", "lon": "120.15", "keywords": ["hangzhou", "杭州", "hz"]},
    {"name": "成都", "adm1": "四川省", "country": "中国", "id": "101270101", "lat": "30.67", "lon": "104.07", "keywords": ["chengdu", "成都", "cd"]},
    {"name": "重庆", "adm1": "重庆市", "country": "中国", "id": "101040100", "lat": "29.56", "lon": "106.55", "keywords": ["chongqing", "重庆", "cq"]},
    {"name": "武汉", "adm1": "湖北省", "country": "中国", "id": "101200101", "lat": "30.59", "lon": "114.30", "keywords": ["wuhan", "武汉", "wh"]},
    {"name": "西安", "adm1": "陕西省", "country": "中国", "id": "101110101", "lat": "34.34", "lon": "108.94", "keywords": ["xian", "西安", "xa"]},
    {"name": "南京", "adm1": "江苏省", "country": "中国", "id": "101190101", "lat": "32.06", "lon": "118.79", "keywords": ["nanjing", "南京", "nj"]},
    {"name": "天津", "adm1": "天津市", "country": "中国", "id": "101030100", "lat": "39.13", "lon": "117.20", "keywords": ["tianjin", "天津", "tj"]},
    {"name": "苏州", "adm1": "江苏省", "country": "中国", "id": "101190401", "lat": "31.30", "lon": "120.58", "keywords": ["suzhou", "苏州", "su"]},
    {"name": "长沙", "adm1": "湖南省", "country": "中国", "id": "101250101", "lat": "28.23", "lon": "112.94", "keywords": ["changsha", "长沙", "cs"]},
    {"name": "郑州", "adm1": "河南省", "country": "中国", "id": "101180101", "lat": "34.77", "lon": "113.62", "keywords": ["zhengzhou", "郑州", "zz"]},
    {"name": "济南", "adm1": "山东省", "country": "中国", "id": "101120101", "lat": "36.65", "lon": "117.12", "keywords": ["jinan", "济南", "jn"]},
    {"name": "青岛", "adm1": "山东省", "country": "中国", "id": "101120201", "lat": "36.07", "lon": "120.38", "keywords": ["qingdao", "青岛", "qd"]},
    {"name": "厦门", "adm1": "福建省", "country": "中国", "id": "101230201", "lat": "24.48", "lon": "118.11", "keywords": ["xiamen", "厦门", "xm"]},
    {"name": "福州", "adm1": "福建省", "country": "中国", "id": "101230101", "lat": "26.08", "lon": "119.30", "keywords": ["fuzhou", "福州", "fz"]},
    {"name": "宁波", "adm1": "浙江省", "country": "中国", "id": "101210401", "lat": "29.87", "lon": "121.55", "keywords": ["ningbo", "宁波", "nb"]},
    {"name": "无锡", "adm1": "江苏省", "country": "中国", "id": "101190201", "lat": "31.49", "lon": "120.30", "keywords": ["wuxi", "无锡", "wx"]},
    
    # 二线城市
    {"name": "大连", "adm1": "辽宁省", "country": "中国", "id": "101070201", "lat": "38.92", "lon": "121.63", "keywords": ["dalian", "大连", "dl"]},
    {"name": "沈阳", "adm1": "辽宁省", "country": "中国", "id": "101070101", "lat": "41.80", "lon": "123.43", "keywords": ["shenyang", "沈阳", "sy"]},
    {"name": "哈尔滨", "adm1": "黑龙江", "country": "中国", "id": "101050101", "lat": "45.80", "lon": "126.53", "keywords": ["haerbin", "哈尔滨", "heb"]},
    {"name": "长春", "adm1": "吉林省", "country": "中国", "id": "101060101", "lat": "43.88", "lon": "125.32", "keywords": ["changchun", "长春", "cc"]},
    {"name": "南昌", "adm1": "江西省", "country": "中国", "id": "101240101", "lat": "28.68", "lon": "115.86", "keywords": ["nanchang", "南昌", "nc"]},
    {"name": "昆明", "adm1": "云南省", "country": "中国", "id": "101290101", "lat": "25.04", "lon": "102.71", "keywords": ["kunming", "昆明", "km"]},
    {"name": "太原", "adm1": "山西省", "country": "中国", "id": "101100101", "lat": "37.87", "lon": "112.55", "keywords": ["taiyuan", "太原", "ty"]},
    {"name": "石家庄", "adm1": "河北省", "country": "中国", "id": "101090101", "lat": "38.03", "lon": "114.51", "keywords": ["shijiazhuang", "石家庄", "sjz"]},
    {"name": "合肥", "adm1": "安徽省", "country": "中国", "id": "101220101", "lat": "31.82", "lon": "117.23", "keywords": ["hefei", "合肥", "hf"]},
    {"name": "南宁", "adm1": "广西", "country": "中国", "id": "101300101", "lat": "22.82", "lon": "108.37", "keywords": ["nanning", "南宁", "nn"]},
    {"name": "贵阳", "adm1": "贵州省", "country": "中国", "id": "101260101", "lat": "26.65", "lon": "106.63", "keywords": ["guiyang", "贵阳", "gy"]},
    {"name": "兰州", "adm1": "甘肃省", "country": "中国", "id": "101160101", "lat": "36.06", "lon": "103.83", "keywords": ["lanzhou", "兰州", "lz"]},
    {"name": "乌鲁木齐", "adm1": "新疆", "country": "中国", "id": "101130101", "lat": "43.83", "lon": "87.62", "keywords": ["wulumuqi", "乌鲁木齐", "wlmq", "urumqi"]},
    {"name": "呼和浩特", "adm1": "内蒙古", "country": "中国", "id": "101080101", "lat": "40.84", "lon": "111.73", "keywords": ["hohhot", "呼和浩特", "huhehaote"]},
    {"name": "银川", "adm1": "宁夏", "country": "中国", "id": "101170101", "lat": "38.47", "lon": "106.27", "keywords": ["yinchuan", "银川", "yc"]},
    {"name": "西宁", "adm1": "青海", "country": "中国", "id": "101150101", "lat": "36.62", "lon": "101.78", "keywords": ["xining", "西宁", "xn"]},
    {"name": "拉萨", "adm1": "西藏", "country": "中国", "id": "101140101", "lat": "29.65", "lon": "91.10", "keywords": ["lasa", "拉萨", "ls", "lhasa"]},
    {"name": "海口", "adm1": "海南省", "country": "中国", "id": "101310101", "lat": "20.04", "lon": "110.20", "keywords": ["haikou", "海口", "hk"]},
    {"name": "三亚", "adm1": "海南省", "country": "中国", "id": "101280201", "lat": "18.25", "lon": "109.51", "keywords": ["sanya", "三亚", "sy"]},
    
    # 热门旅游城市
    {"name": "珠海", "adm1": "广东省", "country": "中国", "id": "101280701", "lat": "22.28", "lon": "113.58", "keywords": ["zhuhai", "珠海", "zh"]},
    {"name": "东莞", "adm1": "广东省", "country": "中国", "id": "101281601", "lat": "23.05", "lon": "113.76", "keywords": ["dongguan", "东莞", "dg"]},
    {"name": "佛山", "adm1": "广东省", "country": "中国", "id": "101280800", "lat": "23.02", "lon": "113.12", "keywords": ["foshan", "佛山", "fs"]},
    {"name": "中山", "adm1": "广东省", "country": "中国", "id": "101281701", "lat": "22.52", "lon": "113.39", "keywords": ["zhongshan", "中山", "zs"]},
    {"name": "温州", "adm1": "浙江省", "country": "中国", "id": "101210501", "lat": "28.00", "lon": "120.69", "keywords": ["wenzhou", "温州", "wz"]},
    {"name": "泉州", "adm1": "福建省", "country": "中国", "id": "101230501", "lat": "24.87", "lon": "118.67", "keywords": ["quanzhou", "泉州", "qz"]},
    {"name": "烟台", "adm1": "山东省", "country": "中国", "id": "101120501", "lat": "37.46", "lon": "121.45", "keywords": ["yantai", "烟台", "yt"]},
    {"name": "威海", "adm1": "山东省", "country": "中国", "id": "101120801", "lat": "37.51", "lon": "122.12", "keywords": ["weihai", "威海", "wh"]},
    {"name": "大理", "adm1": "云南省", "country": "中国", "id": "101291201", "lat": "25.60", "lon": "100.27", "keywords": ["dali", "大理", "dl"]},
    {"name": "丽江", "adm1": "云南省", "country": "中国", "id": "101291401", "lat": "26.87", "lon": "100.23", "keywords": ["lijiang", "丽江", "lj"]},
    {"name": "桂林", "adm1": "广西", "country": "中国", "id": "101300501", "lat": "25.27", "lon": "110.18", "keywords": ["guilin", "桂林", "gl"]},
    {"name": "张家界", "adm1": "湖南省", "country": "中国", "id": "101251101", "lat": "29.12", "lon": "110.48", "keywords": ["zhangjiajie", "张家界", "zjj"]},
    
    # 港澳台
    {"name": "香港", "adm1": "香港", "country": "中国", "id": "101340201", "lat": "22.28", "lon": "114.16", "keywords": ["hongkong", "香港", "hk", "hong kong"]},
    {"name": "澳门", "adm1": "澳门", "country": "中国", "id": "101340301", "lat": "22.20", "lon": "113.55", "keywords": ["macau", "澳门", "am", "macao"]},
    {"name": "台北", "adm1": "台湾", "country": "中国", "id": "101340101", "lat": "25.03", "lon": "121.52", "keywords": ["taipei", "台北", "tb", "taipei"]},
    {"name": "高雄", "adm1": "台湾", "country": "中国", "id": "101340401", "lat": "22.63", "lon": "120.30", "keywords": ["kaohsiung", "高雄", "kx"]},
]


class IPLocationInfo(BaseModel):
    """IP 定位信息"""
    status: str
    country: str
    city: str
    lat: float
    lon: float


async def get_location_from_ip() -> Optional[IPLocationInfo]:
    """
    使用 ip-api.com 获取当前 IP 的位置信息（免费）

    Returns:
        IPLocationInfo 或 None（失败时）
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "http://ip-api.com/json/?fields=status,country,city,lat,lon"
            )
            data = resp.json()
            if data.get("status") == "success":
                return IPLocationInfo(
                    status=data.get("status"),
                    country=data.get("country", ""),
                    city=data.get("city", ""),
                    lat=data.get("lat", 0),
                    lon=data.get("lon", 0)
                )
    except Exception as e:
        print(f"⚠️  IP 定位失败: {e}")
    return None


async def search_city(query: str, limit: int = 10) -> List[CityInfo]:
    """
    搜索城市（使用外部地理编码 API）

    优先使用免费地理编码服务（Nominatim/OpenStreetMap）获取城市坐标，
    然后使用和风天气 API 获取对应的 LocationID。

    Args:
        query: 城市名称关键词（支持中文、英文）
        limit: 返回结果数量限制

    Returns:
        城市信息列表
    """
    # 首先尝试使用 Nominatim API 获取坐标
    coords = await _geocode_city(query)

    if coords:
        # 使用坐标查询和风天气城市 ID
        city_id = await _get_city_id_from_coords(coords["lat"], coords["lon"])
        if city_id:
            return [CityInfo(
                name=coords.get("display_name", query).split(",")[0],
                id=city_id,
                adm1=coords.get("state", ""),
                adm2=coords.get("city", ""),
                country=coords.get("country", ""),
                lat=coords["lat"],
                lon=coords["lon"]
            )]

    # 如果外部 API 失败，降级到预定义列表
    return _search_city_fallback(query, limit)


async def _geocode_city(query: str) -> Optional[dict]:
    """
    使用和风天气城市搜索 API 获取城市信息
    https://dev.qweather.com/docs/api/geoapi/city-lookup/
    """
    try:
        from storage.config_store import load_config
        config = load_config()
        api_key = config.qweather_api_key
        api_host = config.qweather_api_host
    except Exception:
        api_key = os.getenv("QWEATHER_API_KEY")
        api_host = os.getenv("QWEATHER_API_HOST", "devapi.qweather.com")

    if not api_key or api_key == "your_qweather_api_key_here":
        return None

    try:
        url = f"https://geo.qweather.com/v2/cities/lookup"
        params = {
            "location": query,
            "key": api_key,
            "number": 1
        }

        async with httpx.AsyncClient(verify=False, trust_env=True) as client:
            response = await client.get(url, params=params, timeout=10.0)
            response.raise_for_status()
            data = response.json()

            if data.get("code") == "200" and data.get("location"):
                loc = data["location"][0]
                return {
                    "display_name": f"{loc.get('name', query)}, {loc.get('adm1', '')}",
                    "lat": loc.get("lat", "0"),
                    "lon": loc.get("lon", "0"),
                    "country": loc.get("country", ""),
                    "state": loc.get("adm1", ""),
                    "city": loc.get("name", "")
                }
    except Exception as e:
        print(f"⚠️ 和风城市搜索失败: {e}")

    return None


async def _get_city_id_from_coords(lat: str, lon: str) -> Optional[str]:
    """
    使用和风天气 API 根据坐标获取城市 ID
    """
    try:
        from storage.config_store import load_config
        config = load_config()
        api_key = config.qweather_api_key
        api_host = config.qweather_api_host
    except Exception:
        api_key = os.getenv("QWEATHER_API_KEY")
        api_host = os.getenv("QWEATHER_API_HOST", "devapi.qweather.com")

    if not api_key or api_key == "your_qweather_api_key_here":
        return None

    url = f"https://{api_host}/v2/cities/lookup"
    params = {
        "location": f"{lon},{lat}",
        "key": api_key
    }

    try:
        async with httpx.AsyncClient(verify=False, trust_env=True) as client:
            response = await client.get(url, params=params, timeout=10.0)
            response.raise_for_status()
            data = response.json()

            if data.get("code") == "200" and data.get("location"):
                return data["location"][0].get("id")
    except Exception as e:
        print(f"⚠️ 获取城市ID失败: {e}")

    return None


def _search_city_fallback(query: str, limit: int = 10) -> List[CityInfo]:
    """降级方案：使用预定义城市列表"""
    query_lower = query.lower()
    matched_cities = []

    for city_data in COMMON_CITIES:
        if any(query_lower in keyword.lower() for keyword in city_data["keywords"]):
            matched_cities.append(CityInfo(
                name=city_data["name"],
                id=city_data["id"],
                adm1=city_data["adm1"],
                adm2=city_data["adm1"],
                country=city_data["country"],
                lat=city_data.get("lat", "0"),
                lon=city_data.get("lon", "0")
            ))
            if len(matched_cities) >= limit:
                break

    return matched_cities


async def get_qweather_now(location: str) -> Optional[WeatherResponse]:
    """
    调用和风天气 API 获取实时天气
    
    Args:
        location: LocationID 或 经纬度坐标(逗号分隔，如 "116.41,39.92")
        
    Returns:
        WeatherResponse 或 None（失败时）
    
    示例:
        - location="101010100" (北京的LocationID)
        - location="116.41,39.92" (经纬度坐标)
    """
    # 优先从配置系统读取 API Key
    try:
        from storage.config_store import load_config
        config = load_config()
        api_key = config.qweather_api_key
        api_host = config.qweather_api_host
    except Exception:
        # 回退到环境变量
        api_key = os.getenv("QWEATHER_API_KEY")
        api_host = os.getenv("QWEATHER_API_HOST", "devapi.qweather.com")
    
    if not api_key or api_key == "your_qweather_api_key_here":
        print("⚠️  和风天气 API Key 未配置，请在前端设置界面或 .env 文件中配置")
        return None
    
    url = f"https://{api_host}/v7/weather/now"
    params = {
        "location": location,
        "key": api_key
    }
    
    try:
        async with httpx.AsyncClient(verify=False, trust_env=True) as client:
            response = await client.get(url, params=params, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            
            if data.get("code") != "200":
                print(f"❌ 和风天气 API 错误: code={data.get('code')}")
                return None
            
            return WeatherResponse(**data)
    
    except Exception as e:
        print(f"❌ 获取天气信息失败: {e}")
        return None


async def get_weather(location: str = "101020100") -> Optional[WeatherInfo]:
    """
    获取天气信息（简化版）
    
    Args:
        location: LocationID 或 经纬度坐标
                 默认: 101020100 (上海)
                 
    Returns:
        WeatherInfo 或 None
    """
    weather_response = await get_qweather_now(location)
    
    if not weather_response:
        # 返回模拟数据作为降级方案
        print("⚠️  使用模拟天气数据")
        return WeatherInfo(
            temperature=20.0,
            feelsLike=22.0,
            condition="晴",
            icon="100",
            humidity=60.0,
            windDir="南风",
            windScale="2",
            location=location,
            obsTime="2024-01-01T12:00+08:00"
        )
    
    now = weather_response.now
    return WeatherInfo(
        temperature=float(now.temp),
        feelsLike=float(now.feelsLike),
        condition=now.text,
        icon=now.icon,
        humidity=float(now.humidity),
        windDir=now.windDir,
        windScale=now.windScale,
        location=location,
            obsTime=now.obsTime
    )


async def get_weather_forecast(
    location: str,
    days: int = 7
) -> Optional[List[DailyForecast]]:
    """
    获取天气预报（未来几天）

    Args:
        location: LocationID 或 经纬度坐标
        days: 预报天数，支持 3/7/10/15/30 天

    Returns:
        预报列表或 None
    """
    try:
        from storage.config_store import load_config
        config = load_config()
        api_key = config.qweather_api_key
        api_host = config.qweather_api_host
    except Exception:
        api_key = os.getenv("QWEATHER_API_KEY")
        api_host = os.getenv("QWEATHER_API_HOST", "devapi.qweather.com")

    if not api_key or api_key == "your_qweather_api_key_here":
        return None

    # 限制天数
    days = min(max(days, 3), 30)

    url = f"https://{api_host}/v7/weather/{days}d"
    params = {
        "location": location,
        "key": api_key
    }

    try:
        async with httpx.AsyncClient(verify=False, trust_env=True) as client:
            response = await client.get(url, params=params, timeout=10.0)
            response.raise_for_status()
            data = response.json()

            if data.get("code") != "200":
                print(f"❌ 天气预报 API 错误: code={data.get('code')}")
                return None

            forecast_data = ForecastResponse(**data)
            return forecast_data.daily

    except Exception as e:
        print(f"❌ 获取天气预报失败: {e}")
        return None


def get_season_from_weather(weather: WeatherInfo) -> list[str]:
    """
    根据天气推断适合的季节标签
    
    Args:
        weather: 天气信息
        
    Returns:
        季节标签列表
    """
    temp = weather.temperature
    
    if temp < 10:
        return ["冬"]
    elif temp < 20:
        return ["春", "秋"]
    else:
        return ["夏"]


def get_clothing_suggestion(weather: WeatherInfo) -> str:
    """
    根据天气推荐穿搭建议
    
    Args:
        weather: 天气信息
        
    Returns:
        穿搭建议文字
    """
    temp = weather.temperature
    feels_like = weather.feelsLike
    condition = weather.condition
    
    # 基于温度的建议
    if feels_like < 0:
        suggestion = "🧥 建议穿厚羽绒服、棉衣等保暖衣物"
    elif feels_like < 10:
        suggestion = "🧥 建议穿风衣、大衣、夹克等外套"
    elif feels_like < 20:
        suggestion = "👔 建议穿薄外套、长袖衬衫、卫衣"
    elif feels_like < 28:
        suggestion = "👕 建议穿短袖、薄长袖等轻便衣物"
    else:
        suggestion = "👕 建议穿短袖、短裤等夏季清凉衣物"
    
    # 根据天气状况补充建议
    if "雨" in condition:
        suggestion += "，记得带伞☂️"
    elif "雪" in condition:
        suggestion += "，注意防滑保暖❄️"
    elif "晴" in condition and feels_like > 25:
        suggestion += "，注意防晒☀️"
    
    return suggestion
