# -*- coding: utf-8 -*-
"""
FastAPI 路由入口

提供 REST API 接口访问 Agent
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from pydantic import BaseModel

# 导入 Agent
from api.agent import FashionAgent, get_agent

# 创建路由
router = APIRouter()


# ========== 请求/响应模型 ==========
class RecommendationRequest(BaseModel):
    """穿搭推荐请求"""
    query: str
    location: Optional[str] = None
    include_weather: bool = False


class RecommendationResponse(BaseModel):
    """穿搭推荐响应"""
    success: bool
    query: str
    result: dict
    weather: Optional[dict] = None


class WeatherRequest(BaseModel):
    """天气查询请求"""
    location: str = "101020100"
    city_name: Optional[str] = None


class WardrobeSearchRequest(BaseModel):
    """衣柜检索请求"""
    scene: str = "commute"
    top_k: int = 8
    category: Optional[str] = None
    tags: Optional[list] = None


# ========== 路由定义 ==========
@router.get("/")
async def root():
    """根路径"""
    return {
        "name": "AI Fashion Assistant API",
        "version": "1.0.0",
        "description": "AI 穿搭助手 API"
    }


@router.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy"}


@router.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    """
    穿搭推荐接口
    
    根据用户查询推荐穿搭搭配
    """
    try:
        agent = get_agent()
        
        # 运行推荐
        result = agent.run(request.query)
        
        # 如果需要天气信息
        weather_info = None
        if request.include_weather and request.location:
            from tools.weather import get_weather_with_suggestion
            weather_info = get_weather_with_suggestion(request.location)
        
        return RecommendationResponse(
            success=True,
            query=request.query,
            result=result,
            weather=weather_info
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tools")
async def list_tools():
    """
    列出所有可用工具
    """
    agent = get_agent()
    tools = []
    
    for tool in agent.get_tools():
        tools.append({
            "name": tool.name,
            "description": tool.description
        })
    
    return {"tools": tools}


@router.post("/wardrobe/search")
async def search_wardrobe(request: WardrobeSearchRequest):
    """
    衣柜检索接口
    """
    from tools.db_search import search_wardrobe as _search_wardrobe
    
    try:
        result = _search_wardrobe(
            scene=request.scene,
            top_k=request.top_k,
            category=request.category,
            tags=request.tags
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/weather")
async def get_weather(request: WeatherRequest):
    """
    天气查询接口
    """
    from tools.weather import get_weather_info
    
    try:
        result = get_weather_info(
            location=request.location,
            city_name=request.city_name
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/xhs/search")
async def search_xhs(
    keyword: str = Query(...),
    page: str = Query("1"),
    sort_type: str = Query("popularity_descending")
):
    """
    小红书搜索接口
    """
    from tools.xhs_search import search_xhs_notes
    
    try:
        result = search_xhs_notes(
            keyword=keyword,
            page=page,
            sort_type=sort_type
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
