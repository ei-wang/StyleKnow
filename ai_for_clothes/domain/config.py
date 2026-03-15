"""
API 配置模型
"""
from pydantic import BaseModel
from typing import Optional, List, Literal


class LLMConfig(BaseModel):
    """LLM API 配置"""
    api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str = "sk-cdcf6c42b89d4d90a465f8098df68113"
    model: str = "qwen-plus"
    VLM: str = "qwen3.5-flash"
    # 文本 Embedding 模型（通过 OpenAI 客户端调用）
    # 推荐: text-embedding-v3 (1024维), text-embedding-v4
    embedding_model: str = "tongyi-embedding-vision-flash"
    
    # 多模态 Embedding 模型（使用 dashscope 原生 SDK）
    # 推荐: tongyi-embedding-vision-plus (支持 dashscope SDK)
    vision_embedding_model: str = "tongyi-embedding-vision-plus"
    # remove.bg 配置
    removebg_api_key: str = "Eg1nnFvx24FaQevcQXfGBewW"
    bg_removal_method: Literal["local", "removebg"] = "removebg"  # 本地 rembg 或 remove.bg API
    # 和风天气 API 配置
    qweather_api_key: str = "0d69e80c540e43c8bcff72ceaef66c2f"
    #qweather_api_host: str = "devapi.qweather.com"  # 免费版: devapi.qweather.com | 付费版: api.qweather.com
    qweather_api_host: str = "api.qweather.com"  # 免费版: devapi.qweather.com | 付费版: api.qweather.com
    xhs_api_url: str = "https://api.302.ai/tools/xiaohongshu/app/search_notes"
    xhs_api_key: str = "sk-VEZRFPYodaohpXo1qp6GZI6AOmfrz7hiI4tTzaQKmDnGyzIs"
    
    
class LLMConfigUpdate(BaseModel):
    """更新 LLM 配置的请求体"""
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    embedding_model: Optional[str] = None
    vision_embedding_model: Optional[str] = None
    removebg_api_key: Optional[str] = None
    bg_removal_method: Optional[Literal["local", "removebg"]] = None
    qweather_api_key: Optional[str] = None
    qweather_api_host: Optional[str] = None


class AvailableModel(BaseModel):
    """可用模型"""
    id: str
    name: str
    

class ModelListResponse(BaseModel):
    """模型列表响应"""
    models: List[AvailableModel]
