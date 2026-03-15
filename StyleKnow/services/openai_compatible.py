"""
OpenAI 兼容 API 服务
支持任何 OpenAI 风格的 API 接口
"""
import base64
import json
import re
from typing import List, Optional
from openai import AsyncOpenAI
from storage.config_store import load_config
from domain.clothes import CLOTHES_SEMANTIC_PROMPT
from domain.clothes import ClothesSemantics


def _get_openai_client() -> AsyncOpenAI:
    """获取配置好的 AsyncOpenAI 客户端"""
    config = load_config()
    if not config.api_key:
        raise ValueError("缺少 API Key，请在配置中设置")
    
    # 兼容处理 base_url
    base_url = (config.api_base or "").rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1"
    
    return AsyncOpenAI(
        api_key=config.api_key,
        base_url=base_url
    )


async def fetch_available_models() -> List[dict]:
    """
    获取可用模型列表
    """
    config = load_config()
    
    if not config.api_key:
        return []
    
    # 确保 api_base 格式正确
    api_base = config.api_base.rstrip("/")
    if not api_base.endswith("/v1"):
        api_base = api_base + "/v1"
    
    try:
        client = _get_openai_client()
        models = await client.models.list()
        
        # 过滤出支持视觉的模型
        return [
            {"id": m.id, "name": m.id}
            for m in models.data
        ]
    except Exception as e:
        print(f"获取模型列表异常: {e}")
        raise Exception(f"连接异常: {str(e)}")


def extract_json_from_response(text: str) -> dict:
    """
    从响应中提取 JSON
    处理可能的 markdown 代码块包装
    """
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # 尝试提取 markdown 代码块中的 JSON
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # 尝试找到 { } 包围的内容
    brace_match = re.search(r'\{[\s\S]*\}', text)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass
    
    raise ValueError(f"无法从响应中提取 JSON: {text}")


async def analyze_clothes_openai(image_bytes: bytes) -> ClothesSemantics:
    """
    使用 OpenAI 兼容 API 分析衣物图片
    
    Args:
        image_bytes: 图片的字节数据
        
    Returns:
        ClothesSemantics: 衣物语义信息
    """
    config = load_config()
    
    if not config.api_key:
        raise ValueError("请先配置 API Key")
    
    # 获取客户端
    client = _get_openai_client()
    
    # 将图片转换为 base64
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    
    # 构建消息
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": CLOTHES_SEMANTIC_PROMPT
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}"
                    }
                }
            ]
        }
    ]
    
    try:
        # 使用 OpenAI 客户端调用
        response = await client.chat.completions.create(
            model=config.VLM,
            messages=messages,
            max_tokens=1000
        )
        
        # 提取响应内容
        content = response.choices[0].message.content
        
        # 解析 JSON
        result = extract_json_from_response(content)
        
        return ClothesSemantics(**result)
        
    except Exception as e:
        raise ValueError(f"API 请求失败: {str(e)}")
