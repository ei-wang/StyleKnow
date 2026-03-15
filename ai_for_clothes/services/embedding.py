"""
统一的 Embedding 服务（阿里云 DashScope）：

- 文本 embedding：使用 OpenAI 客户端调用 text-embedding-v3/v4
- 多模态 embedding：使用 dashscope SDK 调用 tongyi-embedding-vision-plus

说明：
- 文本向量用于检索时把关键词转成向量
- 图片向量用于入库时把衣物图片转成向量（直接多模态 embedding）

使用方式：
- embed_text("测试文本")  # 文本向量化
- embed_image(image_bytes)  # 图片向量化（多模态）
- embed_image_from_url(image_url)  # URL 图片向量化
"""

from __future__ import annotations

import base64
import os
from typing import List, Optional, Union

from openai import OpenAI
from http import HTTPStatus

from storage.config_store import load_config


def _get_openai_client() -> OpenAI:
    """获取配置好的 OpenAI 客户端"""
    config = load_config()
    if not config.api_key:
        raise ValueError("缺少 API Key，请在配置中设置")
    
    # 兼容处理 base_url
    base_url = (config.api_base or "").rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1"
    
    return OpenAI(
        api_key=config.api_key,
        base_url=base_url
    )


# ========== 文本 Embedding ==========
def embed_text(
    text: str,
    model: Optional[str] = None,
    dimensions: Optional[int] = None
) -> List[float]:
    """
    使用 OpenAI 客户端将文本向量化。

    Args:
        text: 输入文本
        model: embedding 模型名称（如不指定则使用配置中的默认模型）
               推荐: text-embedding-v3, text-embedding-v4
        dimensions: 向量维度（部分模型支持）

    Returns:
        向量列表
    """
    config = load_config()
    client = _get_openai_client()
    
    embedding_model = model or config.embedding_model

    try:
        kwargs = {
            "model": embedding_model,
            "input": text
        }
        if dimensions:
            kwargs["dimensions"] = dimensions
        
        response = client.embeddings.create(**kwargs)
        return response.data[0].embedding

    except Exception as e:
        raise RuntimeError(f"文本向量化失败: {e}") from e


def embed_texts(
    texts: List[str],
    model: Optional[str] = None,
    dimensions: Optional[int] = None
) -> List[List[float]]:
    """
    批量文本向量化。

    Args:
        texts: 输入文本列表
        model: embedding 模型名称
        dimensions: 向量维度

    Returns:
        向量列表
    """
    config = load_config()
    client = _get_openai_client()
    
    embedding_model = model or config.embedding_model

    try:
        kwargs = {
            "model": embedding_model,
            "input": texts
        }
        if dimensions:
            kwargs["dimensions"] = dimensions
        
        response = client.embeddings.create(**kwargs)
        # 按输入顺序返回 embedding
        return [item.embedding for item in response.data]

    except Exception as e:
        raise RuntimeError(f"批量文本向量化失败: {e}") from e


# ========== 多模态 Embedding (使用 dashscope 原生 SDK) ==========
def embed_image(
    image_input: Union[bytes, str],
    model: Optional[str] = None
) -> List[float]:
    """
    使用多模态 Embedding 模型将图片向量化。

    支持的图片输入：
    - bytes: 图片字节数据
    - str: base64 编码的图片字符串（以 "data:..." 开头）或 URL

    Args:
        image_input: 图片数据（字节或 base64 字符串或 URL）
        model: 多模态 embedding 模型名称
               推荐: tongyi-embedding-vision-plus, multimodal-embedding-v1

    Returns:
        向量列表
    """
    import dashscope
    from storage.config_store import load_config
    
    config = load_config()
    
    # 使用配置中的视觉 embedding 模型
    multimodal_model = model or config.vision_embedding_model
    
    # 设置 API Key
    dashscope.api_key = config.api_key
    
    # 处理图片输入
    if isinstance(image_input, bytes):
        # bytes: 转为 base64
        image_b64 = base64.b64encode(image_input).decode("utf-8")
        input_data = [{'image': f'data:image/png;base64,{image_b64}'}]
    elif isinstance(image_input, str):
        if image_input.startswith("data:"):
            # base64 字符串
            image_b64 = image_input.split(",", 1)[1]
            input_data = [{'image': f'data:image/png;base64,{image_b64}'}]
        elif image_input.startswith("http"):
            # URL
            input_data = [{'image': image_input}]
        else:
            # 纯 base64
            input_data = [{'image': f'data:image/png;base64,{image_input}'}]
    else:
        raise ValueError("image_input 必须是 bytes、base64 字符串或 URL")

    try:
        resp = dashscope.MultiModalEmbedding.call(
            model=multimodal_model,
            input=input_data
        )
        
        if resp.status_code == HTTPStatus.OK:
            # 提取 embedding 向量
            embedding = resp.output['embeddings'][0]['embedding']
            return embedding
        else:
            raise RuntimeError(f"图片向量化失败: {resp.code} - {resp.message}")

    except Exception as e:
        raise RuntimeError(f"图片向量化失败: {e}") from e


def embed_images(
    image_inputs: List[Union[bytes, str]],
    model: Optional[str] = None
) -> List[List[float]]:
    """
    批量图片向量化（多模态 Embedding）。

    Args:
        image_inputs: 图片数据列表（字节或 base64 字符串或 URL）
        model: 多模态 embedding 模型名称

    Returns:
        向量列表
    """
    import dashscope
    from storage.config_store import load_config
    
    config = load_config()
    
    # 使用配置中的视觉 embedding 模型
    multimodal_model = model or config.vision_embedding_model
    
    # 设置 API Key
    dashscope.api_key = config.api_key
    
    # 处理所有图片输入
    input_data = []
    for image_input in image_inputs:
        if isinstance(image_input, bytes):
            image_b64 = base64.b64encode(image_input).decode("utf-8")
            input_data.append({'image': f'data:image/png;base64,{image_b64}'})
        elif isinstance(image_input, str):
            if image_input.startswith("data:"):
                image_b64 = image_input.split(",", 1)[1]
                input_data.append({'image': f'data:image/png;base64,{image_b64}'})
            elif image_input.startswith("http"):
                input_data.append({'image': image_input})
            else:
                input_data.append({'image': f'data:image/png;base64,{image_input}'})
        else:
            raise ValueError("image_input 必须是 bytes、base64 字符串或 URL")

    try:
        resp = dashscope.MultiModalEmbedding.call(
            model=multimodal_model,
            input=input_data
        )
        
        if resp.status_code == HTTPStatus.OK:
            # 提取所有 embedding 向量
            embeddings = [item['embedding'] for item in resp.output['embeddings']]
            return embeddings
        else:
            raise RuntimeError(f"批量图片向量化失败: {resp.code} - {resp.message}")

    except Exception as e:
        raise RuntimeError(f"批量图片向量化失败: {e}") from e


def embed_image_from_url(
    image_url: str,
    model: Optional[str] = None
) -> List[float]:
    """
    从 URL 下载图片并向量化（多模态 Embedding）。

    Args:
        image_url: 图片 URL
        model: 多模态 embedding 模型名称

    Returns:
        向量列表
    """
    # 直接使用 embed_image，传入 URL（更高效）
    return embed_image(image_url, model=model)


# ========== 向量相似度计算 ==========
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    计算两个向量的余弦相似度。

    Args:
        vec1: 向量1
        vec2: 向量2

    Returns:
        余弦相似度（-1 到 1）
    """
    if not vec1 or not vec2:
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def search_by_vector(
    query_vector: List[float],
    item_vectors: List[List[float]],
    top_k: int = 5
) -> List[tuple]:
    """
    向量检索：计算查询向量与所有候选向量的相似度，返回 top-k。

    Args:
        query_vector: 查询向量
        item_vectors: 候选向量列表
        top_k: 返回数量

    Returns:
        [(index, similarity), ...] 按相似度降序
    """
    similarities = []
    for i, item_vec in enumerate(item_vectors):
        sim = cosine_similarity(query_vector, item_vec)
        similarities.append((i, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]
