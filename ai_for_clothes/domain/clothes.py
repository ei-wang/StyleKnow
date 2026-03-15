# -*- coding: utf-8 -*-
"""
衣物数据模型

定义衣物的数据结构，与 storage/db.py 中的 InMemoryWardrobeDB 兼容
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime


class BasicInfo(BaseModel):
    """基础信息"""
    name: str                      # 衣物名称
    category: str                  # 品类 (上衣/裤子/裙子/鞋子/配饰/外套)
    image_url: str = ""            # 图片路径
    created_at: Optional[int] = None  # 创建时间戳


class ClothingCreate(BaseModel):
    """衣物创建请求"""
    name: str
    category: str
    image_url: str = ""
    semantic_tags: List[str] = []
    dynamic_metadata: Dict[str, Any] = {}
    vector_embedding: List[float] = []


class ClothingItem(BaseModel):
    """完整衣物数据"""
    id: str
    basic_info: BasicInfo
    semantic_tags: List[str] = []
    dynamic_metadata: Dict[str, Any] = {}
    vector_embedding: List[float] = []


class ClothesSemantics(BaseModel):
    """
    VLM 解析后的衣物语义信息
    
    用于 upload.py 中的图片分析流程
    """
    # 品类信息
    category: str = "未知"           # 大类：上衣、裤子、裙子、鞋子、配饰、外套
    item: str = ""                  # 具体款式
    
    # 语义标签
    style_semantics: List[str] = []   # 风格标签: 简约、复古、商务、街头等
    season_semantics: List[str] = []  # 季节标签: 春夏、秋冬、四季
    usage_semantics: List[str] = []   # 使用场景: 通勤、休闲、运动、正式
    color_semantics: List[str] = []    # 颜色标签
    
    # 描述
    description: str = ""           # 自然语言描述
    
    # 详细属性
    material: str = ""              # 材质
    pattern: str = ""               # 图案/花色
    fit: str = ""                   # 版型
    details: Dict[str, str] = {}    # 其他细节
    
    def to_storage_format(self) -> Dict:
        """
        转换为 storage/db.py 格式
        
        Returns:
            存储格式的字典
        """
        # 构建 semantic_tags
        tags = []
        tags.append(self.category)
        tags.extend(self.color_semantics)
        tags.extend(self.style_semantics)
        tags.extend(self.season_semantics)
        tags.extend(self.usage_semantics)
        
        # 去重
        tags = list(set(tags))
        
        # 构建 dynamic_metadata
        metadata = {
            "color": ", ".join(self.color_semantics) if self.color_semantics else "",
            "material": self.material,
            "style_vibe": " / ".join(self.style_semantics) if self.style_semantics else "",
            "details": self.details.copy()
        }
        
        # 如果有 item，添加到 details
        if self.item:
            metadata["details"]["item"] = self.item
        if self.pattern:
            metadata["details"]["pattern"] = self.pattern
        if self.fit:
            metadata["details"]["fit"] = self.fit
            
        # 添加解析来源
        metadata["parsed_by"] = "vlm"
        
        return {
            "basic_info": {
                "name": self.item or f"{' '.join(self.color_semantics)} {self.category}" if self.color_semantics else self.category,
                "category": self.category,
                "image_url": "",
                "created_at": int(datetime.now().timestamp())
            },
            "semantic_tags": tags,
            "dynamic_metadata": metadata,
            "vector_embedding": []  # 后续生成
        }


# ========== Prompt 模板 ==========
CLOTHES_SEMANTIC_PROMPT = """你是一个专业的衣物识别助手。请分析以下衣物图片，提取详细的语义信息。

请按以下 JSON 格式输出：
{
    "category": "上衣/裤子/裙子/鞋子/配饰/外套",
    "item": "具体款式，如T恤、衬衫、牛仔裤等",
    "style_semantics": ["风格标签列表，如简约、复古、商务"],
    "season_semantics": ["季节标签，如春夏、秋冬"],
    "usage_semantics": ["使用场景，如通勤、休闲、运动"],
    "color_semantics": ["颜色标签，如白色、黑色、蓝色"],
    "description": "一段自然语言描述",
    "material": "材质，如纯棉、羊毛、皮革",
    "pattern": "图案，如纯色、条纹、印花",
    "fit": "版型，如修身、宽松、合体",
    "details": {"其他细节": "值"}
}

请尽可能详细地识别，包括领型、袖长、裤长等细节。"""


__all__ = [
    "BasicInfo",
    "ClothingCreate", 
    "ClothingItem",
    "ClothesSemantics",
    "CLOTHES_SEMANTIC_PROMPT"
]
