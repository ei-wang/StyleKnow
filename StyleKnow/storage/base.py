# -*- coding: utf-8 -*-
"""
衣柜数据库抽象基类

定义所有存储后端必须实现的接口，确保：
- 支持多种后端（内存/SQLite/Milvus/Redis）
- 统一的 API 设计
- 便于后续迁移到生产级存储
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any


class BaseWardrobeDB(ABC):
    """
    衣柜数据库抽象基类
    
    定义衣物存储和检索的标准接口，子类需实现以下方法：
    - add_item: 添加衣物
    - search_hybrid: 混合检索
    - update_user_preference: 更新用户偏好
    - get_user_preference: 获取用户偏好
    """
    
    @abstractmethod
    def add_item(self, item_data: Dict[str, Any]) -> str:
        """
        添加衣物到数据库
        
        Args:
            item_data: 衣物数据字典，格式：
            {
                "basic_info": {"name": str, "category": str, "image_url": str},
                "semantic_tags": List[str],
                "dynamic_metadata": Dict,
                "vector_embedding": List[float]
            }
        
        Returns:
            衣物 ID
        """
        pass
    
    @abstractmethod
    def get_item(self, item_id: str) -> Optional[Dict]:
        """
        根据 ID 获取衣物
        
        Args:
            item_id: 衣物 ID
        
        Returns:
            衣物数据字典，不存在返回 None
        """
        pass
    
    @abstractmethod
    def update_item(self, item_id: str, updates: Dict) -> bool:
        """
        更新衣物信息
        
        Args:
            item_id: 衣物 ID
            updates: 更新字段字典
        
        Returns:
            是否更新成功
        """
        pass
    
    @abstractmethod
    def delete_item(self, item_id: str) -> bool:
        """
        删除衣物
        
        Args:
            item_id: 衣物 ID
        
        Returns:
            是否删除成功
        """
        pass
    
    @abstractmethod
    def list_items(
        self,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        列出衣物
        
        Args:
            category: 按品类筛选
            tags: 按标签筛选（任一匹配）
            limit: 返回数量限制
        
        Returns:
            衣物列表
        """
        pass
    
    @abstractmethod
    def search_hybrid(
        self,
        query_text: str,
        top_k: int = 8,
        category: Optional[str] = None,
        keywords: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        混合检索：结合关键词的稀疏检索与向量的稠密检索
        
        Args:
            query_text: 检索文本
            top_k: 返回数量
            category: 可选，按品类筛选
            keywords: 可选，关键词列表
        
        Returns:
            检索结果列表，每项包含 item_id, similarity, item
        """
        pass
    
    @abstractmethod
    def search_by_scene(
        self,
        scene: str,
        top_k: int = 8,
        category: Optional[str] = None
    ) -> List[Dict]:
        """
        基于场景偏好检索
        
        Args:
            scene: 场景名称 (commute/vacation/casual/sports)
            top_k: 返回数量
            category: 可选，按品类筛选
        
        Returns:
            相似衣物列表
        """
        pass
    
    @abstractmethod
    def update_user_preference(self, user_id: str, prefs: Dict[str, Any]) -> bool:
        """
        更新用户偏好
        
        Args:
            user_id: 用户 ID
            prefs: 偏好字典
        
        Returns:
            是否更新成功
        """
        pass
    
    @abstractmethod
    def get_user_preference(self, user_id: str) -> Dict[str, Any]:
        """
        获取用户偏好
        
        Args:
            user_id: 用户 ID
        
        Returns:
            用户偏好字典
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计字典
        """
        pass
