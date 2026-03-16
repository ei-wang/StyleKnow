# -*- coding: utf-8 -*-
"""
衣柜数据库层

支持多种存储后端：
- InMemoryWardrobeDB: 内存数据库
- SQLiteWardrobeDB: SQLite 持久化存储（可选）

数据模型：
- basic_info: 基础信息 (name, category, image_url, created_at)
- semantic_tags: 语义标签列表
- dynamic_metadata: 动态元数据 (颜色、材质、风格等)
- vector_embedding: 向量嵌入
"""

import uuid
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from storage.base import BaseWardrobeDB


# ========== 数据模型 ==========
class ClothingItem:
    """衣物数据模型"""
    
    def __init__(
        self,
        item_id: str,
        name: str,
        category: str,
        image_url: str = "",
        created_at: Optional[int] = None,
        semantic_tags: Optional[List[str]] = None,
        dynamic_metadata: Optional[Dict] = None,
        vector_embedding: Optional[List[float]] = None
    ):
        self.id = item_id
        self.name = name
        self.category = category
        self.image_url = image_url
        self.created_at = created_at or int(datetime.now().timestamp())
        self.semantic_tags = semantic_tags or []
        self.dynamic_metadata = dynamic_metadata or {}
        self.vector_embedding = vector_embedding
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "id": self.id,
            "basic_info": {
                "name": self.name,
                "category": self.category,
                "image_url": self.image_url,
                "created_at": self.created_at
            },
            "semantic_tags": self.semantic_tags,
            "dynamic_metadata": self.dynamic_metadata,
            "vector_embedding": self.vector_embedding
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ClothingItem":
        """从字典创建"""
        basic = data.get("basic_info", {})
        return cls(
            item_id=data.get("id", ""),
            name=basic.get("name", ""),
            category=basic.get("category", ""),
            image_url=basic.get("image_url", ""),
            created_at=basic.get("created_at"),
            semantic_tags=data.get("semantic_tags", []),
            dynamic_metadata=data.get("dynamic_metadata", {}),
            vector_embedding=data.get("vector_embedding", [])
        )


# ========== 内存数据库实现 ==========
class InMemoryWardrobeDB(BaseWardrobeDB):
    """内存版衣柜数据库，继承自 BaseWardrobeDB"""

    def __init__(self):
        self.items: Dict[str, Dict] = {}  # item_id -> item data
        self.user_preferences: Dict[str, Dict] = {}  # user_id -> preferences
        self.user_preference_vectors: Dict[str, Dict[str, np.ndarray]] = {}  # user_id -> {pref_type -> vector}
        print("[DB] InMemoryWardrobeDB 初始化完成")
    
    # ========== 基础 CRUD ==========
    def add_item(self, item_data: Dict) -> str:
        """
        添加衣物到数据库
        
        Args:
            item_data: 衣物数据字典，格式如下：
            {
                "basic_info": {
                    "name": "美式复古做旧水洗皮夹克",
                    "category": "外套",
                    "image_url": "/uploads/images/item_xxx.png",
                    "created_at": 1710288000
                },
                "semantic_tags": ["外套", "棕色", "皮质", "美式复古", "做旧", "短款", "秋冬", "防风"],
                "dynamic_metadata": {
                    "color": "深棕色",
                    "material": "PU皮/真皮",
                    "style_vibe": "美式复古 / 机车风",
                    "details": {...}
                },
                "vector_embedding": [0.01235, -0.04582, ...]
            }
        
        Returns:
            衣物 ID
        """
        # 生成唯一 ID
        item_id = item_data.get("id") or f"item_{uuid.uuid4().hex[:8]}"
        
        # 构建完整数据
        item = {
            "id": item_id,
            "basic_info": {
                "name": item_data.get("basic_info", {}).get("name", "未命名"),
                "category": item_data.get("basic_info", {}).get("category", "未知"),
                "image_url": item_data.get("basic_info", {}).get("image_url", ""),
                "created_at": item_data.get("basic_info", {}).get("created_at") or int(datetime.now().timestamp())
            },
            "semantic_tags": item_data.get("semantic_tags", []),
            "dynamic_metadata": item_data.get("dynamic_metadata", {}),
            "vector_embedding": item_data.get("vector_embedding", [])
        }
        
        self.items[item_id] = item
        print(f"[DB] 添加衣物成功: {item['basic_info']['name']} (ID: {item_id})")
        
        return item_id
    
    def add_item_from_vlm(self, vlm_payload: Dict) -> str:
        """
        从 VLM 解析结果添加衣物
        
        Args:
            vlm_payload: VLM 返回的数据
        
        Returns:
            衣物 ID
        """
        # 从 VLM payload 构建数据
        item_data = {
            "basic_info": {
                "name": vlm_payload.get("name", vlm_payload.get("category", "未命名")),
                "category": vlm_payload.get("category", "未知"),
                "image_url": vlm_payload.get("image_url", ""),
                "created_at": int(datetime.now().timestamp())
            },
            "semantic_tags": vlm_payload.get("semantic_tags", []),
            "dynamic_metadata": vlm_payload.get("dynamic_metadata", {}),
            "vector_embedding": vlm_payload.get("vector_embedding", [])
        }
        
        # 如果 VLM 返回了 tags，也加入 semantic_tags
        if "tags" in vlm_payload:
            item_data["semantic_tags"].extend(vlm_payload.get("tags", []))
        
        # 从 dynamic_metadata 提取标签
        metadata = vlm_payload.get("dynamic_metadata", {})
        if "color" in metadata:
            item_data["semantic_tags"].append(metadata["color"])
        if "style_vibe" in metadata:
            item_data["semantic_tags"].extend(metadata["style_vibe"].replace("/", " ").split())
        
        # 去重
        item_data["semantic_tags"] = list(set(item_data["semantic_tags"]))
        
        # 如果没有向量，生成随机向量
        if not item_data["vector_embedding"]:
            item_data["vector_embedding"] = self._generate_random_vector()
        
        return self.add_item(item_data)
    
    def get_item(self, item_id: str) -> Optional[Dict]:
        """根据 ID 获取衣物"""
        return self.items.get(item_id)
    
    def update_item(self, item_id: str, updates: Dict) -> bool:
        """更新衣物信息"""
        if item_id not in self.items:
            return False
        
        item = self.items[item_id]
        
        # 更新 basic_info
        if "basic_info" in updates:
            item["basic_info"].update(updates["basic_info"])
        
        # 更新 semantic_tags
        if "semantic_tags" in updates:
            item["semantic_tags"] = updates["semantic_tags"]
        
        # 更新 dynamic_metadata
        if "dynamic_metadata" in updates:
            item["dynamic_metadata"].update(updates["dynamic_metadata"])
        
        # 更新向量
        if "vector_embedding" in updates:
            item["vector_embedding"] = updates["vector_embedding"]
        
        return True
    
    def delete_item(self, item_id: str) -> bool:
        """删除衣物"""
        if item_id in self.items:
            del self.items[item_id]
            return True
        return False
    
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
        results = list(self.items.values())
        
        # 品类筛选
        if category:
            results = [r for r in results if r["basic_info"]["category"] == category]
        
        # 标签筛选
        if tags:
            results = [
                r for r in results 
                if any(tag in r["semantic_tags"] for tag in tags)
            ]
        
        # 按创建时间倒序
        results.sort(key=lambda x: x["basic_info"]["created_at"], reverse=True)
        
        return results[:limit]
    
    # ========== 向量检索 ==========
    def _generate_random_vector(self, dim: int = 128) -> List[float]:
        """生成随机向量（用于测试）"""
        vector = np.random.randn(dim).astype(np.float32)
        vector = vector / np.linalg.norm(vector)
        return vector.tolist()
    
    def search_by_vector(
        self,
        query_vector: List[float],
        top_k: int = 8,
        category: Optional[str] = None
    ) -> List[Dict]:
        """
        向量相似度检索
        
        Args:
            query_vector: 查询向量
            top_k: 返回数量
            category: 可选，按品类筛选
        
        Returns:
            相似衣物列表
        """
        if not query_vector:
            return []
        
        query_vec = np.array(query_vector, dtype=np.float32)
        query_vec = query_vec / np.linalg.norm(query_vec)
        
        similarities = []
        for item_id, item in self.items.items():
            # 品类筛选
            if category and item["basic_info"]["category"] != category:
                continue
            
            # 向量相似度
            item_vec = item.get("vector_embedding", [])
            if not item_vec:
                continue
            
            vec = np.array(item_vec, dtype=np.float32)
            sim = float(np.dot(query_vec, vec))
            
            similarities.append({
                "item_id": item_id,
                "similarity": sim,
                "item": item
            })
        
        # 排序并返回
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_k]
    
    def search_by_tags(
        self,
        tags: List[str],
        top_k: int = 8,
        category: Optional[str] = None,
        match_all: bool = False
    ) -> List[Dict]:
        """
        标签检索
        
        Args:
            tags: 标签列表
            top_k: 返回数量
            category: 可选，按品类筛选
            match_all: True=全匹配，False=任一匹配
        
        Returns:
            匹配的衣物列表
        """
        results = []
        
        for item_id, item in self.items.items():
            # 品类筛选
            if category and item["basic_info"]["category"] != category:
                continue
            
            item_tags = set(item["semantic_tags"])
            
            if match_all:
                # 全匹配
                if all(tag in item_tags for tag in tags):
                    results.append({"item_id": item_id, "item": item})
            else:
                # 任一匹配
                if any(tag in item_tags for tag in tags):
                    results.append({"item_id": item_id, "item": item})
        
        return results[:top_k]

    def search_by_category_and_keywords(
        self,
        category: str,
        keywords: List[str],
        top_k: int = 5,
        use_vector: bool = True
    ) -> List[Dict]:
        """
        按品类 + 关键词检索（用于新版检索节点）
        
        检索逻辑：
        1. 先按品类筛选
        2. 如果 use_vector=True：
           - 把关键词转成向量（使用 embed_text）
           - 与衣物的向量做余弦相似度
        3. 如果 use_vector=False：
           - 只用标签匹配
        
        Args:
            category: 品类
            keywords: 检索关键词列表
            top_k: 返回数量
            use_vector: 是否使用向量检索（默认开启）
        
        Returns:
            检索结果列表
        """
        # 先按品类筛选
        category_items = [
            (item_id, item) for item_id, item in self.items.items()
            if item["basic_info"]["category"] == category
        ]
        
        if not category_items:
            return []
        
        # 向量检索：把关键词转成向量
        query_vector = None
        if use_vector and keywords:
            try:
                from services.embedding import embed_text
                # 将关键词合并成文本并向量化
                query_text = " ".join(keywords)
                query_vector = embed_text(query_text)
                print(f"[DB] 关键词向量化: '{query_text}' -> {len(query_vector)}维")
            except Exception as e:
                print(f"[DB] 向量化失败，回退到标签匹配: {e}")
                use_vector = False
        
        # 计算匹配分数
        scored_results = []
        keywords_set = set(keywords)
        
        for item_id, item in category_items:
            item_tags = set(item["semantic_tags"])
            metadata = item.get("dynamic_metadata", {})
            
            # 1. 基础标签匹配分数（semantic_tags）
            matched_tags = keywords_set & item_tags
            tag_score = len(matched_tags) / max(len(keywords_set), 1) if keywords_set else 0.0
            
            # 2. 扩展匹配：从 dynamic_metadata 中提取更多匹配点
            extra_matches = 0
            metadata_texts = []
            
            # 从 details 中提取所有文本值
            details = metadata.get("details", {})
            for key, value in details.items():
                if value and isinstance(value, str):
                    metadata_texts.append(value)
            
            # 从 material, color, style_vibe 中提取
            if metadata.get("material"):
                metadata_texts.append(metadata["material"])
            if metadata.get("color"):
                metadata_texts.append(metadata["color"])
            if metadata.get("style_vibe"):
                metadata_texts.append(metadata["style_vibe"])
            
            # 检查关键词是否在 metadata 中
            for kw in keywords:
                for meta_text in metadata_texts:
                    if kw.lower() in meta_text.lower():
                        extra_matches += 1
                        matched_tags.add(kw)
            
            # 额外匹配加分（最多加0.3分）
            extra_score = min(extra_matches * 0.1, 0.3)
            
            # 3. 向量相似度分数
            vector_score = 0.0
            if use_vector and query_vector and item.get("vector_embedding"):
                from services.embedding import cosine_similarity
                vector_score = cosine_similarity(query_vector, item["vector_embedding"])
            
            # 综合分数 = 标签匹配 * 0.4 + 额外匹配 * 0.3 + 向量相似 * 0.3
            if use_vector and query_vector:
                final_score = tag_score * 0.4 + extra_score + vector_score * 0.3
            else:
                final_score = tag_score * 0.7 + extra_score
            
            scored_results.append({
                "item_id": item_id,
                "similarity": final_score,
                "tag_score": tag_score,
                "extra_score": extra_score,
                "vector_score": vector_score,
                "matched_tags": list(matched_tags),
                "item": item
            })
        
        # 排序并返回
        scored_results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # 打印日志
        if scored_results:
            method = "向量+标签" if use_vector and query_vector else "仅标签"
            print(f"[DB] 品类'{category}'检索: 关键词={keywords}, 方法={method}, 匹配{len(scored_results)}件, top={top_k}")
        
        return scored_results[:top_k]

    # ========== 场景检索 ==========
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
        # 场景到标签的映射
        scene_tags = {
            "通勤": ["通勤", "简约", "商务", "正式", "衬衫", "西装裤", "乐福鞋"],
            "日常": ["日常", "休闲", "舒适", "T恤", "牛仔裤", "运动鞋"],
            "约会": ["约会", "甜美", "浪漫", "连衣裙", "半身裙", "高跟鞋"],
            "运动": ["运动", "健身", "跑步", "运动服", "运动裤", "运动鞋"],
            "度假": ["度假", "海滩", "宽松", "连衣裙", "短裤", "凉鞋"],
        }
        
        # 获取场景对应的关键词
        keywords = scene_tags.get(scene, [scene])
        
        # 如果没有指定品类，使用向量检索
        if category:
            return self.search_by_category_and_keywords(
                category=category,
                keywords=keywords,
                top_k=top_k,
                use_vector=True
            )
        
        # 否则使用标签检索
        return self.search_by_tags(keywords, top_k=top_k)

    # ========== 并行批量检索 ==========
    def search_batch(
        self,
        search_requests: List[Dict],
        user_id: str = "default_user"
    ) -> Dict[str, List[Dict]]:
        """
        并行批量检索多个品类

        使用场景：
        - 用户需求一套完整穿搭（上衣+裤子+鞋子）
        - 需要同时检索多个品类

        Args:
            search_requests: 检索请求列表，每个元素包含：
                - category: 品类
                - keywords: 关键词列表
                - top_k: 返回数量
                - scene: 场景（可选，用于偏好过滤）
            user_id: 用户ID（用于偏好过滤）

        Returns:
            字典，key 为品类，value 为检索结果列表：
            {
                "上衣": [...],
                "裤子": [...],
                "鞋子": [...]
            }

        Examples:
            >>> db.search_batch([
            >>>     {"category": "上衣", "keywords": ["通勤", "简约"], "top_k": 3},
            >>>     {"category": "裤子", "keywords": ["通勤", "简约"], "top_k": 2},
            >>>     {"category": "鞋子", "keywords": ["通勤", "简约"], "top_k": 2},
            >>> ])
        """
        import concurrent.futures
        import threading

        results = {}
        errors = {}

        def _search_single(request: Dict) -> tuple:
            """执行单次检索"""
            category = request.get("category", "")
            keywords = request.get("keywords", [])
            top_k = request.get("top_k", 3)
            scene = request.get("scene")
            styles = request.get("styles")

            try:
                if styles:
                    # 使用 LLM 提供的风格关键词检索（优先）
                    search_results = self.search_by_styles_with_user_pref(
                        styles=styles,
                        user_id=user_id,
                        top_k=top_k,
                        category=category
                    )
                elif scene:
                    # 使用场景偏好检索（兼容旧逻辑）
                    search_results = self.search_by_scene_with_user_pref(
                        scene=scene,
                        user_id=user_id,
                        top_k=top_k,
                        category=category
                    )
                else:
                    # 使用普通关键词检索
                    search_results = self.search_by_category_and_keywords(
                        category=category,
                        keywords=keywords,
                        top_k=top_k,
                        use_vector=False  # 并行时关闭向量，节省时间
                    )

                return (category, search_results)
            except Exception as e:
                return (category, {"error": str(e)})

        # 并行执行所有检索
        print(f"[DB] 开始并行检索 {len(search_requests)} 个品类...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(search_requests), 8)) as executor:
            future_to_category = {
                executor.submit(_search_single, req): req.get("category")
                for req in search_requests
            }

            for future in concurrent.futures.as_completed(future_to_category):
                category, result = future.result()
                results[category] = result
                if "error" in result:
                    errors[category] = result.get("error")

        # 打印结果摘要
        success_count = len(results) - len(errors)
        print(f"[DB] 并行检索完成: 成功 {success_count}/{len(search_requests)}")

        if errors:
            print(f"[DB] 失败品类: {errors}")

        return results

    # ========== 用户偏好向量管理 ==========
    def set_preference_vector(self, user_id: str, pref_type: str, vector: List[float]) -> None:
        """
        设置用户偏好向量

        Args:
            user_id: 用户ID
            pref_type: 偏好类型 (style_color: 风格颜色偏好, fit: 版型偏好)
            vector: 偏好向量
        """
        if not hasattr(self, 'user_preference_vectors'):
            self.user_preference_vectors = {}

        if user_id not in self.user_preference_vectors:
            self.user_preference_vectors[user_id] = {}

        vec = np.array(vector, dtype=np.float32)
        vec = vec / np.linalg.norm(vec) if np.linalg.norm(vec) > 0 else vec
        self.user_preference_vectors[user_id][pref_type] = vec
        print(f"[DB] 设置用户偏好向量: user={user_id}, type={pref_type}")

    def get_preference_vector(self, user_id: str, pref_type: str) -> Optional[np.ndarray]:
        """获取用户偏好向量"""
        if not hasattr(self, 'user_preference_vectors'):
            return None
        return self.user_preference_vectors.get(user_id, {}).get(pref_type)

    # ========== 场景到风格的映射（仅作为备用兼容） ==========
    # 注意：新的调用流程应由 LLM 通过 styles 参数直接传递风格关键词
    # 此映射仅用于兼容旧代码（如 search_by_scene_with_user_pref）

    # ========== 获取默认风格关键词 ==========
    def get_default_styles(self) -> List[str]:
        """获取默认风格关键词（当 LLM 未提供风格时使用）"""
        return ["百搭", "舒适", "简约"]

    def get_styles_by_scene(self, scene: str) -> List[str]:
        """
        根据场景获取对应的风格关键词（兼容旧逻辑）

        注意：推荐使用 LLM 动态判断风格，此方法仅用于兼容旧代码。
        """
        # 基础风格映射（简化版，用于兜底）
        base_styles = {
            "通勤": ["简约", "干练"],
            "商务": ["正式", "专业"],
            "休闲": ["舒适", "随性"],
            "约会": ["浪漫", "甜美"],
            "度假": ["休闲", "清爽"],
            "运动": ["运动", "功能"],
            "聚会": ["时尚", "潮流"],
            "日常": ["百搭", "简约"],
        }

        # 直接匹配
        if scene in base_styles:
            return base_styles[scene]

        # 模糊匹配
        for key, styles in base_styles.items():
            if key in scene or scene in key:
                return styles

        # 默认返回通用风格
        return self.get_default_styles()

    def search_by_scene_with_user_pref(
        self,
        scene: str,
        user_id: str = "default_user",
        top_k: int = 8,
        category: Optional[str] = None
    ) -> List[Dict]:
        """
        基于场景+用户偏好检索衣物

        检索逻辑：
        1. 根据场景获取风格关键词
        2. 获取用户在该场景的偏好
        3. 过滤掉用户厌恶的标签
        4. 优先匹配用户喜欢的风格/颜色

        Args:
            scene: 场景名称 (通勤/商务/休闲/约会/度假/运动/聚会/日常)
            user_id: 用户ID
            top_k: 返回数量
            category: 可选，按品类筛选

        Returns:
            相似衣物列表
        """
        # 1. 获取场景对应的风格关键词
        scene_styles = self.get_styles_by_scene(scene)

        # 2. 获取用户在该场景的偏好
        scene_prefs = self.get_scene_preference(user_id, scene)
        liked_styles = scene_prefs.get("style", "").split(",") if scene_prefs.get("style") else []
        liked_colors = scene_prefs.get("color", "").split(",") if scene_prefs.get("color") else []
        avoid_items = scene_prefs.get("avoid", "").split(",") if scene_prefs.get("avoid") else []

        # 清理空白
        liked_styles = [s.strip() for s in liked_styles if s.strip()]
        liked_colors = [c.strip() for c in liked_colors if c.strip()]
        avoid_items = [a.strip() for a in avoid_items if a.strip()]

        print(f"[DB] 场景偏好: scene={scene}, liked_styles={liked_styles}, liked_colors={liked_colors}, avoid={avoid_items}")

        # 3. 组合检索关键词
        keywords = scene_styles.copy()

        # 添加用户喜欢的风格和颜色
        if liked_styles:
            keywords.extend(liked_styles)
        if liked_colors:
            keywords.extend(liked_colors)

        # 4. 执行混合检索
        results = self.search_by_category_and_keywords(
            category=category if category else "上衣",
            keywords=keywords,
            top_k=top_k * 2,  # 多检索一些，后面要过滤
            use_vector=False  # 先不用向量，用关键词匹配
        )

        # 5. 过滤和排序
        filtered_results = []
        for r in results:
            item = r["item"]
            item_tags = set(item.get("semantic_tags", []))
            metadata = item.get("dynamic_metadata", {})
            item_color = metadata.get("color", "")
            item_style = metadata.get("style_vibe", "")

            # 检查是否包含用户厌恶的标签
            should_filter = False
            for avoid in avoid_items:
                if avoid in item_tags or avoid in item_color or avoid in item_style:
                    should_filter = True
                    break

            if should_filter:
                continue

            # 计算偏好匹配分数
            pref_score = 0.0

            # 风格匹配
            for style in liked_styles:
                if style in item_tags or style in item_style:
                    pref_score += 0.5
                    break

            # 颜色匹配
            for color in liked_colors:
                if color in item_tags or color in item_color:
                    pref_score += 0.5
                    break

            # 综合分数 = 原分数 + 偏好分数
            final_score = r["similarity"] + pref_score * 0.3

            filtered_results.append({
                **r,
                "similarity": final_score,
                "pref_score": pref_score
            })

        # 排序
        filtered_results.sort(key=lambda x: x["similarity"], reverse=True)

        # 返回 top_k
        print(f"[DB] 场景检索: scene='{scene}', 原始{len(results)}条, 过滤后{len(filtered_results)}条, 返回{top_k}条")

        return filtered_results[:top_k]

    def search_by_styles_with_user_pref(
        self,
        styles: List[str],
        user_id: str = "default_user",
        top_k: int = 8,
        category: Optional[str] = None,
        scene: Optional[str] = None
    ) -> List[Dict]:
        """
        基于风格关键词+用户偏好检索衣物（由 LLM 动态判断风格）

        检索逻辑：
        1. 直接使用 LLM 提供的风格关键词
        2. 获取用户的偏好信息
        3. 过滤掉用户厌恶的标签
        4. 优先匹配用户喜欢的风格/颜色

        Args:
            styles: LLM 动态判断的风格关键词列表，如 ["简约", "干练", "商务"]
            user_id: 用户ID
            top_k: 返回数量
            category: 可选，按品类筛选
            scene: 可选，场景名称（用于获取用户在该场景的偏好）

        Returns:
            相似衣物列表
        """
        # 1. 获取用户偏好（使用 scene 或默认场景）
        scene_prefs = self.get_scene_preference(user_id, scene)

        liked_styles = scene_prefs.get("style", "").split(",") if scene_prefs.get("style") else []
        liked_colors = scene_prefs.get("color", "").split(",") if scene_prefs.get("color") else []
        avoid_items = scene_prefs.get("avoid", "").split(",") if scene_prefs.get("avoid") else []

        # 清理空白
        liked_styles = [s.strip() for s in liked_styles if s.strip()]
        liked_colors = [c.strip() for c in liked_colors if c.strip()]
        avoid_items = [a.strip() for a in avoid_items if a.strip()]

        print(f"[DB] 风格检索: styles={styles}, scene={scene}, liked_styles={liked_styles}, liked_colors={liked_colors}, avoid={avoid_items}")

        # 2. 组合检索关键词
        keywords = styles.copy()

        # 添加用户喜欢的风格和颜色
        if liked_styles:
            keywords.extend(liked_styles)
        if liked_colors:
            keywords.extend(liked_colors)

        # 3. 执行混合检索
        results = self.search_by_category_and_keywords(
            category=category if category else "上衣",
            keywords=keywords,
            top_k=top_k * 2,  # 多检索一些，后面要过滤
            use_vector=False  # 先不用向量，用关键词匹配
        )

        # 4. 过滤和排序
        filtered_results = []
        for r in results:
            item = r["item"]
            item_tags = set(item.get("semantic_tags", []))
            metadata = item.get("dynamic_metadata", {})
            item_color = metadata.get("color", "")
            item_style = metadata.get("style_vibe", "")

            # 检查是否包含用户厌恶的标签
            should_filter = False
            for avoid in avoid_items:
                if avoid in item_tags or avoid in item_color or avoid in item_style:
                    should_filter = True
                    break

            if should_filter:
                continue

            # 计算偏好匹配分数
            pref_score = 0.0

            # 风格匹配
            for style in liked_styles:
                if style in item_tags or style in item_style:
                    pref_score += 0.5
                    break

            # 颜色匹配
            for color in liked_colors:
                if color in item_tags or color in item_color:
                    pref_score += 0.5
                    break

            # 综合分数 = 原分数 + 偏好分数
            final_score = r["similarity"] + pref_score * 0.3

            filtered_results.append({
                **r,
                "similarity": final_score,
                "pref_score": pref_score
            })

        # 排序
        filtered_results.sort(key=lambda x: x["similarity"], reverse=True)

        # 返回 top_k
        print(f"[DB] 风格检索: styles={styles}, 原始{len(results)}条, 过滤后{len(filtered_results)}条, 返回{top_k}条")

        return filtered_results[:top_k]
    
    # ========== 统计 ==========
    def get_stats(self) -> Dict:
        """获取统计信息"""
        by_category = {}
        by_tag = {}

        for item in self.items.values():
            # 按品类统计
            cat = item["basic_info"]["category"]
            by_category[cat] = by_category.get(cat, 0) + 1

            # 按标签统计
            for tag in item["semantic_tags"]:
                by_tag[tag] = by_tag.get(tag, 0) + 1

        return {
            "total_items": len(self.items),
            "by_category": by_category,
            "by_tag": by_tag,
            "scenes": ["通勤", "商务", "休闲", "约会", "度假", "运动", "聚会", "日常"]
        }
    
    # ========== 用户偏好管理 (实现抽象方法) ==========
    # 新版偏好结构：{user_id: {default_scene: str, scenes: {scene: {key: value}}}}

    def update_user_preference(
        self,
        user_id: str,
        prefs: Dict[str, Any],
        scene: str = None
    ) -> bool:
        """
        更新用户偏好（支持场景化）

        新结构：
        {
            "default_scene": "通勤",
            "scenes": {
                "通勤": {"style": "简约", "color": "蓝色", "avoid": "过于正式"},
                "约会": {"style": "甜美", "color": "粉色"}
            }
        }

        Args:
            user_id: 用户 ID
            prefs: 偏好字典，如 {"style": "简约", "color": "蓝色"}
            scene: 可选，指定场景。如果不指定，更新默认场景的偏好。

        Returns:
            是否更新成功
        """
        try:
            if user_id not in self.user_preferences:
                self.user_preferences[user_id] = {
                    "default_scene": "日常",
                    "scenes": {}
                }

            user_prefs = self.user_preferences[user_id]

            # 如果没有指定场景，使用默认场景
            target_scene = scene or user_prefs.get("default_scene", "日常")

            # 确保 scenes 结构存在
            if "scenes" not in user_prefs:
                user_prefs["scenes"] = {}

            # 初始化场景偏好（如果不存在）
            if target_scene not in user_prefs["scenes"]:
                user_prefs["scenes"][target_scene] = {}

            # 更新该场景的偏好
            user_prefs["scenes"][target_scene].update(prefs)

            print(f"[DB] 更新用户偏好: user={user_id}, scene={target_scene}, prefs={prefs}")
            return True
        except Exception as e:
            print(f"[DB] 更新用户偏好失败: {e}")
            return False

    def get_user_preference(self, user_id: str) -> Dict[str, Any]:
        """
        获取用户完整偏好

        Args:
            user_id: 用户 ID

        Returns:
            用户偏好字典（包含 default_scene 和 scenes）
        """
        return self.user_preferences.get(user_id, {
            "default_scene": "日常",
            "scenes": {}
        })

    def get_scene_preference(
        self,
        user_id: str,
        scene: str = None
    ) -> Dict[str, Any]:
        """
        获取特定场景的偏好

        Args:
            user_id: 用户 ID
            scene: 场景名称。如果不指定，使用默认场景。

        Returns:
            场景偏好字典，如 {"style": "简约", "color": "蓝色", "avoid": "过于正式"}
        """
        user_prefs = self.get_user_preference(user_id)
        target_scene = scene or user_prefs.get("default_scene", "日常")

        return user_prefs.get("scenes", {}).get(target_scene, {})

    def set_default_scene(self, user_id: str, scene: str) -> bool:
        """
        设置用户的默认场景

        Args:
            user_id: 用户 ID
            scene: 场景名称

        Returns:
            是否设置成功
        """
        try:
            if user_id not in self.user_preferences:
                self.user_preferences[user_id] = {
                    "default_scene": scene,
                    "scenes": {}
                }
            else:
                self.user_preferences[user_id]["default_scene"] = scene

            print(f"[DB] 设置默认场景: user={user_id}, scene={scene}")
            return True
        except Exception as e:
            print(f"[DB] 设置默认场景失败: {e}")
            return False
    
    # ========== 混合检索 (实现抽象方法) ==========
    def search_hybrid(
        self,
        query_text: str,
        top_k: int = 8,
        category: Optional[str] = None,
        keywords: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        混合检索：结合关键词的稀疏检索与向量的稠密检索
        
        暂时代替向量检索的逻辑使用文本匹配或假数据
        
        Args:
            query_text: 检索文本
            top_k: 返回数量
            category: 可选，按品类筛选
            keywords: 可选，关键词列表
        
        Returns:
            检索结果列表
        """
        # 合并查询文本和关键词
        search_terms = set()
        if query_text:
            search_terms.add(query_text)
        if keywords:
            search_terms.update(keywords)
        
        # 简单的文本匹配检索
        results = []
        for item_id, item in self.items.items():
            # 品类筛选
            if category and item["basic_info"]["category"] != category:
                continue
            
            # 计算匹配分数
            score = 0.0
            matched_tags = []
            
            item_text = " ".join([
                item["basic_info"].get("name", ""),
                " ".join(item.get("semantic_tags", [])),
                " ".join(str(v) for v in item.get("dynamic_metadata", {}).values())
            ]).lower()
            
            for term in search_terms:
                term_lower = term.lower()
                if term_lower in item_text:
                    score += 1.0
                    matched_tags.append(term)
            
            if score > 0:
                results.append({
                    "item_id": item_id,
                    "similarity": score / max(len(search_terms), 1),
                    "matched_tags": matched_tags,
                    "item": item
                })
        
        # 排序并返回
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        print(f"[DB] 混合检索: query='{query_text}', category={category}, 匹配{len(results)}件")
        
        return results[:top_k]
    
    # ========== Mock 数据 ==========
    def mock(self) -> None:
        """生成测试数据"""
        print("\n[DB] 开始生成 Mock 测试数据...")

        mock_items = [
            {
                "basic_info": {
                    "name": "美式复古做旧水洗皮夹克",
                    "category": "外套",
                    "image_url": "/uploads/images/item_001.png",
                    "created_at": 1710288000
                },
                "semantic_tags": ["外套", "棕色", "皮质", "美式复古", "做旧", "短款", "秋冬", "防风"],
                "dynamic_metadata": {
                    "color": "深棕色",
                    "material": "PU皮",
                    "style_vibe": "美式复古 / 机车风",
                    "details": {"collar": "翻领", "fit": "宽松短款"}
                }
            },
            {
                "basic_info": {
                    "name": "纯白T恤",
                    "category": "上衣",
                    "image_url": "/uploads/images/item_002.png",
                    "created_at": 1710191600
                },
                "semantic_tags": ["上衣", "白色", "纯棉", "基础款", "春夏", "百搭"],
                "dynamic_metadata": {
                    "color": "白色",
                    "material": "纯棉",
                    "style_vibe": "简约 / 基础款"
                }
            },
            {
                "basic_info": {
                    "name": "黑色西裤",
                    "category": "裤子",
                    "image_url": "/uploads/images/item_003.png",
                    "created_at": 1710095200
                },
                "semantic_tags": ["裤子", "黑色", "西装", "通勤", "正式", "秋冬"],
                "dynamic_metadata": {
                    "color": "黑色",
                    "material": "羊毛混纺",
                    "style_vibe": "商务 / 通勤"
                }
            }
        ]

        for item_data in mock_items:
            # 生成向量
            if not item_data.get("vector_embedding"):
                item_data["vector_embedding"] = self._generate_random_vector()
            self.add_item(item_data)

        # 初始化用户偏好向量存储
        if not hasattr(self, 'user_preference_vectors'):
            self.user_preference_vectors = {}

        print(f"[DB] 已生成 {len(self.items)} 件测试衣物")


# ========== 持久化存储（可选）==========
class FileWardrobeDB(InMemoryWardrobeDB):
    """文件持久化衣柜数据库"""

    def __init__(self, storage_path: str = "storage/wardrobe_data.json"):
        super().__init__()
        self.storage_path = Path(storage_path)
        self._load()

    def _load(self) -> None:
        """从文件加载数据"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.items = data.get("items", {})

                    # 加载用户偏好向量 (新版)
                    user_vectors = data.get("user_preference_vectors", {})
                    self.user_preference_vectors = {
                        uid: {k: np.array(v) for k, v in prefs.items()}
                        for uid, prefs in user_vectors.items()
                    }

                    print(f"[DB] 已从文件加载 {len(self.items)} 件衣物")
            except Exception as e:
                print(f"[DB] 加载数据失败: {e}")

    def save(self) -> None:
        """保存数据到文件"""
        # 确保目录存在
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # 转换用户偏好向量为列表 (新版)
        user_vectors = {}
        if hasattr(self, 'user_preference_vectors'):
            user_vectors = {
                uid: {k: v.tolist() for k, v in prefs.items()}
                for uid, prefs in self.user_preference_vectors.items()
            }

        data = {
            "items": self.items,
            "user_preference_vectors": user_vectors
        }

        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"[DB] 已保存 {len(self.items)} 件衣物到文件")


# ========== 推荐衣物数据库（带 TTL）==========
class RecommendedClothingDB:
    """
    推荐衣物数据库
    
    与 WardrobeDB 数据格式完全兼容，额外功能：
    - TTL 过期机制：支持设置过期时间
    - 自动清理：定期删除过期衣物
    - 迁移功能：一键迁移到衣柜数据库
    
    数据格式（与 WardrobeDB 兼容）：
    {
        "id": "rec_xxx",
        "basic_info": {
            "name": "xxx",
            "category": "xxx",
            "image_url": "xxx",
            "created_at": 1234567890
        },
        "semantic_tags": [...],
        "dynamic_metadata": {...},
        "vector_embedding": [...],
        # 额外字段
        "expires_at": 1234567890,  # 过期时间戳，0 表示永不过期
        "source": "xhs_search",     # 来源：xhs_search / ai_recommend / user_upload
        "scene": "通勤",             # 推荐场景
        "reason": "xxx"              # 推荐理由
    }
    """
    
    def __init__(self, default_ttl_days: int = 7):
        """
        初始化推荐数据库
        
        Args:
            default_ttl_days: 默认过期天数（默认7天）
        """
        self.items: Dict[str, Dict] = {}  # item_id -> item data
        self.default_ttl_days = default_ttl_days
        print(f"[推荐DB] RecommendedClothingDB 初始化完成 (默认TTL: {default_ttl_days}天)")
    
    def add_item(
        self,
        item_data: Dict,
        ttl_days: int = None,
        source: str = "ai_recommend",
        scene: str = "",
        reason: str = ""
    ) -> str:
        """
        添加推荐衣物
        
        Args:
            item_data: 衣物数据（与 WardrobeDB 格式相同）
            ttl_days: 过期天数（默认使用初始化时的默认值）
            source: 来源 (xhs_search/ai_recommend/user_upload)
            scene: 推荐场景
            reason: 推荐理由
        
        Returns:
            衣物 ID
        """
        import time
        
        # 生成唯一 ID
        item_id = item_data.get("id") or f"rec_{uuid.uuid4().hex[:8]}"
        
        # 计算过期时间
        ttl = ttl_days if ttl_days is not None else self.default_ttl_days
        expires_at = int(time.time()) + ttl * 24 * 3600 if ttl > 0 else 0
        
        # 构建完整数据
        item = {
            "id": item_id,
            "basic_info": {
                "name": item_data.get("basic_info", {}).get("name", "未命名"),
                "category": item_data.get("basic_info", {}).get("category", "未知"),
                "image_url": item_data.get("basic_info", {}).get("image_url", ""),
                "created_at": item_data.get("basic_info", {}).get("created_at") or int(time.time())
            },
            "semantic_tags": item_data.get("semantic_tags", []),
            "dynamic_metadata": item_data.get("dynamic_metadata", {}),
            "vector_embedding": item_data.get("vector_embedding", []),
            # 推荐数据库额外字段
            "expires_at": expires_at,
            "source": source,
            "scene": scene,
            "reason": reason
        }
        
        self.items[item_id] = item
        print(f"[推荐DB] 添加推荐衣物: {item['basic_info']['name']} (ID: {item_id}, 过期: {ttl}天后)")
        
        return item_id
    
    def add_from_xhs(self, xhs_item: Dict, ttl_days: int = 7) -> str:
        """
        从小红书搜索结果添加推荐衣物
        
        Args:
            xhs_item: 小红书物品数据
            ttl_days: 过期天数
        
        Returns:
            衣物 ID
        """
        item_data = {
            "basic_info": {
                "name": xhs_item.get("title", "小红书推荐"),
                "category": xhs_item.get("category", "未知"),
                "image_url": xhs_item.get("image_url", ""),
                "created_at": int(datetime.now().timestamp())
            },
            "semantic_tags": xhs_item.get("semantic_tags", []),
            "dynamic_metadata": xhs_item.get("dynamic_metadata", {}),
            "vector_embedding": xhs_item.get("vector_embedding", [])
        }
        
        return self.add_item(
            item_data,
            ttl_days=ttl_days,
            source="xhs_search",
            scene=xhs_item.get("scene", ""),
            reason=xhs_item.get("reason", "")
        )
    
    def get_item(self, item_id: str) -> Optional[Dict]:
        """根据 ID 获取衣物"""
        return self.items.get(item_id)
    
    def list_items(
        self,
        category: Optional[str] = None,
        include_expired: bool = False,
        limit: int = 100
    ) -> List[Dict]:
        """
        列出推荐衣物
        
        Args:
            category: 按品类筛选
            include_expired: 是否包含已过期的衣物
            limit: 返回数量限制
        
        Returns:
            衣物列表
        """
        import time
        current_time = int(time.time())
        
        results = []
        for item in self.items.values():
            # 品类筛选
            if category and item["basic_info"]["category"] != category:
                continue
            
            # 过期筛选
            expires_at = item.get("expires_at", 0)
            if not include_expired and expires_at > 0 and expires_at < current_time:
                continue
            
            results.append(item)
        
        # 按创建时间倒序
        results.sort(key=lambda x: x["basic_info"]["created_at"], reverse=True)
        
        return results[:limit]
    
    def get_expired_items(self) -> List[Dict]:
        """获取所有已过期的衣物"""
        import time
        current_time = int(time.time())
        
        return [
            item for item in self.items.values()
            if item.get("expires_at", 0) > 0 and item["expires_at"] < current_time
        ]
    
    def cleanup_expired(self) -> int:
        """
        清理所有过期衣物
        
        Returns:
            删除的数量
        """
        expired = self.get_expired_items()
        count = 0
        
        for item in expired:
            item_id = item["id"]
            if item_id in self.items:
                del self.items[item_id]
                count += 1
        
        if count > 0:
            print(f"[推荐DB] 清理了 {count} 件过期推荐衣物")
        
        return count
    
    def renew_item(self, item_id: str, ttl_days: int = None) -> bool:
        """
        刷新推荐衣物的过期时间
        
        Args:
            item_id: 衣物 ID
            ttl_days: 新的过期天数
        
        Returns:
            是否成功
        """
        import time
        
        if item_id not in self.items:
            return False
        
        ttl = ttl_days if ttl_days is not None else self.default_ttl_days
        if ttl > 0:
            self.items[item_id]["expires_at"] = int(time.time()) + ttl * 24 * 3600
        
        return True
    
    def delete_item(self, item_id: str) -> bool:
        """删除推荐衣物"""
        if item_id in self.items:
            del self.items[item_id]
            return True
        return False
    
    def transfer_to_wardrobe(
        self,
        item_id: str,
        wardrobe_db: InMemoryWardrobeDB
    ) -> Optional[str]:
        """
        将推荐衣物迁移到衣柜数据库
        
        Args:
            item_id: 推荐衣物 ID
            wardrobe_db: 目标衣柜数据库
        
        Returns:
            衣柜中的新 ID，失败返回 None
        """
        if item_id not in self.items:
            print(f"[推荐DB] 迁移失败：找不到推荐衣物 {item_id}")
            return None
        
        item = self.items[item_id]
        
        # 移除推荐数据库的额外字段，保持与衣柜格式一致
        wardrobe_item = {
            "basic_info": item["basic_info"],
            "semantic_tags": item["semantic_tags"],
            "dynamic_metadata": item["dynamic_metadata"],
            "vector_embedding": item["vector_embedding"]
        }
        
        # 添加到衣柜
        new_id = wardrobe_db.add_item(wardrobe_item)
        
        # 从推荐数据库删除
        del self.items[item_id]
        
        print(f"[推荐DB] 迁移成功: {item_id} -> 衣柜 {new_id}")
        
        return new_id
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        import time
        current_time = int(time.time())
        
        active_count = 0
        expired_count = 0
        
        for item in self.items.values():
            expires_at = item.get("expires_at", 0)
            if expires_at > 0 and expires_at < current_time:
                expired_count += 1
            else:
                active_count += 1
        
        by_category = {}
        for item in self.items.values():
            cat = item["basic_info"]["category"]
            by_category[cat] = by_category.get(cat, 0) + 1
        
        return {
            "total_items": len(self.items),
            "active_items": active_count,
            "expired_items": expired_count,
            "by_category": by_category
        }


# ========== 全局实例 ==========
_db_instance: Optional[BaseWardrobeDB] = None
_recommended_db_instance: Optional["RecommendedClothingDB"] = None


def get_wardrobe_db() -> BaseWardrobeDB:
    """获取数据库单例
    
    Returns:
        BaseWardrobeDB: 数据库实例（实际返回 InMemoryWardrobeDB）
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = InMemoryWardrobeDB()
    return _db_instance


def init_wardrobe_db() -> InMemoryWardrobeDB:
    """初始化数据库"""
    db = get_wardrobe_db()
    db.mock()
    return db


def get_recommended_db(default_ttl_days: int = 7) -> RecommendedClothingDB:
    """
    获取推荐数据库单例
    
    Args:
        default_ttl_days: 默认过期天数
    
    Returns:
        推荐数据库实例
    """
    global _recommended_db_instance
    if _recommended_db_instance is None:
        _recommended_db_instance = RecommendedClothingDB(default_ttl_days=default_ttl_days)
    return _recommended_db_instance


def init_recommended_db(default_ttl_days: int = 7) -> RecommendedClothingDB:
    """初始化推荐数据库"""
    return get_recommended_db(default_ttl_days=default_ttl_days)


# ========== 单元测试 ==========
if __name__ == "__main__":
    print("=" * 60)
    print("测试新版衣柜数据库")
    print("=" * 60)
    
    db = InMemoryWardrobeDB()
    db.mock()
    
    # 测试添加
    print("\n[测试1] 添加衣物")
    new_item = {
        "basic_info": {
            "name": "测试夹克",
            "category": "外套",
            "image_url": "/test.jpg"
        },
        "semantic_tags": ["测试", "秋冬"],
        "dynamic_metadata": {"color": "黑色"}
    }
    item_id = db.add_item(new_item)
    print(f"添加成功: {item_id}")
    
    # 测试获取
    print("\n[测试2] 获取衣物")
    item = db.get_item(item_id)
    print(f"名称: {item['basic_info']['name']}")
    print(f"标签: {item['semantic_tags']}")
    
    # 测试检索
    print("\n[测试3] 场景检索")
    results = db.search_by_scene("commute", top_k=3)
    for r in results:
        print(f"  - {r['item']['basic_info']['name']} (相似度: {r['similarity']:.4f})")
    
    # 测试统计
    print("\n[测试4] 统计信息")
    stats = db.get_stats()
    print(f"总数: {stats['total_items']}")
    print(f"品类: {stats['by_category']}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)