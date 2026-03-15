# -*- coding: utf-8 -*-
"""
衣柜数据库检索工具

Agent 可直接调用的衣物检索工具：
- search_wardrobe: 根据场景关键词检索衣柜中的衣物
- add_clothing_item: 向衣柜添加新衣物

依赖: storage.db.InMemoryWardrobeDB
"""

from typing import List, Dict, Optional
import sys
import os

# 确保可以导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.db import InMemoryWardrobeDB, get_wardrobe_db, init_wardrobe_db
import datetime


def search_wardrobe(
    scene: str = "commute",
    top_k: int = 8,
    category: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> Dict:
    """
    根据场景检索衣柜中的衣物
    
    Args:
        scene: 场景关键词 (commute/通勤, vacation/度假, casual/休闲, sports/运动)
        top_k: 返回的衣物数量，默认 8
        category: 可选，按品类筛选 (上衣/裤子/裙子/鞋子/配饰/外套)
        tags: 可选，按标签筛选列表
    
    Returns:
        包含检索结果的字典:
        {
            "scene": str,          # 场景名称
            "total": int,         # 结果总数
            "items": List[dict]   # 衣物列表
        }
    
    Examples:
        >>> search_wardrobe("通勤")
        >>> search_wardrobe("vacation", top_k=10)
        >>> search_wardrobe("通勤", category="外套")
    """
    # 标准化场景名称
    scene = normalize_scene(scene)
    
    # 获取数据库实例
    db = get_wardrobe_db()
    
    if db is None or len(db.items) == 0:
        # 如果数据库未初始化或为空，先初始化
        db = init_wardrobe_db()
    
    try:
        # 执行场景检索
        results = db.search_by_scene(scene=scene, top_k=top_k, category=category)
        
        # 如果有标签筛选
        if tags:
            results = [
                r for r in results
                if any(tag in r["item"]["semantic_tags"] for tag in tags)
            ]
        
        # 转换为简化格式
        items = []
        for r in results:
            item = r["item"]
            basic = item.get("basic_info", {})
            
            items.append({
                "item_id": r["item_id"],
                "name": basic.get("name", ""),
                "category": basic.get("category", ""),
                "image_url": basic.get("image_url", ""),
                "tags": item.get("semantic_tags", []),
                "metadata": item.get("dynamic_metadata", {}),
                "similarity": round(r["similarity"], 4)
            })
        
        return {
            "scene": scene,
            "total": len(items),
            "items": items
        }
        
    except Exception as e:
        return {
            "scene": scene,
            "total": 0,
            "items": [],
            "error": str(e)
        }


def search_wardrobe_by_task(
    category: str,
    keywords: List[str],
    top_k: int = 5,
    scene: str = "commute"
) -> Dict:
    """
    按品类 + 关键词检索（新版本检索节点使用）
    
    Args:
        category: 品类 (上衣/裤子/鞋子/配饰/外套等)
        keywords: LLM生成的语义标签关键词列表
        top_k: 返回数量，默认 5
        scene: 场景（用于参考）
    
    Returns:
        检索结果:
        {
            "category": str,
            "keywords": List[str],
            "total": int,
            "items": List[dict]
        }
    
    Examples:
        >>> search_wardrobe_by_task("上衣", ["简约", "通勤", "白色"], top_k=5)
    """
    # 获取数据库
    db = get_wardrobe_db()
    
    if db is None or len(db.items) == 0:
        db = init_wardrobe_db()
    
    try:
        # 按品类+关键词检索（启用向量检索）
        results = db.search_by_category_and_keywords(
            category=category,
            keywords=keywords,
            top_k=top_k,
            use_vector=True  # 启用真正的向量检索
        )
        
        # 转换为简化格式
        items = []
        for r in results:
            item = r["item"]
            basic = item.get("basic_info", {})
            
            items.append({
                "item_id": r["item_id"],
                "name": basic.get("name", ""),
                "category": basic.get("category", ""),
                "image_url": basic.get("image_url", ""),
                "tags": item.get("semantic_tags", []),
                "metadata": item.get("dynamic_metadata", {}),
                "matched_tags": r.get("matched_tags", []),
                "similarity": round(r["similarity"], 4)
            })
        
        return {
            "category": category,
            "keywords": keywords,
            "scene": scene,
            "total": len(items),
            "items": items
        }
        
    except Exception as e:
        return {
            "category": category,
            "keywords": keywords,
            "total": 0,
            "items": [],
            "error": str(e)
        }


def add_clothing_item(
    name: str,
    category: str,
    image_url: str = "",
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict] = None,
    vector_embedding: Optional[List[float]] = None
) -> Dict:
    """
    向衣柜添加新衣物
    
    Args:
        name: 衣物名称
        category: 品类 (上衣/裤子/裙子/鞋子/配饰/外套)
        image_url: 图片路径
        tags: 标签列表
        metadata: 动态元数据字典
        vector_embedding: 向量嵌入
    
    Returns:
        包含添加结果的字典:
        {
            "success": bool,
            "item_id": str,       # 新增衣物的ID
            "message": str        # 结果信息
        }
    
    Examples:
        >>> add_clothing_item("红色连衣裙", "裙子", tags=["度假", "复古"])
    """
    # 验证品类
    valid_categories = ["上衣", "裤子", "裙子", "鞋子", "配饰", "外套", "未知"]
    if category not in valid_categories:
        category = "未知"
    
    # 准备完整数据
    item_data = {
        "basic_info": {
            "name": name,
            "category": category,
            "image_url": image_url,
            "created_at": int(datetime.now().timestamp()) if 'datetime' in globals() else None
        },
        "semantic_tags": tags or [],
        "dynamic_metadata": metadata or {},
        "vector_embedding": vector_embedding or []
    }
    
    # 如果名称包含颜色信息，尝试提取
    colors = ["黑", "白", "红", "蓝", "绿", "黄", "灰", "卡其", "棕", "粉", "紫"]
    for color in colors:
        if color in name and "color" not in item_data["dynamic_metadata"]:
            item_data["dynamic_metadata"]["color"] = color
            if color not in item_data["semantic_tags"]:
                item_data["semantic_tags"].append(color)
            break
    
    # 如果没有标签，添加品类作为标签
    if not item_data["semantic_tags"]:
        item_data["semantic_tags"] = [category]
    
    # 如果没有向量，生成随机向量
    if not item_data["vector_embedding"]:
        import numpy as np
        vector = np.random.randn(128).astype(np.float32)
        vector = vector / np.linalg.norm(vector)
        item_data["vector_embedding"] = vector.tolist()
    
    # 获取数据库
    db = get_wardrobe_db()
    if db is None or len(db.items) == 0:
        db = init_wardrobe_db()
    
    try:
        # 添加衣物
        item_id = db.add_item(item_data)
        
        return {
            "success": True,
            "item_id": item_id,
            "message": f"成功添加衣物: {name} (ID: {item_id})"
        }
    except Exception as e:
        return {
            "success": False,
            "item_id": None,
            "message": f"添加衣物失败: {str(e)}"
        }


def get_wardrobe_stats() -> Dict:
    """
    获取衣柜统计信息
    
    Returns:
        包含统计信息的字典:
        {
            "total_items": int,
            "by_category": Dict[str, int],
            "scenes": List[str]
        }
    """
    db = get_wardrobe_db()
    if db is None or len(db.items) == 0:
        db = init_wardrobe_db()
    
    return db.get_stats()


def get_item_detail(item_id: str) -> Dict:
    """
    获取衣物详情
    
    Args:
        item_id: 衣物ID
    
    Returns:
        衣物详情字典
    """
    db = get_wardrobe_db()
    if db is None:
        db = init_wardrobe_db()
    
    item = db.get_item(item_id)
    if item:
        return {
            "success": True,
            "item": item
        }
    return {
        "success": False,
        "message": f"未找到衣物: {item_id}"
    }


def list_wardrobe_items(
    category: Optional[str] = None,
    tags: Optional[List[str]] = None,
    limit: int = 100
) -> Dict:
    """
    列出衣柜中的衣物
    
    Args:
        category: 按品类筛选
        tags: 按标签筛选
        limit: 返回数量限制
    
    Returns:
        衣物列表
    """
    db = get_wardrobe_db()
    if db is None or len(db.items) == 0:
        db = init_wardrobe_db()
    
    items = db.list_items(category=category, tags=tags, limit=limit)
    
    return {
        "total": len(items),
        "items": items
    }


def normalize_scene(scene: str) -> str:
    """
    标准化场景名称
    
    Args:
        scene: 原始场景输入
    
    Returns:
        标准化后的场景名称
    """
    # 中文关键词映射
    cn_to_en = {
        "通勤": "commute",
        "上班": "commute",
        "商务": "commute",
        "职场": "commute",
        "度假": "vacation",
        "旅游": "vacation",
        "旅行": "vacation",
        "休闲": "casual",
        "运动": "sports",
        "健身": "sports"
    }
    
    scene_lower = scene.lower().strip()
    
    # 直接匹配英文
    valid_scenes = ["commute", "vacation", "casual", "sports"]
    if scene_lower in valid_scenes:
        return scene_lower
    
    # 匹配中文
    for cn_keyword, en_scene in cn_to_en.items():
        if cn_keyword in scene:
            return en_scene
    
    # 默认返回 commute
    return "commute"


# ========== 单元测试 ==========
if __name__ == "__main__":
    print("=" * 60)
    print("测试衣柜检索工具")
    print("=" * 60)
    
    # 初始化数据库
    init_wardrobe_db()
    
    # 测试检索
    print("\n[测试1] 检索通勤衣物")
    result = search_wardrobe("通勤", top_k=5)
    print(f"场景: {result['scene']}")
    print(f"总数: {result['total']}")
    for item in result['items']:
        print(f"  - {item['name']} ({item['category']})")
    
    print("\n[测试2] 检索度假衣物")
    result = search_wardrobe("vacation", top_k=5)
    print(f"场景: {result['scene']}")
    print(f"总数: {result['total']}")
    for item in result['items']:
        print(f"  - {item['name']} ({item['category']})")
    
    print("\n[测试3] 添加新衣物")
    result = add_clothing_item(
        name="酒红色法式连衣裙",
        category="裙子",
        tags=["度假", "复古", "法式"]
    )
    print(f"结果: {result}")
    
    print("\n[测试4] 获取统计信息")
    stats = get_wardrobe_stats()
    print(f"总衣物数: {stats['total_items']}")
    print(f"按品类: {stats['by_category']}")
    print(f"可用场景: {stats['scenes']}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
