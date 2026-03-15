# -*- coding: utf-8 -*-
"""
极简盲盒胶囊衣柜 V1.0 - 内存数据库层

核心设计理念：混合动态数据结构 (Hybrid Schema)
- 核心字段结构化（category 用于强规则过滤）
- 边缘属性标签化（tags 扁平化存储，支持语义检索）
- 语义向量兜底（vector 用于相似度计算）
- 动态 metadata 字典（接收 VLM 的不规则 JSON）
"""

import uuid
import numpy as np
from typing import Dict, List, Optional


class InMemoryWardrobeDB:
    """内存版衣柜数据库"""
    
    def __init__(self):
        self.items_collection: Dict[str, dict] = {}
        self.preference_vectors: Dict[str, np.ndarray] = {}
        print("[DB] InMemoryWardrobeDB 初始化完成")
    
    def mock(self) -> None:
        """生成测试数据：20件衣物 + 2个场景偏好"""
        print("\n[DB] 开始生成 Mock 测试数据...")
        
        item_templates = [
            {"category": "上衣", "names": ["纯白T恤", "黑色衬衫", "条纹Polo", "灰色卫衣", 
                                           "蓝色牛仔外套", "白色衬衫", "黑色针织衫", "印花T恤"],
             "tags_pool": [["通勤", "白色", "基础款"], ["通勤", "黑色", "基础款"],
                          ["休闲", "条纹"], ["秋冬", "灰色"], ["度假", "蓝色"],
                          ["正式", "白色"], ["秋冬", "黑色"], ["休闲", "印花"]]},
            {"category": "裤子", "names": ["黑色西裤", "蓝色牛仔裤", "卡其色休闲裤", "灰色运动裤",
                                           "白色短裤", "黑色牛仔裤", "卡其色短裤"],
             "tags_pool": [["通勤", "黑色", "基础款"], ["通勤", "蓝色"],
                          ["休闲", "卡其色"], ["运动", "灰色"], ["度假", "白色"],
                          ["秋冬", "黑色"], ["度假", "卡其色"]]},
            {"category": "鞋子", "names": ["黑色皮鞋", "白色运动鞋", "棕色皮鞋", "灰色休闲鞋", "黑色靴子"],
             "tags_pool": [["通勤", "黑色", "基础款"], ["运动", "白色", "基础款"],
                          ["商务", "棕色"], ["休闲", "灰色"], ["秋冬", "黑色"]]},
        ]
        
        item_id = 1
        for category_data in item_templates:
            for i, name in enumerate(category_data["names"]):
                item_id_str = f"item_{item_id:03d}"
                tags = category_data["tags_pool"][i].copy()
                
                if i % 3 == 0 and "基础款" not in tags:
                    tags.append("基础款")
                
                vector = np.random.randn(128).astype(np.float32)
                vector = vector / np.linalg.norm(vector)
                
                self.items_collection[item_id_str] = {
                    "id": item_id_str,
                    "name": name,
                    "category": category_data["category"],
                    "tags": tags,
                    "metadata": {},  # 预留：用于存储 VLM 解析出的动态属性
                    "vector": vector
                }
                item_id += 1
        
        print(f"[DB] 已生成 {len(self.items_collection)} 件测试衣物")
        
        basic_count = sum(1 for item in self.items_collection.values() 
                        if "基础款" in item["tags"])
        print(f"[DB] 其中包含 '基础款' 标签的衣物: {basic_count} 件")
        
        # 场景偏好向量
        commute_vector = np.random.randn(128).astype(np.float32)
        commute_vector = commute_vector / np.linalg.norm(commute_vector)
        commute_vector[0:32] += 0.5
        commute_vector = commute_vector / np.linalg.norm(commute_vector)
        
        vacation_vector = np.random.randn(128).astype(np.float32)
        vacation_vector = vacation_vector / np.linalg.norm(vacation_vector)
        vacation_vector[64:96] += 0.5
        vacation_vector = vacation_vector / np.linalg.norm(vacation_vector)
        
        self.preference_vectors["commute"] = commute_vector
        self.preference_vectors["vacation"] = vacation_vector
        
        print(f"[DB] 已生成 {len(self.preference_vectors)} 个场景偏好向量")
        print("[DB] Mock 数据生成完成!\n")
    

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """纯numpy实现余弦相似度计算"""
        return float(np.dot(vec1, vec2))
    
    
    def search_top_k(self, scene: str, top_k: int = 8) -> List[dict]:
        """基于余弦相似度检索，保证至少2件基础款"""
        if scene not in self.preference_vectors:
            raise ValueError(f"[DB] 未知场景: {scene}")
        
        preference_vector = self.preference_vectors[scene]
        
        similarities = []
        for item_id, item_data in self.items_collection.items():
            sim = self.cosine_similarity(preference_vector, item_data["vector"])
            similarities.append({
                "item_id": item_id,
                "similarity": sim,
                "item": item_data
            })
        
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        # 盲盒连通率逻辑
        basic_items = [s for s in similarities if "基础款" in s["item"]["tags"]]
        non_basic_items = [s for s in similarities if "基础款" not in s["item"]["tags"]]
        
        result = basic_items[:2] + non_basic_items[:top_k-2]
        
        if len(result) < top_k:
            used_ids = {r["item_id"] for r in result}
            for s in similarities:
                if s["item_id"] not in used_ids:
                    result.append(s)
                    if len(result) >= top_k:
                        break
        
        result = sorted(result, key=lambda x: x["similarity"], reverse=True)[:top_k]
        
        basic_in_result = sum(1 for r in result if "基础款" in r["item"]["tags"])
        print(f"\n[DB] 检索完成: 场景={scene}, top_k={top_k}")
        print(f"[DB] 检索结果中包含 '基础款' 标签的衣物: {basic_in_result} 件")
        
        return result
    
    def add_item_from_vlm(self, vlm_payload: dict) -> str:
        """
        模拟接收视觉 Agent (VLM) 动态解析出的不规则 JSON，并写入数据库。
        
        核心思路：将 VLM 返回的任意结构拍平成统一的 items_collection 格式。
        - category: 保留为核心结构化字段
        - tags: 将所有动态属性值扁平化后存入
        - metadata: 原汁原味保留原始 JSON，供后续 Agent 读取
        
        vlm_payload 示例: 
        {
            "category": "裙子",
            "color": "酒红",
            "details": {"style": "法式复古", "design": "开叉", "length": "及踝"}
        }
        
        Returns:
            str: 新增衣物的 ID
        """
        # 1. 生成唯一 ID
        item_id_str = f"item_{uuid.uuid4().hex[:8]}"
        
        # 2. 提取基础大类 (如果没有，默认为 '未知品类')
        category = vlm_payload.get("category", "未知品类")
        
        # 3. 动态展平所有属性生成 Tags
        # 将颜色、以及 details 里的所有 values 提取出来，组成扁平列表
        dynamic_tags = [category]  # 先放入品类作为基础标签
        
        if "color" in vlm_payload:
            dynamic_tags.append(vlm_payload["color"])
        
        details = vlm_payload.get("details", {})
        for key, value in details.items():
            dynamic_tags.append(str(value))  # 例如: "法式复古", "开叉", "及踝"
        
        # 4. 模拟文本向量化
        # 未来这里会替换为真正的 embedding(text) 调用
        # 例如: embedding("裙子 酒红 法式复古 开叉 及踝")
        vector = np.random.randn(128).astype(np.float32)
        vector = vector / np.linalg.norm(vector)
        
        # 5. 组装入库
        self.items_collection[item_id_str] = {
            "id": item_id_str,
            "name": f"{vlm_payload.get('color', '')}{category}",
            "category": category,
            "tags": dynamic_tags,       # ["裙子", "酒红", "法式复古", "开叉", "及踝"]
            "metadata": vlm_payload,    # 原汁原味保留动态结构
            "vector": vector
        }
        
        print(f"[DB] 动态衣物入库成功! ID: {item_id_str}")
        print(f"[DB] 提取到的动态标签: {dynamic_tags}")
        return item_id_str
    
    def get_item_by_id(self, item_id: str) -> Optional[dict]:
        """根据 ID 获取单件衣物详情"""
        return self.items_collection.get(item_id)


if __name__ == "__main__":
    print("=" * 60)
    print("开始测试 InMemoryWardrobeDB")
    print("=" * 60)
    
    db = InMemoryWardrobeDB()
    db.mock()
    
    print("\n" + "=" * 60)
    print("测试场景: commute (通勤)")
    print("=" * 60)
    
    results = db.search_top_k(scene="commute", top_k=8)
    
    print("\n检索结果详情:")
    print("-" * 60)
    for i, r in enumerate(results, 1):
        item = r["item"]
        print(f"{i}. [{item['id']}] {item['name']} ({item['category']})")
        print(f"   标签: {', '.join(item['tags'])}")
        print(f"   相似度: {r['similarity']:.4f}")
    
    print("\n" + "=" * 60)
    print("测试场景: vacation (度假)")
    print("=" * 60)
    
    results = db.search_top_k(scene="vacation", top_k=8)
    
    print("\n检索结果详情:")
    print("-" * 60)
    for i, r in enumerate(results, 1):
        item = r["item"]
        print(f"{i}. [{item['id']}] {item['name']} ({item['category']})")
        print(f"   标签: {', '.join(item['tags'])}")
        print(f"   相似度: {r['similarity']:.4f}")
    
    # ========== 新增：测试 VLM 动态注入 ==========
    print("\n" + "=" * 60)
    print("测试 VLM 动态注入功能")
    print("=" * 60)
    
    # 模拟 VLM 解析出的不规则 JSON
    vlm_outputs = [
        {
            "category": "配饰",
            "color": "银色",
            "details": {
                "material": "钛钢",
                "vibe": "赛博朋克",
                "element": "十字架"
            }
        },
        {
            "category": "裙子",
            "color": "酒红",
            "details": {
                "style": "法式复古",
                "design": "开叉",
                "length": "及踝"
            }
        },
        {
            "category": "上衣",
            "color": "黑色",
            "details": {
                "neckline": "V领",
                "pattern": "破洞",
                "sleeve": "五分袖"
            }
        }
    ]
    
    for vlm_payload in vlm_outputs:
        new_id = db.add_item_from_vlm(vlm_payload)
        # 验证入库
        item = db.get_item_by_id(new_id)
        print(f"   验证: {item['name']}, tags={item['tags']}, metadata={item['metadata']}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
