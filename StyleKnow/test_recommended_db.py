# -*- coding: utf-8 -*-
"""测试推荐衣物数据库"""

from storage.db import RecommendedClothingDB, InMemoryWardrobeDB


def test_recommended_db():
    print("=" * 60)
    print("测试推荐衣物数据库")
    print("=" * 60)
    
    # 1. 创建推荐数据库（TTL = 1 天）
    rec_db = RecommendedClothingDB(default_ttl_days=1)
    wardrobe_db = InMemoryWardrobeDB()
    
    # 2. 添加推荐衣物
    print("\n[测试1] 添加推荐衣物")
    item1 = {
        "basic_info": {
            "name": "白色衬衫",
            "category": "上衣",
            "image_url": "/uploads/shirt.png"
        },
        "semantic_tags": ["白色", "衬衫", "通勤", "简约"],
        "dynamic_metadata": {"color": "白色", "material": "纯棉"}
    }
    
    item2 = {
        "basic_info": {
            "name": "黑色西裤",
            "category": "裤子",
            "image_url": "/uploads/pants.png"
        },
        "semantic_tags": ["黑色", "西裤", "通勤", "正式"],
        "dynamic_metadata": {"color": "黑色", "material": "羊毛"}
    }
    
    id1 = rec_db.add_item(item1, ttl_days=1, source="ai_recommend", scene="通勤", reason="百搭经典")
    id2 = rec_db.add_item(item2, ttl_days=7, source="xhs_search", scene="通勤", reason="小红书热门")
    
    # 3. 列出推荐衣物
    print("\n[测试2] 列出推荐衣物")
    items = rec_db.list_items()
    print(f"共 {len(items)} 件推荐衣物")
    for item in items:
        print(f"  - {item['basic_info']['name']} (ID: {item['id']}, 场景: {item.get('scene')})")
    
    # 4. 统计信息
    print("\n[测试3] 统计信息")
    stats = rec_db.get_stats()
    print(f"总数: {stats['total_items']}, 活跃: {stats['active_items']}, 过期: {stats['expired_items']}")
    
    # 5. 迁移到衣柜
    print("\n[测试4] 迁移到衣柜")
    new_id = rec_db.transfer_to_wardrobe(id1, wardrobe_db)
    print(f"迁移成功: 推荐 {id1} -> 衣柜 {new_id}")
    
    # 验证衣柜
    wardrobe_item = wardrobe_db.get_item(new_id)
    print(f"衣柜中物品: {wardrobe_item['basic_info']['name']}")
    
    # 验证推荐数据库
    rec_item = rec_db.get_item(id1)
    print(f"推荐数据库中是否存在: {rec_item is None}")  # 应该不存在
    
    # 6. 清理过期
    print("\n[测试5] 清理过期衣物")
    # 手动设置一个过期的
    rec_db.items[id2]["expires_at"] = 0  # 设置为已过期
    count = rec_db.cleanup_expired()
    print(f"清理了 {count} 件过期衣物")
    
    # 剩余
    items = rec_db.list_items()
    print(f"剩余 {len(items)} 件推荐衣物")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    test_recommended_db()
