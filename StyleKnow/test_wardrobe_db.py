# -*- coding: utf-8 -*-
"""
测试衣柜数据库 + 保存数据用于调试向量化问题
"""
import sys
import io
import json

# 修复 Windows 控制台编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from agent.graph import run_workflow

# 测试查询
query = "今天出去玩，怎么穿搭呢"

# 运行工作流
result = run_workflow(
    user_query=query,
    user_images=[],
    uploaded_images=[]
)

# 打印结果
print("\n" + "=" * 70)
print("最终结果:")
print("=" * 70)
print(result.get("final_outfit", ""))
print("\n迭代次数:", result.get("iterations"))

# ========== 保存数据库数据用于调试 ==========
print("\n" + "=" * 70)
print("保存数据库数据...")
print("=" * 70)

# 获取数据库实例
from agent.ultis import get_wardrobe_db
db = get_wardrobe_db()

# 保存衣物数据
items_data = []
for item_id, item in db.items_collection.items():
    items_data.append({
        "id": item["id"],
        "name": item["name"],
        "category": item["category"],
        "tags": item.get("tags", []),
        "metadata": item.get("metadata", {}),
        "has_vector": "vector" in item and item["vector"] is not None
    })

# 保存场景偏好向量
preference_data = {}
for scene, vector in db.preference_vectors.items():
    preference_data[scene] = {
        "has_vector": vector is not None,
        "shape": vector.shape if vector is not None else None
    }

db_state = {
    "items": items_data,
    "preference_vectors": preference_data,
    "total_items": len(items_data)
}

# 保存到文件
output_file = "wardrobe_db_state.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(db_state, f, ensure_ascii=False, indent=2)

print(f"数据库状态已保存到: {output_file}")
print(f"衣物总数: {len(items_data)}")
print(f"场景偏好: {list(preference_data.keys())}")
