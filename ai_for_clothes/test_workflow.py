# -*- coding: utf-8 -*-
"""
直接运行 LangGraph 多 Agent 工作流测试
"""
import sys
import io

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
