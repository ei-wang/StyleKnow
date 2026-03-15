# -*- coding: utf-8 -*-
"""测试重构后的 Agent 模块"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置 UTF-8 输出
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from agent.graph import run_workflow
from agent.tools import update_user_preference


def main():
    print("=" * 70)
    print("测试重构后的 Agent 模块")
    print("=" * 70)
    
    # 运行工作流
    result = run_workflow("明天去公司开会，帮我搭配")
    
    print("\n" + "=" * 70)
    print("工作流执行完成!")
    print("=" * 70)
    print(f"最终迭代次数: {result['iterations']}")
    print(f"\n最终搭配结果:")
    print("-" * 70)
    print(result["final_outfit"])
    print("=" * 70)
    
    # 演示用户反馈
    print("\n" + "=" * 70)
    print("演示：用户点赞后更新偏好")
    print("=" * 70)
    
    scene = result["parsed_intent"]["scene"]
    items = result["retrieved_items"]
    
    print(f"场景: {scene}")
    print(f"搭配衣物: {len(items)} 件")
    
    # 用户点赞
    update_user_preference(
        scene=scene,
        outfit_items=items,
        is_like=True
    )
    
    print("\n偏好更新完成!")


if __name__ == "__main__":
    main()
