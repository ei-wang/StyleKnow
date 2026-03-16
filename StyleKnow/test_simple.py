# -*- coding: utf-8 -*-
"""
简化版多轮对话测试
"""
import sys

# 确保 UTF-8 输出
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from agent.graph import run_workflow


def test_simple():
    """简单测试"""
    print("=" * 50)
    print("【测试开始】")
    print("=" * 50)

    # 测试单轮
    print("\n>>> 测试 1: 初始推荐")
    result1 = run_workflow(
        user_query="明天去上班，怎么穿搭？",
        thread_id="test_001",
        user_id="user_001"
    )
    print(f"结果: {result1.get('final_response', '无')[:200]}...")

    # 测试偏好更新
    print("\n>>> 测试 2: 更新偏好")
    result2 = run_workflow(
        user_query="我喜欢蓝色的衣服",
        thread_id="test_001",
        user_id="user_001"
    )
    print(f"结果: {result2.get('final_response', '无')[:200]}...")

    # 测试查询偏好
    print("\n>>> 测试 3: 查询偏好")
    result3 = run_workflow(
        user_query="我的偏好是什么？",
        thread_id="test_001",
        user_id="user_001"
    )
    print(f"结果: {result3.get('final_response', '无')[:200]}...")

    print("\n" + "=" * 50)
    print("【测试完成】")
    print("=" * 50)


if __name__ == "__main__":
    test_simple()
