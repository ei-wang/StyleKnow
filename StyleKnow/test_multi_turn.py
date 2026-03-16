# -*- coding: utf-8 -*-
"""
多轮对话测试脚本

测试 LangGraph Checkpoint 的多轮对话能力：
- 相同 thread_id 会自动恢复之前的状态
- 支持追问、修改等连续对话场景
- 测试多租户隔离（user_id 分流）
"""
import sys
import io

# 修复 Windows 控制台编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 强制使用 Mock 模式（不调用真实 LLM）
import os
os.environ["OPENAI_API_KEY"] = ""  # 空 API key 强制使用 mock

from agent.graph import run_workflow


def test_multi_turn_conversation():
    """测试多轮对话（单用户）"""
    thread_id = "user_session_001"
    user_id = "user_001"

    # 第一轮：初始推荐
    print("\n" + "="*70)
    print("【第一轮对话】初始推荐 - 明天去上班")
    print("="*70)

    result1 = run_workflow(
        user_query="明天去上班，怎么穿搭？",
        thread_id=thread_id,
        user_id=user_id
    )

    print("\n>>> 第一轮结果:")
    print(result1.get("final_response", "")[:500] if result1.get("final_response") else "无结果")
    print(f"\n当前用户: {result1.get('user_id', '')}")

    # 第二轮：追问/修改
    print("\n" + "="*70)
    print("【第二轮对话】追问 - 想要更年轻一点")
    print("="*70)

    result2 = run_workflow(
        user_query="能不能搭配得更年轻时尚一点？",
        thread_id=thread_id,
        user_id=user_id
    )

    print("\n>>> 第二轮结果:")
    print(result2.get("final_response", "")[:500] if result2.get("final_response") else "无结果")

    # 第三轮：场景变化
    print("\n" + "="*70)
    print("【第三轮对话】场景变化 - 周末出游")
    print("="*70)

    result3 = run_workflow(
        user_query="周末出去玩也想这样穿",
        thread_id=thread_id,
        user_id=user_id
    )

    print("\n>>> 第三轮结果:")
    print(result3.get("final_response", "")[:500] if result3.get("final_response") else "无结果")

    # 第四轮：更新偏好
    print("\n" + "="*70)
    print("【第四轮对话】更新偏好 - 我不喜欢黑色")
    print("="*70)

    result4 = run_workflow(
        user_query="我不喜欢黑色的衣服，以后推荐尽量避免",
        thread_id=thread_id,
        user_id=user_id
    )

    print("\n>>> 第四轮结果:")
    print(result4.get("final_response", "")[:500] if result4.get("final_response") else "无结果")

    print("\n" + "="*70)
    print("【单用户多轮对话测试完成】")
    print("="*70)


def test_multi_tenant_isolation():
    """测试多租户隔离"""
    print("\n\n" + "="*70)
    print("【多租户隔离测试】")
    print("="*70)

    # 用户 A 的会话
    thread_id_a = "user_a_session"
    user_id_a = "user_A"

    # 用户 B 的会话
    thread_id_b = "user_b_session"
    user_id_b = "user_B"

    # 用户 A 更新偏好
    print("\n>>> 用户 A 更新偏好：喜欢蓝色")
    result_a1 = run_workflow(
        user_query="我喜欢蓝色的衣服",
        thread_id=thread_id_a,
        user_id=user_id_a
    )
    print(f"用户 A 回复: {result_a1.get('final_response', '')[:200]}...")

    # 用户 B 更新偏好
    print("\n>>> 用户 B 更新偏好：喜欢红色")
    result_b1 = run_workflow(
        user_query="我喜欢红色的衣服",
        thread_id=thread_id_b,
        user_id=user_id_b
    )
    print(f"用户 B 回复: {result_b1.get('final_response', '')[:200]}...")

    # 用户 A 查询偏好
    print("\n>>> 用户 A 查询偏好")
    result_a2 = run_workflow(
        user_query="我的偏好是什么？",
        thread_id=thread_id_a,
        user_id=user_id_a
    )
    print(f"用户 A 偏好: {result_a2.get('final_response', '')}")

    # 用户 B 查询偏好
    print("\n>>> 用户 B 查询偏好")
    result_b2 = run_workflow(
        user_query="我的偏好是什么？",
        thread_id=thread_id_b,
        user_id=user_id_b
    )
    print(f"用户 B 偏好: {result_b2.get('final_response', '')}")

    print("\n" + "="*70)
    print("【多租户隔离测试完成】")
    print("="*70)


if __name__ == "__main__":
    test_multi_turn_conversation()
    test_multi_tenant_isolation()
