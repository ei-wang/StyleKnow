# -*- coding: utf-8 -*-
"""
多轮对话测试脚本

测试 LangGraph Checkpoint 的多轮对话能力：
- 相同 thread_id 会自动恢复之前的状态
- 支持追问、修改等连续对话场景
"""
import sys
import io

# 修复 Windows 控制台编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from agent.graph import run_workflow


def test_multi_turn_conversation():
    """测试多轮对话"""
    thread_id = "user_session_001"
    
    # 第一轮：初始推荐
    print("\n" + "="*70)
    print("【第一轮对话】初始推荐 - 明天去上班")
    print("="*70)
    
    result1 = run_workflow(
        user_query="明天去上班，怎么穿搭？",
        thread_id=thread_id
    )
    
    print("\n>>> 第一轮结果:")
    print(result1.get("final_outfit", "")[:500] if result1.get("final_outfit") else "无结果")
    print(f"\n场景: {result1.get('parsed_intent', {}).get('scene', '')}")
    
    # 第二轮：追问/修改
    print("\n" + "="*70)
    print("【第二轮对话】追问 - 想要更年轻一点")
    print("="*70)
    
    result2 = run_workflow(
        user_query="能不能搭配得更年轻时尚一点？",
        thread_id=thread_id
    )
    
    print("\n>>> 第二轮结果:")
    print(result2.get("final_outfit", "")[:500] if result2.get("final_outfit") else "无结果")
    print(f"\n场景: {result2.get('parsed_intent', {}).get('scene', '')}")
    
    # 第三轮：场景变化
    print("\n" + "="*70)
    print("【第三轮对话】场景变化 - 周末出游")
    print("="*70)
    
    result3 = run_workflow(
        user_query="周末出去玩也想这样穿",
        thread_id=thread_id
    )
    
    print("\n>>> 第三轮结果:")
    print(result3.get("final_outfit", "")[:500] if result3.get("final_outfit") else "无结果")
    print(f"\n场景: {result3.get('parsed_intent', {}).get('scene', '')}")
    
    # 第四轮：完全不同的场景
    print("\n" + "="*70)
    print("【第四轮对话】新场景 - 参加朋友婚礼")
    print("="*70)
    
    result4 = run_workflow(
        user_query="下周参加朋友的婚礼怎么穿？",
        thread_id=thread_id
    )
    
    print("\n>>> 第四轮结果:")
    print(result4.get("final_outfit", "")[:500] if result4.get("final_outfit") else "无结果")
    print(f"\n场景: {result4.get('parsed_intent', {}).get('scene', '')}")
    
    print("\n" + "="*70)
    print("【多轮对话测试完成】")
    print("="*70)


if __name__ == "__main__":
    test_multi_turn_conversation()
