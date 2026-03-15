# -*- coding: utf-8 -*-
"""
LangGraph 边（Edge）路由函数集合

包含条件边路由逻辑：
- should_continue: critic 评估后判断走向
"""

from typing import Literal

from agent.state import GraphState


# ========== 常量定义 ==========
MAX_ITERATIONS: int = 3  # 最大迭代次数


# ========== 条件边路由 ==========
def should_continue(state: GraphState) -> Literal["generate_outfit", "end"]:
    """
    条件边路由：critic 评估之后直接判断走向
    
    逻辑：
    - 评估通过：直接结束
    - 评估未通过且迭代次数 < 3：回到 generate_outfit 继续生成
    - 评估未通过且迭代次数 >= 3：强制结束
    
    注意：此函数不应直接修改 state，迭代计数由 generate_outfit_node 维护
    
    Args:
        state: 当前 GraphState
    
    Returns:
        下一个节点名称 "generate_outfit" 或 "end"
    """
    critic_feedback = state.get("critic_feedback", "")
    iterations = state.get("iterations", 0)
    
    is_approved = "通过" in critic_feedback or "[PASS]" in critic_feedback
    
    if is_approved:
        print(f"\n[Route] 评估通过 -> END")
        return "end"
    
    if iterations >= MAX_ITERATIONS:
        print(f"\n[Route] 达到最大迭代次数({MAX_ITERATIONS}) -> END")
        return "end"
    
    # 增加迭代计数，准备下一轮
    # 注意：在 LangGraph 中，应该在目标节点中更新状态，这里返回目标节点名称
    print(f"\n[Route] 评估未通过，迭代次数={iterations} -> generate_outfit (重新生成)")
    return "generate_outfit"
