# -*- coding: utf-8 -*-
"""
LangGraph 边（Edge）路由函数集合

基于 Supervisor 模式的条件路由：

- route_after_router: router 节点后的路由
- route_after_stylist: stylist 节点后的路由
- route_after_critic: critic 节点后的路由
"""

from typing import Literal

from langchain_core.messages import AIMessage

from agent.state import GraphState


# ========== 常量定义 ==========
MAX_ITERATIONS: int = 3  # 最大迭代次数


# ========== 条件边1: router 之后的路由 ==========
def route_after_router(state: GraphState) -> Literal["stylist", "direct_action"]:
    """
    Router 节点之后的路由
    
    根据 current_intent 决定后续处理方式：
    - recommend -> stylist (需要生成穿搭方案)
    - wardrobe_add / update_preference / chat -> direct_action (直接处理)
    
    Args:
        state: GraphState
        
    Returns:
        目标节点名称
    """
    intent = state.get("current_intent", "")
    
    print(f"[Edge: route_after_router] current_intent={intent}")
    
    if intent == "recommend":
        print("[Edge] -> stylist (生成穿搭方案)")
        return "stylist"
    else:
        print("[Edge] -> direct_action (直接处理)")
        return "direct_action"


# ========== 条件边2: stylist 之后的路由 ==========
def route_after_stylist(state: GraphState) -> Literal["tools", "critic"]:
    """
    Stylist 节点之后的路由
    
    检查最后一条 AIMessage 是否有 tool_calls：
    - 有 tool_calls -> tools (执行工具)
    - 无 tool_calls -> critic (评估生成结果)
    
    Args:
        state: GraphState
        
    Returns:
        目标节点名称
    """
    messages = state.get("messages", [])
    
    # 获取最后一条 AI 消息
    last_ai_msg = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            last_ai_msg = msg
            break
    
    if last_ai_msg and hasattr(last_ai_msg, 'tool_calls') and last_ai_msg.tool_calls:
        tool_names = [tc.get('name') for tc in last_ai_msg.tool_calls]
        print(f"[Edge: route_after_stylist] 有 tool_calls: {tool_names}")
        print("[Edge] -> tools (执行工具)")
        return "tools"
    else:
        print("[Edge: route_after_stylist] 无 tool_calls，直接进入评估")
        print("[Edge] -> critic (评估)")
        return "critic"


# ========== 条件边3: critic 之后的路由 ==========
def route_after_critic(state: GraphState) -> Literal["end", "stylist"]:
    """
    Critic 节点之后的路由
    
    逻辑：
    - 评估通过 -> END
    - 评估未通过且迭代次数 < 3 -> stylist (重试)
    - 评估未通过且迭代次数 >= 3 -> END (强制结束)
    
    Args:
        state: GraphState
        
    Returns:
        目标节点名称
    """
    messages = state.get("messages", [])
    iterations = state.get("iterations", 0)
    
    # 获取 critic 的反馈
    # 如果 critic 返回了修改意见（HumanMessage），说明未通过
    has_feedback = False
    for msg in reversed(messages):
        from langchain_core.messages import HumanMessage
        if isinstance(msg, HumanMessage) and "【修改意见】" in msg.content:
            has_feedback = True
            print(f"[Edge: route_after_critic] 收到修改意见: {msg.content[:50]}...")
            break
    
    if has_feedback:
        if iterations >= MAX_ITERATIONS:
            print(f"[Edge] -> END (达到最大迭代次数 {MAX_ITERATIONS})")
            return "end"
        else:
            print(f"[Edge] -> stylist (重试，迭代次数: {iterations})")
            return "stylist"
    else:
        print("[Edge] -> END (评估通过)")
        return "end"
