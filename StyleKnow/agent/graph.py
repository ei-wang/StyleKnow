# -*- coding: utf-8 -*-
"""
LangGraph 工作流构建器

基于 Supervisor 模式的 StateGraph 工作流：
- router: 意图路由
- stylist: 穿搭生成 Agent (ReAct)
- tools: 工具执行节点
- critic: 评估 Agent
- direct_action: 直接处理节点

使用 LangGraph 原生的 add_messages Reducer 实现多轮对话。

支持同步和异步调用：
- run_workflow: 同步调用
- run_workflow_async: 异步调用（推荐，使用 AsyncChatOpenAI 持久化连接）
"""

import sys
import os
import asyncio
from typing import Dict, Any

# 确保可以导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

from agent.state import GraphState
from agent.nodes import (
    router_node,
    direct_action_node,
    stylist_agent_node,
    critic_agent_node
)
from agent.edges import (
    route_after_router,
    route_after_stylist,
    route_after_critic
)
from agent.tools import (
    search_xhs_tool,
    search_wardrobe_tool,
    update_preference_tool,
    get_user_preference_tool,
)
from agent.weather_tools import AGENT_WEATHER_TOOLS
from agent.ultis import init_wardrobe_db
from storage.db import get_wardrobe_db


# ========== 全局单例 ==========
_workflow_graph = None
_async_workflow_graph = None


def get_workflow_graph():
    """获取全局工作流实例（单例模式，同步版本）"""
    global _workflow_graph
    if _workflow_graph is None:
        _workflow_graph = build_workflow()
    return _workflow_graph


async def get_async_workflow_graph():
    """获取全局异步工作流实例（单例模式）

    异步版本使用异步节点函数，支持 AsyncChatOpenAI 持久化连接。
    """
    global _async_workflow_graph
    if _async_workflow_graph is None:
        _async_workflow_graph = build_async_workflow()
    return _async_workflow_graph


# ========== 工作流构建 ==========
def build_workflow() -> StateGraph:
    """
    构建 StateGraph 工作流 (Supervisor 模式)
    
    流程：
    START 
      → router (意图识别)
      → [条件边] → stylist (推荐) / direct_action (直接处理)
      
      → stylist:
          → tools (执行工具) → stylist (循环，直到不需要工具)
          → critic (评估)
          → [条件边] → END / stylist (重试)
      
      → direct_action:
          → END
    
    Returns:
        编译后的 StateGraph
    """
    
    # 1. 合并所有工具
    all_tools = [
        search_xhs_tool,
        search_wardrobe_tool,
        update_preference_tool,
        get_user_preference_tool,
    ] + AGENT_WEATHER_TOOLS
    
    # 2. 使用 ToolNode 包装工具
    tools_node = ToolNode(all_tools)
    
    # 3. 创建 Checkpoint
    checkpoint = MemorySaver()
    
    # 4. 构建 StateGraph
    workflow = StateGraph(GraphState)
    
    # 5. 添加节点
    workflow.add_node("router", router_node)
    workflow.add_node("direct_action", direct_action_node)
    workflow.add_node("stylist", stylist_agent_node)
    workflow.add_node("tools", tools_node)
    workflow.add_node("critic", critic_agent_node)
    
    # 6. 设置入口点
    workflow.set_entry_point("router")
    
    # 7. 条件边：router -> stylist / direct_action
    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "stylist": "stylist",
            "direct_action": "direct_action"
        }
    )
    
    # 8. 条件边：stylist -> tools / critic
    workflow.add_conditional_edges(
        "stylist",
        route_after_stylist,
        {
            "tools": "tools",
            "critic": "critic"
        }
    )
    
    # 9. 边：tools -> stylist（工具执行完后回到 Stylist）
    workflow.add_edge("tools", "stylist")
    
    # 10. 条件边：critic -> END / stylist
    workflow.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "end": "__end__",
            "stylist": "stylist"
        }
    )
    
    # 11. 边：direct_action -> END
    workflow.add_edge("direct_action", "__end__")
    
    # 12. 编译
    return workflow.compile(checkpointer=checkpoint)


def build_async_workflow() -> StateGraph:
    """
    构建异步 StateGraph 工作流

    与 build_workflow 相同，但使用异步节点函数，
    配合 AsyncChatOpenAI 实现持久化连接。

    Returns:
        编译后的异步 StateGraph
    """
    # 1. 合并所有工具
    all_tools = [
        search_xhs_tool,
        search_wardrobe_tool,
        update_preference_tool,
        get_user_preference_tool,
    ] + AGENT_WEATHER_TOOLS

    # 2. 使用 ToolNode 包装工具
    tools_node = ToolNode(all_tools)

    # 3. 创建 Checkpoint
    checkpoint = MemorySaver()

    # 4. 构建 StateGraph
    workflow = StateGraph(GraphState)

    # 5. 添加节点（使用异步版本）
    workflow.add_node("router", router_node)
    workflow.add_node("direct_action", direct_action_node)
    workflow.add_node("stylist", stylist_agent_node)
    workflow.add_node("tools", tools_node)
    workflow.add_node("critic", critic_agent_node)

    # 6. 设置入口点
    workflow.set_entry_point("router")

    # 7. 条件边：router -> stylist / direct_action
    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "stylist": "stylist",
            "direct_action": "direct_action"
        }
    )

    # 8. 条件边：stylist -> tools / critic
    workflow.add_conditional_edges(
        "stylist",
        route_after_stylist,
        {
            "tools": "tools",
            "critic": "critic"
        }
    )

    # 9. 边：tools -> stylist（工具执行完后回到 Stylist）
    workflow.add_edge("tools", "stylist")

    # 10. 条件边：critic -> END / stylist
    workflow.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "end": "__end__",
            "stylist": "stylist"
        }
    )

    # 11. 边：direct_action -> END
    workflow.add_edge("direct_action", "__end__")

    # 12. 编译
    return workflow.compile(checkpointer=checkpoint)


# ========== 工作流运行入口 ==========
def run_workflow(
    user_query: str,
    user_images: list = None,
    uploaded_images: list = None,
    thread_id: str = "default_thread",
    user_id: str = "default_user"
) -> Dict[str, Any]:
    """
    运行完整的工作流

    使用 LangGraph 原生的 Checkpoint 和 add_messages Reducer：
    - 相同 thread_id 自动恢复历史状态
    - messages 自动拼接
    - 无需手动管理 history

    Args:
        user_query: 用户查询
        user_images: 用户上传的图片列表（可选，暂未使用）
        uploaded_images: 预处理上传的图片信息（可选，暂未使用）
        thread_id: 线程ID，用于多轮对话记忆（默认 "default_thread"）
        user_id: 用户ID，用于多租户隔离（默认 "default_user"）

    Returns:
        包含最终结果的字典
    """
    # 初始化数据库
    init_wardrobe_db()

    # 获取工作流
    graph = get_workflow_graph()

    # 配置
    config = {
        "recursion_limit": 50,
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id
        }
    }

    print("\n" + "=" * 70)
    print(f"开始执行工作流: {user_query}")
    print(f"Thread ID: {thread_id}, User ID: {user_id}")
    print("=" * 70)

    # 偏[关键修复] 从数据库读取用户偏好（状态水合）
    # 避免每次请求都用空字典覆盖用户好
    db = get_wardrobe_db()
    prefs = db.get_user_preference(user_id)

    # 构建输入状态
    # LangGraph 会自动根据 thread_id 恢复历史消息
    input_state = {
        "messages": [HumanMessage(content=user_query)],
        "user_preferences": prefs,  # 注入真实的偏好数据
        "current_intent": "",
        "draft_outfit": "",
        "iterations": 0,
        "user_id": user_id
    }
    
    # 执行工作流
    final_state = graph.invoke(input_state, config=config)
    
    # 提取最终回复
    final_response = ""
    messages = final_state.get("messages", [])
    
    # 获取最后一条 AI 消息
    for msg in reversed(messages):
        from langchain_core.messages import AIMessage
        if isinstance(msg, AIMessage):
            # 排除 tool 调用结果
            if not hasattr(msg, 'tool_calls') or not msg.tool_calls:
                final_response = msg.content
                break
    
    return {
        "messages": messages,
        "final_response": final_response,
        "iterations": final_state.get("iterations", 0),
        "current_intent": final_state.get("current_intent", ""),
        "user_id": final_state.get("user_id", user_id)
    }


# ========== 异步工作流运行入口 ==========
async def run_workflow_async(
    user_query: str,
    user_images: list = None,
    uploaded_images: list = None,
    thread_id: str = "default_thread",
    user_id: str = "default_user"
) -> Dict[str, Any]:
    """
    异步运行完整的工作流（推荐使用）

    使用 AsyncChatOpenAI 持久化连接，大幅提升调用速度。
    支持多轮对话记忆和用户偏好。

    Args:
        user_query: 用户查询
        user_images: 用户上传的图片列表（可选，暂未使用）
        uploaded_images: 预处理上传的图片信息（可选，暂未使用）
        thread_id: 线程ID，用于多轮对话记忆（默认 "default_thread"）
        user_id: 用户ID，用于多租户隔离（默认 "default_user"）

    Returns:
        包含最终结果的字典
    """
    # 初始化数据库
    init_wardrobe_db()

    # 获取异步工作流
    graph = await get_async_workflow_graph()

    # 配置
    config = {
        "recursion_limit": 50,
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id
        }
    }

    print("\n" + "=" * 70)
    print(f"[异步模式] 开始执行工作流: {user_query}")
    print(f"Thread ID: {thread_id}, User ID: {user_id}")
    print("=" * 70)

    # 从数据库读取用户偏好
    db = get_wardrobe_db()
    prefs = db.get_user_preference(user_id)

    # 构建输入状态
    input_state = {
        "messages": [HumanMessage(content=user_query)],
        "user_preferences": prefs,
        "current_intent": "",
        "draft_outfit": "",
        "iterations": 0,
        "user_id": user_id
    }

    # 异步执行工作流
    final_state = await graph.ainvoke(input_state, config=config)

    # 提取最终回复
    final_response = ""
    messages = final_state.get("messages", [])

    # 获取最后一条 AI 消息
    for msg in reversed(messages):
        from langchain_core.messages import AIMessage
        if isinstance(msg, AIMessage):
            # 排除 tool 调用结果
            if not hasattr(msg, 'tool_calls') or not msg.tool_calls:
                final_response = msg.content
                break

    return {
        "messages": messages,
        "final_response": final_response,
        "iterations": final_state.get("iterations", 0),
        "current_intent": final_state.get("current_intent", ""),
        "user_id": final_state.get("user_id", user_id)
    }
