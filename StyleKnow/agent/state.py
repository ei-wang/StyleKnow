# -*- coding: utf-8 -*-
"""
GraphState 定义

LangGraph 工作流的状态类型定义
引入 LangGraph 原生的消息归约机制（Reducer）
"""

from typing import TypedDict, Annotated, Dict, Any, Optional, List
from enum import Enum

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage


class IntentEnum(str, Enum):
    """用户意图枚举"""
    RECOMMEND = "recommend"           # 穿搭推荐
    WARDROBE_ADD = "wardrobe_add"     # 添加衣物
    UPDATE_PREFERENCE = "update_preference"  # 更新偏好
    CHAT = "chat"                     # 闲聊/通用问答


class GraphState(TypedDict):
    """
    LangGraph 工作流状态定义
    
    使用 LangGraph 原生的 add_messages Reducer 实现消息自动归约，
    支持多轮对话和状态恢复。
    
    字段说明：
    - messages: 对话消息列表（使用 add_messages Reducer）
    - user_preferences: 用户长期偏好（从数据库加载）
    - current_intent: 当前路由意图
    - draft_outfit: 生成的草稿方案
    - iterations: 反思迭代次数
    """
    
    # 核心对话流 - 使用 add_messages Reducer
    # LangGraph 会自动处理消息的追加和替换
    messages: Annotated[list[BaseMessage], add_messages]
    
    # 长期记忆 - 用户偏好（从数据库加载）
    user_preferences: Dict[str, Any]
    
    # 路由意图：recommend / wardrobe_add / update_preference / chat
    current_intent: str
    
    # 生成的草稿方案
    draft_outfit: str
    
    # 反思迭代次数
    iterations: int


# ========== 辅助函数 ==========
def create_human_message(content: str) -> HumanMessage:
    """创建人类消息"""
    return HumanMessage(content=content)


def create_ai_message(content: str) -> AIMessage:
    """创建 AI 消息"""
    return AIMessage(content=content)


def create_system_message(content: str) -> SystemMessage:
    """创建系统消息"""
    return SystemMessage(content=content)
