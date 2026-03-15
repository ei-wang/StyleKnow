# -*- coding: utf-8 -*-
"""
LangGraph Agent 节点函数集合

基于 ReAct 和 Supervisor 模式的独立 Agent 节点：

- router_node: 意图路由（Supervisor）
- direct_action_node: 直接处理（wardrobe_add / chat）
- stylist_agent_node: 核心穿搭生成师（ReAct 模式）
- critic_agent_node: 评估师（质量把控）

核心变化：
- 使用 LangGraph 原生的 add_messages Reducer
- 工具通过 llm.bind_tools() 绑定
- Pydantic Schema 进行结构化输出
"""

import sys
import os
from typing import List, Dict, Any, Optional

# 确保可以导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.state import GraphState
from agent.ultis import get_llm, get_wardrobe_db
from agent.prompts import (
    build_intent_recognition_prompt,
    RouterOutput,
    CriticOutput,
    IntentType
)
from agent.tools import (
    search_xhs_tool,
    search_wardrobe_tool,
    update_preference_tool,
    get_user_preference_tool,
)
from agent.weather_tools import AGENT_WEATHER_TOOLS
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# ========== System Prompt ==========
STYLIST_SYSTEM_PROMPT = """你是一位专业的时尚造型师助手。

你的核心任务是：根据用户的需求，从衣柜中挑选合适的衣物，生成个性化的穿搭建议。

【工作流程】
1. 理解用户需求（场景、风格、天气等）
2. 必要时调用工具获取信息（天气、小红书灵感、衣柜检索）
3. 结合用户偏好和可用衣物，生成穿搭建议

【可用工具】
- search_xhs_tool: 搜索小红书热门穿搭，获取灵感
- search_wardrobe_tool: 检索用户衣柜中的衣物
- get_weather_tool: 获取天气信息（需要知道用户在哪个城市）
- update_preference_tool: 当用户表达喜好/厌恶时，记录到偏好中

【重要规则】
- 先了解用户需求，再决定调用哪些工具
- 可以一次调用多个工具收集信息
- 收集完信息后，给出具体的穿搭建议
- 如果衣柜缺少某品类，明确告知用户

【回复要求】
- 语气专业但亲切
- 给出具体的搭配方案
- 指出每件衣服的特点和搭配理由"""


# ========== 节点1: 路由节点 (Router) ==========
def router_node(state: GraphState) -> Dict[str, Any]:
    """
    路由节点 - 识别用户意图并决定后续处理方式
    
    使用 Pydantic Schema 进行结构化输出。
    
    Args:
        state: GraphState，包含 messages
        
    Returns:
        更新后的状态，包含 current_intent
    """
    messages = state.get("messages", [])
    
    # 获取最后一条人类消息
    user_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break
    
    if not user_message:
        # 没有人类消息，使用默认意图
        return {"current_intent": "chat"}
    
    # 构建提示词
    prompt_text = build_intent_recognition_prompt(
        user_query=user_message,
        has_image=False,  # TODO: 处理图片
        conversation_history=[]
    )
    
    # 调用 LLM 进行意图识别
    llm = get_llm()
    
    if llm is None:
        # Mock 模式
        intent = _mock_route_intent(user_message)
        return {"current_intent": intent}
    
    try:
        # 使用 Pydantic Schema 进行结构化输出
        llm_with_structured = llm.with_structured_output(RouterOutput)
        
        response = llm_with_structured.invoke([
            SystemMessage(content=prompt_text)
        ])
        
        intent = response.intent.value if hasattr(response.intent, 'value') else response.intent
        
        print(f"[Router] 识别意图: {intent}, 理由: {response.reason}")
        
        return {"current_intent": intent}
        
    except Exception as e:
        print(f"[Router] 意图识别失败: {e}")
        return {"current_intent": "chat"}


def _mock_route_intent(user_message: str) -> str:
    """Mock 意图路由"""
    user_lower = user_message.lower()
    
    if any(kw in user_lower for kw in ["添加", "入库", "上传", "新衣服"]):
        return "wardrobe_add"
    elif any(kw in user_lower for kw in ["不喜欢", "偏好", "喜欢", "不要"]):
        return "update_preference"
    elif any(kw in user_lower for kw in ["推荐", "穿什么", "搭配", " outfit"]):
        return "recommend"
    else:
        return "chat"


# ========== 节点2: 直接处理节点 (Direct Action) ==========
def direct_action_node(state: GraphState) -> Dict[str, Any]:
    """
    直接处理节点 - 处理 wardrobe_add、update_preference、chat 意图
    
    不需要复杂的工具调用，直接生成回复。
    
    Args:
        state: GraphState
        
    Returns:
        更新后的状态，包含新的 AI 消息
    """
    current_intent = state.get("current_intent", "chat")
    messages = state.get("messages", [])
    
    # 获取用户消息
    user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break
    
    llm = get_llm()
    
    if llm is None:
        # Mock 模式
        response_content = _mock_direct_response(current_intent, user_message)
    else:
        try:
            # 构建提示词
            prompt = _build_direct_action_prompt(current_intent, user_message)
            
            response = llm.invoke([
                SystemMessage(content="你是一个友好的穿搭助手，请简洁回复用户。"),
                HumanMessage(content=prompt)
            ])
            response_content = response.content
            
        except Exception as e:
            print(f"[DirectAction] 生成回复失败: {e}")
            response_content = _mock_direct_response(current_intent, user_message)
    
    # 返回 AI 消息
    return {
        "messages": [AIMessage(content=response_content)]
    }


def _build_direct_action_prompt(intent: str, user_message: str) -> str:
    """构建直接操作的提示词"""
    if intent == "wardrobe_add":
        return f"""用户想要添加衣物到衣柜。请回复用户：
- 确认收到添加衣物的请求
- 引导用户上传衣物图片或提供详细信息
- 可以给出示例格式

用户消息: {user_message}"""
    
    elif intent == "update_preference":
        return f"""用户想要更新个人偏好。请回复用户：
- 确认收到偏好更新
- 可以根据用户消息提取具体偏好并告知已记录

用户消息: {user_message}"""
    
    else:  # chat
        return f"""用户只是闲聊或询问问题。请友好回复，可以适当引导到穿搭相关话题。

用户消息: {user_message}"""


def _mock_direct_response(intent: str, user_message: str) -> str:
    """Mock 直接处理回复"""
    if intent == "wardrobe_add":
        return "好的，请上传您要添加的衣物图片，我会帮您分析并入库。"
    elif intent == "update_preference":
        return "好的，我已经记下您的偏好了，后续推荐会优先考虑您的喜好。"
    else:
        return "有什么穿搭问题我可以帮您解答的吗？"


# ========== 节点3: 穿搭设计师节点 (Stylist Agent) ==========
def stylist_agent_node(state: GraphState) -> Dict[str, Any]:
    """
    核心穿搭生成 Agent - 使用 ReAct 模式
    
    绑定工具，让 LLM 自行决定调用哪些工具来收集信息，
    直到收集到足够信息后生成穿搭建议。
    
    Args:
        state: GraphState
        
    Returns:
        更新后的状态，包含 AI 消息（可能包含 tool_calls）
    """
    messages = state.get("messages", [])
    user_preferences = state.get("user_preferences", {})
    
    # 获取用户消息
    user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break
    
    llm = get_llm()
    
    if llm is None:
        # Mock 模式
        response_content = _mock_stylist_response(user_message, user_preferences)
        return {"messages": [AIMessage(content=response_content)]}
    
    try:
        # 合并所有工具
        all_tools = [
            search_xhs_tool,
            search_wardrobe_tool,
            update_preference_tool,
            get_user_preference_tool,
        ] + AGENT_WEATHER_TOOLS
        
        # 绑定工具到 LLM
        llm_with_tools = llm.bind_tools(all_tools)
        
        # 构建上下文
        context_messages = [
            SystemMessage(content=STYLIST_SYSTEM_PROMPT)
        ]
        
        # 如果有用户偏好，添加到上下文
        if user_preferences:
            prefs_str = ", ".join([f"{k}={v}" for k, v in user_preferences.items()])
            context_messages.append(SystemMessage(
                content=f"【用户偏好】{prefs_str}"
            ))
        
        # 添加历史消息
        context_messages.extend(messages)
        
        # 调用 LLM（可能返回 tool_calls 或直接回复）
        response = llm_with_tools.invoke(context_messages)
        
        print(f"[Stylist] 响应类型: {type(response).__name__}")
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"[Stylist] 调用工具: {[tc['name'] for tc in response.tool_calls]}")
        
        # 返回响应（包含 tool_calls）
        return {"messages": [response]}
        
    except Exception as e:
        print(f"[Stylist] 生成失败: {e}")
        return {
            "messages": [AIMessage(content="抱歉，生成穿搭建议时遇到问题，请稍后重试。")]
        }


def _mock_stylist_response(user_message: str, user_preferences: str) -> str:
    """Mock 设计师回复"""
    return f"""根据您的需求，我为您推荐以下穿搭方案：

上衣: 白色简约衬衫 + 黑色西装外套
下装: 深蓝色直筒牛仔裤
鞋子: 黑色乐福鞋
配饰: 简约手表

搭配理由：
- 白色衬衫提亮整体色调
- 西装外套增加专业感
- 深蓝牛仔裤平衡正式与休闲
- 整体风格简约干练，适合通勤"""


# ========== 节点4: 评估节点 (Critic Agent) ==========
def critic_agent_node(state: GraphState) -> Dict[str, Any]:
    """
    评估节点 - 审核穿搭方案
    
    读取生成的穿搭建议，使用 Pydantic Schema 进行结构化评估。
    如果不通过，返回包含修改意见的消息，触发重试。
    
    Args:
        state: GraphState
        
    Returns:
        更新后的状态，可能包含新的修改指令消息
    """
    messages = state.get("messages", [])
    iterations = state.get("iterations", 0)
    
    # 获取最后一条 AI 消息（穿搭建议）
    outfit_proposal = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            # 排除包含 tool_calls 的消息
            if not hasattr(msg, 'tool_calls') or not msg.tool_calls:
                outfit_proposal = msg.content
                break
    
    if not outfit_proposal:
        # 没有穿搭建议，跳过评估
        return {"messages": [], "iterations": iterations}
    
    # 获取用户偏好
    user_preferences = state.get("user_preferences", {})
    
    llm = get_llm()
    
    if llm is None:
        # Mock 模式
        is_pass, feedback = _mock_critic_evaluate(iterations)
    else:
        try:
            # 构建评估提示词
            prompt = _build_critic_prompt(outfit_proposal, user_preferences)
            
            # 使用 Pydantic Schema 进行结构化输出
            llm_with_structured = llm.with_structured_output(CriticOutput)
            
            response = llm_with_structured.invoke([
                SystemMessage(content="你是一位严格的时尚评论家，请评估以下穿搭方案。"),
                HumanMessage(content=prompt)
            ])
            
            is_pass = response.is_pass
            feedback = response.feedback_reason
            
            print(f"[Critic] 评估结果: {'通过' if is_pass else '需修改'}")
            print(f"[Critic] 反馈: {feedback}")
            
        except Exception as e:
            print(f"[Critic] 评估失败: {e}")
            # 评估失败时默认通过
            is_pass = True
            feedback = ""
    
    if is_pass:
        # 评估通过，不做修改
        return {"messages": [], "iterations": iterations}
    else:
        # 评估不通过，返回修改指令
        iterations += 1
        feedback_msg = HumanMessage(
            content=f"【修改意见】上一轮穿搭方案需要修改：{feedback}\n请重新生成穿搭建议。"
        )
        
        return {
            "messages": [feedback_msg],
            "iterations": iterations
        }


def _build_critic_prompt(outfit_proposal: str, user_preferences: Dict) -> str:
    """构建评估提示词"""
    prefs_text = ""
    if user_preferences:
        prefs_text = f"\n【用户偏好】{user_preferences}"
    
    return f"""请评估以下穿搭方案是否存在问题：

【穿搭方案】
{outfit_proposal}
{prefs_text}

请检查以下方面：
1. 颜色搭配是否协调
2. 风格是否统一
3. 场合是否合适
4. 是否有明显的搭配漏洞

请直接给出评估结果，不需要 JSON 格式。"""


def _mock_critic_evaluate(iterations: int) -> tuple:
    """Mock 评估结果"""
    import random
    
    # 前两次迭代随机返回修改，第三次强制通过
    if iterations < 2 and random.random() < 0.5:
        return False, "颜色搭配可以更活泼一些，建议添加一些亮色配饰。"
    else:
        return True, ""


# ========== 工具执行节点 ==========
def tool_executor_node(state: GraphState) -> Dict[str, Any]:
    """
    工具执行节点 - 执行 Stylist Agent 返回的 tool_calls
    
    当 Stylist Agent 返回 tool_calls 时，LangGraph 会自动调用此节点
    执行工具并将结果添加到消息中。
    
    Args:
        state: GraphState
        
    Returns:
        更新后的状态，包含工具执行结果
    """
    messages = state.get("messages", [])
    
    # 获取最后一条 AI 消息
    last_ai_msg = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            last_ai_msg = msg
            break
    
    if not last_ai_msg or not hasattr(last_ai_msg, 'tool_calls') or not last_ai_msg.tool_calls:
        # 没有工具调用，直接返回
        return {}
    
    # 合并所有工具
    all_tools = {
        search_xhs_tool.name: search_xhs_tool,
        search_wardrobe_tool.name: search_wardrobe_tool,
        update_preference_tool.name: update_preference_tool,
        get_user_preference_tool.name: get_user_preference_tool,
    }
    for wt in AGENT_WEATHER_TOOLS:
        all_tools[wt.name] = wt
    
    # 执行工具调用
    tool_results = []
    
    for tool_call in last_ai_msg.tool_calls:
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        
        print(f"[ToolExecutor] 执行工具: {tool_name}")
        print(f"[ToolExecutor] 参数: {tool_args}")
        
        tool_func = all_tools.get(tool_name)
        
        if tool_func:
            try:
                result = tool_func.invoke(tool_args)
                print(f"[ToolExecutor] 结果: {str(result)[:100]}...")
                tool_results.append({
                    "tool_call_id": tool_call.get("id"),
                    "result": str(result)
                })
            except Exception as e:
                print(f"[ToolExecutor] 工具执行失败: {e}")
                tool_results.append({
                    "tool_call_id": tool_call.get("id"),
                    "result": f"工具执行失败: {str(e)}"
                })
        else:
            print(f"[ToolExecutor] 未找到工具: {tool_name}")
    
    # 返回工具结果（作为新的 AI 消息）
    if tool_results:
        result_contents = []
        for tr in tool_results:
            result_contents.append(tr["result"])
        
        return {
            "messages": [AIMessage(content="\n\n".join(result_contents))]
        }
    
    return {}
