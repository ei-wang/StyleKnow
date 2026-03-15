# -*- coding: utf-8 -*-
"""
LangGraph 工作流构建器

包含：
- build_workflow: 构建完整的 StateGraph
- run_workflow: 运行工作流的入口函数
"""

from typing import Dict, Any

from langgraph.graph import StateGraph

from agent.state import GraphState
from agent.nodes import (
    recognize_intent_node,
    generate_search_tasks_node,
    retrieve_web_node,
    process_xhs_results_node,
    retrieve_wardrobe_node,
    generate_outfit_node,
    critic_evaluate_node
)
from agent.edges import should_continue
from agent.ultis import init_wardrobe_db
from langgraph.checkpoint.memory import MemorySaver


# ========== 全局单例：确保多轮对话共享同一个 checkpoint ==========
_workflow_graph = None


def get_workflow_graph():
    """获取全局工作流实例（单例模式）"""
    global _workflow_graph
    if _workflow_graph is None:
        _workflow_graph = build_workflow()
    return _workflow_graph


# ========== 工作流构建 ==========
def build_workflow() -> StateGraph:
    """
    构建 StateGraph 工作流
    
    流程（先小红书后衣柜）：
    START 
      → recognize_intent (意图识别)
      → generate_search_tasks (LLM生成搜索任务)
      → retrieve_web (小红书整体穿搭搜索)
      → process_xhs_results (小红书结果处理：重排序+VLM提取)
      → retrieve_wardrobe (按品类+关键词检索衣柜)
      → generate_outfit (生成搭配)
      → critic_evaluate (评估)
      → [条件边] → END / 重新生成
    
    节点说明：
    1. recognize_intent: LLM 识别用户意图（纯文字/图片+文字）
    2. generate_search_tasks: LLM 生成搜索任务（按品类分配关键词）
    3. retrieve_web: 小红书整体穿搭搜索（先执行）
    4. process_xhs_results: 小红书结果处理：重排序→top-3→VLM提取穿搭公式
    5. retrieve_wardrobe: 按品类+关键词检索衣柜（结合小红书灵感，后执行）
    6. generate_outfit: LLM 生成搭配文案
    7. critic_evaluate: LLM 评估（Self-Correction）
    
    Returns:
        编译后的 StateGraph
    """

    checkpoint = MemorySaver()

    workflow = StateGraph(GraphState)
    
    # 添加节点
    workflow.add_node("recognize_intent", recognize_intent_node)
    workflow.add_node("generate_search_tasks", generate_search_tasks_node)
    workflow.add_node("retrieve_web", retrieve_web_node)           # 小红书搜索
    workflow.add_node("process_xhs_results", process_xhs_results_node)  # 小红书结果处理
    workflow.add_node("retrieve_wardrobe", retrieve_wardrobe_node)  # 衣柜检索
    workflow.add_node("generate_outfit", generate_outfit_node)
    workflow.add_node("critic_evaluate", critic_evaluate_node)
    
    # 定义边（线性流程）
    workflow.set_entry_point("recognize_intent")
    workflow.add_edge("recognize_intent", "generate_search_tasks")
    workflow.add_edge("generate_search_tasks", "retrieve_web")
    workflow.add_edge("retrieve_web", "process_xhs_results")
    workflow.add_edge("process_xhs_results", "retrieve_wardrobe")
    workflow.add_edge("retrieve_wardrobe", "generate_outfit")
    workflow.add_edge("generate_outfit", "critic_evaluate")
    
    # 条件边：critic -> generate_outfit / end
    workflow.add_conditional_edges(
        "critic_evaluate",
        should_continue,
        {
            "generate_outfit": "generate_outfit",  # 继续迭代
            "end": "__end__"                      # 结束
        }
    )
    
    return workflow.compile(checkpointer=checkpoint)


# ========== 工作流运行入口 ==========
def run_workflow(
    user_query: str, 
    user_images: list = None,
    uploaded_images: list = None,
    thread_id: str = "default_thread"
) -> Dict[str, Any]:
    """
    运行完整的工作流
    
    使用 checkpoint 实现多轮对话持久化：
    - 相同 thread_id 会自动恢复之前的状态
    - 支持追问、修改等连续对话场景
    
    Args:
        user_query: 用户查询
        user_images: 用户上传的图片列表（可选）
        uploaded_images: 预处理上传的图片信息（可选）
        thread_id: 线程ID，用于多轮对话记忆（默认 "default_thread"）
    
    Returns:
        包含最终结果的字典
    """
    # 初始化数据库
    init_wardrobe_db()
    
    # 使用全局单例，确保多轮对话共享同一个 checkpoint
    graph = get_workflow_graph()
    
    # 配置
    config = {
        "recursion_limit": 50,
        "configurable": {
            "thread_id": thread_id
        }
    }
    
    # 检查是否有已保存的状态（多轮对话恢复）
    existing_state = graph.get_state(config)
    
    if existing_state is not None:
        # 恢复已有状态，更新用户查询
        current_state = dict(existing_state.values)
        
        # 保存上一轮对话到历史记录
        previous_query = current_state.get("user_query", "")
        previous_response = current_state.get("draft_outfit", "") or current_state.get("final_outfit", "")
        
        if previous_query and previous_response:
            history = current_state.get("conversation_history", [])
            history.append({"role": "user", "content": previous_query})
            history.append({"role": "assistant", "content": previous_response})
            current_state["conversation_history"] = history
        
        # 更新当前查询
        current_state["user_query"] = user_query
        if user_images:
            current_state["user_images"] = user_images
        if uploaded_images:
            current_state["uploaded_images"] = uploaded_images
        
        # 重置迭代相关状态，准备新一轮生成
        current_state["draft_outfit"] = ""
        current_state["critic_feedback"] = ""
        current_state["iterations"] = 0
        
        print("\n" + "=" * 70)
        print(f"【多轮对话】恢复状态，继续执行: {user_query}")
        print(f"【对话历史】{len(current_state.get('conversation_history', []))} 条记录")
        print("=" * 70)
    else:
        # 首次对话，初始化状态
        current_state: GraphState = {
            "user_query": user_query,
            "user_images": user_images or [],
            "conversation_history": [],  # 初始化对话历史
            "intent": {},
            "uploaded_images": uploaded_images or [],
            "parsed_intent": {},
            "search_tasks": [],
            "web_formulas": [],
            "xhs_extracted_outfits": [],
            "wardrobe_results": [],
            "draft_outfit": "",
            "critic_feedback": "",
            "iterations": 0,
            "final_outfit": "",
            "fallback_recommendations": [],
            "missing_recommendations": []
        }
        
        print("\n" + "=" * 70)
        print(f"开始执行工作流: {user_query}")
        print("=" * 70)
    
    # 执行（LangGraph 会自动保存 checkpoint）
    final_state = graph.invoke(current_state, config=config)
    
    # 将结果保存到 final_outfit
    final_state["final_outfit"] = final_state.get("draft_outfit", "")
    
    return final_state
