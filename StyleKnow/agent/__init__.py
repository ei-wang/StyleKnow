# -*- coding: utf-8 -*-
"""
Agent 模块

LangGraph 工作流组件

文件结构：
- state.py   : GraphState 类型定义
- prompts.py : LLM Prompt 模板
- tools.py   : 工具函数（LLM、DB、偏好更新）
- nodes.py   : LangGraph 节点函数
- edges.py   : 边路由函数
- graph.py   : 工作流构建器

使用示例：
```python
from agent.graph import run_workflow
from agent.tools import update_user_preference

# 运行工作流
result = run_workflow("明天去公司开会，帮我搭配")
print(result["final_outfit"])

# 用户点赞后更新偏好
update_user_preference(
    scene="commute",
    outfit_items=result["retrieved_items"],
    is_like=True
)
```
"""

# State
from agent.state import GraphState

# Nodes
from agent.nodes import (
    recognize_intent_node,
    generate_search_tasks_node,
    retrieve_web_node,
    process_xhs_results_node,
    retrieve_wardrobe_node,
    generate_outfit_node,
    critic_evaluate_node
)

# Edges
from agent.edges import should_continue

# Graph
from agent.graph import build_workflow, run_workflow

# Tools (from ultis.py)
from agent.ultis import (
    get_llm,
    init_wardrobe_db,
    get_wardrobe_db,
    update_user_preference,
    parse_llm_json_response,
    clean_xhs_response as _clean_xhs_response
)

# Prompts
from agent.prompts import (
    build_generate_prompt,
    build_critic_system_prompt,
    build_critic_human_prompt
)

__all__ = [
    # State
    "GraphState",
    
    # Nodes
    "recognize_intent_node",
    "generate_search_tasks_node",
    "retrieve_web_node",
    "process_xhs_results_node",
    "retrieve_wardrobe_node",
    "generate_outfit_node",
    "critic_evaluate_node",
    
    # Edges
    "should_continue",
    
    # Graph
    "build_workflow",
    "run_workflow",
    
    # Tools
    "get_llm",
    "init_wardrobe_db",
    "get_wardrobe_db",
    "update_user_preference",
    "parse_llm_json_response",
    "clean_xhs_response",
    
    # Prompts
    "build_generate_prompt",
    "build_critic_system_prompt",
    "build_critic_human_prompt",
]
