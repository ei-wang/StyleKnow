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
from agent.tools import update_preference_tool

# 运行工作流
result = run_workflow("明天去公司开会，帮我搭配", user_id="user_001")
print(result["final_response"])
```
"""

# State
from agent.state import GraphState

# Nodes
from agent.nodes import (
    router_node,
    direct_action_node,
    stylist_agent_node,
    critic_agent_node
)

# Edges
from agent.edges import (
    route_after_router,
    route_after_stylist,
    route_after_critic
)

# Graph
from agent.graph import build_workflow, run_workflow

# Tools
from agent.tools import (
    search_xhs_tool,
    search_wardrobe_tool,
    update_preference_tool,
    get_user_preference_tool,
    AGENT_TOOLS
)

# Utils
from agent.ultis import (
    get_llm,
    init_wardrobe_db,
    get_wardrobe_db,
    update_user_preference,
    parse_llm_json_response,
    clean_xhs_response
)

# Prompts
from agent.prompts import (
    build_generate_prompt,
    build_critic_system_prompt,
    build_critic_human_prompt,
    IntentType,
    RouterOutput,
    CriticOutput
)

__all__ = [
    # State
    "GraphState",

    # Nodes
    "router_node",
    "direct_action_node",
    "stylist_agent_node",
    "critic_agent_node",

    # Edges
    "route_after_router",
    "route_after_stylist",
    "route_after_critic",

    # Graph
    "build_workflow",
    "run_workflow",

    # Tools
    "search_xhs_tool",
    "search_wardrobe_tool",
    "update_preference_tool",
    "get_user_preference_tool",
    "AGENT_TOOLS",

    # Utils
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
    "IntentType",
    "RouterOutput",
    "CriticOutput",
]
