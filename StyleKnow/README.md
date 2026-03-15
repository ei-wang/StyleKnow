# AI 穿搭助手

基于 LLM 和多源检索的智能穿搭推荐系统

## 项目概述

AI 穿搭助手是一个基于 LangGraph 工作流的智能穿搭推荐 Agent。它结合了：

- 本地衣柜数据库检索（语义向量检索）
- 小红书穿搭公式提取（网络检索）
- 天气信息查询（上下文感知）
- LLM 生成与评估（Self-Correction 循环）

## 项目结构

```
w-study-ai-for-clothes/
├── api/                    # API 层
│   ├── agent.py           # Agent 主类
│   └── routes.py          # FastAPI 路由
├── core/                  # 核心模块
│   └── __init__.py       # 核心模块导出
├── tools/                 # Agent 工具
│   ├── __init__.py       # 工具集合 (@tool 封装)
│   ├── db_search.py       # 衣柜检索
│   ├── weather.py         # 天气查询
│   └── xhs_search.py     # 小红书搜索
├── agent/                 # LangGraph 工作流
│   ├── state.py           # 状态定义
│   ├── nodes.py           # 节点函数
│   ├── edges.py           # 边路由
│   ├── graph.py           # 工作流构建
│   ├── ultis.py           # 工具函数
│   └── prompts.py         # Prompt 模板
├── services/              # 外部服务
│   └── weather.py         # 和风天气 API
├── storage/               # 存储
│   └── config_store.py   # 配置存储
├── domain/                # 领域模型
│   └── config.py         # 配置模型
├── db/                   # 数据库
│   └── db.py             # 内存衣柜数据库
├── main.py                # FastAPI 入口
└── requirements.txt       # 依赖
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Keys

在 `storage/llm_config.json` 中配置必要的 API 密钥

### 3. 启动服务

```bash
python main.py
```

### 4. 使用 Agent

#### Python 代码调用

```python
from api.agent import get_agent

agent = get_agent()
result = agent.run("帮我搭配一套通勤穿搭")
print(result["final_outfit"])
```

#### REST API 调用

```bash
curl -X POST http://localhost:8000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "帮我搭配一套通勤穿搭"}'
```

## 工具列表

- `search_wardrobe`: 衣柜检索
- `add_clothing_item`: 添加衣物
- `get_wardrobe_stats`: 衣柜统计
- `get_weather_info`: 天气查询
- `get_weather_with_suggestion`: 天气+穿搭建议
- `search_city_info`: 城市搜索
- `search_xhs_notes`: 小红书搜索

## 工作流

1. **解析上下文**: 提取场景关键词
2. **检索衣柜**: 从本地数据库检索衣物
3. **生成搭配**: LLM 生成搭配文案
4. **批评评估**: LLM 评估（Self-Correction）
5. 评估不通过则返回步骤3重新生成（最多3次）
