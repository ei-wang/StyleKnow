这是一份面向生产环境的、真正的**多智能体（Multi-Agent）穿搭推荐系统总体架构设计方案**。

针对你之前代码中存在的“单体线性流水线”、“状态与业务强耦合”、“伪工具调用”等问题，这套新架构引入了 **Supervisor（路由/主管）模式**、**ReAct 范式**，并深度利用了 LangGraph 的**状态归约（Reducer）**和**并行处理（Send API）**机制。同时，在底层存储上为接入 Redis 和 Milvus（向量数据库）做好了接口抽象。

---

# 👔 StyleKnow: 基于 LangGraph 的多智能体穿搭推荐系统架构设计

## 一、 核心图拓扑：Supervisor 分布式路由架构

摒弃之前的单向线性流（意图->查网->查库->生成->评估），我们采用星型或条件分支型拓扑结构。每个 Node 都是一个职责单一的独立智能体或工具执行器。

### 1. 架构拓扑图

```text
                          [ START ]
                              │
                              ▼
                      ┌───────────────┐
                      │ Router_Agent  │ (路由节点：识别意图，分配任务)
                      └───────┬───────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Direct_Action  │ │  Stylist_Agent  │<│   Chat_Agent    │
│ (直接入库/更新) │ │ (核心穿搭生成师)│>│ (闲聊与通用问答)│
└────────┬────────┘ └───────┬─▲───────┘ └────────┬────────┘
         │                  │ │                  │
         │          (调用)  │ │ (返回)           │
         │                  ▼ │                  │
         │          ┌─────────────────┐          │
         │          │    Tool_Node    │          │
         │          │(执行搜索/查天气)│          │
         │          └─────────────────┘          │
         │                  │                    │
         │                  ▼                    │
         │          ┌─────────────────┐          │
         │          │  Critic_Agent   │          │
         │          │  (评估与反思)   │          │
         │          └────────┬─▲──────┘          │
         │  (修改意见)       │ │ (重试)          │
         │ ┌─────────────────┘ │                 │
         │ │                   │                 │
         ▼ ▼                   ▼                 ▼
        [ END ]             [ END ]           [ END ]

```

### 2. 核心智能体 (Agents) 职责定义

* **Router_Agent (主管/路由)**：
* **职责**：系统的“大脑门面”。接收用户输入，利用 LLM（搭配 Structured Output）判定当前对话属于哪种工作流（`recommend` 推荐穿搭、`wardrobe_add` 衣服入库、`preference_update` 更新偏好、`casual_chat` 闲聊）。
* **输出**：决定图的下一步走向（Conditional Edge）。


* **Stylist_Agent (穿搭生成师 - ReAct 模式)**：
* **职责**：核心业务 Agent。它被绑定了多个外部工具（小红书检索、衣柜检索、天气查询）。它通过内部思考决定需要调用哪些工具、调用几次，直到收集到足够的信息后，生成最终的穿搭方案。


* **Critic_Agent (评估师)**：
* **职责**：质量把控。对 Stylist 的输出进行审核（如：颜色是否冲突、是否符合天气）。如果不通过，将反馈意见作为一条 `AIMessage` 压入状态，并让状态回流给 Stylist 重新生成。



---

## 二、 全局状态管理 (GraphState) 重构

不要手动在节点中通过 `history.append()` 拼接对话。必须拥抱 LangGraph 原生的 Reducer 机制和 Checkpoint 机制，实现多轮对话和长短期记忆的无缝切换。

```python
from typing import TypedDict, Annotated, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class GraphState(TypedDict):
    # 1. 短期记忆 (Conversation History)
    # 使用 add_messages 归约器，LangGraph 底层会自动处理消息的 append 和替换
    messages: Annotated[list[BaseMessage], add_messages]
    
    # 2. 长期记忆 (User Profile & Preferences)
    # 存储用户的静态画像、风格偏好、身型数据。这部分数据可以在每次会话开始前从数据库加载
    user_preferences: dict[str, Any]
    
    # 3. 路由与控制状态 (Control Flow)
    current_intent: str      # Router 决定的当前意图
    iterations: int          # Critic 触发的重试次数
    
    # 4. 业务数据 (Business Context)
    weather_context: dict    # 天气缓存，避免重复调用
    missing_items: list[str] # 记录缺少的品类，用于生成购买建议

```

---

## 三、 工具链解耦与动态感知 (Tools Layer)

之前的代码中，小红书检索和衣柜检索是在 Node 里面用 Python `for` 循环写死的。在新架构中，必须将它们封装为标准的 LangChain Tools，交给 `Stylist_Agent` 动态调用。

### 工具清单 (注册给 `ToolNode`)

1. **`search_xhs_tool(query: str, category: str = None)`**：获取外部灵感。
2. **`search_wardrobe_tool(keywords: list[str], category: str)`**：基于多路召回（关键词+向量）检索个人衣柜。
3. **`get_weather_tool(location: str, date: str)`**：获取环境约束。
4. **`update_preference_tool(key: str, value: str)`**：**关键工具！** 当用户在对话中说“我不喜欢暗黑风”时，Agent 可以主动调用此工具，更新 `user_preferences` 状态，并持久化到数据库。

**动态调用的优势**：Agent 会自己决定“先查天气，发现下雨，于是构造搜索词‘雨天穿搭’去查小红书，最后再查衣柜里的防水外套”。这就是 Agentic 工作流，而不是流水线。

---

## 四、 存储与检索架构设计 (RAG 混合检索引擎)

为了后续平滑迁移到生产级的 Redis 和 Milvus 架构，必须在项目初期就引入**策略模式 (Strategy Pattern)** 和**依赖注入 (DI)** 来隔离存储层。

### 1. 接口抽象层 (`storage/base.py`)

定义抽象基类，规范所有底层操作：

```python
from abc import ABC, abstractmethod

class BaseWardrobeDB(ABC):
    @abstractmethod
    def add_item(self, item_data: dict) -> str: pass
    
    @abstractmethod
    def search_hybrid(self, query_text: str, query_vector: list[float], top_k: int) -> list[dict]: 
        """混合检索：结合关键词的稀疏检索与向量的稠密检索"""
        pass
        
    @abstractmethod
    def update_user_preference(self, user_id: str, prefs: dict): pass

```

### 2. 具体实现层

* **阶段一 (当前)**：`InMemoryWardrobeDB(BaseWardrobeDB)` - 使用 Python 字典和简单的余弦相似度计算（如 `numpy`）。
* **阶段二 (未来生产级)**：
* `MilvusWardrobeDB(BaseWardrobeDB)`：利用 Milvus 存储衣物的图像 Embedding 和文本 Embedding，支持大规模的高性能向量召回。
* `RedisSessionManager`：结合 LangGraph 的 `Checkpointer` 接口，将多轮对话的 `messages` 序列化并存入 Redis，确保分布式环境下的会话一致性。



---

## 五、 性能、高并发与高可用优化

在一个成熟的 AI 应用中，必须解决检索慢和 API 容易崩溃的问题。

### 1. 利用 LangGraph 的 Map-Reduce 解决多品类检索性能瓶颈

在原来的逻辑中，查询多个品类（上衣、裤子、鞋）是串行执行的。
**优化方案**：当 `Stylist_Agent` 生成了一份包含多个品类的检索清单后，不要串行调用 Tool。使用 LangGraph 的 **`Send` API** 进行扇出（Fan-out）：

```python
# 伪代码：在生成检索任务的节点
def dispatch_searches(state: GraphState):
    tasks = state.get("search_tasks", [])
    # 动态并发启动多个检索节点
    return [Send("parallel_wardrobe_search", task) for task in tasks]

```

底层的 asyncio 会同时向 Milvus/数据库 发起并发查询，将检索耗时从 `O(N)` 降至 `O(1)`。

### 2. VLM 视觉解析任务异步化 (非阻塞架构)

处理用户上传的全身照并抠图提取特征是非常耗时的动作。
**优化方案**：绝不能让 Agent 节点等待图片解析。

* 上传接口只负责把图片存入 OSS，产生 URL，返回给前端，并在消息队列（如 Redis Stream 或 Celery）投递一个解析任务。
* 后台 Worker 异步调用 VLM 提取语义和 Vector，写入 Milvus。
* LangGraph 仅在需要时通过 `search_wardrobe_tool` 去读取已经解析好的结构化数据。

### 3. Pydantic 结构化输出与 Fallback (容错)

* **彻底消灭正则匹配 JSON**：在 `Router` 和 `Critic` 节点，强制使用 LangChain 的 `llm.with_structured_output(PydanticModel)`，这能在模型层面（如 OpenAI 的 Tool Calling 或 JSON Mode）保证输出数据结构的 100% 确定性。
* **API 降级机制**：小红书搜索等外部 API 极其不稳定。在 LangGraph 中，为调用外部 API 的 Node 添加 `fallbacks`，一旦捕获 `Timeout` 或 `APIError`，静默路由到一个备用节点（例如仅使用衣柜数据和 LLM 内部知识生成推荐）。