---
title: LangGraph 基础概念深度解析
date: 2025-09-30
permalink: /ai/langgraph/basic-concepts.html
categories:
  - AI
  - LangGraph
---

# LangGraph 基础概念深度解析

## 核心架构概览

LangGraph 的核心架构基于有向图（Directed Graph）理论，将复杂的 AI 工作流抽象为图结构。理解这些基础概念是掌握 LangGraph 的关键。

```python
# LangGraph 的核心组件关系
"""
┌─────────────────────────────────────────────────┐
│                  StateGraph                      │
│  ┌──────────┐      ┌──────────┐     ┌────────┐ │
│  │  Node A  │──────>│  Node B  │────>│  END   │ │
│  └──────────┘      └──────────┘     └────────┘ │
│       ↑                  │                       │
│       │                  ↓                       │
│       └─────────────  State                      │
│                    (共享状态)                     │
└─────────────────────────────────────────────────┘
"""
```

## 1. Graph（图）

### 概念理解

图是 LangGraph 的核心数据结构，由节点（Nodes）和边（Edges）组成。每个图代表一个完整的工作流程。

```python
from langgraph.graph import StateGraph
from typing import TypedDict, Annotated
from operator import add

# 定义状态类型
class WorkflowState(TypedDict):
    messages: Annotated[List[str], add]  # 使用 add 操作符进行消息累加
    current_step: str
    context: dict

# 创建图实例
workflow = StateGraph(WorkflowState)
```

### 图的类型

1. **StateGraph**：带状态的图（最常用）
```python
# StateGraph 维护全局状态
graph = StateGraph(State)
```

2. **MessageGraph**：专门处理消息流的图
```python
from langgraph.graph import MessageGraph

# MessageGraph 是 StateGraph 的特殊化版本
graph = MessageGraph()
```

3. **子图（SubGraph）**：可以嵌套的图
```python
# 创建子图用于模块化设计
sub_workflow = StateGraph(SubState)
main_workflow.add_node("sub_process", sub_workflow.compile())
```

## 2. Node（节点）

### 节点的本质

节点是图中的执行单元，每个节点封装了特定的逻辑。节点可以是：
- 函数
- LangChain 工具
- 另一个编译后的图
- 任何可调用对象

```python
# 节点函数的标准签名
def node_function(state: State) -> State:
    """
    节点函数接收当前状态，返回更新后的状态
    """
    # 处理逻辑
    new_data = process_something(state)

    # 返回状态更新（部分更新即可）
    return {"field_to_update": new_data}

# 添加节点到图
workflow.add_node("process_node", node_function)
```

### 节点类型详解

1. **普通节点**
```python
def simple_node(state):
    # 基础处理逻辑
    result = state["input"] * 2
    return {"output": result}
```

2. **LLM 节点**
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")

def llm_node(state):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}
```

3. **工具节点**
```python
from langchain.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

def search_node(state):
    query = state["query"]
    results = search.run(query)
    return {"search_results": results}
```

4. **条件节点**
```python
def decision_node(state):
    """决策节点，不修改状态，只用于路由"""
    if state["score"] > 0.8:
        return "high_quality"
    else:
        return "needs_improvement"
```

## 3. Edge（边）

### 边的类型

边定义了节点之间的连接关系和执行流程。

1. **普通边（Direct Edge）**
```python
# A 执行完后直接执行 B
workflow.add_edge("node_a", "node_b")
```

2. **条件边（Conditional Edge）**
```python
def routing_function(state):
    """根据状态决定下一个节点"""
    if state["needs_tool"]:
        return "tool_node"
    return "llm_node"

workflow.add_conditional_edges(
    "decision_node",
    routing_function,
    {
        "tool_node": "use_tool",
        "llm_node": "use_llm"
    }
)
```

3. **入口边（Entry Point）**
```python
# 设置图的入口点
workflow.set_entry_point("start_node")
```

4. **结束边（End Edge）**
```python
from langgraph.graph import END

# 连接到结束节点
workflow.add_edge("final_node", END)
```

### 高级边配置

```python
# 带权重的边（用于概率路由）
def weighted_routing(state):
    import random
    rand = random.random()
    if rand < 0.7:
        return "primary_path"
    return "alternative_path"

# 多条件边
workflow.add_conditional_edges(
    "router",
    weighted_routing,
    {
        "primary_path": "main_process",
        "alternative_path": "backup_process",
        "error": "error_handler"  # 错误处理路径
    }
)
```

## 4. State（状态）

### 状态管理机制

状态是 LangGraph 的核心创新之一，它在整个图的执行过程中被传递和更新。

```python
from typing import TypedDict, Annotated
from operator import add

class State(TypedDict):
    # 简单字段：直接覆盖
    current_step: str

    # 累加字段：使用 Annotated 和操作符
    messages: Annotated[List[str], add]

    # 复杂字段：自定义合并逻辑
    metadata: dict
```

### 状态更新策略

1. **覆盖更新**
```python
def override_node(state):
    return {"field": "new_value"}  # 完全覆盖
```

2. **累加更新**
```python
class State(TypedDict):
    items: Annotated[List[str], add]

def append_node(state):
    return {"items": ["new_item"]}  # 添加到列表
```

3. **自定义更新**
```python
def custom_reducer(old, new):
    """自定义合并逻辑"""
    return {**old, **new, "updated_at": datetime.now()}

class State(TypedDict):
    data: Annotated[dict, custom_reducer]
```

### 状态的生命周期

```python
# 状态在整个执行过程中的流转
"""
初始状态 → Node A (修改) → Node B (读取) → Node C (修改) → 最终状态
    ↑                           ↓                      ↑
    └──────── 持久化存储（可选）──────────────────────┘
"""

# 实现状态检查点
from langgraph.checkpoint import MemorySaver

# 创建检查点保存器
checkpointer = MemorySaver()

# 编译时启用检查点
app = workflow.compile(checkpointer=checkpointer)

# 运行时可以恢复状态
config = {"configurable": {"thread_id": "conversation_1"}}
result = app.invoke(initial_state, config)
```

## 5. Channels（通道）

### 通道概念

通道是状态中字段的抽象，定义了如何更新和合并状态。

```python
from langgraph.channels import BinaryChannel, LastValue, Context

class AdvancedState(TypedDict):
    # LastValue: 保留最后一个值
    current: Annotated[str, LastValue]

    # Context: 上下文累积
    history: Annotated[List, Context]

    # BinaryChannel: 二进制操作
    flags: Annotated[int, BinaryChannel]
```

### 自定义通道

```python
class CustomChannel:
    def __init__(self):
        self.values = []

    def update(self, new_value):
        # 自定义更新逻辑
        if new_value not in self.values:
            self.values.append(new_value)
        return self.values

    def get(self):
        return self.values
```

## 6. Checkpointer（检查点）

### 检查点机制

检查点允许保存和恢复图的执行状态，实现持久化和故障恢复。

```python
from langgraph.checkpoint import MemorySaver, SqliteSaver

# 内存检查点（开发测试）
memory_checkpointer = MemorySaver()

# SQLite 检查点（生产环境）
sqlite_checkpointer = SqliteSaver.from_conn_string("checkpoint.db")

# 使用检查点
app = workflow.compile(checkpointer=sqlite_checkpointer)

# 保存执行状态
config = {"configurable": {"thread_id": "session_123"}}
state = app.invoke(input_data, config)

# 后续可以恢复执行
resumed_state = app.invoke(None, config)  # 从上次状态继续
```

### 检查点的应用场景

1. **长时间运行的工作流**
```python
# 定期保存进度
for step in long_running_steps:
    state = app.invoke(step_input, config)
    # 自动保存检查点
```

2. **人工审批流程**
```python
# 执行到需要审批的节点
state = app.invoke(initial_input, config)

# 等待人工审批...
human_decision = get_human_input()

# 从检查点恢复并继续
state = app.invoke({"decision": human_decision}, config)
```

## 完整示例：综合运用

```python
from typing import TypedDict, Annotated, List
from operator import add
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver
from langchain_openai import ChatOpenAI

# 1. 定义状态
class ChatState(TypedDict):
    messages: Annotated[List[str], add]
    turn_count: int
    should_end: bool

# 2. 定义节点
llm = ChatOpenAI(model="gpt-4")

def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm.invoke(messages)
    turn_count = state.get("turn_count", 0) + 1

    should_end = turn_count >= 5 or "goodbye" in response.content.lower()

    return {
        "messages": [response],
        "turn_count": turn_count,
        "should_end": should_end
    }

def router(state: ChatState):
    if state["should_end"]:
        return "end"
    return "continue"

# 3. 构建图
workflow = StateGraph(ChatState)

# 4. 添加节点
workflow.add_node("chat", chat_node)

# 5. 设置入口
workflow.set_entry_point("chat")

# 6. 添加条件边
workflow.add_conditional_edges(
    "chat",
    router,
    {
        "continue": "chat",
        "end": END
    }
)

# 7. 编译with检查点
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# 8. 执行
config = {"configurable": {"thread_id": "chat_session_1"}}
initial_state = {
    "messages": ["Hello! How can I help you today?"],
    "turn_count": 0,
    "should_end": False
}

result = app.invoke(initial_state, config)
print(f"Final state: {result}")
```

## 调试和可视化

```python
# 可视化图结构
from IPython.display import Image, display

# 生成图的可视化
display(Image(app.get_graph().draw_mermaid_png()))

# 调试模式
import logging
logging.basicConfig(level=logging.DEBUG)

# 追踪执行过程
for step in app.stream(initial_state, config):
    print(f"Step: {step}")
```

## 总结

LangGraph 的基础概念构成了一个完整的图计算框架：

- **Graph** 定义整体结构
- **Node** 执行具体任务
- **Edge** 控制执行流程
- **State** 维护共享数据
- **Channel** 管理状态更新
- **Checkpointer** 提供持久化支持

理解这些概念及其相互关系，是构建复杂 AI 应用的基础。在下一篇文章中，我们将深入探讨状态管理的高级特性。