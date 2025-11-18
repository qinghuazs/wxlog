---
title: LangGraph API 参考手册
date: 2025-01-30
permalink: /ai/langgraph/api-reference.html
categories:
  - AI
  - LangGraph
---

# LangGraph API 参考手册

## 一、核心API

### 1.1 StateGraph

```python
from langgraph.graph import StateGraph

class StateGraph:
    """状态图 - LangGraph 的核心类"""

    def __init__(self, state_schema: Type[TypedDict]):
        """
        初始化状态图

        参数:
            state_schema: TypedDict 类型的状态模式
        """
        pass

    def add_node(self, name: str, action: Callable) -> None:
        """
        添加节点

        参数:
            name: 节点名称（字符串）
            action: 节点函数，接受状态并返回状态更新

        示例:
            def my_node(state: State) -> dict:
                return {"field": "value"}

            graph.add_node("my_node", my_node)
        """
        pass

    def add_edge(self, start: str, end: str) -> None:
        """
        添加边

        参数:
            start: 起始节点名称
            end: 结束节点名称或 END

        示例:
            graph.add_edge("node1", "node2")
            graph.add_edge("node2", END)
        """
        pass

    def add_conditional_edges(
        self,
        source: str,
        router: Callable,
        mapping: Dict[str, str]
    ) -> None:
        """
        添加条件边

        参数:
            source: 源节点名称
            router: 路由函数，返回映射的键
            mapping: 路由键到目标节点的映射

        示例:
            def router(state):
                if state["value"] > 10:
                    return "high"
                return "low"

            graph.add_conditional_edges(
                "check",
                router,
                {"high": "handler_a", "low": "handler_b"}
            )
        """
        pass

    def set_entry_point(self, name: str) -> None:
        """
        设置入口点

        参数:
            name: 入口节点名称

        示例:
            graph.set_entry_point("start")
        """
        pass

    def set_conditional_entry_point(
        self,
        router: Callable,
        mapping: Dict[str, str]
    ) -> None:
        """
        设置条件入口点

        参数:
            router: 路由函数
            mapping: 路由映射

        示例:
            def route_entry(state):
                return "path_a" if state["type"] == "A" else "path_b"

            graph.set_conditional_entry_point(
                route_entry,
                {"path_a": "node_a", "path_b": "node_b"}
            )
        """
        pass

    def set_finish_point(self, name: str) -> None:
        """
        设置结束点（已废弃，使用 END）

        参数:
            name: 结束节点名称
        """
        pass

    def compile(
        self,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        interrupt_before: Optional[list[str]] = None,
        interrupt_after: Optional[list[str]] = None,
        debug: bool = False
    ) -> CompiledGraph:
        """
        编译图

        参数:
            checkpointer: 检查点保存器
            interrupt_before: 在这些节点前中断
            interrupt_after: 在这些节点后中断
            debug: 启用调试模式

        返回:
            CompiledGraph: 编译后的可执行图

        示例:
            from langgraph.checkpoint.memory import MemorySaver

            app = graph.compile(
                checkpointer=MemorySaver(),
                interrupt_before=["review"]
            )
        """
        pass
```

### 1.2 CompiledGraph

```python
class CompiledGraph:
    """编译后的图"""

    def invoke(
        self,
        input: dict,
        config: Optional[dict] = None
    ) -> dict:
        """
        同步执行图

        参数:
            input: 初始状态
            config: 配置（如 thread_id）

        返回:
            dict: 最终状态

        示例:
            result = app.invoke(
                {"input": "hello"},
                config={"configurable": {"thread_id": "1"}}
            )
        """
        pass

    async def ainvoke(
        self,
        input: dict,
        config: Optional[dict] = None
    ) -> dict:
        """
        异步执行图

        参数:
            input: 初始状态
            config: 配置

        返回:
            dict: 最终状态

        示例:
            result = await app.ainvoke({"input": "hello"})
        """
        pass

    def stream(
        self,
        input: dict,
        config: Optional[dict] = None
    ) -> Iterator[dict]:
        """
        流式执行图

        参数:
            input: 初始状态
            config: 配置

        返回:
            Iterator[dict]: 每个节点的输出

        示例:
            for chunk in app.stream({"input": "hello"}):
                print(chunk)
        """
        pass

    async def astream(
        self,
        input: dict,
        config: Optional[dict] = None
    ) -> AsyncIterator[dict]:
        """
        异步流式执行图

        参数:
            input: 初始状态
            config: 配置

        返回:
            AsyncIterator[dict]: 每个节点的输出

        示例:
            async for chunk in app.astream({"input": "hello"}):
                print(chunk)
        """
        pass

    def get_state(
        self,
        config: dict
    ) -> StateSnapshot:
        """
        获取当前状态

        参数:
            config: 配置（必须包含 thread_id）

        返回:
            StateSnapshot: 当前状态快照

        示例:
            state = app.get_state(
                config={"configurable": {"thread_id": "1"}}
            )
        """
        pass

    def get_state_history(
        self,
        config: dict,
        limit: Optional[int] = None
    ) -> Iterator[StateSnapshot]:
        """
        获取状态历史

        参数:
            config: 配置
            limit: 返回的最大数量

        返回:
            Iterator[StateSnapshot]: 状态历史

        示例:
            for state in app.get_state_history(config):
                print(state.values)
        """
        pass

    def update_state(
        self,
        config: dict,
        values: dict,
        as_node: Optional[str] = None
    ) -> None:
        """
        更新状态

        参数:
            config: 配置
            values: 要更新的值
            as_node: 作为哪个节点更新

        示例:
            app.update_state(
                config={"configurable": {"thread_id": "1"}},
                values={"approved": True}
            )
        """
        pass

    def get_graph(self) -> Graph:
        """
        获取图结构

        返回:
            Graph: 图对象

        示例:
            graph_obj = app.get_graph()
            print(graph_obj.nodes)
        """
        pass
```

## 二、Checkpointer API

### 2.1 MemorySaver

```python
from langgraph.checkpoint.memory import MemorySaver

class MemorySaver:
    """内存检查点保存器"""

    def __init__(self):
        """
        初始化内存保存器

        示例:
            checkpointer = MemorySaver()
            app = graph.compile(checkpointer=checkpointer)
        """
        pass
```

### 2.2 SqliteSaver

```python
from langgraph.checkpoint.sqlite import SqliteSaver

class SqliteSaver:
    """SQLite 检查点保存器"""

    @classmethod
    def from_conn_string(cls, conn_string: str) -> "SqliteSaver":
        """
        从连接字符串创建

        参数:
            conn_string: SQLite 数据库路径

        示例:
            checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
        """
        pass
```

## 三、常量

```python
from langgraph.graph import END, START

# 特殊节点标识
END: str = "__end__"
START: str = "__start__"

# 使用示例
graph.add_edge("final_node", END)
graph.set_entry_point(START)  # 或直接使用节点名
```

## 四、工具函数

### 4.1 Send API

```python
from langgraph.constants import Send

def router(state):
    """
    使用 Send 动态创建并行任务

    示例:
        def router(state):
            return [
                Send("process", {"item": item})
                for item in state["items"]
            ]

        graph.add_conditional_edges("split", router)
    """
    return [
        Send("target_node", {"data": "value"})
    ]
```

## 五、类型定义

### 5.1 StateSnapshot

```python
from typing import TypedDict, Any

class StateSnapshot:
    """状态快照"""

    values: dict
    """当前状态值"""

    next: tuple[str, ...]
    """下一步要执行的节点"""

    config: dict
    """配置"""

    metadata: dict
    """元数据"""

    created_at: str
    """创建时间"""

    parent_config: Optional[dict]
    """父配置"""
```

## 六、常用模式速查

### 6.1 基础图

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class State(TypedDict):
    value: int

def node_func(state: State) -> dict:
    return {"value": state["value"] + 1}

graph = StateGraph(State)
graph.add_node("process", node_func)
graph.set_entry_point("process")
graph.add_edge("process", END)

app = graph.compile()
result = app.invoke({"value": 0})
```

### 6.2 条件路由

```python
def router(state: State) -> str:
    if state["value"] > 10:
        return "high"
    return "low"

graph.add_conditional_edges(
    "check",
    router,
    {"high": "handle_high", "low": "handle_low"}
)
```

### 6.3 循环图

```python
def should_continue(state: State) -> str:
    if state["count"] < 10:
        return "continue"
    return "end"

graph.add_conditional_edges(
    "process",
    should_continue,
    {"continue": "process", "end": END}
)
```

### 6.4 带检查点

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# 第一次执行
result1 = app.invoke(
    {"value": 0},
    config={"configurable": {"thread_id": "1"}}
)

# 继续执行
result2 = app.invoke(
    None,  # 从检查点继续
    config={"configurable": {"thread_id": "1"}}
)
```

### 6.5 中断与恢复

```python
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["approval"]
)

# 执行到中断点
state = app.invoke(input, config=config)

# 更新状态后继续
app.update_state(config, {"approved": True})
result = app.invoke(None, config=config)
```

### 6.6 流式处理

```python
for chunk in app.stream({"input": "data"}):
    if "node_name" in chunk:
        print(f"节点输出: {chunk['node_name']}")
```

### 6.7 并行执行

```python
# 方法1：多个入口点
graph.set_entry_point("task_a")
graph.set_entry_point("task_b")
graph.set_entry_point("task_c")

# 方法2：Send API
from langgraph.constants import Send

def parallel_router(state):
    return [
        Send("worker", {"id": i})
        for i in range(10)
    ]

graph.add_conditional_edges("split", parallel_router)
```

## 七、错误处理

### 7.1 常见错误

```python
# InvalidUpdateError
# 原因：节点返回了状态中不存在的字段
# 解决：确保返回的字段在 TypedDict 中定义

# GraphRecursionError
# 原因：图执行超过最大递归深度
# 解决：检查循环逻辑，确保有退出条件

# NodeInterrupt
# 原因：执行到了中断点
# 解决：这是预期行为，更新状态后继续执行
```

### 7.2 调试技巧

```python
# 启用调试模式
app = graph.compile(debug=True)

# 查看图结构
print(app.get_graph().nodes)
print(app.get_graph().edges)

# 查看状态历史
for state in app.get_state_history(config):
    print(state.values)

# 可视化图
mermaid_code = app.get_graph().draw_mermaid()
print(mermaid_code)
```

## 八、性能优化

```python
# 1. 使用批处理
def batch_process(state):
    # 一次处理多个项目
    results = process_batch(state["items"])
    return {"results": results}

# 2. 启用缓存
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_operation(input_data):
    return compute(input_data)

# 3. 使用连接池
# 在节点外部创建共享资源
db_pool = create_connection_pool()

def node_with_pool(state):
    with db_pool.get_connection() as conn:
        result = conn.query(state["sql"])
    return {"result": result}
```

## 九、最佳实践

1. **状态设计**
   - 保持状态最小化
   - 使用 TypedDict 定义清晰的结构
   - 对列表字段使用 Reducer

2. **节点设计**
   - 单一职责
   - 纯函数（无副作用）
   - 返回部分更新而非完整状态

3. **错误处理**
   - 在节点中捕获异常
   - 在状态中保存错误信息
   - 使用条件路由处理错误

4. **性能优化**
   - 使用并行处理
   - 添加适当的缓存
   - 限制状态大小

5. **测试**
   - 为每个节点编写单元测试
   - 测试完整的图流转
   - 使用 Mock 隔离外部依赖

---

**恭喜！** 您已完成 LangGraph 完整学习路径！🎉

**推荐下一步：**
- 实践一个完整项目
- 加入 LangGraph 社区
- 贡献开源项目
