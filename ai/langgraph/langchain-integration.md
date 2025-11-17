---
title: LangGraph 与 LangChain 集成
date: 2025-01-30
categories:
  - AI
  - LangGraph
---

# LangGraph 与 LangChain 集成

**为什么需要集成 LangChain？**

LangGraph 和 LangChain 是互补的框架:
- **LangGraph**: 提供灵活的图结构和状态管理,擅长复杂的工作流编排
- **LangChain**: 提供丰富的组件生态(LLM、工具、记忆、链等)

将二者集成可以:
1. **借力生态**: 使用 LangChain 丰富的工具和组件库
2. **最佳组合**: LangGraph 的编排能力 + LangChain 的组件能力
3. **快速开发**: 避免重复造轮子,专注业务逻辑
4. **灵活扩展**: 轻松添加新的 LLM、工具或记忆系统

## 一、集成架构

**整体架构概览**

LangGraph 作为编排引擎,可以无缝集成 LangChain 的各种组件:

```mermaid
graph TD
    A[LangGraph 图] --> B[LangChain LLM]
    A --> C[LangChain Tools]
    A --> D[LangChain Memory]
    A --> E[LangChain Chains]
    A --> F[LangChain Agents]

    B --> B1[ChatOpenAI]
    B --> B2[ChatAnthropic]
    B --> B3[自定义 LLM]

    C --> C1[搜索工具]
    C --> C2[数据库工具]
    C --> C3[API 工具]

    D --> D1[BufferMemory]
    D --> D2[VectorStoreMemory]
    D --> D3[ConversationMemory]

    E --> E1[LLMChain]
    E --> E2[SequentialChain]
    E --> E3[RouterChain]

    style A fill:#e1f5fe
    style B fill:#b3e5fc
    style C fill:#81d4fa
    style D fill:#4fc3f7
```

**集成层次**

| 集成类型 | 复杂度 | 灵活性 | 适用场景 |
|---------|-------|-------|---------|
| **LLM 集成** | 低 | 高 | 需要调用大语言模型 |
| **Tools 集成** | 低 | 高 | 需要外部工具调用 |
| **Memory 集成** | 中 | 中 | 需要对话记忆 |
| **Chains 集成** | 中 | 中 | 复用现有链逻辑 |
| **Agents 集成** | 高 | 低 | 需要自主决策能力 |

**集成原则**

1. **状态优先**: LangGraph 的状态管理是核心,LangChain 组件作为工具使用
2. **明确边界**: 清晰定义每个组件的输入输出
3. **错误处理**: LangChain 组件可能抛出异常,需要妥善处理
4. **成本控制**: LLM 调用有成本,需要合理设计调用策略

## 二、集成 LLM

**什么是 LLM 集成？**

LLM(Large Language Model)集成是指在 LangGraph 图中使用 LangChain 提供的各种大语言模型。LangChain 支持 OpenAI、Anthropic、Google、本地模型等多种 LLM。

**为什么要集成 LLM？**

1. **统一接口**: LangChain 为不同 LLM 提供统一的调用接口
2. **易于切换**: 可以轻松切换不同的 LLM 提供商
3. **丰富功能**: 支持流式输出、函数调用、结构化输出等
4. **成本优化**: 根据任务选择合适价格的模型

**LLM 集成模式**

```mermaid
graph LR
    A[用户输入] --> B[LangGraph 节点]
    B --> C[LangChain LLM]
    C --> D[模型响应]
    D --> E[状态更新]
    E --> F[下一个节点]

    style A fill:#e1f5fe
    style B fill:#b3e5fc
    style C fill:#81d4fa
    style D fill:#4fc3f7
    style E fill:#c8e6c9
```

### 2.1 基础 LLM 集成

**使用场景**

- 问答系统
- 文本生成
- 对话机器人
- 内容摘要
- 翻译服务

**完整实现示例**

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END

# 定义聊天状态
class ChatState(TypedDict):
    messages: Annotated[list, operator.add]  # 消息历史(使用 Reducer 累积)
    llm_response: str                         # LLM 响应

def create_llm_graph():
    """
    集成 LangChain LLM 的图

    特点:
    1. 使用 LangChain 的 ChatOpenAI
    2. 支持消息历史
    3. 自动状态管理
    """
    # 初始化 LLM
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.7  # 控制输出随机性
    )

    def chat_node(state: ChatState) -> dict:
        """
        使用 LangChain LLM 的节点

        功能:
        1. 从状态获取消息历史
        2. 调用 LLM 生成响应
        3. 更新状态
        """
        # 调用 LLM
        response = llm.invoke(state["messages"])

        print(f"[LLM] 收到 {len(state['messages'])} 条消息")
        print(f"[LLM] 响应: {response.content[:100]}...")

        return {
            "messages": [AIMessage(content=response.content)],
            "llm_response": response.content
        }

    # 构建图
    graph = StateGraph(ChatState)
    graph.add_node("chat", chat_node)
    graph.set_entry_point("chat")
    graph.add_edge("chat", END)

    return graph.compile()

# 使用示例
def test_llm_integration():
    """
    测试 LLM 集成

    演示:
    1. 使用系统消息设置角色
    2. 发送用户消息
    3. 获取 AI 响应
    """
    print("=== 基础 LLM 集成测试 ===\n")

    app = create_llm_graph()

    result = app.invoke({
        "messages": [
            SystemMessage(content="你是一个有帮助的 AI 助手,专门解释技术概念"),
            HumanMessage(content="什么是 LangGraph? 请用简单的语言解释")
        ],
        "llm_response": ""
    })

    print(f"\n✅ LLM 回复:\n{result['llm_response']}")
    print(f"\n消息总数: {len(result['messages'])}")
```

**关键技术点**

1. **消息类型**
```python
from langchain_core.messages import (
    SystemMessage,   # 系统提示
    HumanMessage,    # 用户消息
    AIMessage,       # AI 响应
    FunctionMessage  # 函数调用结果
)
```

2. **LLM 配置**
```python
llm = ChatOpenAI(
    model="gpt-4",           # 模型选择
    temperature=0.7,         # 随机性控制
    max_tokens=1000,         # 最大token数
    streaming=True,          # 流式输出
    api_key="your-key"       # API密钥
)
```

3. **错误处理**
```python
from langchain_core.exceptions import OutputParserException

def safe_llm_call(llm, messages):
    """安全的 LLM 调用"""
    try:
        return llm.invoke(messages)
    except Exception as e:
        return AIMessage(content=f"错误: {str(e)}")
```

### 2.2 多 LLM 协作

**什么是多 LLM 协作？**

多 LLM 协作是指在同一个工作流中使用多个不同的大语言模型,利用各个模型的优势,通过协作产生更好的结果。

**为什么需要多 LLM 协作？**

1. **取长补短**: 不同模型擅长不同任务
2. **交叉验证**: 多个模型的回答可以互相验证
3. **质量提升**: 综合多个模型的输出提高答案质量
4. **成本优化**: 简单任务用便宜模型,复杂任务用高级模型

**协作模式**

```mermaid
graph TB
    A[用户查询] --> B[GPT-4 分析]
    A --> C[Claude 分析]
    B --> D[综合节点]
    C --> D
    D --> E[最终答案]

    style A fill:#e1f5fe
    style B fill:#b3e5fc
    style C fill:#b3e5fc
    style D fill:#81d4fa
    style E fill:#c8e6c9
```

**完整实现示例**

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from typing import TypedDict

class MultiLLMState(TypedDict):
    query: str
    gpt_response: str
    claude_response: str
    final_answer: str

def create_multi_llm_graph():
    """多个 LLM 协作的图"""
    gpt = ChatOpenAI(model="gpt-4")
    claude = ChatAnthropic(model="claude-3-sonnet-20240229")

    def gpt_node(state: MultiLLMState) -> dict:
        """GPT 节点"""
        response = gpt.invoke(state["query"])
        return {"gpt_response": response.content}

    def claude_node(state: MultiLLMState) -> dict:
        """Claude 节点"""
        response = claude.invoke(state["query"])
        return {"claude_response": response.content}

    def synthesize_node(state: MultiLLMState) -> dict:
        """综合两个 LLM 的回答"""
        synthesis_prompt = f"""
        对于问题: {state['query']}

        GPT-4 的回答: {state['gpt_response']}
        Claude 的回答: {state['claude_response']}

        请综合这两个回答,给出最佳答案。
        """

        final_response = gpt.invoke(synthesis_prompt)
        return {"final_answer": final_response.content}

    graph = StateGraph(MultiLLMState)
    graph.add_node("gpt", gpt_node)
    graph.add_node("claude", claude_node)
    graph.add_node("synthesize", synthesize_node)

    # GPT 和 Claude 并行执行
    graph.set_entry_point("gpt")
    graph.set_entry_point("claude")

    graph.add_edge("gpt", "synthesize")
    graph.add_edge("claude", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()
```

## 三、集成 Tools

### 3.1 基础工具集成

```python
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool
from typing import TypedDict

class ToolState(TypedDict):
    query: str
    search_results: str
    wiki_results: str
    final_answer: str

@tool
def custom_calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

def create_tool_graph():
    """集成工具的图"""
    # LangChain 工具
    search = DuckDuckGoSearchRun()
    wikipedia = WikipediaAPIWrapper()

    def search_node(state: ToolState) -> dict:
        """搜索节点"""
        results = search.run(state["query"])
        return {"search_results": results}

    def wiki_node(state: ToolState) -> dict:
        """维基百科节点"""
        results = wikipedia.run(state["query"])
        return {"wiki_results": results}

    def synthesize_node(state: ToolState) -> dict:
        """综合工具结果"""
        answer = f"""
        基于搜索结果和维基百科:

        搜索: {state['search_results'][:200]}...
        维基: {state['wiki_results'][:200]}...

        综合答案: [这里应该用 LLM 生成综合答案]
        """
        return {"final_answer": answer}

    graph = StateGraph(ToolState)
    graph.add_node("search", search_node)
    graph.add_node("wiki", wiki_node)
    graph.add_node("synthesize", synthesize_node)

    # 并行执行工具
    graph.set_entry_point("search")
    graph.set_entry_point("wiki")

    graph.add_edge("search", "synthesize")
    graph.add_edge("wiki", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()
```

### 3.2 动态工具调用

```python
from langchain_core.tools import BaseTool
from typing import TypedDict, List

class ToolCallState(TypedDict):
    query: str
    available_tools: List[str]
    selected_tool: str
    tool_result: str

# 定义多个工具
@tool
def weather_tool(location: str) -> str:
    """获取天气信息"""
    return f"{location} 的天气: 晴天 25°C"

@tool
def news_tool(topic: str) -> str:
    """获取新闻"""
    return f"关于 {topic} 的新闻: ..."

@tool
def stock_tool(symbol: str) -> str:
    """获取股票信息"""
    return f"{symbol} 股票价格: $100"

def create_dynamic_tool_graph():
    """动态工具调用图"""
    llm = ChatOpenAI(model="gpt-4")

    # 工具映射
    tools = {
        "weather": weather_tool,
        "news": news_tool,
        "stock": stock_tool
    }

    def select_tool(state: ToolCallState) -> dict:
        """LLM 选择合适的工具"""
        prompt = f"""
        对于查询: {state['query']}
        可用工具: {', '.join(state['available_tools'])}

        选择最合适的工具名称（只返回工具名）。
        """

        response = llm.invoke(prompt)
        return {"selected_tool": response.content.strip()}

    def execute_tool(state: ToolCallState) -> dict:
        """执行选定的工具"""
        tool = tools.get(state["selected_tool"])

        if tool:
            # 从查询中提取参数（简化示例）
            result = tool.invoke(state["query"])
        else:
            result = "未找到合适的工具"

        return {"tool_result": result}

    graph = StateGraph(ToolCallState)
    graph.add_node("select", select_tool)
    graph.add_node("execute", execute_tool)

    graph.set_entry_point("select")
    graph.add_edge("select", "execute")
    graph.add_edge("execute", END)

    return graph.compile()
```

## 四、集成 Memory

### 4.1 对话记忆

```python
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, Annotated
import operator

class MemoryState(TypedDict):
    messages: Annotated[list, operator.add]
    memory_summary: str

def create_memory_graph():
    """集成记忆的图"""
    llm = ChatOpenAI(model="gpt-4")
    memory = ConversationBufferMemory(return_messages=True)

    def chat_with_memory(state: MemoryState) -> dict:
        """带记忆的聊天"""
        # 从状态获取消息
        current_messages = state["messages"]

        # 保存到记忆
        if len(current_messages) > 0:
            last_message = current_messages[-1]
            if isinstance(last_message, HumanMessage):
                memory.chat_memory.add_user_message(last_message.content)

        # 获取记忆上下文
        memory_context = memory.load_memory_variables({})
        history = memory_context.get("history", [])

        # 调用 LLM（包含历史）
        all_messages = history + current_messages
        response = llm.invoke(all_messages)

        # 保存 AI 回复到记忆
        memory.chat_memory.add_ai_message(response.content)

        return {
            "messages": [AIMessage(content=response.content)],
            "memory_summary": f"对话轮次: {len(memory.chat_memory.messages)}"
        }

    graph = StateGraph(MemoryState)
    graph.add_node("chat", chat_with_memory)
    graph.set_entry_point("chat")
    graph.add_edge("chat", END)

    return graph.compile()

# 测试多轮对话
def test_memory():
    """测试记忆功能"""
    app = create_memory_graph()

    # 第一轮
    result1 = app.invoke({
        "messages": [HumanMessage(content="我叫张三")],
        "memory_summary": ""
    })
    print(f"回复1: {result1['messages'][-1].content}")

    # 第二轮（LLM 应该记得名字）
    result2 = app.invoke({
        "messages": [HumanMessage(content="我叫什么名字?")],
        "memory_summary": ""
    })
    print(f"回复2: {result2['messages'][-1].content}")
```

### 4.2 向量记忆

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.memory import VectorStoreRetrieverMemory
from typing import TypedDict

class VectorMemoryState(TypedDict):
    query: str
    relevant_history: str
    response: str

def create_vector_memory_graph():
    """使用向量记忆的图"""
    # 创建向量存储
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(
        ["初始化向量存储"],
        embedding=embeddings
    )

    # 创建记忆
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    memory = VectorStoreRetrieverMemory(retriever=retriever)

    llm = ChatOpenAI(model="gpt-4")

    def query_with_vector_memory(state: VectorMemoryState) -> dict:
        """带向量记忆的查询"""
        # 检索相关历史
        relevant = memory.load_memory_variables(
            {"query": state["query"]}
        )

        # 构建提示
        prompt = f"""
        相关历史记录:
        {relevant.get('history', '无')}

        当前问题: {state['query']}

        请回答问题。
        """

        response = llm.invoke(prompt)

        # 保存到记忆
        memory.save_context(
            {"input": state["query"]},
            {"output": response.content}
        )

        return {
            "relevant_history": str(relevant),
            "response": response.content
        }

    graph = StateGraph(VectorMemoryState)
    graph.add_node("query", query_with_vector_memory)
    graph.set_entry_point("query")
    graph.add_edge("query", END)

    return graph.compile()
```

## 五、集成 Chains

### 5.1 LLMChain 集成

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from typing import TypedDict

class ChainState(TypedDict):
    topic: str
    outline: str
    content: str

def create_chain_graph():
    """集成 LangChain Chain 的图"""
    llm = ChatOpenAI(model="gpt-4")

    # 创建 Chain
    outline_prompt = PromptTemplate(
        input_variables=["topic"],
        template="为主题 '{topic}' 创建一个文章大纲。"
    )
    outline_chain = LLMChain(llm=llm, prompt=outline_prompt)

    content_prompt = PromptTemplate(
        input_variables=["topic", "outline"],
        template="""
        主题: {topic}
        大纲: {outline}

        根据以上大纲,写一篇详细的文章。
        """
    )
    content_chain = LLMChain(llm=llm, prompt=content_prompt)

    def outline_node(state: ChainState) -> dict:
        """生成大纲"""
        outline = outline_chain.run(topic=state["topic"])
        return {"outline": outline}

    def content_node(state: ChainState) -> dict:
        """生成内容"""
        content = content_chain.run(
            topic=state["topic"],
            outline=state["outline"]
        )
        return {"content": content}

    graph = StateGraph(ChainState)
    graph.add_node("outline", outline_node)
    graph.add_node("content", content_node)

    graph.set_entry_point("outline")
    graph.add_edge("outline", "content")
    graph.add_edge("content", END)

    return graph.compile()
```

### 5.2 SequentialChain 集成

```python
from langchain.chains import SequentialChain
from typing import TypedDict

class SequentialState(TypedDict):
    input: str
    output: str

def create_sequential_chain_graph():
    """集成 SequentialChain"""
    llm = ChatOpenAI(model="gpt-4")

    # 第一个 chain
    chain1 = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            input_variables=["text"],
            template="总结以下文本: {text}"
        ),
        output_key="summary"
    )

    # 第二个 chain
    chain2 = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            input_variables=["summary"],
            template="将以下摘要翻译成英文: {summary}"
        ),
        output_key="translation"
    )

    # 组合成 SequentialChain
    sequential_chain = SequentialChain(
        chains=[chain1, chain2],
        input_variables=["text"],
        output_variables=["translation"]
    )

    def process_node(state: SequentialState) -> dict:
        """使用 Sequential Chain 处理"""
        result = sequential_chain({"text": state["input"]})
        return {"output": result["translation"]}

    graph = StateGraph(SequentialState)
    graph.add_node("process", process_node)
    graph.set_entry_point("process")
    graph.add_edge("process", END)

    return graph.compile()
```

## 六、集成 Agents

### 6.1 ReAct Agent 集成

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from typing import TypedDict

class AgentState(TypedDict):
    task: str
    agent_output: str

def create_agent_graph():
    """集成 LangChain Agent"""
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    # 定义工具
    tools = [
        Tool(
            name="Calculator",
            func=lambda x: str(eval(x)),
            description="用于数学计算"
        ),
        Tool(
            name="Search",
            func=lambda x: f"搜索结果: {x}",
            description="用于搜索信息"
        )
    ]

    # 创建 agent
    prompt = PromptTemplate.from_template("""
    回答以下问题,你可以使用这些工具:

    {tools}

    使用以下格式:

    Question: 需要回答的问题
    Thought: 思考该做什么
    Action: 要采取的行动,应该是 [{tool_names}] 中的一个
    Action Input: 行动的输入
    Observation: 行动的结果
    ... (这个 Thought/Action/Action Input/Observation 可以重复 N 次)
    Thought: 我现在知道最终答案了
    Final Answer: 原始问题的最终答案

    开始!

    Question: {input}
    Thought: {agent_scratchpad}
    """)

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=3
    )

    def agent_node(state: AgentState) -> dict:
        """Agent 节点"""
        result = agent_executor.invoke({"input": state["task"]})
        return {"agent_output": result["output"]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)

    return graph.compile()

# 测试 Agent
def test_agent():
    """测试 Agent 集成"""
    app = create_agent_graph()

    result = app.invoke({
        "task": "计算 25 * 4 然后搜索这个数字的含义",
        "agent_output": ""
    })

    print(f"Agent 输出: {result['agent_output']}")
```

## 七、完整集成示例

### RAG 系统

```python
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from typing import TypedDict, Annotated
import operator

class RAGState(TypedDict):
    query: str
    documents: Annotated[list, operator.add]
    context: str
    answer: str

def create_rag_graph():
    """完整的 RAG 系统"""
    # 初始化组件
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(model="gpt-4")

    # 假设已经有向量存储
    # vectorstore = FAISS.load_local("./vectorstore")

    def retrieve_node(state: RAGState) -> dict:
        """检索文档"""
        # vectorstore.similarity_search(state["query"], k=3)
        # 模拟检索
        docs = [
            {"content": "文档1内容", "score": 0.9},
            {"content": "文档2内容", "score": 0.8},
        ]
        return {"documents": docs}

    def rerank_node(state: RAGState) -> dict:
        """重排序文档"""
        # 使用 LLM 重排序
        sorted_docs = sorted(
            state["documents"],
            key=lambda x: x["score"],
            reverse=True
        )
        return {"documents": sorted_docs[:3]}

    def generate_node(state: RAGState) -> dict:
        """生成答案"""
        # 构建上下文
        context = "\n\n".join([
            doc["content"] for doc in state["documents"]
        ])

        prompt = f"""
        基于以下上下文回答问题:

        上下文:
        {context}

        问题: {state["query"]}

        答案:
        """

        response = llm.invoke(prompt)
        return {
            "context": context,
            "answer": response.content
        }

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", END)

    return graph.compile()
```

## 八、最佳实践

### 8.1 集成清单

**设计阶段**

- ✅ **明确需求**: 清楚知道需要哪些 LangChain 组件
- ✅ **选择组件**: 根据场景选择合适的 LLM、工具、记忆等
- ✅ **定义状态**: 设计清晰的状态结构,包含所有必要字段
- ✅ **规划流程**: 画出图结构,明确数据流向

**开发阶段**

- ✅ **状态转换**: 在节点中正确处理 LangChain 输出到状态的转换
- ✅ **异常处理**: 为 LLM 调用、工具执行添加 try-except
- ✅ **类型安全**: 使用 TypedDict 定义状态,避免类型错误
- ✅ **日志记录**: 添加详细日志,方便调试

**优化阶段**

- ✅ **缓存机制**: 对重复的 LLM 调用使用缓存
- ✅ **并行执行**: 识别可并行的 LLM/工具调用
- ✅ **成本控制**: 监控 API 调用次数,使用性价比高的模型
- ✅ **性能监控**: 使用 LangSmith 或自定义监控

**测试阶段**

- ✅ **单元测试**: 测试每个节点的功能
- ✅ **集成测试**: 测试整个图的执行流程
- ✅ **边界测试**: 测试异常情况和边界条件
- ✅ **性能测试**: 测试响应时间和资源消耗

### 8.2 常见问题 FAQ

**Q1: 如何处理 LangChain 和 LangGraph 的状态差异?**

A: 在节点中进行状态转换,明确输入输出格式。

```python
def llm_node(state: MyState) -> dict:
    # 从 LangGraph 状态提取数据
    messages = state["messages"]

    # 调用 LangChain 组件
    response = llm.invoke(messages)

    # 转换为 LangGraph 状态格式
    return {
        "messages": [AIMessage(content=response.content)],
        "llm_response": response.content
    }
```

**Q2: 如何优化 LLM 调用成本?**

A: 使用多种策略降低成本:

1. **使用缓存**
```python
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

set_llm_cache(InMemoryCache())
```

2. **选择合适模型**
```python
# 简单任务用便宜模型
simple_llm = ChatOpenAI(model="gpt-3.5-turbo")

# 复杂任务用高级模型
complex_llm = ChatOpenAI(model="gpt-4")
```

3. **批处理**
```python
# 批量调用
responses = llm.batch([msg1, msg2, msg3])
```

**Q3: 如何调试集成问题?**

A: 使用多种调试工具:

1. **启用 verbose 模式**
```python
llm = ChatOpenAI(model="gpt-4", verbose=True)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

2. **添加详细日志**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

def debug_node(state):
    logging.debug(f"节点输入: {state}")
    result = process(state)
    logging.debug(f"节点输出: {result}")
    return result
```

3. **使用 LangSmith**
```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-key"
```

**Q4: 如何处理 LLM 调用失败?**

A: 实现重试和降级机制:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def robust_llm_call(llm, messages):
    """带重试的 LLM 调用"""
    return llm.invoke(messages)
```

**Q5: 如何在 LangGraph 中使用 LangChain 的流式输出?**

A: 使用异步流式:

```python
async def streaming_node(state: ChatState):
    """流式 LLM 节点"""
    response_content = ""

    async for chunk in llm.astream(state["messages"]):
        response_content += chunk.content
        print(chunk.content, end="", flush=True)

    return {"llm_response": response_content}
```

**Q6: 如何管理多个 LLM 的 API 密钥?**

A: 使用环境变量或配置文件:

```python
import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# 方式1: 环境变量
os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"

# 方式2: 显式传递
gpt = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
claude = ChatAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
```

**Q7: LangChain 组件在 LangGraph 中是否线程安全?**

A: 大多数 LangChain 组件是线程安全的,但需要注意:

- Memory 组件可能不是线程安全的,建议每个会话使用独立实例
- 向量存储的并发写入需要加锁
- LLM 调用本身是线程安全的

```python
from threading import Lock

memory_lock = Lock()

def thread_safe_memory_node(state):
    with memory_lock:
        # 安全地访问 memory
        memory.save_context(...)
```

**Q8: 如何监控集成系统的性能?**

A: 实现性能监控:

```python
import time
from collections import defaultdict

class PerformanceMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)

    def track(self, node_name: str, duration: float):
        self.metrics[node_name].append(duration)

    def report(self):
        for node, durations in self.metrics.items():
            avg = sum(durations) / len(durations)
            print(f"{node}: 平均 {avg:.2f}s, 调用 {len(durations)} 次")

monitor = PerformanceMonitor()

def monitored_node(state):
    start = time.time()
    result = process(state)
    monitor.track("process_node", time.time() - start)
    return result
```

### 8.3 集成模式总结

**推荐的集成模式**

1. **简单问答系统**
```
LangGraph(状态管理) + ChatOpenAI(LLM) + BufferMemory(记忆)
```

2. **RAG 系统**
```
LangGraph(编排) + FAISS(向量库) + OpenAIEmbeddings + ChatOpenAI
```

3. **Agent 系统**
```
LangGraph(流程控制) + Tools(工具集) + ReActAgent(决策) + Memory(记忆)
```

4. **多模态系统**
```
LangGraph(编排) + ChatOpenAI(文本) + DALL-E(图像) + Whisper(语音)
```

**性能对比**

| 集成方式 | 延迟 | 成本 | 复杂度 | 适用场景 |
|---------|------|------|-------|---------|
| 单 LLM | 低 | 中 | 低 | 简单对话 |
| 多 LLM 协作 | 中 | 高 | 中 | 质量要求高 |
| LLM + Tools | 中 | 中 | 中 | 需要外部数据 |
| Agent | 高 | 高 | 高 | 复杂任务 |
| RAG | 中 | 中 | 中 | 知识库问答 |

### 8.4 下一步学习

**进阶主题**

1. **自定义 LangChain 组件**
   - 实现自定义 LLM
   - 创建自定义工具
   - 开发自定义 Memory

2. **高级集成技巧**
   - 流式集成
   - 并行 LLM 调用
   - 动态工具选择

3. **生产部署**
   - 性能优化
   - 错误处理
   - 监控告警

**相关资源**

- [LangChain 官方文档](https://python.langchain.com/)
- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- [LangSmith 平台](https://smith.langchain.com/)
- [LangChain Hub](https://smith.langchain.com/hub)

---

**总结** 📚

LangGraph 与 LangChain 的集成为构建复杂 AI 应用提供了强大的工具组合:

- ✅ **LangGraph** 提供灵活的流程编排和状态管理
- ✅ **LangChain** 提供丰富的组件生态和工具链
- ✅ **两者结合** 实现了最佳的开发体验和功能完整性

通过本章学习,你已经掌握了:
1. 如何集成各种 LLM
2. 如何使用 LangChain 工具
3. 如何集成记忆系统
4. 如何使用 Chains 和 Agents
5. 如何构建完整的 RAG 系统

**下一步:** 学习 [12.多智能体系统](./12.多智能体系统.md) 构建协作系统! 🚀
