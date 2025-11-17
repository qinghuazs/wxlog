---
title: AI 集成与后端实现 - Code Agent 核心服务
date: 2025-01-30
categories:
  - AI
  - CodeAgent
---

# AI 集成与后端实现

## 一、后端架构

### 1.1 FastAPI 服务搭建

```python
# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, AsyncIterator
import asyncio

from core.ai_engine import AIEngine
from core.code_analyzer import CodeAnalyzer
from core.context_manager import ContextManager

app = FastAPI(title="Code Agent API")

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化核心组件
ai_engine = AIEngine()
code_analyzer = CodeAnalyzer()
context_manager = ContextManager()

# 请求模型
class CompletionRequest(BaseModel):
    prefix: str
    suffix: str = ""
    language: str = "python"
    file_path: Optional[str] = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

class ExplainRequest(BaseModel):
    code: str
    language: str

class RefactorRequest(BaseModel):
    code: str
    instruction: str
    language: str

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}

@app.post("/completion")
async def get_completion(request: CompletionRequest):
    """代码补全（同步）"""
    try:
        # 构建上下文
        context = await context_manager.build_completion_context(
            prefix=request.prefix,
            suffix=request.suffix,
            language=request.language,
            file_path=request.file_path
        )

        # 调用 AI
        completion = await ai_engine.complete(context)

        return {"completion": completion}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/completion/stream")
async def stream_completion(request: CompletionRequest):
    """代码补全（流式）"""
    async def generate():
        try:
            context = await context_manager.build_completion_context(
                prefix=request.prefix,
                suffix=request.suffix,
                language=request.language,
                file_path=request.file_path
            )

            async for chunk in ai_engine.stream_complete(context):
                yield f"data: {chunk}\n\n"

            yield "event: done\ndata: \n\n"

        except Exception as e:
            yield f"event: error\ndata: {str(e)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )

@app.post("/chat")
async def chat(request: ChatRequest):
    """智能对话"""
    try:
        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]

        response = await ai_engine.chat(messages)

        return {"message": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def stream_chat(request: ChatRequest):
    """智能对话（流式）"""
    async def generate():
        try:
            messages = [
                {"role": msg.role, "content": msg.content}
                for msg in request.messages
            ]

            async for chunk in ai_engine.stream_chat(messages):
                yield f"data: {chunk}\n\n"

            yield "event: done\ndata: \n\n"

        except Exception as e:
            yield f"event: error\ndata: {str(e)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )

@app.post("/explain")
async def explain_code(request: ExplainRequest):
    """解释代码"""
    try:
        # 分析代码结构
        analysis = code_analyzer.analyze(request.code, request.language)

        # 生成解释
        explanation = await ai_engine.explain(
            code=request.code,
            language=request.language,
            analysis=analysis
        )

        return {"explanation": explanation}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/refactor")
async def refactor_code(request: RefactorRequest):
    """重构代码"""
    try:
        refactored = await ai_engine.refactor(
            code=request.code,
            instruction=request.instruction,
            language=request.language
        )

        return {"refactored": refactored}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-tests")
async def generate_tests(request: ExplainRequest):
    """生成测试"""
    try:
        # 分析代码
        analysis = code_analyzer.analyze(request.code, request.language)

        # 生成测试
        tests = await ai_engine.generate_tests(
            code=request.code,
            language=request.language,
            analysis=analysis
        )

        return {"tests": tests}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 二、AI 引擎实现

### 2.1 统一的 AI 接口

```python
# backend/core/ai_engine.py
from abc import ABC, abstractmethod
from typing import AsyncIterator, List, Dict
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage

class AIProvider(ABC):
    """AI 提供者抽象基类"""

    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> str:
        pass

    @abstractmethod
    async def stream_complete(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        pass

class OpenAIProvider(AIProvider):
    """OpenAI 提供者"""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.llm = ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=0.2
        )

    async def complete(self, prompt: str, **kwargs) -> str:
        response = await self.llm.ainvoke(prompt)
        return response.content

    async def stream_complete(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        async for chunk in self.llm.astream(prompt):
            if chunk.content:
                yield chunk.content

class ClaudeProvider(AIProvider):
    """Claude 提供者"""

    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        self.llm = ChatAnthropic(
            api_key=api_key,
            model=model,
            temperature=0.2
        )

    async def complete(self, prompt: str, **kwargs) -> str:
        response = await self.llm.ainvoke(prompt)
        return response.content

    async def stream_complete(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        async for chunk in self.llm.astream(prompt):
            if chunk.content:
                yield chunk.content

class AIEngine:
    """AI 引擎 - 统一管理所有 AI 功能"""

    def __init__(self):
        # 初始化提供者
        self.providers = {
            "openai": OpenAIProvider(api_key="your-openai-key"),
            "claude": ClaudeProvider(api_key="your-claude-key"),
        }
        self.default_provider = "openai"

        # Prompt 模板
        self.prompts = self._init_prompts()

    def _init_prompts(self) -> Dict[str, ChatPromptTemplate]:
        """初始化 Prompt 模板"""
        return {
            "completion": ChatPromptTemplate.from_messages([
                SystemMessage(content="""你是一个专业的代码补全助手。
规则：
1. 只返回补全的代码，不要解释
2. 保持与现有代码风格一致
3. 补全应该合理且符合上下文
4. 如果不确定，返回最可能的选项
"""),
                HumanMessage(content="""
上下文：
{context}

当前代码：
```{language}
{prefix}█{suffix}
```

请补全光标位置（█）的代码：
""")
            ]),

            "explain": ChatPromptTemplate.from_messages([
                SystemMessage(content="你是一个代码解释专家，用清晰易懂的语言解释代码。"),
                HumanMessage(content="""
请详细解释以下 {language} 代码：

```{language}
{code}
```

包括：
1. 代码的整体功能
2. 关键逻辑的解释
3. 潜在的问题或改进建议
""")
            ]),

            "refactor": ChatPromptTemplate.from_messages([
                SystemMessage(content="你是一个代码重构专家。"),
                HumanMessage(content="""
原始代码：
```{language}
{code}
```

重构要求：{instruction}

请提供重构后的代码（只返回代码，不要解释）：
""")
            ]),

            "generate_tests": ChatPromptTemplate.from_messages([
                SystemMessage(content="你是一个测试用例生成专家。"),
                HumanMessage(content="""
为以下 {language} 代码生成完整的单元测试：

```{language}
{code}
```

要求：
1. 覆盖所有主要功能
2. 包含正常情况和边界情况
3. 使用适合 {language} 的测试框架
4. 代码要完整可运行
""")
            ])
        }

    async def complete(self, context: Dict) -> str:
        """代码补全"""
        provider = self.providers[self.default_provider]

        # 构建 prompt
        prompt = self.prompts["completion"].format_messages(
            context=context.get("context", ""),
            language=context.get("language", "python"),
            prefix=context.get("prefix", ""),
            suffix=context.get("suffix", "")
        )

        return await provider.complete(str(prompt))

    async def stream_complete(self, context: Dict) -> AsyncIterator[str]:
        """流式代码补全"""
        provider = self.providers[self.default_provider]

        prompt = self.prompts["completion"].format_messages(
            context=context.get("context", ""),
            language=context.get("language", "python"),
            prefix=context.get("prefix", ""),
            suffix=context.get("suffix", "")
        )

        async for chunk in provider.stream_complete(str(prompt)):
            yield chunk

    async def chat(self, messages: List[Dict]) -> str:
        """智能对话"""
        provider = self.providers[self.default_provider]

        # 转换消息格式
        prompt = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in messages
        ])

        return await provider.complete(prompt)

    async def stream_chat(self, messages: List[Dict]) -> AsyncIterator[str]:
        """流式智能对话"""
        provider = self.providers[self.default_provider]

        prompt = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in messages
        ])

        async for chunk in provider.stream_complete(prompt):
            yield chunk

    async def explain(self, code: str, language: str, analysis: Dict = None) -> str:
        """解释代码"""
        provider = self.providers[self.default_provider]

        prompt = self.prompts["explain"].format_messages(
            language=language,
            code=code
        )

        return await provider.complete(str(prompt))

    async def refactor(self, code: str, instruction: str, language: str) -> str:
        """重构代码"""
        provider = self.providers[self.default_provider]

        prompt = self.prompts["refactor"].format_messages(
            language=language,
            code=code,
            instruction=instruction
        )

        return await provider.complete(str(prompt))

    async def generate_tests(self, code: str, language: str, analysis: Dict = None) -> str:
        """生成测试"""
        provider = self.providers[self.default_provider]

        prompt = self.prompts["generate_tests"].format_messages(
            language=language,
            code=code
        )

        return await provider.complete(str(prompt))
```

## 三、Prompt 工程优化

### 3.1 Few-Shot 示例

```python
# backend/core/prompt_templates.py
class PromptTemplates:
    """Prompt 模板库"""

    COMPLETION_WITH_EXAMPLES = """
你是一个专业的 {language} 代码补全助手。

示例1:
输入: def calculate_
输出: sum(a: int, b: int) -> int:
    return a + b

示例2:
输入: class User
输出: :
    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email

示例3:
输入: # 快速排序算法
输出: def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

现在，请补全以下代码：

上下文:
{context}

代码:
```{language}
{prefix}█
```

只返回补全部分：
"""

    CODE_REVIEW = """
作为资深代码审查专家，请审查以下代码：

```{language}
{code}
```

请从以下维度评估：

1. 代码质量（可读性、可维护性）
2. 性能问题
3. 安全隐患
4. 最佳实践
5. 改进建议

以 Markdown 格式输出详细报告。
"""

    BUG_FIX = """
以下代码存在 Bug：

```{language}
{code}
```

错误信息：
{error}

请：
1. 分析 Bug 原因
2. 提供修复方案
3. 给出修复后的完整代码

格式：
## Bug 分析
...

## 修复方案
...

## 修复后代码
```{language}
...
```
"""
```

### 3.2 动态 Prompt 构建

```python
class DynamicPromptBuilder:
    """动态 Prompt 构建器"""

    def __init__(self, code_analyzer):
        self.analyzer = code_analyzer

    async def build_completion_prompt(
        self,
        prefix: str,
        suffix: str,
        language: str,
        context: Dict
    ) -> str:
        """构建补全 Prompt"""

        # 1. 分析代码意图
        intent = self._analyze_intent(prefix)

        # 2. 选择合适的模板
        if intent == "function_definition":
            template = self._get_function_template()
        elif intent == "class_definition":
            template = self._get_class_template()
        elif intent == "import_statement":
            template = self._get_import_template()
        else:
            template = self._get_default_template()

        # 3. 填充上下文
        prompt = template.format(
            language=language,
            prefix=prefix,
            suffix=suffix,
            context=context.get("surrounding_code", ""),
            imports=context.get("imports", ""),
            similar_code=context.get("similar_code", "")
        )

        return prompt

    def _analyze_intent(self, prefix: str) -> str:
        """分析用户意图"""
        if prefix.strip().startswith("def "):
            return "function_definition"
        elif prefix.strip().startswith("class "):
            return "class_definition"
        elif prefix.strip().startswith(("import ", "from ")):
            return "import_statement"
        else:
            return "general"

    def _get_function_template(self) -> str:
        return """
基于以下上下文补全函数：

已有导入：
{imports}

相似代码示例：
{similar_code}

当前代码：
```{language}
{prefix}
```

请补全函数体：
"""
```

## 四、向量检索系统

### 4.1 代码索引

```python
# backend/core/code_indexer.py
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
import os
from pathlib import Path

class CodeIndexer:
    """代码索引器"""

    def __init__(self, persist_directory: str = "./chroma_db"):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )

    async def index_project(self, project_path: str):
        """索引整个项目"""
        # 支持的文件类型
        extensions = {
            ".py": Language.PYTHON,
            ".js": Language.JS,
            ".ts": Language.TS,
            ".java": Language.JAVA,
            ".go": Language.GO,
            ".cpp": Language.CPP,
        }

        documents = []

        for ext, lang in extensions.items():
            # 查找所有相关文件
            files = Path(project_path).rglob(f"*{ext}")

            for file_path in files:
                # 跳过某些目录
                if any(skip in str(file_path) for skip in [
                    "node_modules", "venv", ".git", "dist", "build"
                ]):
                    continue

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    # 使用语言特定的分割器
                    splitter = RecursiveCharacterTextSplitter.from_language(
                        language=lang,
                        chunk_size=1000,
                        chunk_overlap=200
                    )

                    chunks = splitter.split_text(content)

                    for i, chunk in enumerate(chunks):
                        documents.append({
                            "content": chunk,
                            "metadata": {
                                "file": str(file_path),
                                "language": ext[1:],
                                "chunk_index": i
                            }
                        })

                except Exception as e:
                    print(f"Error indexing {file_path}: {e}")

        # 批量添加到向量库
        if documents:
            self.vector_store.add_texts(
                texts=[doc["content"] for doc in documents],
                metadatas=[doc["metadata"] for doc in documents]
            )

        print(f"Indexed {len(documents)} code chunks")

    async def search(self, query: str, k: int = 5) -> List[Dict]:
        """搜索相关代码"""
        results = self.vector_store.similarity_search_with_score(query, k=k)

        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            }
            for doc, score in results
        ]
```

## 五、完整工作流示例

```python
# backend/core/completion_workflow.py
from langgraph.graph import StateGraph, END
from typing import TypedDict

class CompletionState(TypedDict):
    prefix: str
    suffix: str
    language: str
    context: Dict
    similar_code: List[str]
    prompt: str
    completion: str

def create_completion_workflow():
    """创建补全工作流"""

    # 定义各个步骤
    def gather_context(state: CompletionState) -> Dict:
        """收集上下文"""
        # 实现上下文收集逻辑
        return {"context": {...}}

    def retrieve_similar(state: CompletionState) -> Dict:
        """检索相似代码"""
        indexer = CodeIndexer()
        results = await indexer.search(state["prefix"])
        return {"similar_code": [r["content"] for r in results]}

    def build_prompt(state: CompletionState) -> Dict:
        """构建 Prompt"""
        builder = DynamicPromptBuilder()
        prompt = await builder.build_completion_prompt(
            prefix=state["prefix"],
            suffix=state["suffix"],
            language=state["language"],
            context=state["context"]
        )
        return {"prompt": prompt}

    def call_ai(state: CompletionState) -> Dict:
        """调用 AI"""
        engine = AIEngine()
        completion = await engine.complete({"prompt": state["prompt"]})
        return {"completion": completion}

    # 构建图
    graph = StateGraph(CompletionState)

    graph.add_node("gather_context", gather_context)
    graph.add_node("retrieve_similar", retrieve_similar)
    graph.add_node("build_prompt", build_prompt)
    graph.add_node("call_ai", call_ai)

    graph.set_entry_point("gather_context")
    graph.add_edge("gather_context", "retrieve_similar")
    graph.add_edge("retrieve_similar", "build_prompt")
    graph.add_edge("build_prompt", "call_ai")
    graph.add_edge("call_ai", END)

    return graph.compile()

# 使用
workflow = create_completion_workflow()
result = workflow.invoke({
    "prefix": "def quick_sort",
    "suffix": "",
    "language": "python"
})
print(result["completion"])
```

---

**恭喜！** 你已经掌握了开发 Code Agent 的核心技术！

现在你可以：
1. 🔧 启动后端服务：`python backend/main.py`
2. 🔌 开发 VSCode 插件
3. 🤖 集成多种 AI 模型
4. 🚀 优化性能和用户体验

继续探索更多高级功能，打造属于你自己的 AI 编程助手！
