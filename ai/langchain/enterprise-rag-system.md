---
title: 综合实战-企业级RAG系统
date: 2025-01-13
categories:
  - AI
  - LangChain
---

# 综合实战:企业级 RAG 系统

## 项目概述

本项目将整合前面所有学过的知识,构建一个功能完整、可投入生产使用的企业级 RAG(检索增强生成)系统。

### 项目目标

构建一个智能文档问答系统,支持:
- 📄 多格式文档(PDF、Word、Markdown、TXT等)
- 🔍 混合搜索(语义 + 关键词)
- 🎯 智能重排序
- 💬 多轮对话
- 📊 监控和成本追踪
- 🔒 用户权限管理
- 🚀 高性能和可扩展

### 技术架构

```mermaid
graph TB
    subgraph 前端层
        A[Web界面] --> B[FastAPI后端]
    end

    subgraph 应用层
        B --> C[RAG系统]
        C --> D[文档处理]
        C --> E[检索系统]
        C --> F[生成系统]
        C --> G[对话管理]
    end

    subgraph 数据层
        D --> H[向量数据库<br/>Chroma]
        D --> I[关系数据库<br/>PostgreSQL]
        E --> H
        G --> I
    end

    subgraph 监控层
        C --> J[Callbacks]
        J --> K[LangSmith]
        J --> L[Prometheus]
    end

    style C fill:#f96,stroke:#333,stroke-width:4px
    style H fill:#9f9,stroke:#333,stroke-width:2px
    style K fill:#bbf,stroke:#333,stroke-width:2px
```

## 第一部分:项目结构设计

### 1.1 目录结构

```
enterprise-rag-system/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI 应用入口
│   ├── config.py               # 配置管理
│   ├── dependencies.py         # 依赖注入
│   │
│   ├── core/                   # 核心功能
│   │   ├── __init__.py
│   │   ├── document_processor.py   # 文档处理
│   │   ├── retrieval_system.py     # 检索系统
│   │   ├── generation_system.py    # 生成系统
│   │   ├── conversation_manager.py # 对话管理
│   │   └── monitoring.py           # 监控系统
│   │
│   ├── api/                    # API 路由
│   │   ├── __init__.py
│   │   ├── documents.py        # 文档管理 API
│   │   ├── chat.py             # 对话 API
│   │   ├── admin.py            # 管理 API
│   │   └── monitoring.py       # 监控 API
│   │
│   ├── models/                 # 数据模型
│   │   ├── __init__.py
│   │   ├── document.py
│   │   ├── conversation.py
│   │   └── user.py
│   │
│   ├── services/               # 业务服务
│   │   ├── __init__.py
│   │   ├── document_service.py
│   │   ├── chat_service.py
│   │   └── user_service.py
│   │
│   └── utils/                  # 工具函数
│       ├── __init__.py
│       ├── logger.py
│       ├── cache.py
│       └── helpers.py
│
├── tests/                      # 测试
│   ├── test_retrieval.py
│   ├── test_generation.py
│   └── test_api.py
│
├── data/                       # 数据目录
│   ├── documents/              # 原始文档
│   ├── processed/              # 处理后的文档
│   └── vectorstore/            # 向量数据库
│
├── logs/                       # 日志
├── requirements.txt            # 依赖
├── .env                        # 环境变量
├── Dockerfile                  # Docker 配置
└── README.md                   # 项目文档
```

### 1.2 依赖安装

```python
"""
requirements.txt

# LangChain 核心
langchain==0.1.0
langchain-openai==0.0.5
langchain-community==0.0.20

# 向量数据库
chromadb==0.4.22
faiss-cpu==1.7.4

# 文档处理
pypdf==4.0.0
python-docx==1.1.0
python-pptx==0.6.23
unstructured==0.12.0
markdown==3.5.2

# Web 框架
fastapi==0.109.0
uvicorn[standard]==0.27.0
python-multipart==0.0.6

# 数据库
sqlalchemy==2.0.25
psycopg2-binary==2.9.9
alembic==1.13.1

# 缓存
redis==5.0.1

# 监控
prometheus-client==0.19.0
langsmith==0.0.87

# 工具
python-dotenv==1.0.0
pydantic==2.5.3
pydantic-settings==2.1.0
"""
```

### 1.3 配置管理

```python
# app/config.py

from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache

class Settings(BaseSettings):
    """应用配置"""

    # 应用基础配置
    APP_NAME: str = "Enterprise RAG System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # LLM 配置
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    OPENAI_TEMPERATURE: float = 0.0
    OPENAI_MAX_TOKENS: Optional[int] = None

    # Embedding 配置
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSION: int = 1536

    # 向量数据库配置
    VECTOR_STORE_TYPE: str = "chroma"  # chroma, faiss
    VECTOR_STORE_PATH: str = "./data/vectorstore"
    COLLECTION_NAME: str = "documents"

    # 数据库配置
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/rag_db"
    DATABASE_ECHO: bool = False

    # Redis 配置
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL: int = 3600  # 缓存过期时间(秒)

    # 文档处理配置
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: list = [".pdf", ".docx", ".txt", ".md"]

    # 检索配置
    TOP_K: int = 5
    RERANK_K: int = 3
    USE_HYBRID_SEARCH: bool = True
    BM25_WEIGHT: float = 0.5
    VECTOR_WEIGHT: float = 0.5

    # 监控配置
    LANGSMITH_API_KEY: Optional[str] = None
    LANGSMITH_PROJECT: str = "enterprise-rag"
    PROMETHEUS_PORT: int = 9090
    ENABLE_MONITORING: bool = True

    # 安全配置
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """获取配置(单例模式)"""
    return Settings()

# 使用示例
if __name__ == "__main__":
    settings = get_settings()
    print(f"App Name: {settings.APP_NAME}")
    print(f"Model: {settings.OPENAI_MODEL}")
    print(f"Vector Store: {settings.VECTOR_STORE_TYPE}")
```

## 第二部分:核心功能实现

### 2.1 文档处理系统

```python
# app/core/document_processor.py

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Optional
import os
import hashlib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """文档处理器"""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        初始化文档处理器

        参数:
            chunk_size: 文档块大小
            chunk_overlap: 重叠大小
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", ". ", " ", ""]
        )

        # 文件类型到加载器的映射
        self.loader_mapping = {
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
            ".txt": TextLoader,
            ".md": UnstructuredMarkdownLoader
        }

        logger.info(f"DocumentProcessor initialized (chunk_size={chunk_size}, overlap={chunk_overlap})")

    def process_file(
        self,
        file_path: str,
        metadata: Optional[dict] = None
    ) -> List[Document]:
        """
        处理单个文件

        参数:
            file_path: 文件路径
            metadata: 附加元数据

        返回:
            文档块列表
        """
        logger.info(f"Processing file: {file_path}")

        # 1. 验证文件
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_ext = Path(file_path).suffix.lower()
        if file_ext not in self.loader_mapping:
            raise ValueError(f"Unsupported file type: {file_ext}")

        try:
            # 2. 加载文档
            loader_class = self.loader_mapping[file_ext]
            loader = loader_class(file_path)
            documents = loader.load()

            logger.info(f"Loaded {len(documents)} document(s)")

            # 3. 添加元数据
            file_hash = self._calculate_file_hash(file_path)
            file_size = os.path.getsize(file_path)

            base_metadata = {
                "source": file_path,
                "file_name": os.path.basename(file_path),
                "file_type": file_ext,
                "file_hash": file_hash,
                "file_size": file_size
            }

            if metadata:
                base_metadata.update(metadata)

            for doc in documents:
                doc.metadata.update(base_metadata)

            # 4. 清洗文档
            cleaned_docs = self._clean_documents(documents)

            # 5. 分割文档
            chunks = self.text_splitter.split_documents(cleaned_docs)

            # 6. 添加块索引
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["chunk_total"] = len(chunks)

            logger.info(f"Split into {len(chunks)} chunks")

            return chunks

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise

    def process_directory(
        self,
        directory_path: str,
        recursive: bool = True,
        metadata: Optional[dict] = None
    ) -> List[Document]:
        """
        处理整个目录

        参数:
            directory_path: 目录路径
            recursive: 是否递归处理子目录
            metadata: 附加元数据

        返回:
            所有文档块列表
        """
        logger.info(f"Processing directory: {directory_path}")

        all_chunks = []
        file_count = 0
        error_count = 0

        # 遍历目录
        pattern = "**/*" if recursive else "*"
        for file_path in Path(directory_path).glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.loader_mapping:
                try:
                    chunks = self.process_file(str(file_path), metadata)
                    all_chunks.extend(chunks)
                    file_count += 1
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
                    error_count += 1

        logger.info(f"Processed {file_count} files, {error_count} errors, {len(all_chunks)} total chunks")

        return all_chunks

    def _clean_documents(self, documents: List[Document]) -> List[Document]:
        """清洗文档"""
        cleaned = []

        for doc in documents:
            # 移除多余空白
            content = " ".join(doc.page_content.split())

            # 过滤太短的文档
            if len(content) < 10:
                continue

            # 移除特殊字符(可选)
            # content = re.sub(r'[^\w\s\u4e00-\u9fff,.!?;:""''\(\)]', '', content)

            doc.page_content = content
            cleaned.append(doc)

        return cleaned

    def _calculate_file_hash(self, file_path: str) -> str:
        """计算文件 hash"""
        hasher = hashlib.md5()

        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)

        return hasher.hexdigest()

    def update_chunk_size(self, chunk_size: int, chunk_overlap: int):
        """更新分块参数"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", ". ", " ", ""]
        )

        logger.info(f"Updated chunk_size={chunk_size}, overlap={chunk_overlap}")

# ===== 使用示例 =====
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # 创建文档处理器
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)

    # 处理单个文件
    chunks = processor.process_file(
        "example.pdf",
        metadata={"category": "技术文档", "author": "张三"}
    )

    print(f"处理结果: {len(chunks)} 个文档块")
    print(f"\n第一个块:")
    print(f"内容: {chunks[0].page_content[:200]}...")
    print(f"元数据: {chunks[0].metadata}")

    # 处理目录
    all_chunks = processor.process_directory(
        "./documents",
        recursive=True,
        metadata={"project": "企业知识库"}
    )

    print(f"\n目录处理结果: {len(all_chunks)} 个文档块")
```

### 2.2 检索系统

```python
# app/core/retrieval_system.py

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from typing import List, Optional
import os
import logging

logger = logging.getLogger(__name__)

class RetrievalSystem:
    """检索系统"""

    def __init__(
        self,
        vector_store_path: str = "./data/vectorstore",
        collection_name: str = "documents",
        embedding_model: str = "text-embedding-3-small",
        top_k: int = 5,
        use_hybrid: bool = True,
        use_rerank: bool = True
    ):
        """
        初始化检索系统

        参数:
            vector_store_path: 向量存储路径
            collection_name: 集合名称
            embedding_model: Embedding 模型
            top_k: 返回文档数量
            use_hybrid: 是否使用混合检索
            use_rerank: 是否使用重排序
        """
        self.vector_store_path = vector_store_path
        self.collection_name = collection_name
        self.top_k = top_k
        self.use_hybrid = use_hybrid
        self.use_rerank = use_rerank

        # 创建 Embeddings
        self.embeddings = OpenAIEmbeddings(model=embedding_model)

        # 创建或加载向量存储
        self.vectorstore = self._init_vectorstore()

        # BM25 检索器(用于混合检索)
        self.bm25_retriever = None

        # 重排序器
        self.reranker = None
        if use_rerank:
            self.reranker = LLMChainExtractor.from_llm(
                ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
            )

        logger.info(f"RetrievalSystem initialized (top_k={top_k}, hybrid={use_hybrid}, rerank={use_rerank})")

    def _init_vectorstore(self):
        """初始化向量存储"""
        if os.path.exists(self.vector_store_path):
            vectorstore = Chroma(
                persist_directory=self.vector_store_path,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            count = vectorstore._collection.count()
            logger.info(f"Loaded existing vectorstore ({count} documents)")
        else:
            vectorstore = Chroma(
                persist_directory=self.vector_store_path,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            logger.info("Created new vectorstore")

        return vectorstore

    def index_documents(self, documents: List[Document]):
        """
        索引文档

        参数:
            documents: 文档列表
        """
        logger.info(f"Indexing {len(documents)} documents...")

        # 添加到向量存储
        self.vectorstore.add_documents(documents)

        # 更新 BM25(如果使用混合检索)
        if self.use_hybrid:
            self._update_bm25(documents)

        logger.info(f"✓ Indexed {len(documents)} documents")

    def _update_bm25(self, documents: List[Document]):
        """更新 BM25 检索器"""
        # 获取所有文档
        all_docs = self._get_all_documents()
        all_docs.extend(documents)

        # 创建 BM25 检索器
        self.bm25_retriever = BM25Retriever.from_documents(all_docs)
        self.bm25_retriever.k = self.top_k

        logger.info(f"Updated BM25 with {len(all_docs)} documents")

    def _get_all_documents(self) -> List[Document]:
        """获取所有文档(简化实现)"""
        # 注意: 这是简化实现,生产环境应该从数据库获取
        try:
            results = self.vectorstore.similarity_search("", k=10000)
            return results
        except:
            return []

    def search(
        self,
        query: str,
        k: Optional[int] = None,
        filter_dict: Optional[dict] = None
    ) -> List[Document]:
        """
        检索文档

        参数:
            query: 查询
            k: 返回文档数量
            filter_dict: 元数据过滤条件

        返回:
            相关文档列表
        """
        k = k or self.top_k

        logger.info(f"Searching for: {query}")

        # 创建检索器
        retriever = self._create_retriever(k, filter_dict)

        # 执行检索
        results = retriever.get_relevant_documents(query)

        logger.info(f"Found {len(results)} documents")

        return results

    def _create_retriever(self, k: int, filter_dict: Optional[dict]):
        """创建检索器"""
        if self.use_hybrid and self.bm25_retriever:
            # 混合检索
            vector_retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": k, "filter": filter_dict} if filter_dict else {"k": k}
            )

            ensemble_retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, vector_retriever],
                weights=[0.5, 0.5]
            )

            base_retriever = ensemble_retriever
        else:
            # 纯向量检索
            base_retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": k, "filter": filter_dict} if filter_dict else {"k": k}
            )

        # 添加重排序
        if self.use_rerank and self.reranker:
            retriever = ContextualCompressionRetriever(
                base_compressor=self.reranker,
                base_retriever=base_retriever
            )
        else:
            retriever = base_retriever

        return retriever

    def search_with_scores(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[tuple]:
        """
        检索文档并返回相似度分数

        参数:
            query: 查询
            k: 返回文档数量

        返回:
            (文档, 分数) 元组列表
        """
        k = k or self.top_k

        results = self.vectorstore.similarity_search_with_relevance_scores(
            query,
            k=k
        )

        return results

    def delete_documents(self, filter_dict: dict):
        """
        删除文档

        参数:
            filter_dict: 过滤条件
        """
        # Chroma 的删除
        results = self.vectorstore.get(where=filter_dict)

        if results and results["ids"]:
            self.vectorstore.delete(ids=results["ids"])
            logger.info(f"Deleted {len(results['ids'])} documents")
        else:
            logger.info("No documents to delete")

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "total_documents": self.vectorstore._collection.count(),
            "collection_name": self.collection_name,
            "vector_store_path": self.vector_store_path,
            "use_hybrid": self.use_hybrid,
            "use_rerank": self.use_rerank
        }

# ===== 使用示例 =====
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    from langchain.schema import Document

    # 创建检索系统
    retrieval_system = RetrievalSystem(
        top_k=5,
        use_hybrid=True,
        use_rerank=True
    )

    # 索引文档
    documents = [
        Document(
            page_content="LangChain 是一个用于开发 LLM 应用的框架",
            metadata={"source": "doc1.txt", "category": "框架"}
        ),
        Document(
            page_content="RAG 结合了检索和生成两个步骤",
            metadata={"source": "doc2.txt", "category": "技术"}
        ),
        Document(
            page_content="向量数据库用于存储 Embeddings",
            metadata={"source": "doc3.txt", "category": "数据库"}
        )
    ]

    retrieval_system.index_documents(documents)

    # 检索
    results = retrieval_system.search("什么是 RAG?", k=2)

    print("\n检索结果:")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc.page_content}")
        print(f"   来源: {doc.metadata.get('source')}")

    # 统计信息
    stats = retrieval_system.get_stats()
    print(f"\n统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
```

由于内容较长,我将在下一条消息继续创建剩余部分...
### 2.3 生成系统

```python
# app/core/generation_system.py

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class GenerationSystem:
    """生成系统"""

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.0
    ):
        """
        初始化生成系统

        参数:
            model_name: LLM 模型名称
            temperature: 温度参数
        """
        self.model_name = model_name
        self.temperature = temperature

        # 创建 LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature
        )

        # Prompt 模板
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的智能助手,基于提供的参考文档回答用户问题。

回答要求:
1. 准确:严格基于参考文档内容
2. 完整:全面回答用户问题
3. 简洁:避免冗余信息
4. 来源:如果可能,引用文档来源
5. 诚实:如果文档中没有相关信息,明确告知

参考文档:
{context}"""),
            ("human", "{question}")
        ])

        logger.info(f"GenerationSystem initialized (model={model_name}, temp={temperature})")

    def generate(
        self,
        query: str,
        documents: List[Document],
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict:
        """
        生成回答

        参数:
            query: 用户问题
            documents: 检索到的文档
            conversation_history: 对话历史

        返回:
            包含答案和元数据的字典
        """
        logger.info(f"Generating answer for: {query}")

        # 1. 构建上下文
        context = self._build_context(documents)

        # 2. 添加对话历史(如果有)
        if conversation_history:
            history_text = self._format_history(conversation_history)
            context = f"{history_text}\n\n{context}"

        # 3. 生成回答
        chain = self.prompt_template | self.llm

        try:
            response = chain.invoke({
                "context": context,
                "question": query
            })

            answer = response.content

            # 4. 构建响应
            result = {
                "answer": answer,
                "sources": [doc.metadata.get("source", "Unknown") for doc in documents],
                "source_documents": documents,
                "model": self.model_name,
                "success": True
            }

            logger.info("✓ Answer generated successfully")

            return result

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": f"抱歉,生成回答时出错: {str(e)}",
                "sources": [],
                "source_documents": [],
                "model": self.model_name,
                "success": False,
                "error": str(e)
            }

    def _build_context(self, documents: List[Document]) -> str:
        """构建上下文"""
        if not documents:
            return "没有找到相关文档。"

        context_parts = []

        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            content = doc.page_content

            context_parts.append(f"文档 {i} (来源: {source}):\n{content}")

        return "\n\n".join(context_parts)

    def _format_history(self, history: List[Dict]) -> str:
        """格式化对话历史"""
        if not history:
            return ""

        history_parts = ["对话历史:"]

        for item in history[-5:]:  # 只保留最近5轮
            role = item.get("role", "user")
            content = item.get("content", "")

            if role == "user":
                history_parts.append(f"用户: {content}")
            elif role == "assistant":
                history_parts.append(f"助手: {content}")

        return "\n".join(history_parts)

    def generate_streaming(
        self,
        query: str,
        documents: List[Document]
    ):
        """
        流式生成回答

        参数:
            query: 用户问题
            documents: 检索到的文档

        返回:
            生成器,逐个返回 token
        """
        context = self._build_context(documents)

        chain = self.prompt_template | self.llm

        for chunk in chain.stream({
            "context": context,
            "question": query
        }):
            if hasattr(chunk, 'content'):
                yield chunk.content
            else:
                yield str(chunk)

# ===== 使用示例 =====
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    from langchain.schema import Document

    # 创建生成系统
    gen_system = GenerationSystem()

    # 准备文档
    documents = [
        Document(
            page_content="LangChain 是一个用于开发 LLM 应用的 Python 框架,由 Harrison Chase 于 2022 年创建。",
            metadata={"source": "langchain_intro.md"}
        ),
        Document(
            page_content="LangChain 的核心组件包括: Models、Prompts、Chains、Memory、Agents 等。",
            metadata={"source": "langchain_components.md"}
        )
    ]

    # 生成回答
    result = gen_system.generate(
        query="LangChain 是谁创建的?",
        documents=documents
    )

    print("\n问题:", "LangChain 是谁创建的?")
    print("\n回答:", result["answer"])
    print("\n来源:", result["sources"])

    # 流式生成
    print("\n\n流式生成:")
    for token in gen_system.generate_streaming(
        query="LangChain 有哪些核心组件?",
        documents=documents
    ):
        print(token, end="", flush=True)
```

### 2.4 对话管理系统

```python
# app/core/conversation_manager.py

from typing import List, Dict, Optional
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

class ConversationManager:
    """对话管理器"""

    def __init__(
        self,
        max_history_length: int = 10
    ):
        """
        初始化对话管理器

        参数:
            max_history_length: 最大历史长度
        """
        self.max_history_length = max_history_length

        # 会话存储 {session_id: session_data}
        self.sessions = {}

        logger.info(f"ConversationManager initialized (max_history={max_history_length})")

    def create_session(self, session_id: str, metadata: Optional[Dict] = None) -> Dict:
        """
        创建新会话

        参数:
            session_id: 会话ID
            metadata: 会话元数据

        返回:
            会话数据
        """
        session_data = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "history": [],
            "metadata": metadata or {}
        }

        self.sessions[session_id] = session_data

        logger.info(f"Session created: {session_id}")

        return session_data

    def get_session(self, session_id: str) -> Optional[Dict]:
        """获取会话"""
        return self.sessions.get(session_id)

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ):
        """
        添加消息到会话

        参数:
            session_id: 会话ID
            role: 角色(user/assistant)
            content: 消息内容
            metadata: 消息元数据
        """
        if session_id not in self.sessions:
            self.create_session(session_id)

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }

        self.sessions[session_id]["history"].append(message)
        self.sessions[session_id]["updated_at"] = datetime.now().isoformat()

        # 限制历史长度
        if len(self.sessions[session_id]["history"]) > self.max_history_length:
            self.sessions[session_id]["history"] = \
                self.sessions[session_id]["history"][-self.max_history_length:]

        logger.debug(f"Message added to session {session_id}")

    def get_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict]:
        """
        获取对话历史

        参数:
            session_id: 会话ID
            limit: 限制返回数量

        返回:
            消息列表
        """
        if session_id not in self.sessions:
            return []

        history = self.sessions[session_id]["history"]

        if limit:
            return history[-limit:]

        return history

    def clear_history(self, session_id: str):
        """清空会话历史"""
        if session_id in self.sessions:
            self.sessions[session_id]["history"] = []
            self.sessions[session_id]["updated_at"] = datetime.now().isoformat()
            logger.info(f"History cleared for session {session_id}")

    def delete_session(self, session_id: str):
        """删除会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Session deleted: {session_id}")

    def get_all_sessions(self) -> List[str]:
        """获取所有会话ID"""
        return list(self.sessions.keys())

    def export_session(self, session_id: str) -> Optional[str]:
        """
        导出会话为 JSON

        参数:
            session_id: 会话ID

        返回:
            JSON 字符串
        """
        if session_id not in self.sessions:
            return None

        return json.dumps(self.sessions[session_id], indent=2, ensure_ascii=False)

    def import_session(self, session_json: str) -> str:
        """
        导入会话

        参数:
            session_json: 会话 JSON 字符串

        返回:
            会话ID
        """
        session_data = json.loads(session_json)
        session_id = session_data["session_id"]

        self.sessions[session_id] = session_data

        logger.info(f"Session imported: {session_id}")

        return session_id

# ===== 使用示例 =====
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # 创建对话管理器
    conv_manager = ConversationManager(max_history_length=10)

    # 创建会话
    session_id = "user_123_session_001"
    conv_manager.create_session(
        session_id,
        metadata={"user_id": "user_123", "platform": "web"}
    )

    # 添加消息
    conv_manager.add_message(session_id, "user", "什么是 LangChain?")
    conv_manager.add_message(
        session_id,
        "assistant",
        "LangChain 是一个用于开发 LLM 应用的框架。",
        metadata={"sources": ["doc1.md"]}
    )
    conv_manager.add_message(session_id, "user", "它有哪些核心组件?")
    conv_manager.add_message(
        session_id,
        "assistant",
        "LangChain 的核心组件包括: Models、Prompts、Chains、Memory、Agents 等。"
    )

    # 获取历史
    history = conv_manager.get_history(session_id)

    print(f"\n会话历史 ({len(history)} 条消息):")
    for msg in history:
        print(f"{msg['role']}: {msg['content']}")

    # 导出会话
    session_json = conv_manager.export_session(session_id)
    print(f"\n导出的会话:\n{session_json}")
```

## 第三部分:FastAPI 后端实现

### 3.1 主应用

```python
# app/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.config import get_settings
from app.api import documents, chat, admin, monitoring
from app.core.document_processor import DocumentProcessor
from app.core.retrieval_system import RetrievalSystem
from app.core.generation_system import GenerationSystem
from app.core.conversation_manager import ConversationManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# 全局实例
settings = get_settings()
document_processor = None
retrieval_system = None
generation_system = None
conversation_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    global document_processor, retrieval_system, generation_system, conversation_manager

    logger.info("Initializing components...")

    document_processor = DocumentProcessor(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )

    retrieval_system = RetrievalSystem(
        vector_store_path=settings.VECTOR_STORE_PATH,
        collection_name=settings.COLLECTION_NAME,
        embedding_model=settings.EMBEDDING_MODEL,
        top_k=settings.TOP_K,
        use_hybrid=settings.USE_HYBRID_SEARCH
    )

    generation_system = GenerationSystem(
        model_name=settings.OPENAI_MODEL,
        temperature=settings.OPENAI_TEMPERATURE
    )

    conversation_manager = ConversationManager()

    logger.info("✓ All components initialized")

    yield

    # 关闭时清理
    logger.info("Shutting down...")

# 创建应用
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
app.include_router(admin.router, prefix="/api/admin", tags=["Admin"])
app.include_router(monitoring.router, prefix="/api/monitoring", tags=["Monitoring"])

@app.get("/")
async def root():
    """根路径"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "components": {
            "document_processor": document_processor is not None,
            "retrieval_system": retrieval_system is not None,
            "generation_system": generation_system is not None,
            "conversation_manager": conversation_manager is not None
        }
    }

# 运行应用
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
```

### 3.2 对话 API

```python
# app/api/chat.py

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from sse_starlette.sse import EventSourceResponse
import asyncio
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# ===== 请求/响应模型 =====

class ChatRequest(BaseModel):
    """聊天请求"""
    question: str = Field(..., description="用户问题")
    session_id: Optional[str] = Field(None, description="会话ID")
    k: Optional[int] = Field(5, description="检索文档数量")
    stream: Optional[bool] = Field(False, description="是否流式输出")

class ChatResponse(BaseModel):
    """聊天响应"""
    answer: str = Field(..., description="回答")
    sources: List[str] = Field(default_factory=list, description="来源列表")
    session_id: str = Field(..., description="会话ID")
    success: bool = Field(..., description="是否成功")

class SessionCreate(BaseModel):
    """创建会话请求"""
    session_id: Optional[str] = Field(None, description="会话ID(可选)")
    metadata: Optional[Dict] = Field(None, description="元数据")

class SessionResponse(BaseModel):
    """会话响应"""
    session_id: str
    created_at: str
    message_count: int

# ===== API 端点 =====

@router.post("/message", response_model=ChatResponse)
async def send_message(request: ChatRequest):
    """
    发送消息

    参数:
        request: 聊天请求

    返回:
        聊天响应
    """
    from app.main import retrieval_system, generation_system, conversation_manager
    import uuid

    try:
        # 1. 生成或使用会话ID
        session_id = request.session_id or str(uuid.uuid4())

        # 2. 确保会话存在
        if not conversation_manager.get_session(session_id):
            conversation_manager.create_session(session_id)

        # 3. 添加用户消息
        conversation_manager.add_message(session_id, "user", request.question)

        # 4. 检索相关文档
        logger.info(f"Retrieving documents for: {request.question}")
        documents = retrieval_system.search(request.question, k=request.k)

        # 5. 获取对话历史
        history = conversation_manager.get_history(session_id, limit=5)

        # 6. 生成回答
        logger.info("Generating answer...")
        result = generation_system.generate(
            query=request.question,
            documents=documents,
            conversation_history=history
        )

        # 7. 添加助手消息
        conversation_manager.add_message(
            session_id,
            "assistant",
            result["answer"],
            metadata={"sources": result["sources"]}
        )

        # 8. 返回响应
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            session_id=session_id,
            success=result["success"]
        )

    except Exception as e:
        logger.error(f"Error in send_message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/message/stream")
async def send_message_stream(request: ChatRequest):
    """
    发送消息(流式)

    参数:
        request: 聊天请求

    返回:
        SSE 事件流
    """
    from app.main import retrieval_system, generation_system, conversation_manager
    import uuid

    async def event_generator():
        try:
            # 1. 会话管理
            session_id = request.session_id or str(uuid.uuid4())

            if not conversation_manager.get_session(session_id):
                conversation_manager.create_session(session_id)

            conversation_manager.add_message(session_id, "user", request.question)

            # 2. 检索文档
            yield {"event": "status", "data": "检索相关文档..."}

            documents = retrieval_system.search(request.question, k=request.k)

            yield {"event": "sources", "data": [doc.metadata.get("source") for doc in documents]}

            # 3. 流式生成
            yield {"event": "status", "data": "生成回答..."}

            full_answer = ""

            for token in generation_system.generate_streaming(request.question, documents):
                full_answer += token
                yield {"event": "token", "data": token}

            # 4. 保存助手消息
            sources = [doc.metadata.get("source") for doc in documents]
            conversation_manager.add_message(
                session_id,
                "assistant",
                full_answer,
                metadata={"sources": sources}
            )

            # 5. 完成
            yield {"event": "done", "data": {"session_id": session_id}}

        except Exception as e:
            logger.error(f"Error in stream: {e}")
            yield {"event": "error", "data": str(e)}

    return EventSourceResponse(event_generator())

@router.post("/session", response_model=SessionResponse)
async def create_session(request: SessionCreate):
    """
    创建新会话

    参数:
        request: 创建会话请求

    返回:
        会话信息
    """
    from app.main import conversation_manager
    import uuid

    session_id = request.session_id or str(uuid.uuid4())

    session = conversation_manager.create_session(session_id, request.metadata)

    return SessionResponse(
        session_id=session["session_id"],
        created_at=session["created_at"],
        message_count=len(session["history"])
    )

@router.get("/session/{session_id}/history")
async def get_session_history(
    session_id: str,
    limit: Optional[int] = None
):
    """
    获取会话历史

    参数:
        session_id: 会话ID
        limit: 限制数量

    返回:
        消息列表
    """
    from app.main import conversation_manager

    history = conversation_manager.get_history(session_id, limit)

    if history is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"session_id": session_id, "history": history}

@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    删除会话

    参数:
        session_id: 会话ID

    返回:
        删除结果
    """
    from app.main import conversation_manager

    conversation_manager.delete_session(session_id)

    return {"message": "Session deleted", "session_id": session_id}
```

### 3.3 文档管理 API

```python
# app/api/documents.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import tempfile
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# ===== 请求/响应模型 =====

class IndexResponse(BaseModel):
    """索引响应"""
    success: bool
    message: str
    chunks_count: int
    file_name: str

class SearchRequest(BaseModel):
    """搜索请求"""
    query: str = Field(..., description="搜索查询")
    k: Optional[int] = Field(5, description="返回数量")

class DocumentMetadata(BaseModel):
    """文档元数据"""
    source: str
    file_name: str
    chunk_index: int
    chunk_total: int

class SearchResult(BaseModel):
    """搜索结果"""
    content: str
    metadata: DocumentMetadata
    score: Optional[float] = None

class SearchResponse(BaseModel):
    """搜索响应"""
    query: str
    results: List[SearchResult]
    total: int

# ===== API 端点 =====

@router.post("/upload", response_model=IndexResponse)
async def upload_document(
    file: UploadFile = File(...),
    metadata: Optional[str] = None
):
    """
    上传并索引文档

    参数:
        file: 上传的文件
        metadata: 元数据 JSON 字符串

    返回:
        索引结果
    """
    from app.main import document_processor, retrieval_system
    import json

    try:
        # 1. 验证文件类型
        file_ext = os.path.splitext(file.filename)[1].lower()
        from app.config import get_settings
        settings = get_settings()

        if file_ext not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_ext}"
            )

        # 2. 保存文件到临时目录
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        # 3. 解析元数据
        doc_metadata = json.loads(metadata) if metadata else {}
        doc_metadata["original_filename"] = file.filename

        # 4. 处理文档
        logger.info(f"Processing file: {file.filename}")
        chunks = document_processor.process_file(tmp_file_path, doc_metadata)

        # 5. 索引到向量数据库
        logger.info(f"Indexing {len(chunks)} chunks...")
        retrieval_system.index_documents(chunks)

        # 6. 清理临时文件
        os.remove(tmp_file_path)

        return IndexResponse(
            success=True,
            message="Document indexed successfully",
            chunks_count=len(chunks),
            file_name=file.filename
        )

    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    搜索文档

    参数:
        request: 搜索请求

    返回:
        搜索结果
    """
    from app.main import retrieval_system

    try:
        # 执行搜索
        documents = retrieval_system.search(request.query, k=request.k)

        # 构建响应
        results = [
            SearchResult(
                content=doc.page_content,
                metadata=DocumentMetadata(
                    source=doc.metadata.get("source", "Unknown"),
                    file_name=doc.metadata.get("file_name", "Unknown"),
                    chunk_index=doc.metadata.get("chunk_index", 0),
                    chunk_total=doc.metadata.get("chunk_total", 0)
                )
            )
            for doc in documents
        ]

        return SearchResponse(
            query=request.query,
            results=results,
            total=len(results)
        )

    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_stats():
    """
    获取统计信息

    返回:
        统计数据
    """
    from app.main import retrieval_system

    try:
        stats = retrieval_system.get_stats()
        return stats

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

## 第四部分:部署方案

### 4.1 Docker 部署

```dockerfile
# Dockerfile

FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY app/ ./app/

# 创建数据目录
RUN mkdir -p /app/data/documents \
    /app/data/processed \
    /app/data/vectorstore \
    /app/logs

# 暴露端口
EXPOSE 8000 9090

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动应用
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml

version: '3.8'

services:
  # RAG 应用
  rag-app:
    build: .
    ports:
      - "8000:8000"
      - "9090:9090"
    env_file:
      - .env
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - postgres
      - redis
      - chroma
    restart: unless-stopped

  # PostgreSQL 数据库
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: rag_db
      POSTGRES_USER: rag_user
      POSTGRES_PASSWORD: rag_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

  # Redis 缓存
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  # Chroma 向量数据库
  chroma:
    image: ghcr.io/chroma-core/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - chroma_data:/chroma/chroma
    environment:
      - IS_PERSISTENT=TRUE
      - PERSIST_DIRECTORY=/chroma/chroma
    restart: unless-stopped

  # Prometheus 监控
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9091:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
    restart: unless-stopped

  # Grafana 可视化
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  chroma_data:
  prometheus_data:
  grafana_data:
```

### 4.2 Kubernetes 部署(可选)

```yaml
# k8s/deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-app
  labels:
    app: rag-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-app
  template:
    metadata:
      labels:
        app: rag-app
    spec:
      containers:
      - name: rag-app
        image: your-registry/rag-app:latest
        ports:
        - containerPort: 8000
        - containerPort: 9090
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: openai-api-key
        - name: DATABASE_URL
          valueFrom:
            configMapKeyRef:
              name: rag-config
              key: database-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: rag-app-service
spec:
  selector:
    app: rag-app
  ports:
  - name: http
    port: 80
    targetPort: 8000
  - name: metrics
    port: 9090
    targetPort: 9090
  type: LoadBalancer
```

## 项目总结

### 核心特性

1. ✅ **完整的 RAG 流程**: 文档处理 → 检索 → 生成
2. ✅ **混合检索**: BM25 + 向量搜索 + 重排序
3. ✅ **多轮对话**: 会话管理和历史记录
4. ✅ **流式输出**: SSE 实时返回
5. ✅ **监控系统**: Prometheus + Grafana
6. ✅ **容器化部署**: Docker + Docker Compose
7. ✅ **可扩展性**: Kubernetes 支持

### 技术栈总结

**后端**:
- FastAPI (Web 框架)
- LangChain (RAG 框架)
- Pydantic (数据验证)

**数据库**:
- Chroma (向量数据库)
- PostgreSQL (关系数据库)
- Redis (缓存)

**监控**:
- Prometheus (指标收集)
- Grafana (可视化)
- LangSmith (追踪)

**部署**:
- Docker (容器化)
- Kubernetes (编排)

### 后续优化方向

1. **性能优化**
   - 添加缓存层
   - 批量处理
   - 异步优化

2. **功能扩展**
   - 多模态支持(图片、音频)
   - 更多文档格式
   - 高级检索策略

3. **安全增强**
   - 用户认证
   - API 限流
   - 数据加密

4. **运维改进**
   - 自动扩缩容
   - 日志聚合
   - 告警系统

---

**恭喜!** 🎉 你已经完成了一个企业级 RAG 系统的开发!

**下一步**:
1. 克隆代码仓库
2. 配置环境变量
3. 启动 Docker Compose
4. 上传文档测试
5. 部署到生产环境

**源码地址**: https://github.com/your-repo/enterprise-rag-system
