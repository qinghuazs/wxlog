---
title: LangGraph 项目模板与脚手架
date: 2025-09-30
permalink: /ai/langgraph/project-template.html
categories:
  - AI
  - LangGraph
---

# LangGraph 项目模板与脚手架

## 概述

本文档提供完整的 LangGraph 项目模板和脚手架工具，帮助快速启动新项目。

```mermaid
graph TB
    subgraph "项目结构"
        A[项目根目录] --> B[src/]
        A --> C[tests/]
        A --> D[configs/]
        A --> E[docs/]
        A --> F[scripts/]

        B --> G[workflows/]
        B --> H[nodes/]
        B --> I[utils/]
        B --> J[api/]

        C --> K[unit/]
        C --> L[integration/]
        C --> M[e2e/]
    end

    style A fill:#f9f,stroke:#333,stroke-width:4px
    style B fill:#9cf,stroke:#333,stroke-width:2px
    style G fill:#fc9,stroke:#333,stroke-width:2px
```

## 1. 基础项目模板

### 1.1 项目目录结构

```bash
langgraph-project/
├── src/
│   ├── __init__.py
│   ├── workflows/           # 工作流定义
│   │   ├── __init__.py
│   │   ├── base.py         # 基础工作流类
│   │   ├── main_workflow.py
│   │   └── sub_workflows/
│   ├── nodes/              # 节点实现
│   │   ├── __init__.py
│   │   ├── llm_nodes.py
│   │   ├── tool_nodes.py
│   │   ├── validation_nodes.py
│   │   └── custom_nodes.py
│   ├── states/             # 状态定义
│   │   ├── __init__.py
│   │   ├── base_state.py
│   │   └── workflow_states.py
│   ├── utils/              # 工具函数
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   ├── validators.py
│   │   └── helpers.py
│   ├── api/                # API接口
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── routes.py
│   │   └── models.py
│   └── config/             # 配置管理
│       ├── __init__.py
│       ├── settings.py
│       └── constants.py
├── tests/                   # 测试
│   ├── __init__.py
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── configs/                 # 配置文件
│   ├── development.yaml
│   ├── production.yaml
│   └── test.yaml
├── docs/                    # 文档
│   ├── API.md
│   ├── ARCHITECTURE.md
│   └── DEPLOYMENT.md
├── scripts/                 # 脚本
│   ├── setup.sh
│   ├── deploy.sh
│   └── test.sh
├── docker/                  # Docker相关
│   ├── Dockerfile
│   └── docker-compose.yml
├── .env.example            # 环境变量示例
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── Makefile
└── README.md
```

### 1.2 基础工作流类

```python
# src/workflows/base.py
from abc import ABC, abstractmethod
from typing import TypedDict, Any, Dict, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import BaseCheckpointSaver
import logging

logger = logging.getLogger(__name__)

class BaseWorkflow(ABC):
    """基础工作流抽象类"""

    def __init__(
        self,
        name: str,
        state_class: type[TypedDict],
        checkpointer: Optional[BaseCheckpointSaver] = None
    ):
        self.name = name
        self.state_class = state_class
        self.checkpointer = checkpointer
        self.workflow = StateGraph(state_class)
        self.app = None
        self._setup_workflow()

    @abstractmethod
    def _setup_workflow(self):
        """设置工作流结构"""
        pass

    @abstractmethod
    def _add_nodes(self):
        """添加节点"""
        pass

    @abstractmethod
    def _add_edges(self):
        """添加边"""
        pass

    def compile(self) -> Any:
        """编译工作流"""
        if self.app is None:
            self.app = self.workflow.compile(checkpointer=self.checkpointer)
            logger.info(f"Workflow '{self.name}' compiled successfully")
        return self.app

    def invoke(self, input_state: Dict, config: Optional[Dict] = None) -> Dict:
        """执行工作流"""
        if self.app is None:
            self.compile()

        logger.info(f"Invoking workflow '{self.name}'")
        try:
            result = self.app.invoke(input_state, config)
            logger.info(f"Workflow '{self.name}' completed successfully")
            return result
        except Exception as e:
            logger.error(f"Workflow '{self.name}' failed: {e}")
            raise

    async def ainvoke(self, input_state: Dict, config: Optional[Dict] = None) -> Dict:
        """异步执行工作流"""
        if self.app is None:
            self.compile()

        logger.info(f"Async invoking workflow '{self.name}'")
        try:
            result = await self.app.ainvoke(input_state, config)
            logger.info(f"Workflow '{self.name}' completed successfully")
            return result
        except Exception as e:
            logger.error(f"Workflow '{self.name}' failed: {e}")
            raise

    def get_graph_image(self) -> bytes:
        """获取工作流图像"""
        if self.app is None:
            self.compile()
        return self.app.get_graph().draw_mermaid_png()
```

### 1.3 状态管理模板

```python
# src/states/base_state.py
from typing import TypedDict, List, Dict, Optional, Annotated, Any
from operator import add
from datetime import datetime

class BaseState(TypedDict):
    """基础状态类"""
    # 基础字段
    id: str
    created_at: datetime
    updated_at: datetime

    # 执行状态
    status: str  # pending, running, completed, failed
    error: Optional[str]

    # 追踪信息
    execution_path: Annotated[List[str], add]
    metrics: Dict[str, Any]

class WorkflowState(BaseState):
    """工作流状态"""
    # 输入输出
    input_data: Dict
    output_data: Dict

    # 中间结果
    intermediate_results: Annotated[List[Dict], add]

    # 上下文
    context: Dict
    metadata: Dict

class ConversationState(BaseState):
    """对话状态"""
    messages: Annotated[List[Dict], add]
    user_info: Dict
    session_id: str
    turn_count: int
    conversation_summary: Optional[str]

class ProcessingState(BaseState):
    """处理状态"""
    documents: List[Dict]
    processed_chunks: Annotated[List[Dict], add]
    embeddings: Optional[List[List[float]]]
    search_results: List[Dict]
    final_result: Optional[str]
```

### 1.4 配置管理

```python
# src/config/settings.py
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import Optional, List, Dict
import os
from pathlib import Path

class Settings(BaseSettings):
    """应用配置"""

    # 基础配置
    app_name: str = Field(default="LangGraph App", env="APP_NAME")
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # API配置
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_prefix: str = Field(default="/api/v1", env="API_PREFIX")
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")

    # LLM配置
    llm_provider: str = Field(default="openai", env="LLM_PROVIDER")
    llm_model: str = Field(default="gpt-3.5-turbo", env="LLM_MODEL")
    llm_temperature: float = Field(default=0.7, env="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=2000, env="LLM_MAX_TOKENS")
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")

    # 数据库配置
    database_url: str = Field(
        default="sqlite:///./app.db",
        env="DATABASE_URL"
    )
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")

    # 向量数据库配置
    vector_db_type: str = Field(default="chroma", env="VECTOR_DB_TYPE")
    vector_db_url: Optional[str] = Field(default=None, env="VECTOR_DB_URL")
    embedding_model: str = Field(
        default="text-embedding-ada-002",
        env="EMBEDDING_MODEL"
    )

    # 性能配置
    max_concurrent_workflows: int = Field(default=10, env="MAX_CONCURRENT_WORKFLOWS")
    workflow_timeout: int = Field(default=300, env="WORKFLOW_TIMEOUT")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")

    # 安全配置
    secret_key: str = Field(..., env="SECRET_KEY")
    api_keys: List[str] = Field(default=[], env="API_KEYS")
    enable_auth: bool = Field(default=True, env="ENABLE_AUTH")

    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @validator("api_keys", pre=True)
    def parse_api_keys(cls, v):
        if isinstance(v, str):
            return [key.strip() for key in v.split(",")]
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# 单例模式
_settings = None

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
```

## 2. CLI 脚手架工具

### 2.1 脚手架主程序

```python
# scripts/langgraph_cli.py
#!/usr/bin/env python3
"""
LangGraph CLI - 项目脚手架工具
"""

import click
import os
import shutil
from pathlib import Path
import subprocess
import yaml
import json

TEMPLATE_DIR = Path(__file__).parent / "templates"

@click.group()
def cli():
    """LangGraph 项目脚手架工具"""
    pass

@cli.command()
@click.argument("project_name")
@click.option("--template", default="basic", help="项目模板类型")
@click.option("--llm", default="openai", help="LLM提供商")
@click.option("--with-docker", is_flag=True, help="包含Docker配置")
@click.option("--with-tests", is_flag=True, help="包含测试框架")
def new(project_name, template, llm, with_docker, with_tests):
    """创建新的LangGraph项目"""
    click.echo(f"Creating new LangGraph project: {project_name}")

    # 创建项目目录
    project_path = Path(project_name)
    if project_path.exists():
        click.echo(f"Error: Directory {project_name} already exists", err=True)
        return

    project_path.mkdir()

    # 复制模板文件
    template_path = TEMPLATE_DIR / template
    if not template_path.exists():
        click.echo(f"Error: Template {template} not found", err=True)
        return

    # 复制基础结构
    for item in template_path.iterdir():
        if item.is_dir():
            shutil.copytree(item, project_path / item.name)
        else:
            shutil.copy2(item, project_path)

    # 生成配置文件
    generate_config(project_path, project_name, llm)

    # 添加Docker支持
    if with_docker:
        add_docker_support(project_path, project_name)

    # 添加测试框架
    if with_tests:
        add_test_framework(project_path)

    # 初始化git仓库
    subprocess.run(["git", "init"], cwd=project_path, check=True)

    # 安装依赖
    click.echo("Installing dependencies...")
    subprocess.run(["pip", "install", "-r", "requirements.txt"], cwd=project_path)

    click.echo(f"""
✅ Project {project_name} created successfully!

Next steps:
1. cd {project_name}
2. cp .env.example .env
3. Edit .env with your configuration
4. python -m src.api.app (to start the server)

Happy coding! 🚀
    """)

@cli.command()
@click.option("--name", prompt="Workflow name", help="工作流名称")
@click.option("--type", default="basic", help="工作流类型")
def add_workflow(name, type):
    """添加新的工作流"""
    click.echo(f"Adding workflow: {name}")

    workflow_file = f"src/workflows/{name.lower()}_workflow.py"
    state_file = f"src/states/{name.lower()}_state.py"

    # 生成工作流代码
    workflow_code = generate_workflow_code(name, type)
    with open(workflow_file, 'w') as f:
        f.write(workflow_code)

    # 生成状态代码
    state_code = generate_state_code(name)
    with open(state_file, 'w') as f:
        f.write(state_code)

    click.echo(f"✅ Workflow {name} added successfully!")

@cli.command()
@click.option("--name", prompt="Node name", help="节点名称")
@click.option("--type", default="process", help="节点类型")
def add_node(name, type):
    """添加新的节点"""
    click.echo(f"Adding node: {name}")

    node_code = generate_node_code(name, type)

    # 添加到相应的节点文件
    node_file = f"src/nodes/{type}_nodes.py"

    with open(node_file, 'a') as f:
        f.write(f"\n\n{node_code}")

    click.echo(f"✅ Node {name} added successfully!")

def generate_config(project_path: Path, project_name: str, llm: str):
    """生成配置文件"""

    # .env.example
    env_content = f"""
# Application
APP_NAME={project_name}
ENVIRONMENT=development
DEBUG=True
LOG_LEVEL=INFO

# API
API_HOST=0.0.0.0
API_PORT=8000
API_PREFIX=/api/v1
CORS_ORIGINS=*

# LLM
LLM_PROVIDER={llm}
LLM_MODEL=gpt-3.5-turbo
LLM_TEMPERATURE=0.7
OPENAI_API_KEY=your_api_key_here

# Database
DATABASE_URL=sqlite:///./app.db
REDIS_URL=redis://localhost:6379

# Security
SECRET_KEY=your_secret_key_here
API_KEYS=
ENABLE_AUTH=False
"""

    with open(project_path / ".env.example", 'w') as f:
        f.write(env_content.strip())

    # requirements.txt
    requirements = """
langgraph>=0.0.20
langchain>=0.1.0
langchain-openai>=0.0.5
pydantic>=2.0.0
pydantic-settings>=2.0.0
fastapi>=0.104.0
uvicorn>=0.24.0
python-dotenv>=1.0.0
redis>=5.0.0
sqlalchemy>=2.0.0
chromadb>=0.4.0
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
black>=23.0.0
"""

    with open(project_path / "requirements.txt", 'w') as f:
        f.write(requirements.strip())

def generate_workflow_code(name: str, workflow_type: str) -> str:
    """生成工作流代码"""

    return f"""
from typing import Dict, Optional
from langgraph.graph import END
from src.workflows.base import BaseWorkflow
from src.states.{name.lower()}_state import {name}State
import logging

logger = logging.getLogger(__name__)

class {name}Workflow(BaseWorkflow):
    \"\"\"
    {name} 工作流实现
    \"\"\"

    def __init__(self, checkpointer=None):
        super().__init__(
            name="{name}",
            state_class={name}State,
            checkpointer=checkpointer
        )

    def _setup_workflow(self):
        \"\"\"设置工作流\"\"\"
        self._add_nodes()
        self._add_edges()

    def _add_nodes(self):
        \"\"\"添加节点\"\"\"
        self.workflow.add_node("initialize", self._initialize_node)
        self.workflow.add_node("process", self._process_node)
        self.workflow.add_node("finalize", self._finalize_node)

    def _add_edges(self):
        \"\"\"添加边\"\"\"
        self.workflow.set_entry_point("initialize")
        self.workflow.add_edge("initialize", "process")
        self.workflow.add_edge("process", "finalize")
        self.workflow.add_edge("finalize", END)

    def _initialize_node(self, state: {name}State) -> Dict:
        \"\"\"初始化节点\"\"\"
        logger.info("Initializing {name} workflow")
        return {{
            "status": "initialized",
            "execution_path": ["initialize"]
        }}

    def _process_node(self, state: {name}State) -> Dict:
        \"\"\"处理节点\"\"\"
        logger.info("Processing in {name} workflow")
        # 实现处理逻辑
        return {{
            "status": "processing",
            "execution_path": ["process"]
        }}

    def _finalize_node(self, state: {name}State) -> Dict:
        \"\"\"完成节点\"\"\"
        logger.info("Finalizing {name} workflow")
        return {{
            "status": "completed",
            "execution_path": ["finalize"],
            "output_data": {{"result": "success"}}
        }}
"""

def generate_state_code(name: str) -> str:
    """生成状态代码"""

    return f"""
from typing import TypedDict, List, Dict, Optional, Annotated
from operator import add
from src.states.base_state import BaseState

class {name}State(BaseState):
    \"\"\"
    {name} 工作流状态
    \"\"\"

    # 添加自定义状态字段
    custom_field: str
    processing_results: Annotated[List[Dict], add]

    # 可选字段
    optional_data: Optional[Dict]
"""

def generate_node_code(name: str, node_type: str) -> str:
    """生成节点代码"""

    return f"""
def {name.lower()}_node(state: Dict) -> Dict:
    \"\"\"
    {name} 节点

    Args:
        state: 当前状态

    Returns:
        状态更新
    \"\"\"
    import logging
    logger = logging.getLogger(__name__)

    logger.info("Executing {name} node")

    # 实现节点逻辑
    try:
        # TODO: 实现具体逻辑
        result = {{"processed": True}}

        return {{
            "execution_path": ["{name.lower()}"],
            "{name.lower()}_result": result
        }}
    except Exception as e:
        logger.error(f"Error in {name} node: {{e}}")
        return {{"error": str(e)}}
"""
```

## 3. Docker 模板

### 3.1 Dockerfile

```dockerfile
# docker/Dockerfile
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建非root用户
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# 启动命令
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 3.2 Docker Compose

```yaml
# docker/docker-compose.yml
version: '3.8'

services:
  app:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=development
      - DATABASE_URL=postgresql://user:password@postgres:5432/langgraph
      - REDIS_URL=redis://redis:6379
    volumes:
      - ../src:/app/src
      - ../configs:/app/configs
    depends_on:
      - postgres
      - redis
    networks:
      - langgraph-network

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: langgraph
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    networks:
      - langgraph-network

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - langgraph-network

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - app
    networks:
      - langgraph-network

volumes:
  postgres_data:
  redis_data:

networks:
  langgraph-network:
    driver: bridge
```

## 4. API 模板

### 4.1 FastAPI 应用

```python
# src/api/app.py
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from typing import Dict, Any, List

from src.config.settings import get_settings
from src.api.routes import workflow_router, health_router
from src.api.models import WorkflowRequest, WorkflowResponse
from src.utils.logger import setup_logging

# 设置日志
setup_logging()
logger = logging.getLogger(__name__)

# 获取配置
settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("Starting LangGraph application...")
    # 初始化资源
    yield
    # 清理资源
    logger.info("Shutting down LangGraph application...")

# 创建应用
app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    lifespan=lifespan
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(health_router, tags=["health"])
app.include_router(
    workflow_router,
    prefix=settings.api_prefix,
    tags=["workflows"]
)

@app.get("/")
async def root():
    """根路径"""
    return {
        "name": settings.app_name,
        "environment": settings.environment,
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )
```

### 4.2 路由定义

```python
# src/api/routes.py
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, List
import logging

from src.api.models import (
    WorkflowRequest,
    WorkflowResponse,
    HealthResponse
)
from src.workflows.registry import WorkflowRegistry

logger = logging.getLogger(__name__)

# 健康检查路由
health_router = APIRouter()

@health_router.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    return HealthResponse(
        status="healthy",
        services={
            "api": "up",
            "workflows": "up"
        }
    )

# 工作流路由
workflow_router = APIRouter()
registry = WorkflowRegistry()

@workflow_router.post("/workflows/{workflow_name}/invoke", response_model=WorkflowResponse)
async def invoke_workflow(
    workflow_name: str,
    request: WorkflowRequest,
    background_tasks: BackgroundTasks
):
    """执行工作流"""
    try:
        workflow = registry.get_workflow(workflow_name)
        if not workflow:
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_name} not found")

        result = await workflow.ainvoke(request.input_data, request.config)

        return WorkflowResponse(
            workflow_name=workflow_name,
            status="completed",
            result=result
        )
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@workflow_router.get("/workflows", response_model=List[str])
async def list_workflows():
    """列出所有工作流"""
    return registry.list_workflows()

@workflow_router.get("/workflows/{workflow_name}/graph")
async def get_workflow_graph(workflow_name: str):
    """获取工作流图"""
    workflow = registry.get_workflow(workflow_name)
    if not workflow:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_name} not found")

    image = workflow.get_graph_image()
    return {"graph": image.decode() if isinstance(image, bytes) else image}
```

## 5. 测试模板

### 5.1 测试配置

```python
# tests/conftest.py
import pytest
import asyncio
from typing import Generator, AsyncGenerator
from unittest.mock import Mock, AsyncMock

from src.config.settings import Settings
from src.workflows.base import BaseWorkflow

@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def test_settings():
    """测试配置"""
    return Settings(
        environment="test",
        debug=True,
        openai_api_key="test_key",
        secret_key="test_secret"
    )

@pytest.fixture
def mock_workflow():
    """模拟工作流"""
    workflow = Mock(spec=BaseWorkflow)
    workflow.invoke.return_value = {"result": "success"}
    workflow.ainvoke = AsyncMock(return_value={"result": "success"})
    return workflow

@pytest.fixture
def sample_state():
    """示例状态"""
    return {
        "id": "test_id",
        "input_data": {"test": "data"},
        "status": "pending"
    }
```

### 5.2 工作流测试

```python
# tests/unit/test_workflows.py
import pytest
from src.workflows.base import BaseWorkflow
from src.states.base_state import WorkflowState

class TestBaseWorkflow:
    """基础工作流测试"""

    def test_workflow_creation(self):
        """测试工作流创建"""

        class TestWorkflow(BaseWorkflow):
            def _setup_workflow(self):
                self._add_nodes()
                self._add_edges()

            def _add_nodes(self):
                self.workflow.add_node("test", lambda s: {"result": "test"})

            def _add_edges(self):
                self.workflow.set_entry_point("test")
                self.workflow.add_edge("test", "__end__")

        workflow = TestWorkflow("test", WorkflowState)
        assert workflow.name == "test"
        assert workflow.state_class == WorkflowState

    @pytest.mark.asyncio
    async def test_async_invoke(self, mock_workflow, sample_state):
        """测试异步执行"""
        result = await mock_workflow.ainvoke(sample_state)
        assert result["result"] == "success"
        mock_workflow.ainvoke.assert_called_once_with(sample_state, None)
```

## 6. Makefile

```makefile
# Makefile
.PHONY: help install test lint format run docker-build docker-up docker-down clean

help:
	@echo "Available commands:"
	@echo "  install      Install dependencies"
	@echo "  test         Run tests"
	@echo "  lint         Run linters"
	@echo "  format       Format code"
	@echo "  run          Run the application"
	@echo "  docker-build Build Docker image"
	@echo "  docker-up    Start Docker services"
	@echo "  docker-down  Stop Docker services"
	@echo "  clean        Clean up temporary files"

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v --cov=src --cov-report=html

lint:
	pylint src/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

run:
	python -m src.api.app

docker-build:
	docker-compose -f docker/docker-compose.yml build

docker-up:
	docker-compose -f docker/docker-compose.yml up -d

docker-down:
	docker-compose -f docker/docker-compose.yml down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
```

## 7. 项目初始化脚本

```bash
#!/bin/bash
# scripts/init_project.sh

echo "🚀 Initializing LangGraph Project..."

# 创建虚拟环境
echo "Creating virtual environment..."
python -m venv venv
source venv/bin/activate

# 安装依赖
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 复制环境变量
echo "Setting up environment..."
cp .env.example .env

# 初始化数据库
echo "Initializing database..."
python scripts/init_db.py

# 运行测试
echo "Running tests..."
pytest tests/

# 格式化代码
echo "Formatting code..."
black src/ tests/

echo "✅ Project initialization complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your configuration"
echo "2. Run 'make run' to start the application"
echo "3. Visit http://localhost:8000/docs for API documentation"
```

## 总结

这个项目模板和脚手架提供了：

1. **完整的项目结构**：清晰的目录组织
2. **基础类和抽象**：可复用的工作流基类
3. **配置管理**：灵活的配置系统
4. **CLI工具**：快速创建项目和组件
5. **Docker支持**：容器化部署
6. **API模板**：FastAPI集成
7. **测试框架**：完整的测试支持
8. **开发工具**：Makefile和初始化脚本

通过这个脚手架，可以快速启动一个生产级的 LangGraph 项目，专注于业务逻辑的实现。