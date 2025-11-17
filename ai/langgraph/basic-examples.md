---
title: LangGraph 基础案例实战
date: 2025-01-30
categories:
  - AI
  - LangGraph
---

# LangGraph 基础案例实战

## 一、概述

本章通过三个循序渐进的实战案例，帮助你掌握 LangGraph 的基础应用开发。

## 二、案例一：智能待办事项管理器

### 2.1 需求说明

创建一个智能待办事项管理器，支持添加、完成、优先级排序等功能。

### 2.2 完整实现

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Optional, Annotated
import operator
from datetime import datetime
from enum import Enum

# 定义优先级枚举
class Priority(Enum):
    HIGH = 1
    MEDIUM = 2
    LOW = 3

# 定义任务结构
class Task(TypedDict):
    id: int
    title: str
    description: str
    priority: Priority
    completed: bool
    created_at: str
    completed_at: Optional[str]

# 定义状态
class TodoState(TypedDict):
    tasks: List[Task]
    current_command: str
    command_params: Dict
    last_action: str
    message: Annotated[List[str], operator.add]

# 节点函数
def parse_command(state: TodoState) -> TodoState:
    """解析用户命令"""
    command = state["current_command"].lower().strip()
    params = {}

    if command.startswith("add"):
        parts = command[3:].strip().split("|")
        params = {
            "action": "add",
            "title": parts[0] if len(parts) > 0 else "",
            "description": parts[1] if len(parts) > 1 else "",
            "priority": parts[2] if len(parts) > 2 else "medium"
        }
    elif command.startswith("complete"):
        task_id = command[8:].strip()
        params = {"action": "complete", "task_id": int(task_id) if task_id.isdigit() else 0}
    elif command == "list":
        params = {"action": "list"}
    elif command == "list high":
        params = {"action": "list", "filter": "high"}
    elif command.startswith("delete"):
        task_id = command[6:].strip()
        params = {"action": "delete", "task_id": int(task_id) if task_id.isdigit() else 0}
    else:
        params = {"action": "unknown"}

    return {
        "command_params": params,
        "message": [f"解析命令: {params.get('action', 'unknown')}"]
    }

def add_task(state: TodoState) -> TodoState:
    """添加新任务"""
    params = state["command_params"]

    # 生成新ID
    max_id = max([t["id"] for t in state["tasks"]], default=0)

    # 转换优先级
    priority_map = {
        "high": Priority.HIGH,
        "medium": Priority.MEDIUM,
        "low": Priority.LOW
    }

    new_task = Task(
        id=max_id + 1,
        title=params["title"],
        description=params["description"],
        priority=priority_map.get(params["priority"], Priority.MEDIUM),
        completed=False,
        created_at=datetime.now().isoformat(),
        completed_at=None
    )

    state["tasks"].append(new_task)
    return {
        "last_action": "add",
        "message": [f"✅ 添加任务: {new_task['title']} (ID: {new_task['id']})"]
    }

def complete_task(state: TodoState) -> TodoState:
    """完成任务"""
    task_id = state["command_params"]["task_id"]

    for task in state["tasks"]:
        if task["id"] == task_id:
            task["completed"] = True
            task["completed_at"] = datetime.now().isoformat()
            return {
                "last_action": "complete",
                "message": [f"✅ 完成任务: {task['title']}"]
            }

    return {
        "last_action": "complete",
        "message": [f"❌ 未找到任务 ID: {task_id}"]
    }

def delete_task(state: TodoState) -> TodoState:
    """删除任务"""
    task_id = state["command_params"]["task_id"]

    for i, task in enumerate(state["tasks"]):
        if task["id"] == task_id:
            deleted = state["tasks"].pop(i)
            return {
                "last_action": "delete",
                "message": [f"🗑️ 删除任务: {deleted['title']}"]
            }

    return {
        "last_action": "delete",
        "message": [f"❌ 未找到任务 ID: {task_id}"]
    }

def list_tasks(state: TodoState) -> TodoState:
    """列出任务"""
    filter_type = state["command_params"].get("filter")
    tasks = state["tasks"]

    # 过滤任务
    if filter_type == "high":
        tasks = [t for t in tasks if t["priority"] == Priority.HIGH]

    # 排序任务（按优先级和创建时间）
    tasks.sort(key=lambda x: (x["priority"].value, x["created_at"]))

    # 格式化输出
    messages = ["📋 任务列表:"]
    if not tasks:
        messages.append("  暂无任务")
    else:
        for task in tasks:
            status = "✅" if task["completed"] else "⏳"
            priority_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
            p_emoji = priority_emoji.get(task["priority"].name, "")
            messages.append(
                f"  {status} [{task['id']}] {p_emoji} {task['title']}"
            )
            if task["description"]:
                messages.append(f"      {task['description']}")

    return {
        "last_action": "list",
        "message": messages
    }

def handle_unknown(state: TodoState) -> TodoState:
    """处理未知命令"""
    return {
        "last_action": "unknown",
        "message": [
            "❓ 未知命令。支持的命令:",
            "  - add 标题|描述|优先级",
            "  - complete ID",
            "  - delete ID",
            "  - list",
            "  - list high"
        ]
    }

# 路由函数
def route_command(state: TodoState) -> str:
    action = state["command_params"].get("action")

    route_map = {
        "add": "add_task",
        "complete": "complete_task",
        "delete": "delete_task",
        "list": "list_tasks",
        "unknown": "handle_unknown"
    }

    return route_map.get(action, "handle_unknown")

# 创建图
def create_todo_manager():
    graph = StateGraph(TodoState)

    # 添加节点
    graph.add_node("parse", parse_command)
    graph.add_node("add_task", add_task)
    graph.add_node("complete_task", complete_task)
    graph.add_node("delete_task", delete_task)
    graph.add_node("list_tasks", list_tasks)
    graph.add_node("handle_unknown", handle_unknown)

    # 添加边
    graph.add_conditional_edges(
        "parse",
        route_command,
        {
            "add_task": "add_task",
            "complete_task": "complete_task",
            "delete_task": "delete_task",
            "list_tasks": "list_tasks",
            "handle_unknown": "handle_unknown"
        }
    )

    # 所有操作节点都结束
    for node in ["add_task", "complete_task", "delete_task", "list_tasks", "handle_unknown"]:
        graph.add_edge(node, END)

    graph.set_entry_point("parse")

    return graph.compile()

# 使用示例
def run_todo_manager():
    app = create_todo_manager()

    # 初始状态
    state = {
        "tasks": [],
        "message": []
    }

    # 测试命令
    commands = [
        "add 学习 LangGraph|阅读官方文档|high",
        "add 写代码|完成实战项目|medium",
        "add 休息|喝杯咖啡|low",
        "list",
        "complete 1",
        "list high",
        "delete 3",
        "list"
    ]

    for cmd in commands:
        print(f"\n执行命令: {cmd}")
        state["current_command"] = cmd
        state["message"] = []

        result = app.invoke(state)
        state = result

        for msg in result["message"]:
            print(msg)

# 运行
if __name__ == "__main__":
    run_todo_manager()
```

## 三、案例二：多步骤表单验证系统

### 3.1 需求说明

创建一个多步骤表单验证系统，包括格式验证、业务规则验证和数据完整性检查。

### 3.2 实现代码

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Annotated
import operator
import re
from datetime import datetime

class FormState(TypedDict):
    # 表单数据
    name: str
    email: str
    phone: str
    age: int
    country: str

    # 验证结果
    validation_errors: Annotated[List[str], operator.add]
    validation_warnings: Annotated[List[str], operator.add]
    validation_passed: bool

    # 处理步骤
    current_step: str
    steps_completed: Annotated[List[str], operator.add]

def validate_format(state: FormState) -> FormState:
    """格式验证"""
    errors = []
    warnings = []

    # 姓名验证
    if not state.get("name"):
        errors.append("姓名不能为空")
    elif len(state["name"]) < 2:
        errors.append("姓名至少需要2个字符")
    elif len(state["name"]) > 50:
        warnings.append("姓名过长，建议不超过50个字符")

    # 邮箱验证
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not state.get("email"):
        errors.append("邮箱不能为空")
    elif not re.match(email_pattern, state["email"]):
        errors.append("邮箱格式不正确")

    # 电话验证
    phone_pattern = r'^[\d\s\-\+\(\)]+$'
    if not state.get("phone"):
        errors.append("电话不能为空")
    elif not re.match(phone_pattern, state["phone"]):
        errors.append("电话格式不正确")
    elif len(re.sub(r'\D', '', state["phone"])) < 10:
        errors.append("电话号码至少需要10位数字")

    return {
        "validation_errors": errors,
        "validation_warnings": warnings,
        "steps_completed": ["format_validation"],
        "current_step": "format_validation"
    }

def validate_business_rules(state: FormState) -> FormState:
    """业务规则验证"""
    errors = []
    warnings = []

    # 年龄验证
    if state.get("age") is None:
        errors.append("年龄不能为空")
    elif state["age"] < 0:
        errors.append("年龄不能为负数")
    elif state["age"] < 18:
        errors.append("必须年满18岁")
    elif state["age"] > 120:
        warnings.append("年龄似乎不太合理")

    # 国家验证
    valid_countries = ["中国", "美国", "英国", "日本", "德国", "法国"]
    if not state.get("country"):
        errors.append("国家不能为空")
    elif state["country"] not in valid_countries:
        warnings.append(f"国家 '{state['country']}' 不在常用列表中")

    # 特殊业务规则
    if state.get("age") and state["age"] < 21 and state.get("country") == "美国":
        warnings.append("在美国，21岁以下可能有某些限制")

    return {
        "validation_errors": errors,
        "validation_warnings": warnings,
        "steps_completed": ["business_validation"],
        "current_step": "business_validation"
    }

def validate_data_integrity(state: FormState) -> FormState:
    """数据完整性检查"""
    errors = []
    warnings = []

    # 检查邮箱唯一性（模拟）
    existing_emails = ["test@example.com", "admin@company.com"]
    if state.get("email") in existing_emails:
        errors.append("该邮箱已被注册")

    # 检查数据一致性
    if state.get("country") == "中国":
        if state.get("phone") and not state["phone"].startswith("+86"):
            warnings.append("中国电话通常以 +86 开头")

    # 交叉验证
    if state.get("email"):
        email_domain = state["email"].split("@")[-1]
        if ".cn" in email_domain and state.get("country") != "中国":
            warnings.append("邮箱域名与国家不匹配")

    return {
        "validation_errors": errors,
        "validation_warnings": warnings,
        "steps_completed": ["integrity_check"],
        "current_step": "integrity_check"
    }

def generate_summary(state: FormState) -> FormState:
    """生成验证摘要"""
    total_errors = len(state.get("validation_errors", []))
    total_warnings = len(state.get("validation_warnings", []))

    validation_passed = total_errors == 0

    summary = [
        "=" * 40,
        "表单验证摘要",
        "=" * 40,
        f"错误数量: {total_errors}",
        f"警告数量: {total_warnings}",
        f"验证结果: {'✅ 通过' if validation_passed else '❌ 未通过'}"
    ]

    if state.get("validation_errors"):
        summary.append("\n❌ 错误列表:")
        for error in state["validation_errors"]:
            summary.append(f"  - {error}")

    if state.get("validation_warnings"):
        summary.append("\n⚠️ 警告列表:")
        for warning in state["validation_warnings"]:
            summary.append(f"  - {warning}")

    print("\n".join(summary))

    return {
        "validation_passed": validation_passed,
        "steps_completed": ["summary"],
        "current_step": "summary"
    }

def route_after_format(state: FormState) -> str:
    """格式验证后的路由"""
    # 获取当前步骤的错误
    errors = [e for e in state.get("validation_errors", [])
              if "格式" in e or "不能为空" in e]

    if len(errors) > 3:  # 如果格式错误太多，直接结束
        return "summary"
    return "business_rules"

def route_after_business(state: FormState) -> str:
    """业务规则验证后的路由"""
    critical_errors = [e for e in state.get("validation_errors", [])
                      if "必须" in e or "不能" in e]

    if critical_errors:
        return "summary"
    return "data_integrity"

# 创建验证流程
def create_form_validator():
    graph = StateGraph(FormState)

    # 添加节点
    graph.add_node("format", validate_format)
    graph.add_node("business_rules", validate_business_rules)
    graph.add_node("data_integrity", validate_data_integrity)
    graph.add_node("summary", generate_summary)

    # 添加条件边
    graph.add_conditional_edges(
        "format",
        route_after_format,
        {
            "business_rules": "business_rules",
            "summary": "summary"
        }
    )

    graph.add_conditional_edges(
        "business_rules",
        route_after_business,
        {
            "data_integrity": "data_integrity",
            "summary": "summary"
        }
    )

    graph.add_edge("data_integrity", "summary")
    graph.add_edge("summary", END)

    graph.set_entry_point("format")

    return graph.compile()

# 测试表单验证
def test_form_validation():
    validator = create_form_validator()

    # 测试用例
    test_cases = [
        {
            "name": "张三",
            "email": "zhangsan@example.cn",
            "phone": "+86 138 0000 0000",
            "age": 25,
            "country": "中国"
        },
        {
            "name": "J",
            "email": "invalid-email",
            "phone": "123",
            "age": 15,
            "country": "火星"
        },
        {
            "name": "John Doe",
            "email": "test@example.com",  # 已存在
            "phone": "+1 555 123 4567",
            "age": 30,
            "country": "美国"
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"测试用例 {i}: {test_case['name']}")
        print('='*50)

        initial_state = {
            **test_case,
            "validation_errors": [],
            "validation_warnings": [],
            "steps_completed": []
        }

        result = validator.invoke(initial_state)
        print(f"\n完成步骤: {result['steps_completed']}")

if __name__ == "__main__":
    test_form_validation()
```

## 四、案例三：简单对话机器人

### 4.1 需求说明

创建一个具有意图识别、上下文管理和多轮对话能力的简单机器人。

### 4.2 实现代码

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List, Annotated, Optional
import operator
import random

class ChatState(TypedDict):
    messages: Annotated[List[Dict[str, str]], operator.add]
    user_input: str
    intent: str
    context: Dict[str, any]
    turn_count: int
    should_end: bool

class IntentClassifier:
    """简单的意图分类器"""

    @staticmethod
    def classify(text: str) -> str:
        text_lower = text.lower()

        # 问候
        if any(word in text_lower for word in ["你好", "hello", "hi", "嗨"]):
            return "greeting"

        # 告别
        if any(word in text_lower for word in ["再见", "bye", "拜拜", "退出"]):
            return "farewell"

        # 询问天气
        if "天气" in text_lower or "weather" in text_lower:
            return "weather"

        # 询问时间
        if any(word in text_lower for word in ["时间", "几点", "time"]):
            return "time"

        # 询问名字
        if "名字" in text_lower or "name" in text_lower:
            return "name"

        # 帮助
        if any(word in text_lower for word in ["帮助", "help", "功能"]):
            return "help"

        # 闲聊
        return "chitchat"

def get_user_input(state: ChatState) -> ChatState:
    """获取用户输入"""
    user_input = input("\n👤 用户: ")

    return {
        "user_input": user_input,
        "messages": [{"role": "user", "content": user_input}],
        "turn_count": state.get("turn_count", 0) + 1
    }

def classify_intent(state: ChatState) -> ChatState:
    """意图识别"""
    intent = IntentClassifier.classify(state["user_input"])

    return {
        "intent": intent,
        "messages": [{"role": "system", "content": f"识别意图: {intent}"}]
    }

def handle_greeting(state: ChatState) -> ChatState:
    """处理问候"""
    greetings = [
        "你好！很高兴见到你！😊",
        "嗨！有什么我可以帮助你的吗？",
        "你好呀！今天过得怎么样？"
    ]

    response = random.choice(greetings)

    # 如果是第一次问候，记录在上下文中
    if not state.get("context", {}).get("greeted"):
        state["context"] = state.get("context", {})
        state["context"]["greeted"] = True
        response += " 我是你的智能助手。"

    return {
        "messages": [{"role": "assistant", "content": response}]
    }

def handle_farewell(state: ChatState) -> ChatState:
    """处理告别"""
    farewells = [
        "再见！祝你有个美好的一天！👋",
        "拜拜！随时欢迎回来聊天！",
        "再见！很高兴和你交谈！"
    ]

    response = random.choice(farewells)

    # 如果聊天时间较长，添加额外的告别语
    if state.get("turn_count", 0) > 5:
        response += " 感谢这次愉快的对话！"

    return {
        "messages": [{"role": "assistant", "content": response}],
        "should_end": True
    }

def handle_weather(state: ChatState) -> ChatState:
    """处理天气查询"""
    cities = ["北京", "上海", "广州", "深圳"]
    weather_types = ["晴天", "多云", "小雨", "阴天"]
    temps = range(15, 30)

    city = random.choice(cities)
    weather = random.choice(weather_types)
    temp = random.choice(temps)

    response = f"今天{city}的天气是{weather}，温度约{temp}°C。"

    # 添加建议
    if "雨" in weather:
        response += " 记得带伞哦！☔"
    elif weather == "晴天":
        response += " 适合出门活动！☀️"

    return {
        "messages": [{"role": "assistant", "content": response}]
    }

def handle_time(state: ChatState) -> ChatState:
    """处理时间查询"""
    from datetime import datetime

    now = datetime.now()
    response = f"现在是 {now.strftime('%Y年%m月%d日 %H:%M:%S')}"

    hour = now.hour
    if hour < 6:
        response += " 夜深了，注意休息！"
    elif hour < 12:
        response += " 早上好！"
    elif hour < 18:
        response += " 下午好！"
    else:
        response += " 晚上好！"

    return {
        "messages": [{"role": "assistant", "content": response}]
    }

def handle_name(state: ChatState) -> ChatState:
    """处理名字询问"""
    response = "我是 LangGraph 助手，一个基于状态图的智能对话系统！"

    # 如果用户之前提到过自己的名字，记住它
    if "user_name" in state.get("context", {}):
        response += f" 我记得你是 {state['context']['user_name']}！"

    return {
        "messages": [{"role": "assistant", "content": response}]
    }

def handle_help(state: ChatState) -> ChatState:
    """处理帮助请求"""
    response = """我可以帮你：
    1. 🌤️ 查询天气
    2. ⏰ 告诉你时间
    3. 💬 和你闲聊
    4. 👋 问候和告别

    你可以试试说：
    - "今天天气怎么样？"
    - "现在几点了？"
    - "你叫什么名字？"
    """

    return {
        "messages": [{"role": "assistant", "content": response}]
    }

def handle_chitchat(state: ChatState) -> ChatState:
    """处理闲聊"""
    responses = [
        "这很有趣！能详细说说吗？",
        "我明白了，还有什么想聊的吗？",
        "听起来不错！",
        f"你说的是：{state['user_input']}。这个话题很有意思！",
        "嗯嗯，我在听呢！"
    ]

    response = random.choice(responses)

    return {
        "messages": [{"role": "assistant", "content": response}]
    }

def route_by_intent(state: ChatState) -> str:
    """根据意图路由"""
    intent_routes = {
        "greeting": "greeting",
        "farewell": "farewell",
        "weather": "weather",
        "time": "time",
        "name": "name",
        "help": "help",
        "chitchat": "chitchat"
    }

    return intent_routes.get(state["intent"], "chitchat")

def should_continue_chat(state: ChatState) -> str:
    """决定是否继续对话"""
    if state.get("should_end"):
        return "end"

    if state.get("turn_count", 0) >= 10:
        return "farewell"  # 对话轮数过多，主动结束

    return "continue"

def print_response(state: ChatState) -> ChatState:
    """打印机器人响应"""
    for msg in state.get("messages", []):
        if msg["role"] == "assistant":
            print(f"🤖 助手: {msg['content']}")

    return state

# 创建聊天机器人
def create_chatbot():
    graph = StateGraph(ChatState)

    # 添加节点
    graph.add_node("input", get_user_input)
    graph.add_node("classify", classify_intent)
    graph.add_node("greeting", handle_greeting)
    graph.add_node("farewell", handle_farewell)
    graph.add_node("weather", handle_weather)
    graph.add_node("time", handle_time)
    graph.add_node("name", handle_name)
    graph.add_node("help", handle_help)
    graph.add_node("chitchat", handle_chitchat)
    graph.add_node("print", print_response)

    # 设置入口
    graph.set_entry_point("input")

    # 从输入到分类
    graph.add_edge("input", "classify")

    # 根据意图路由
    graph.add_conditional_edges(
        "classify",
        route_by_intent,
        {
            "greeting": "greeting",
            "farewell": "farewell",
            "weather": "weather",
            "time": "time",
            "name": "name",
            "help": "help",
            "chitchat": "chitchat"
        }
    )

    # 所有处理节点都到打印
    for handler in ["greeting", "weather", "time", "name", "help", "chitchat"]:
        graph.add_edge(handler, "print")

    # 告别特殊处理
    graph.add_edge("farewell", "print")

    # 打印后决定是否继续
    graph.add_conditional_edges(
        "print",
        should_continue_chat,
        {
            "continue": "input",
            "farewell": "farewell",
            "end": END
        }
    )

    # 使用内存检查点保存对话历史
    memory = MemorySaver()

    return graph.compile(checkpointer=memory)

# 运行聊天机器人
def run_chatbot():
    print("="*50)
    print("🤖 LangGraph 聊天机器人")
    print("="*50)
    print("输入 '再见' 或 'bye' 退出")
    print("输入 '帮助' 或 'help' 查看功能")
    print("-"*50)

    chatbot = create_chatbot()

    # 配置会话
    config = {"configurable": {"thread_id": "chat-001"}}

    # 初始状态
    initial_state = {
        "messages": [],
        "context": {},
        "turn_count": 0,
        "should_end": False
    }

    try:
        # 运行对话循环
        chatbot.invoke(initial_state, config)
    except KeyboardInterrupt:
        print("\n\n👋 对话被中断，再见！")
    except Exception as e:
        print(f"\n❌ 发生错误：{e}")

if __name__ == "__main__":
    run_chatbot()
```

## 五、案例总结与对比

### 5.1 三个案例的特点对比

| 特性 | 待办事项管理器 | 表单验证系统 | 对话机器人 |
|------|--------------|-------------|-----------|
| **状态复杂度** | 中等 | 简单 | 复杂 |
| **节点数量** | 6个 | 4个 | 10+ |
| **路由复杂度** | 简单条件 | 多条件 | 意图路由 |
| **状态持久化** | 否 | 否 | 是 |
| **用户交互** | 命令式 | 批处理 | 对话式 |
| **适用场景** | CRUD操作 | 数据验证 | 交互系统 |

### 5.2 学到的关键技能

1. **状态设计**
   - 使用 TypedDict 定义结构化状态
   - 合理使用 Annotated 和 reducer
   - 区分必要状态和临时状态

2. **节点开发**
   - 单一职责原则
   - 错误处理
   - 返回部分更新

3. **流程控制**
   - 条件路由
   - 循环控制
   - 提前退出

4. **实践技巧**
   - 模块化设计
   - 测试用例编写
   - 调试和日志

## 六、练习建议

### 6.1 扩展练习

1. **待办事项管理器**
   - 添加截止日期功能
   - 实现任务分类
   - 添加持久化存储

2. **表单验证系统**
   - 添加异步验证（如API调用）
   - 实现自定义验证规则
   - 生成验证报告

3. **对话机器人**
   - 集成真实的NLP模型
   - 添加多轮对话记忆
   - 实现个性化响应

### 6.2 进阶挑战

1. 结合三个案例，创建一个任务助手机器人
2. 添加可视化界面展示执行流程
3. 实现错误恢复和重试机制

## 七、最佳实践总结

1. **先设计后编码**：画出流程图再实现
2. **渐进式开发**：从简单功能开始逐步完善
3. **充分测试**：准备多样的测试用例
4. **文档注释**：保持代码的可读性
5. **性能意识**：避免不必要的状态更新

---

**下一步：** 深入学习 [04.状态管理详解](./04.状态管理详解.md)，掌握更高级的状态管理技巧！