import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

// https://vitepress.dev/reference/site-config
export default withMermaid(defineConfig({
  title: "冬眠日记",
  description: "祝我们平日都快乐，做平凡的人",

  // 忽略死链接检查（中文文件名导致的误报）
  ignoreDeadLinks: true,

  // 启用缓存目录
  cacheDir: '.vitepress/cache',

  // Vite 性能优化
  vite: {
    optimizeDeps: {
      include: ['vitepress-plugin-mermaid']
    },
    server: {
      hmr: {
        overlay: false // 减少 HMR 开销
      }
    },
    build: {
      chunkSizeWarningLimit: 2000 // 提高警告阈值以减少构建警告
    }
  },

  rewrites: {
    // === docs 目录 ===
    // MySQL
    'docs/database/mysql/提示词.md': 'docs/database/mysql/prompts.md',
    // Kafka
    'docs/kafka/00.提示词.md': 'docs/kafka/00.prompts.md',
    // TailwindCSS
    'docs/htmlcssjs/TailwindCSS/00提示词.md': 'docs/htmlcssjs/TailwindCSS/00prompts.md',
    // Redis
    'docs/database/redis/数据结构/:page': 'docs/database/redis/data-structures/:page',
    // 区块链
    'docs/区块链/:dir/:page': 'docs/blockchain/:dir/:page',
    // AI 提示词
    'docs/ai/提示词/:dir/:page': 'docs/ai/prompts/:dir/:page',

    // === 03.工具 目录 ===
    // Git
    '03.工具/02.Git/01.git推送本地提交到远程仓库.md': 'git/remote.md',
    '03.工具/02.Git/02.git分支常用命令.md': 'git/branch.md',
    '03.工具/02.Git/03.本地项目取消git仓库关联.md': 'git/unlink.md',
    '03.工具/02.Git/04.git重置本地的修改.md': 'git/revert.md',
    '03.工具/02.Git/05.代码提交规范.md': 'git/commit.md',
    // Docker
    '03.工具/03.Docker/01.CentOS安装Docker.md': 'docker/centos-install.md',

    // === 04.中间件 目录 ===
    // MySQL
    '04.中间件/01.数据库/01.MySQL/01.实现原理/01.索引.md': 'mysql/index.md',
    '04.中间件/01.数据库/01.MySQL/01.实现原理/02.mvcc.md': 'mysql/mvcc.md',
    '04.中间件/01.数据库/01.MySQL/01.实现原理/03.Group By 的执行原理.md': 'mysql/group-by.md',
    '04.中间件/01.数据库/01.MySQL/01.实现原理/04.SQL 解析.md': 'mysql/sql-parse.md',
    '04.中间件/01.数据库/01.MySQL/01.实现原理/05.SQL 查询优化.md': 'mysql/sql-optimization.md',
    '04.中间件/01.数据库/01.MySQL/02.应用操作/01.数据类型.md': 'mysql/data-types.md',
    '04.中间件/01.数据库/01.MySQL/02.应用操作/02.创建用户后为用户授权.md': 'mysql/user-grant.md',
    '04.中间件/01.数据库/01.MySQL/03.性能优化/01.Explain的使用.md': 'mysql/explain.md',
    '04.中间件/01.数据库/01.MySQL/04.问题处理/01.死锁.md': 'mysql/deadlock.md',
    // Redis
    '04.中间件/01.数据库/02.Redis/01.配置应用/01.安装部署.md': 'redis/installation.md',
    '04.中间件/01.数据库/02.Redis/02.数据结构/01.sds.md': 'redis/sds.md',
    '04.中间件/01.数据库/02.Redis/02.数据结构/02.Redis 代码整体架构.md': 'redis/architecture.md',
    '04.中间件/01.数据库/02.Redis/03.高级特性/01.哨兵模式.md': 'redis/sentinel.md',
    // Kafka
    '04.中间件/02.消息中间件/01.Kafka/01.应用配置/01Kafka整体介绍.md': 'kafka/introduction.md',
    '04.中间件/02.消息中间件/01.Kafka/01.应用配置/02Kafka安装部署.md': 'kafka/installation.md',
    '04.中间件/02.消息中间件/01.Kafka/01.应用配置/02Kafka安装部署2.md': 'kafka/installation-2.md',
    '04.中间件/02.消息中间件/01.Kafka/02.源码分析/01.Producer/00.前置准备.md': 'kafka/producer-preparation.md',
    '04.中间件/02.消息中间件/01.Kafka/02.源码分析/01.Producer/01.指标.md': 'kafka/producer-metrics.md',
    '04.中间件/02.消息中间件/01.Kafka/02.源码分析/01.Producer/02.配置参数.md': 'kafka/producer-config.md',
    '04.中间件/02.消息中间件/01.Kafka/02.源码分析/01.Producer/03.重要的配置参数.md': 'kafka/producer-important-config.md',
    '04.中间件/02.消息中间件/01.Kafka/02.源码分析/01.Producer/04.分区计算.md': 'kafka/producer-partitioning.md',
    '04.中间件/02.消息中间件/01.Kafka/02.源码分析/01.Producer/05.发送消息.md': 'kafka/producer-send.md',
    '04.中间件/02.消息中间件/01.Kafka/02.源码分析/01.Producer/06.消息追加到消息累加器.md': 'kafka/producer-accumulator.md',
    '04.中间件/02.消息中间件/01.Kafka/02.源码分析/01.Producer/07.粘性分区.md': 'kafka/producer-sticky-partition.md',
    '04.中间件/02.消息中间件/01.Kafka/03.面试真题/01.Kafka为什么那么快.md': 'kafka/why-so-fast.md',
    '04.中间件/02.消息中间件/01.Kafka/03.面试真题/02.零拷贝技术.md': 'kafka/zero-copy.md',

    // === 02.系统设计 目录 ===
    // 注册中心
    '02.系统设计/01.注册中心/01.注册中心的设计.md': 'system-design/service-registry.md',
    // 负载均衡
    '02.系统设计/03.负载均衡/01.负载均衡详解.md': 'system-design/load-balancing.md',
    '02.系统设计/03.负载均衡/02.OpenFeign详解.md': 'system-design/openfeign.md',
    // 熔断限流降级
    '02.系统设计/04.熔断限流降级/01.分布式熔断算法和组件详解.md': 'system-design/circuit-breaker.md',
    '02.系统设计/04.熔断限流降级/02.分布式限流算法和组件详解.md': 'system-design/rate-limiting.md',
    '02.系统设计/04.熔断限流降级/03.分布式降级算法和组件详解.md': 'system-design/degradation.md',
    // 场景设计
    '02.系统设计/08.场景设计/01.短URL系统设计.md': 'system-design/short-url.md',
    '02.系统设计/08.场景设计/08.短连接URL系统设计详解.md': 'system-design/short-url-detailed.md',
    '02.系统设计/08.场景设计/09.互联网企业面试场景题大全.md': 'system-design/interview-questions.md',
    '02.系统设计/08.场景设计/10.库存扣减一致性解决方案.md': 'system-design/inventory-consistency.md',
    '02.系统设计/08.场景设计/11.秒杀系统设计.md': 'system-design/flash-sale.md',
    '02.系统设计/08.场景设计/12.抢红包系统设计.md': 'system-design/red-packet.md',
    '02.系统设计/08.场景设计/13.12306火车票系统设计.md': 'system-design/train-ticket-system.md',
    '02.系统设计/08.场景设计/14.分布式缓存系统设计.md': 'system-design/distributed-cache.md',
    '02.系统设计/08.场景设计/15.打车系统设计.md': 'system-design/ride-hailing.md',
    '02.系统设计/08.场景设计/16.大数据量下数据库分库分表方案.md': 'system-design/sharding.md',
    '02.系统设计/08.场景设计/17.HTTP请求完整流程详解.md': 'system-design/http-flow.md',
    '02.系统设计/08.场景设计/18.CDN实现原理详解.md': 'system-design/cdn.md',
    '02.系统设计/08.场景设计/19.DNS实现原理详解.md': 'system-design/dns.md',
    '02.系统设计/08.场景设计/20.ClickHouse和Doris技术选型对比.md': 'system-design/clickhouse-vs-doris.md',
    '02.系统设计/08.场景设计/21.MySQL主从同步方案和对比分析.md': 'system-design/mysql-replication.md',
    '02.系统设计/08.场景设计/22.技术选型标准和方法.md': 'system-design/tech-selection.md',
    '02.系统设计/08.场景设计/支付系统架构设计.md': 'system-design/payment-system.md',
    '02.系统设计/08.场景设计/滴滴配送引擎业务.md': 'system-design/delivery-engine.md',

    // === 01.AI 目录 ===
    // Claude Code
    '01.AI/01.AI开发工具/01. ClaudeCode/Agent工作流引擎详解.md': 'ai/claude-code/agent-workflow-engine.md',
    // 架构详解系列
    '01.AI/01.AI开发工具/01. ClaudeCode/01.架构详解/01.整体架构设计.md': 'ai/claude-code/architecture-01-overall-architecture.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/01.架构详解/02.核心引擎实现.md': 'ai/claude-code/architecture-02-core-engine.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/01.架构详解/03.上下文管理系统.md': 'ai/claude-code/architecture-03-context-management.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/01.架构详解/04.会话与状态管理.md': 'ai/claude-code/architecture-04-session-management.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/01.架构详解/05.工具系统架构.md': 'ai/claude-code/architecture-05-tool-system-architecture.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/01.架构详解/06.文件操作工具实现.md': 'ai/claude-code/architecture-06-file-operations.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/01.架构详解/07.代码搜索与分析工具.md': 'ai/claude-code/architecture-07-code-search-analysis.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/01.架构详解/08.终端集成实现.md': 'ai/claude-code/architecture-08-terminal-integration.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/01.架构详解/09.MCP协议实现.md': 'ai/claude-code/architecture-09-mcp-protocol.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/01.架构详解/10.浏览器自动化集成.md': 'ai/claude-code/architecture-10-browser-automation.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/01.架构详解/11.Agent工作流引擎.md': 'ai/claude-code/architecture-11-agent-workflow-engine.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/01.架构详解/12.多模态交互实现.md': 'ai/claude-code/architecture-12-multimodal-interaction.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/01.架构详解/13.性能优化深度剖析.md': 'ai/claude-code/architecture-13-performance-optimization.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/01.架构详解/14.错误处理与容错设计.md': 'ai/claude-code/architecture-14-error-handling.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/01.架构详解/15.安全机制设计.md': 'ai/claude-code/architecture-15-security-design.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/01.架构详解/16.测试与质量保证.md': 'ai/claude-code/architecture-16-testing-quality-assurance.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/01.架构详解/17.自定义工具开发实战.md': 'ai/claude-code/architecture-17-custom-tool-development.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/01.架构详解/18.MCP-Server开发实战.md': 'ai/claude-code/architecture-18-mcp-server-practice.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/01.架构详解/20.构建自己的AI编程助手.md': 'ai/claude-code/architecture-20-build-your-own-ai-assistant.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/01.架构详解/21.Claude Code A2A 机制详解.md': 'ai/claude-code/architecture-21-a2a-mechanism.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/01.架构详解/22.Claude Code MCP 服务加载机制详解.md': 'ai/claude-code/architecture-22-mcp-loading-mechanism.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/01.架构详解/Claude Code 测试与质量保证.md': 'ai/claude-code/architecture-testing-quality-assurance.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/01.架构详解/Completion Summary.md': 'ai/claude-code/architecture-completion-summary.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/01.架构详解/Performance Optimization.md': 'ai/claude-code/architecture-performance-optimization.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/01.架构详解/index.md': 'ai/claude-code/architecture-index.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/01.架构详解/navigation.md': 'ai/claude-code/architecture-navigation.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/01.架构详解/代码搜索与分析工具.md': 'ai/claude-code/architecture-code-search-analysis.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/01.架构详解/文件操作工具实现.md': 'ai/claude-code/architecture-file-operations.md',
    // 其他 Claude Code 文件
    '01.AI/01.AI开发工具/01. ClaudeCode/02.安装配置/02.ClaudeCode命令/00.Claude Code安装.md': 'ai/claude-code/installation.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/02.安装配置/02.ClaudeCode命令/01.模型切换.md': 'ai/claude-code/model-switch.md',
    // MCP 安装配置
    '01.AI/01.AI开发工具/01. ClaudeCode/02.安装配置/03.MCP 安装配置/01.Vercel MCP 安装配置.md': 'ai/claude-code/vercel-mcp-setup.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/安全机制设计.md': 'ai/claude-code/security-design.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/终端集成实现.md': 'ai/claude-code/terminal-integration.md',
    // LangChain
    '01.AI/02.AI 开发框架/01.LangChain/1.环境搭建与核心概念.md': 'ai/langchain/week1-setup-core-concepts.md',
    '01.AI/02.AI 开发框架/01.LangChain/2.Messages与Prompts.md': 'ai/langchain/week2-messages-prompts.md',
    '01.AI/02.AI 开发框架/01.LangChain/3.Models层深入探讨.md': 'ai/langchain/week3-models-deep-dive.md',
    '01.AI/02.AI 开发框架/01.LangChain/4.Chains链式基础.md': 'ai/langchain/week4-chains-basics.md',
    '01.AI/02.AI 开发框架/01.LangChain/5.Documents文档处理.md': 'ai/langchain/week5-documents.md',
    '01.AI/02.AI 开发框架/01.LangChain/6.Memory记忆系统.md': 'ai/langchain/week6-memory.md',
    '01.AI/02.AI 开发框架/01.LangChain/7.Tools工具集成.md': 'ai/langchain/week7-tools.md',
    '01.AI/02.AI 开发框架/01.LangChain/8.Agents代理器.md': 'ai/langchain/week8-agents.md',
    '01.AI/02.AI 开发框架/01.LangChain/9.Embeddings和Vector Stores详解.md': 'ai/langchain/week9-embeddings-vectorstores.md',
    '01.AI/02.AI 开发框架/01.LangChain/10.RAG系统详解.md': 'ai/langchain/week10-rag-system.md',
    '01.AI/02.AI 开发框架/01.LangChain/11.Callbacks和监控系统.md': 'ai/langchain/week11-callbacks-monitoring.md',
    '01.AI/02.AI 开发框架/01.LangChain/16.部署与优化.md': 'ai/langchain/week16-deployment-optimization.md',
    '01.AI/02.AI 开发框架/01.LangChain/17.企业级RAG系统实战.md': 'ai/langchain/enterprise-rag-system.md',
    '01.AI/02.AI 开发框架/01.LangChain/18.召回质量评测.md': 'ai/langchain/recall-concept.md',
    '01.AI/02.AI 开发框架/01.LangChain/LangChain学习路线图.md': 'ai/langchain/learning-roadmap.md',
    '01.AI/02.AI 开发框架/01.LangChain/LlamaIndex与LangChain对比分析.md': 'ai/langchain/llamaindex-vs-langchain.md',
    '01.AI/02.AI 开发框架/01.LangChain/index.md': 'ai/langchain/index.md',
    // LangGraph
    '01.AI/02.AI 开发框架/02.LangGraph/1.学习路线图.md': 'ai/langgraph/learning-roadmap.md',
    '01.AI/02.AI 开发框架/02.LangGraph/2.入门介绍.md': 'ai/langgraph/introduction.md',
    '01.AI/02.AI 开发框架/02.LangGraph/3.快速入门指南.md': 'ai/langgraph/quickstart.md',
    '01.AI/02.AI 开发框架/02.LangGraph/4.基础概念深度解析.md': 'ai/langgraph/basic-concepts.md',
    '01.AI/02.AI 开发框架/02.LangGraph/5.核心概念解析.md': 'ai/langgraph/core-concepts.md',
    '01.AI/02.AI 开发框架/02.LangGraph/6.核心概念详解与作用.md': 'ai/langgraph/core-concepts-detail.md',
    '01.AI/02.AI 开发框架/02.LangGraph/7.基础案例实战.md': 'ai/langgraph/basic-examples.md',
    '01.AI/02.AI 开发框架/02.LangGraph/8.状态管理详解.md': 'ai/langgraph/state-management.md',
    '01.AI/02.AI 开发框架/02.LangGraph/9.节点开发指南.md': 'ai/langgraph/node-development.md',
    '01.AI/02.AI 开发框架/02.LangGraph/10.路由与控制流.md': 'ai/langgraph/routing-control.md',
    '01.AI/02.AI 开发框架/02.LangGraph/11.项目开发实战.md': 'ai/langgraph/project-development.md',
    '01.AI/02.AI 开发框架/02.LangGraph/12.实战案例详解.md': 'ai/langgraph/practical-examples.md',
    '01.AI/02.AI 开发框架/02.LangGraph/13.实际项目案例.md': 'ai/langgraph/real-projects.md',
    '01.AI/02.AI 开发框架/02.LangGraph/14.与LangChain集成.md': 'ai/langgraph/langchain-integration.md',
    '01.AI/02.AI 开发框架/02.LangGraph/15.测试与调试.md': 'ai/langgraph/testing-debugging.md',
    '01.AI/02.AI 开发框架/02.LangGraph/16.测试策略与最佳实践.md': 'ai/langgraph/testing-strategy.md',
    '01.AI/02.AI 开发框架/02.LangGraph/17.性能优化指南.md': 'ai/langgraph/performance-optimization.md',
    '01.AI/02.AI 开发框架/02.LangGraph/18.性能测试与基准.md': 'ai/langgraph/performance-testing.md',
    '01.AI/02.AI 开发框架/02.LangGraph/19.多智能体系统.md': 'ai/langgraph/multi-agent-system.md',
    '01.AI/02.AI 开发框架/02.LangGraph/20.高级特性.md': 'ai/langgraph/advanced-features.md',
    '01.AI/02.AI 开发框架/02.LangGraph/21.企业级案例.md': 'ai/langgraph/enterprise-cases.md',
    '01.AI/02.AI 开发框架/02.LangGraph/22.生产部署实践.md': 'ai/langgraph/production-deployment.md',
    '01.AI/02.AI 开发框架/02.LangGraph/23.项目模板与脚手架.md': 'ai/langgraph/project-template.md',
    '01.AI/02.AI 开发框架/02.LangGraph/24.常见问题与解决方案.md': 'ai/langgraph/troubleshooting.md',
    '01.AI/02.AI 开发框架/02.LangGraph/25.最佳实践总结.md': 'ai/langgraph/best-practices.md',
    '01.AI/02.AI 开发框架/02.LangGraph/26.与其他框架对比分析.md': 'ai/langgraph/framework-comparison.md',
    '01.AI/02.AI 开发框架/02.LangGraph/27.API参考手册.md': 'ai/langgraph/api-reference.md',
    '01.AI/02.AI 开发框架/02.LangGraph/28.学习指南-目录索引.md': 'ai/langgraph/index.md',
    // 提示词工程
    '01.AI/03.提示词工程/01.小红书提示词/Xiaohongshu Style.md': 'ai/prompts/xiaohongshu-style.md',
    // CodeAgent
    '01.AI/codeagent/AI 集成与后端实现 - Code Agent 核心服务.md': 'ai/codeagent/ai-integration.md',
    '01.AI/codeagent/Code Agent 代码补全实现详解.md': 'ai/codeagent/code-completion.md',
    '01.AI/codeagent/Code Agent 开发指南 - 构建类似 Cursor 的 AI 编程助手.md': 'ai/codeagent/overview.md',
    '01.AI/codeagent/Code Agent 架构设计详解.md': 'ai/codeagent/architecture.md',
    '01.AI/codeagent/VSCode 插件开发 - Code Agent 前端实现.md': 'ai/codeagent/vscode-extension.md',
    // Other
    '01.AI/claudecode/architecture/Claude Code 错误处理与容错设计.md': 'ai/claudecode/architecture-error-handling.md',

    // === 05.开发语言及框架/01.Java 目录 ===
    // 集合
    '05.开发语言及框架/01.Java/01.集合/01.HashMap 源码分析及面试题.md': 'java/hashmap.md',
    '05.开发语言及框架/01.Java/01.集合/02.ConcurrentHashMap 源码分析及面试题.md': 'java/concurrent-hashmap.md',
    // IO
    '05.开发语言及框架/01.Java/02.IO操作/01.IO分类.md': 'java/io-classification.md',
    '05.开发语言及框架/01.Java/02.IO操作/02.多路复用.md': 'java/io-multiplexing.md',
    // JUC - 队列
    '05.开发语言及框架/01.Java/03.并发包/01.队列/01.JUC中的队列.md': 'java/juc-queue.md',
    // JUC - 锁
    '05.开发语言及框架/01.Java/03.并发包/02.Java 锁/01.JUC中的锁.md': 'java/juc-lock.md',
    '05.开发语言及框架/01.Java/03.并发包/02.Java 锁/02.AbstractQueuedSynchronizer.md': 'java/aqs.md',
    '05.开发语言及框架/01.Java/03.并发包/02.Java 锁/02.ReentrantLock源码解析和面试题.md': 'java/reentrant-lock.md',
    '05.开发语言及框架/01.Java/03.并发包/02.Java 锁/03.独占锁和共享锁.md': 'java/exclusive-shared-lock.md',
    '05.开发语言及框架/01.Java/03.并发包/02.Java 锁/04.CountDownLatch 源码分析和面试题.md': 'java/countdownlatch.md',
    '05.开发语言及框架/01.Java/03.并发包/02.Java 锁/05.Semphore源码分析和面试题.md': 'java/semaphore.md',
    // JUC - 原子类
    '05.开发语言及框架/01.Java/03.并发包/03.原子类/01.原子类.md': 'java/atomic.md',
    // 线程池
    '05.开发语言及框架/01.Java/03.并发包/04.线程池/01.自定义线程池工厂类.md': 'java/threadpool-factory.md',
    '05.开发语言及框架/01.Java/03.并发包/04.线程池/01.ThreadLocal.md': 'java/threadlocal.md',
    '05.开发语言及框架/01.Java/03.并发包/04.线程池/03.线程池.md': 'java/thread-threadpool.md',
    // JVM
    '05.开发语言及框架/01.Java/04.JVM/01.线程FullGC问题排查.md': 'java/jvm-fullgc01.md',
    '05.开发语言及框架/01.Java/04.JVM/02.Java虚拟机配置.md': 'java/jvm-memory-config.md',
    '05.开发语言及框架/01.Java/04.JVM/03.元空间 MetaSpace.md': 'java/jvm-metaspace.md',
    // Spring
    '05.开发语言及框架/01.Java/05.Spring/01.SpringBoot/01.配置/01.数据库连接的配置.md': 'spring/database-connection-configuration.md',
    '05.开发语言及框架/01.Java/05.Spring/01.SpringBoot/01.配置/01.SpringBoot日志配置.md': 'spring/logback.md',
    '05.开发语言及框架/01.Java/05.Spring/01.SpringBoot/01.配置/02.Redis的配置.md': 'spring/redis-configuration.md',
    '05.开发语言及框架/01.Java/05.Spring/01.SpringBoot/02.事务管理/01.PlatformTransactionManager.md': 'spring/PlatformTransactionManager.md',
    '05.开发语言及框架/01.Java/05.Spring/01.SpringBoot/03.Bean 管理/01.Spring Bean 的加载过程.md': 'spring/bean-loading.md',
    '05.开发语言及框架/01.Java/05.Spring/01.SpringBoot/04.扩展机制/01.Spring AOP实现机制.md': 'spring/aop.md',
    '05.开发语言及框架/01.Java/05.Spring/01.SpringBoot/04.扩展机制/02.Spring扩展点.md': 'spring/extension.md',
    '05.开发语言及框架/01.Java/05.Spring/01.SpringBoot/04.扩展机制/02.SpringBoot Starter.md': 'spring/starter.md',
    '05.开发语言及框架/01.Java/05.Spring/01.SpringBoot/04.扩展机制/03.Endpoint.md': 'spring/endpoint.md',
    // JWT
    '05.开发语言及框架/01.Java/06.JWT/工作流程.md': 'java/jwt-workflow.md',
    '05.开发语言及框架/01.Java/06.JWT/JWT实战应用.md': 'java/jwt-practice.md',
    '05.开发语言及框架/01.Java/06.JWT/JWT基础知识.md': 'java/jwt-basics.md',
    '05.开发语言及框架/01.Java/06.JWT/JWT常见问题与故障排除.md': 'java/jwt-troubleshooting.md',
    '05.开发语言及框架/01.Java/06.JWT/使用了JWT之后还需要auth2吗.md': 'java/jwt-oauth2.md',
    // MyBatis
    '05.开发语言及框架/01.Java/07.MyBatis/first.md': 'java/mybatis-interceptor.md',

    // === SpringCloud ===
    // Gateway
    'docs/spring/SpringCloud/Gateway/01.Spring Cloud Gateway.md': 'spring/gateway.md',
    'docs/spring/SpringCloud/Gateway/02.Spring Cloud Gateway的使用.md': 'spring/gateway-usage.md',
    'docs/spring/SpringCloud/Gateway/03.文章汇总.md': 'spring/gateway-articles.md',
    // OpenFeign
    'docs/spring/SpringCloud/OpenFeign/01.Spring Cloud OpenFeign.md': 'spring/openfeign.md',
    // 断路器
    'docs/spring/SpringCloud/断路器/01. Resilience4J.md': 'spring/resilience4j.md',
    'docs/spring/SpringCloud/断路器/02.Resilience4J熔断器使用.md': 'spring/resilience4j-circuit-breaker.md',
    'docs/spring/SpringCloud/断路器/03.Resilience4J限流器使用.md': 'spring/resilience4j-rate-limiter.md',
    // 注册中心
    'docs/spring/SpringCloud/注册中心/01.Eureka.md': 'spring/eureka.md',
    'docs/spring/SpringCloud/注册中心/02.Eureka的面试题.md': 'spring/eureka-interview.md',
    'docs/spring/SpringCloud/注册中心/03.Consul.md': 'spring/consul.md',
    'docs/spring/SpringCloud/注册中心/04.Consul面试题.md': 'spring/consul-interview.md',
    'docs/spring/SpringCloud/注册中心/05.Raft算法.md': 'spring/raft.md',
    // 负载均衡
    'docs/spring/SpringCloud/负载均衡/01.Ribbon.md': 'spring/ribbon.md',
    'docs/spring/SpringCloud/负载均衡/02.Spring Cloud LoadBalancer.md': 'spring/loadbalancer.md',
  },

  // Markdown 配置
  markdown: {
    theme: {
      light: 'github-light',
      dark: 'github-dark'
    }
  },

  // Mermaid 配置
  mermaid: {
    // 可选：配置 mermaid 主题和其他选项
  },

  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: '首页', link: '/' },
      {
        text: '开发语言及框架',
        items: [
          { text: 'Java', link: '/java/hashmap' },
          { text: 'Spring', link: '/spring/bean-loading' },
          { text: 'MyBatis', link: '/java/mybatis-interceptor' },
        ]
      },
      {
        text: '中间件',
        items: [
          { text: 'MySQL', link: '/mysql/index' },
          { text: 'Redis', link: '/redis/sds' },
          { text: 'Kafka', link: '/kafka/introduction' },
        ]
      },
      {
        text: '前端技术',
        items: [
          { text: 'React', link: '/docs/htmlcssjs/React/01useclient' },
          { text: 'TailwindCSS', link: '/docs/htmlcssjs/TailwindCSS/00prompts' },
          { text: 'Daisyui', link: '/docs/htmlcssjs/Daisyui/03hero' },
          { text: 'Monaco', link: '/docs/htmlcssjs/monaco/01EditorProps' },
          { text: 'VuePress', link: '/docs/htmlcssjs/vuepress/主题开发与发布流程' },
        ]
      },
      {
        text: 'AI',
        items: [
          { text: 'Claude Code', link: '/ai/claude-code/model-switch' },
          { text: 'CodeAgent', link: '/docs/ai/codeagent/00.命令指令参数表' },
          { text: 'LangChain', link: '/docs/ai/langchain/README' },
          { text: 'LangGraph', link: '/docs/ai/langgraph/00.目录概览' },
          { text: 'MCP', link: '/docs/ai/mcp/01.MCP的底层原理' },
          { text: 'Dify', link: '/docs/ai/dify/01.安装dify' },
          { text: 'Ollama', link: '/docs/ai/ollma/ollama安装' },
        ]
      },
      {
        text: '算法',
        items: [
          { text: 'LeetCode', link: '/docs/leetcode/first' },
        ]
      },
      {
        text: '系统设计',
        items: [
          { text: '注册中心', link: '/system-design/service-registry' },
          { text: '负载均衡', link: '/system-design/load-balancing' },
          { text: '熔断限流降级', link: '/system-design/circuit-breaker' },
          { text: '场景设计', link: '/system-design/short-url' },
        ]
      },
      {
        text: '工具',
        items: [
          { text: 'Docker', link: '/docker/centos-install' },
          { text: 'Git', link: '/git/remote' },
        ]
      },
      {
        text: '其他',
        items: [
          { text: 'JWT', link: '/docs/JWT/JWT基础知识' },
          { text: 'Swift', link: '/docs/swift/01.在VSCode中编写运行Swift' },
          { text: '小程序', link: '/miniprogram/monetization.html' },
          { text: '我的经历', link: '/my-experience/management.html' },
          { text: '我的玩具', link: '/my-projects/free-apps.html' },
          { text: '面试', link: '/interview/self-introduction.html' },
        ]
      }
    ],

    sidebar: {
      '/ai/': [
        {
          text: 'Claude Code',
          collapsed: false,
          items: [
            { text: '安装配置',
              collapsed: true,
              items: [
                { text: 'Claude Code安装', link: '/ai/claude-code/installation' },
                { text: '模型切换', link: '/ai/claude-code/model-switch' },
              ]
            },
            {
                text: 'MCP 安装配置',
                collapsed: true,
                items: [
                { text: 'Vercel MCP 安装配置', link: '/ai/claude-code/vercel-mcp-setup' },
                ]
            },
            {
              text: '架构详解',
              collapsed: true,
              items: [
                { text: '整体架构设计', link: '/ai/claude-code/architecture-01-overall-architecture' },
                { text: '核心引擎实现', link: '/ai/claude-code/architecture-02-core-engine' },
                { text: '上下文管理系统', link: '/ai/claude-code/architecture-03-context-management' },
                { text: '会话与状态管理', link: '/ai/claude-code/architecture-04-session-management' },
                { text: '工具系统架构', link: '/ai/claude-code/architecture-05-tool-system-architecture' },
                { text: '文件操作工具实现', link: '/ai/claude-code/architecture-06-file-operations' },
                { text: '代码搜索与分析工具', link: '/ai/claude-code/architecture-07-code-search-analysis' },
                { text: '终端集成实现', link: '/ai/claude-code/architecture-08-terminal-integration' },
                { text: 'MCP协议深入解析', link: '/ai/claude-code/architecture-09-mcp-protocol' },
                { text: '浏览器自动化集成', link: '/ai/claude-code/architecture-10-browser-automation' },
                { text: 'Agent工作流引擎', link: '/ai/claude-code/architecture-11-agent-workflow-engine' },
                { text: '多模态交互实现', link: '/ai/claude-code/architecture-12-multimodal-interaction' },
                { text: '性能优化深度剖析', link: '/ai/claude-code/architecture-13-performance-optimization' },
                { text: '错误处理与容错设计', link: '/ai/claude-code/architecture-14-error-handling' },
                { text: '安全机制设计', link: '/ai/claude-code/architecture-15-security-design' },
                { text: '测试与质量保证', link: '/ai/claude-code/architecture-16-testing-quality-assurance' },
                { text: '自定义工具开发实战', link: '/ai/claude-code/architecture-17-custom-tool-development' },
                { text: 'MCP Server开发实战', link: '/ai/claude-code/architecture-18-mcp-server-practice' },
                { text: '构建自己的AI编程助手', link: '/ai/claude-code/architecture-20-build-your-own-ai-assistant' },
                { text: 'A2A 机制详解', link: '/ai/claude-code/architecture-21-a2a-mechanism' },
                { text: 'MCP 服务加载机制详解', link: '/ai/claude-code/architecture-22-mcp-loading-mechanism' },
              ]
            }
          ]
        },
        {
          text: 'CodeAgent',
          collapsed: true,
          items: [
            { text: '命令指令参数表', link: '/docs/ai/codeagent/00.命令指令参数表' },
            { text: '架构概述', link: '/docs/ai/codeagent/01.架构概述' },
            { text: 'VSCode工具详解', link: '/docs/ai/codeagent/02.VSCode工具详解' },
            { text: 'AI编程基础实践', link: '/docs/ai/codeagent/03.AI编程基础实践' },
            { text: '代码安全实践指南', link: '/docs/ai/codeagent/04.代码安全实践指南' },
          ]
        },
        {
          text: 'LangChain',
          collapsed: true,
          items: [
            { text: '学习路线图', link: '/ai/langchain/learning-roadmap' },
            { text: '环境搭建与核心概念', link: '/ai/langchain/week1-setup-core-concepts' },
            { text: 'Messages与Prompts', link: '/ai/langchain/week2-messages-prompts' },
            { text: 'Models层深入探讨', link: '/ai/langchain/week3-models-deep-dive' },
            { text: 'Chains链式基础', link: '/ai/langchain/week4-chains-basics' },
            { text: 'Documents文档处理', link: '/ai/langchain/week5-documents' },
            { text: 'Memory记忆系统', link: '/ai/langchain/week6-memory' },
            { text: 'Tools工具集成', link: '/ai/langchain/week7-tools' },
            { text: 'Agents代理器', link: '/ai/langchain/week8-agents' },
            { text: 'Embeddings和Vector Stores详解', link: '/ai/langchain/week9-embeddings-vectorstores' },
            { text: 'RAG系统详解', link: '/ai/langchain/week10-rag-system' },
            { text: 'Callbacks和监控系统', link: '/ai/langchain/week11-callbacks-monitoring' },
            { text: '部署与优化', link: '/ai/langchain/week16-deployment-optimization' },
            { text: '企业级RAG系统实战', link: '/ai/langchain/enterprise-rag-system' },
            { text: '召回质量评测', link: '/ai/langchain/recall-concept' },
            { text: 'LlamaIndex与LangChain对比分析', link: '/ai/langchain/llamaindex-vs-langchain' },
          ]
        },
        {
          text: 'LangGraph',
          collapsed: true,
          items: [
            { text: '学习路线图', link: '/ai/langgraph/learning-roadmap' },
            { text: '入门介绍', link: '/ai/langgraph/introduction' },
            { text: '快速入门指南', link: '/ai/langgraph/quickstart' },
            { text: '基础概念深度解析', link: '/ai/langgraph/basic-concepts' },
            { text: '核心概念解析', link: '/ai/langgraph/core-concepts' },
            { text: '核心概念详解与作用', link: '/ai/langgraph/core-concepts-detail' },
            { text: '基础案例实战', link: '/ai/langgraph/basic-examples' },
            { text: '状态管理详解', link: '/ai/langgraph/state-management' },
            { text: '节点开发指南', link: '/ai/langgraph/node-development' },
            { text: '路由与控制流', link: '/ai/langgraph/routing-control' },
            { text: '项目开发实战', link: '/ai/langgraph/project-development' },
            { text: '实战案例详解', link: '/ai/langgraph/practical-examples' },
            { text: '实际项目案例', link: '/ai/langgraph/real-projects' },
            { text: '与LangChain集成', link: '/ai/langgraph/langchain-integration' },
            { text: '测试与调试', link: '/ai/langgraph/testing-debugging' },
            { text: '测试策略与最佳实践', link: '/ai/langgraph/testing-strategy' },
            { text: '性能优化指南', link: '/ai/langgraph/performance-optimization' },
            { text: '性能测试与基准', link: '/ai/langgraph/performance-testing' },
            { text: '多智能体系统', link: '/ai/langgraph/multi-agent-system' },
            { text: '高级特性', link: '/ai/langgraph/advanced-features' },
            { text: '企业级案例', link: '/ai/langgraph/enterprise-cases' },
            { text: '生产部署实践', link: '/ai/langgraph/production-deployment' },
            { text: '项目模板与脚手架', link: '/ai/langgraph/project-template' },
            { text: '常见问题与解决方案', link: '/ai/langgraph/troubleshooting' },
            { text: '最佳实践总结', link: '/ai/langgraph/best-practices' },
            { text: '与其他框架对比分析', link: '/ai/langgraph/framework-comparison' },
            { text: 'API参考手册', link: '/ai/langgraph/api-reference' },
            { text: '学习指南-目录索引', link: '/ai/langgraph/index' },
          ]
        },
        {
          text: 'MCP',
          collapsed: true,
          items: [
            { text: 'MCP的底层原理', link: '/docs/ai/mcp/01.MCP的底层原理' },
            { text: '通过SpringAI构建一个MCP', link: '/docs/ai/mcp/02.通过SpringAI构建一个MCP' },
            { text: '深入分析playwright_mcp', link: '/docs/ai/mcp/03.深入分析playwright_mcp' },
          ]
        },
        {
          text: 'Dify',
          collapsed: true,
          items: [
            { text: '安装dify', link: '/docs/ai/dify/01.安装dify' },
          ]
        },
        {
          text: 'Ollama',
          collapsed: true,
          items: [
            { text: 'ollama安装', link: '/docs/ai/ollma/ollama安装' },
            { text: '模型调优', link: '/docs/ai/ollma/模型调优' },
          ]
        },
        {
          text: '提示词',
          collapsed: true,
          items: [
            {
              text: 'UI',
              items: [
                { text: '小红书配图页面', link: '/docs/ai/prompts/UI/01.小红书配图页面' },
              ]
            }
          ]
        }
      ],

      '/java/': [
        {
          text: '集合',
          collapsed: false,
          items: [
            { text: 'HashMap 源码分析', link: '/java/hashmap' },
            { text: 'ConcurrentHashMap 源码分析', link: '/java/concurrent-hashmap' },
          ]
        },
        {
          text: 'IO',
          collapsed: false,
          items: [
            { text: 'IO分类', link: '/java/io-classification' },
            { text: '多路复用', link: '/java/io-multiplexing' },
          ]
        },
        {
          text: 'JUC并发包',
          collapsed: false,
          items: [
            { text: 'JUC中的队列', link: '/java/juc-queue' },
            { text: 'JUC中的锁', link: '/java/juc-lock' },
            { text: 'AQS源码解析', link: '/java/aqs' },
            { text: 'ReentrantLock源码解析', link: '/java/reentrant-lock' },
            { text: '独占锁和共享锁', link: '/java/exclusive-shared-lock' },
            { text: 'CountDownLatch源码分析', link: '/java/countdownlatch' },
            { text: 'Semaphore源码分析', link: '/java/semaphore' },
            { text: '原子类', link: '/java/atomic' },
          ]
        },
        {
          text: '线程池',
          collapsed: false,
          items: [
            { text: 'ThreadLocal', link: '/java/threadlocal' },
            { text: '自定义线程池工厂类', link: '/java/threadpool-factory' },
            { text: '线程池', link: '/java/thread-threadpool' },
          ]
        },
        {
          text: 'JVM',
          collapsed: false,
          items: [
            { text: 'FullGC问题排查', link: '/java/jvm-fullgc01' },
            { text: 'JVM内存配置', link: '/java/jvm-memory-config' },
            { text: '元空间 MetaSpace', link: '/java/jvm-metaspace' },
          ]
        },
        {
          text: 'JWT',
          collapsed: false,
          items: [
            { text: 'JWT基础知识', link: '/java/jwt-basics' },
            { text: 'JWT工作流程', link: '/java/jwt-workflow' },
            { text: 'JWT实战应用', link: '/java/jwt-practice' },
            { text: 'JWT常见问题', link: '/java/jwt-troubleshooting' },
            { text: 'JWT与OAuth2', link: '/java/jwt-oauth2' },
          ]
        },
        {
          text: 'MyBatis',
          collapsed: false,
          items: [
            { text: 'MyBatis Interceptor扩展', link: '/java/mybatis-interceptor' },
          ]
        }
      ],

      '/spring/': [
        {
          text: 'Spring核心',
          collapsed: false,
          items: [
            { text: 'Bean加载过程', link: '/spring/bean-loading' },
            { text: 'AOP实现机制', link: '/spring/aop' },
            { text: 'Spring扩展点', link: '/spring/extension' },
          ]
        },
        {
          text: 'SpringBoot配置',
          collapsed: false,
          items: [
            { text: '日志配置', link: '/spring/logback' },
            { text: '数据库连接配置', link: '/spring/database-connection-configuration' },
            { text: 'Redis配置', link: '/spring/redis-configuration' },
          ]
        },
        {
          text: 'SpringBoot事务',
          collapsed: false,
          items: [
            { text: 'PlatformTransactionManager', link: '/spring/PlatformTransactionManager' },
          ]
        },
        {
          text: 'SpringBoot扩展',
          collapsed: false,
          items: [
            { text: 'SpringBoot Starter', link: '/spring/starter' },
            { text: 'Endpoint扩展', link: '/spring/endpoint' },
          ]
        },
        {
          text: 'SpringCloud',
          collapsed: true,
          items: [
            {
              text: 'Gateway',
              items: [
                { text: 'Spring Cloud Gateway', link: '/spring/gateway' },
                { text: 'Gateway使用', link: '/spring/gateway-usage' },
                { text: '文章汇总', link: '/spring/gateway-articles' },
              ]
            },
            {
              text: 'OpenFeign',
              items: [
                { text: 'Spring Cloud OpenFeign', link: '/spring/openfeign' },
              ]
            },
            {
              text: '断路器',
              items: [
                { text: 'Resilience4J', link: '/spring/resilience4j' },
                { text: 'Resilience4J熔断器使用', link: '/spring/resilience4j-circuit-breaker' },
                { text: 'Resilience4J限流器使用', link: '/spring/resilience4j-rate-limiter' },
              ]
            },
            {
              text: '注册中心',
              items: [
                { text: 'Eureka', link: '/spring/eureka' },
                { text: 'Eureka面试题', link: '/spring/eureka-interview' },
                { text: 'Consul', link: '/spring/consul' },
                { text: 'Consul面试题', link: '/spring/consul-interview' },
                { text: 'Raft算法', link: '/spring/raft' },
              ]
            },
            {
              text: '负载均衡',
              items: [
                { text: 'Ribbon', link: '/spring/ribbon' },
                { text: 'Spring Cloud LoadBalancer', link: '/spring/loadbalancer' },
              ]
            }
          ]
        }
      ],

      '/kafka/': [
        {
          text: '应用配置',
          collapsed: false,
          items: [
            { text: 'Kafka整体介绍', link: '/kafka/introduction' },
            { text: 'Kafka安装部署', link: '/kafka/installation' },
            { text: 'Kafka安装部署2', link: '/kafka/installation-2' },
          ]
        },
        {
          text: 'Producer源码分析',
          collapsed: false,
          items: [
            { text: '前置准备', link: '/kafka/producer-preparation' },
            { text: '指标', link: '/kafka/producer-metrics' },
            { text: '配置参数', link: '/kafka/producer-config' },
            { text: '重要的配置参数', link: '/kafka/producer-important-config' },
            { text: '分区计算', link: '/kafka/producer-partitioning' },
            { text: '发送消息', link: '/kafka/producer-send' },
            { text: '消息追加到消息累加器', link: '/kafka/producer-accumulator' },
            { text: '粘性分区', link: '/kafka/producer-sticky-partition' },
          ]
        },
        {
          text: '面试真题',
          collapsed: false,
          items: [
            { text: 'Kafka为什么那么快', link: '/kafka/why-so-fast' },
            { text: '零拷贝技术', link: '/kafka/zero-copy' },
          ]
        }
      ],

      '/docs/leetcode/': [
        {
          text: 'LeetCode',
          collapsed: false,
          items: [
            { text: 'first', link: '/docs/leetcode/first' },
            {
              text: '二叉树',
              items: [
                { text: '二叉树的层序遍历', link: '/docs/leetcode/二叉树/01.二叉树的层序遍历' },
                { text: '二叉树的中序遍历', link: '/docs/leetcode/二叉树/02.二叉树的中序遍历' },
                { text: '二叉树的最近公共祖先', link: '/docs/leetcode/二叉树/03.二叉树的最近公共祖先' },
                { text: '二叉树的层序遍历二', link: '/docs/leetcode/二叉树/04.二叉树的层序遍历二' },
                { text: '二叉树遍历', link: '/docs/leetcode/二叉树/二叉树遍历' },
              ]
            },
            {
              text: '前缀和',
              items: [
                { text: '前缀和', link: '/docs/leetcode/前缀和/01.前缀和' },
                { text: '买卖股票的最佳时机', link: '/docs/leetcode/前缀和/02.买卖股票的最佳时机' },
              ]
            },
            {
              text: '动态规划',
              items: [
                { text: '动态规划题单', link: '/docs/leetcode/动态规划/01.动态规划题单' },
              ]
            },
            {
              text: '多线程题目',
              items: [
                { text: '2 两线程循环打印0~100', link: '/docs/leetcode/多线程题目/01.2 两线程循环打印0~100' },
              ]
            },
            {
              text: '字符串',
              items: [
                { text: '最长子字符串', link: '/docs/leetcode/字符串/01.最长子字符串' },
                { text: '滑动窗口技巧', link: '/docs/leetcode/字符串/02.滑动窗口技巧' },
              ]
            },
            {
              text: '数组',
              items: [
                { text: '合并区间', link: '/docs/leetcode/数组/01.合并区间' },
                { text: '两数之和', link: '/docs/leetcode/数组/01.两数之和' },
                { text: '搜索旋转排序数', link: '/docs/leetcode/数组/02.搜索旋转排序数' },
                { text: '盛最多水的容器', link: '/docs/leetcode/数组/03.盛最多水的容器' },
                { text: '数组中的第 K 个最大元素', link: '/docs/leetcode/数组/04.数组中的第 K 个最大元素' },
                { text: '三数之和', link: '/docs/leetcode/数组/05.三数之和' },
                { text: '字母异位词', link: '/docs/leetcode/数组/06.字母异位词' },
                { text: '全排列', link: '/docs/leetcode/数组/07.全排列' },
                { text: '合并两个有序数组', link: '/docs/leetcode/数组/08.合并两个有序数组' },
                {
                  text: '一维数组',
                  items: [
                    { text: '最大子数组和', link: '/docs/leetcode/数组/一维数组/01.最大子数组和' },
                  ]
                },
                {
                  text: '二维数组',
                  items: [
                    { text: '搜索二维矩阵', link: '/docs/leetcode/数组/二维数组/01.搜索二维矩阵' },
                    { text: '全排列', link: '/docs/leetcode/数组/二维数组/02.全排列' },
                    { text: '旋转矩阵', link: '/docs/leetcode/数组/二维数组/03.旋转矩阵' },
                    { text: '螺旋矩阵', link: '/docs/leetcode/数组/二维数组/04.螺旋矩阵' },
                  ]
                }
              ]
            },
            {
              text: '查找',
              items: [
                { text: '二分查找', link: '/docs/leetcode/查找/01.二分查找' },
              ]
            },
            {
              text: '栈',
              items: [
                { text: '有效的括号', link: '/docs/leetcode/栈/01.有效的括号' },
              ]
            },
            {
              text: '链表',
              items: [
                { text: '反转链表', link: '/docs/leetcode/链表/01.反转链表' },
                { text: 'K个一组翻转链表', link: '/docs/leetcode/链表/02.K个一组翻转链表' },
                { text: '合并两个有序链表', link: '/docs/leetcode/链表/03.合并两个有序链表' },
                { text: '反转链表ii', link: '/docs/leetcode/链表/04.反转链表ii' },
                { text: '删除链表', link: '/docs/leetcode/链表/05.删除链表' },
              ]
            },
            {
              text: '滑动窗口',
              items: [
                { text: '滑动窗口', link: '/docs/leetcode/滑动窗口/01.滑动窗口' },
                { text: '无重复字符的最长子串', link: '/docs/leetcode/滑动窗口/02.无重复字符的最长子串' },
              ]
            }
          ]
        }
      ],

      '/docs/htmlcssjs/': [
        {
          text: '前端技术',
          collapsed: false,
          items: [
            {
              text: 'Daisyui',
              items: [
                { text: 'hero', link: '/docs/htmlcssjs/Daisyui/03hero' },
                { text: 'tooltip', link: '/docs/htmlcssjs/Daisyui/100tooltip' },
              ]
            },
            {
              text: 'monaco',
              items: [
                { text: 'EditorProps', link: '/docs/htmlcssjs/monaco/01EditorProps' },
              ]
            },
            {
              text: 'React',
              items: [
                { text: 'useclient', link: '/docs/htmlcssjs/React/01useclient' },
                { text: 'Client Component 中的数据获取与性能策略', link: '/docs/htmlcssjs/React/02.Client Component 中的数据获取与性能策略' },
                { text: 'useState的使用', link: '/docs/htmlcssjs/React/03.useState的使用' },
                { text: 'useEffect的使用', link: '/docs/htmlcssjs/React/04.useEffect的使用' },
                { text: 'useContext的使用', link: '/docs/htmlcssjs/React/05.useContext的使用' },
                { text: 'Metadata的使用', link: '/docs/htmlcssjs/React/06.Metadata的使用' },
              ]
            },
            {
              text: 'TailwindCSS',
              items: [
                { text: '提示词', link: '/docs/htmlcssjs/TailwindCSS/00prompts' },
                { text: 'all', link: '/docs/htmlcssjs/TailwindCSS/01all' },
                { text: 'buttong', link: '/docs/htmlcssjs/TailwindCSS/02buttong' },
                { text: 'background', link: '/docs/htmlcssjs/TailwindCSS/03background' },
                { text: 'layout', link: '/docs/htmlcssjs/TailwindCSS/04layout' },
              ]
            },
            {
              text: 'vuepress',
              items: [
                { text: '主题开发与发布流程', link: '/docs/htmlcssjs/vuepress/主题开发与发布流程' },
              ]
            }
          ]
        }
      ],

      '/system-design/': [
        {
          text: '注册中心',
          collapsed: false,
          items: [
            { text: '注册中心的设计', link: '/system-design/service-registry' },
          ]
        },
        {
          text: '负载均衡',
          collapsed: false,
          items: [
            { text: '负载均衡详解', link: '/system-design/load-balancing' },
            { text: 'OpenFeign详解', link: '/system-design/openfeign' },
          ]
        },
        {
          text: '熔断限流降级',
          collapsed: false,
          items: [
            { text: '分布式熔断算法和组件详解', link: '/system-design/circuit-breaker' },
            { text: '分布式限流算法和组件详解', link: '/system-design/rate-limiting' },
            { text: '分布式降级算法和组件详解', link: '/system-design/degradation' },
          ]
        },
        {
          text: '场景设计',
          collapsed: false,
          items: [
            { text: '短URL系统设计', link: '/system-design/short-url' },
            { text: '短连接URL系统设计详解', link: '/system-design/short-url-detailed' },
            { text: '互联网企业面试场景题大全', link: '/system-design/interview-questions' },
            { text: '库存扣减一致性解决方案', link: '/system-design/inventory-consistency' },
            { text: '秒杀系统设计', link: '/system-design/flash-sale' },
            { text: '抢红包系统设计', link: '/system-design/red-packet' },
            { text: '12306火车票系统设计', link: '/system-design/train-ticket-system' },
            { text: '分布式缓存系统设计', link: '/system-design/distributed-cache' },
            { text: '打车系统设计', link: '/system-design/ride-hailing' },
            { text: '大数据量下数据库分库分表方案', link: '/system-design/sharding' },
            { text: 'HTTP请求完整流程详解', link: '/system-design/http-flow' },
            { text: 'CDN实现原理详解', link: '/system-design/cdn' },
            { text: 'DNS实现原理详解', link: '/system-design/dns' },
            { text: 'ClickHouse和Doris技术选型对比', link: '/system-design/clickhouse-vs-doris' },
            { text: 'MySQL主从同步方案和对比分析', link: '/system-design/mysql-replication' },
            { text: '技术选型标准和方法', link: '/system-design/tech-selection' },
            { text: '支付系统架构设计', link: '/system-design/payment-system' },
            { text: '滴滴配送引擎业务', link: '/system-design/delivery-engine' },
          ]
        }
      ],

      '/docs/blockchain/': [
        {
          text: '区块链',
          collapsed: false,
          items: [
            {
              text: '基础知识',
              items: [
                { text: '比特币知识学习路线路', link: '/docs/blockchain/基础知识/00.比特币知识学习路线路' },
                { text: '比特币系统中用到的密码学原理', link: '/docs/blockchain/基础知识/01.比特币系统中用到的密码学原理' },
                { text: '比特币系统中用到的数据结构', link: '/docs/blockchain/基础知识/02.比特币系统中用到的数据结构' },
                { text: '比特币协议栈', link: '/docs/blockchain/基础知识/03.比特币协议栈' },
                { text: '比特币挖矿原理', link: '/docs/blockchain/基础知识/04.比特币挖矿原理' },
                { text: '比特币白皮书精华解读', link: '/docs/blockchain/基础知识/05.比特币白皮书精华解读' },
                { text: '比特币网络协议详解', link: '/docs/blockchain/基础知识/06.比特币网络协议详解' },
                { text: '比特币钱包地址实现', link: '/docs/blockchain/基础知识/07.比特币钱包地址实现' },
                { text: '共识机制原理对比', link: '/docs/blockchain/基础知识/08.共识机制原理对比' },
                { text: '比特币脚本编程语言', link: '/docs/blockchain/基础知识/09.比特币脚本编程语言' },
                // ... 更多区块链文章
              ]
            }
          ]
        }
      ],

      '/docs/docker/': [
        {
          text: 'Docker',
          collapsed: false,
          items: [
            { text: 'CentOS安装Docker', link: '/docs/docker/01.CentOS安装Docker' },
            { text: 'Docker安装Redis', link: '/docs/docker/02.Docker安装Redis' },
          ]
        }
      ],

      '/docs/JWT/': [
        {
          text: 'JWT',
          collapsed: false,
          items: [
            { text: 'JWT基础知识', link: '/docs/JWT/JWT基础知识' },
            { text: 'JWT实战应用', link: '/docs/JWT/JWT实战应用' },
            { text: 'JWT常见问题及其排除', link: '/docs/JWT/JWT常见问题及其排除' },
            { text: '使用了JWT之后还需要auth2吗', link: '/docs/JWT/使用了JWT之后还需要auth2吗' },
            { text: '单点登录', link: '/docs/JWT/单点登录' },
          ]
        }
      ],

      '/docs/microservice/': [
        {
          text: '微服务',
          collapsed: false,
          items: [
            { text: 'first', link: '/docs/microservice/first' },
            { text: 'rpcandrestful', link: '/docs/microservice/rpcandrestful' },
          ]
        }
      ],

      '/docs/mybatis/': [
        {
          text: 'MyBatis',
          collapsed: false,
          items: [
            { text: 'first', link: '/docs/mybatis/first' },
          ]
        }
      ],

      '/docs/scheduler/': [
        {
          text: '调度任务',
          collapsed: false,
          items: [
            { text: 'first', link: '/docs/scheduler/first' },
            { text: 'xxlexecutor', link: '/docs/scheduler/xxlexecutor' },
            { text: 'xxlrouter', link: '/docs/scheduler/xxlrouter' },
            { text: 'xxlthread', link: '/docs/scheduler/xxlthread' },
          ]
        }
      ],

      '/docs/swift/': [
        {
          text: 'Swift',
          collapsed: false,
          items: [
            { text: '在VSCode中编写运行Swift', link: '/docs/swift/01.在VSCode中编写运行Swift' },
            { text: 'SwiftUI中的布局方式', link: '/docs/swift/02.SwiftUI中的布局方式' },
            { text: 'SwiftUI中的组合方式', link: '/docs/swift/03.SwiftUI中的组合方式' },
          ]
        }
      ],

      '/docs/systemdesign/': [
        {
          text: '系统设计',
          collapsed: false,
          items: [
            { text: 'job', link: '/docs/systemdesign/job' },
            { text: '企业内部事件中心设计', link: '/docs/systemdesign/企业内部事件中心设计' },
          ]
        }
      ],

      '/miniprogram/': [
        {
          text: '小程序',
          collapsed: false,
          items: [
            { text: '保险行业小程序赚钱', link: '/miniprogram/monetization' },
            { text: '小程序备案通过查询', link: '/miniprogram/beian-progress' },
            { text: '重新迭代', link: '/miniprogram/development-process' },
          ]
        }
      ],

      '/git/': [
        {
          text: 'Git',
          collapsed: false,
          items: [
            { text: '推送本地提交到远程仓库', link: '/git/remote' },
            { text: 'Git分支常用命令', link: '/git/branch' },
            { text: '本地项目取消git仓库关联', link: '/git/unlink' },
            { text: 'git重置本地的修改', link: '/git/revert' },
            { text: 'git commit规范', link: '/git/commit' },
          ]
        }
      ],

      '/docker/': [
        {
          text: 'Docker',
          collapsed: false,
          items: [
            { text: 'CentOS安装Docker', link: '/docker/centos-install' },
          ]
        }
      ],

      '/mysql/': [
        {
          text: '实现原理',
          collapsed: false,
          items: [
            { text: '索引', link: '/mysql/index' },
            { text: 'MVCC', link: '/mysql/mvcc' },
            { text: 'Group By 的执行原理', link: '/mysql/group-by' },
            { text: 'SQL 解析', link: '/mysql/sql-parse' },
            { text: 'SQL 查询优化', link: '/mysql/sql-optimization' },
          ]
        },
        {
          text: '应用操作',
          collapsed: false,
          items: [
            { text: '数据类型', link: '/mysql/data-types' },
            { text: '创建用户后为用户授权', link: '/mysql/user-grant' },
          ]
        },
        {
          text: '性能优化',
          collapsed: false,
          items: [
            { text: 'Explain的使用', link: '/mysql/explain' },
          ]
        },
        {
          text: '问题处理',
          collapsed: false,
          items: [
            { text: '死锁', link: '/mysql/deadlock' },
          ]
        }
      ],

      '/redis/': [
        {
          text: '配置应用',
          collapsed: false,
          items: [
            { text: '安装部署', link: '/redis/installation' },
          ]
        },
        {
          text: '数据结构',
          collapsed: false,
          items: [
            { text: 'SDS (Simple Dynamic String)', link: '/redis/sds' },
            { text: 'Redis 代码整体架构', link: '/redis/architecture' },
          ]
        },
        {
          text: '高级特性',
          collapsed: false,
          items: [
            { text: '哨兵模式', link: '/redis/sentinel' },
          ]
        }
      ],

 

      '/my-projects/': [
        {
          text: '我的玩具',
          collapsed: false,
          items: [
            { text: '苹果 App Store 限时免费应用', link: '/my-projects/free-apps' },
            { text: '功能需求文档-按优先级详细说明', link: '/my-projects/requirements-prioritized' },
            { text: '第一期开发需求文档', link: '/my-projects/phase1-requirements' },
            { text: '数据库规划-第一期优化版', link: '/my-projects/database-design-v2' },
            { text: '数据库规划', link: '/my-projects/database-design' },
          ]
        }
      ],

      '/my-experience/': [
        {
          text: '我的经历',
          collapsed: false,
          items: [
            { text: '管理经验', link: '/my-experience/management' },
            { text: '个人亮点分析', link: '/my-experience/highlights' },
          ]
        }
      ],

      '/interview/': [
        {
          text: '面试',
          collapsed: false,
          items: [
            {
              text: '个人情况',
              items: [
                { text: '自我介绍', link: '/interview/self-introduction' },
              ]
            },
            {
              text: '反问环节',
              items: [
                { text: '反问环节可以问哪些问题', link: '/interview/reverse' },
              ]
            },
            {
              text: '面试记录',
              items: [
                {
                  text: '202509',
                  items: [
                    { text: '滴滴一面', link: '/interview/didi-delivery-engine-01' },
                  ]
                }
              ]
            },
          ]
        }
      ],

      '/projects/': [
        {
          text: '项目',
          collapsed: false,
          items: [
            { text: '衢州安邦车载', link: '/projects/quzhou-anbang' },
            { text: '架构设计', link: '/projects/architecture' },
            { text: '对账功能', link: '/projects/reconciliation' },
            { text: '资金调度', link: '/projects/fund-management' },
          ]
        }
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/qinghuazs' }
    ],

    // 搜索配置
    search: {
      provider: 'local'
    },

    // 页脚配置
    footer: {
      message: '祝我们平日都快乐，做平凡的人',
      copyright: 'Copyright © 2025-present'
    },

    // 最后更新时间
    lastUpdated: {
      text: '最后更新于',
      formatOptions: {
        dateStyle: 'short',
        timeStyle: 'short'
      }
    }
  }
}))
