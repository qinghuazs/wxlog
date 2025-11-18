import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

// https://vitepress.dev/reference/site-config
export default withMermaid(defineConfig({
  title: "冬眠日记",
  description: "祝我们平日都快乐，做平凡的人",

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

    // === 01.AI 目录 ===
    // Claude Code
    '01.AI/01.AI开发工具/01. ClaudeCode/Agent工作流引擎详解.md': 'ai/claude-code/agent-workflow-engine.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/Claude Code A2A 机制详解.md': 'ai/claude-code/a2a-mechanism.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/Claude Code MCP 服务加载机制详解.md': 'ai/claude-code/mcp-loading-mechanism.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/architecture/03 Context Management.md': 'ai/claude-code/architecture-03-context-management.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/architecture/04 Session Management.md': 'ai/claude-code/architecture-04-session-management.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/architecture/05 Tool System Architecture.md': 'ai/claude-code/architecture-05-tool-system-architecture.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/architecture/09 Mcp Protocol.md': 'ai/claude-code/architecture-09-mcp-protocol.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/architecture/10 Browser Automation.md': 'ai/claude-code/architecture-10-browser-automation.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/architecture/12 Multimodal Interaction.md': 'ai/claude-code/architecture-12-multimodal-interaction.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/architecture/20 Build Your Own Ai Assistant.md': 'ai/claude-code/architecture-20-build-your-own-ai-assistant.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/architecture/Claude Code 测试与质量保证.md': 'ai/claude-code/architecture-testing-quality-assurance.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/architecture/Completion Summary.md': 'ai/claude-code/architecture-completion-summary.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/architecture/MCP Server开发实战.md': 'ai/claude-code/architecture-18-mcp-server-practice.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/architecture/Performance Optimization.md': 'ai/claude-code/architecture-performance-optimization.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/architecture/index.md': 'ai/claude-code/architecture-index.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/architecture/navigation.md': 'ai/claude-code/architecture-navigation.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/architecture/代码搜索与分析工具.md': 'ai/claude-code/architecture-code-search-analysis.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/architecture/文件操作工具实现.md': 'ai/claude-code/architecture-file-operations.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/安全机制设计.md': 'ai/claude-code/security-design.md',
    '01.AI/01.AI开发工具/01. ClaudeCode/终端集成实现.md': 'ai/claude-code/terminal-integration.md',
    // LangChain
    '01.AI/02.AI 开发框架/01.LangChain/Enterprise Rag System.md': 'ai/langchain/enterprise-rag-system.md',
    '01.AI/02.AI 开发框架/01.LangChain/Learning Roadmap.md': 'ai/langchain/learning-roadmap.md',
    '01.AI/02.AI 开发框架/01.LangChain/Llamaindex Vs Langchain.md': 'ai/langchain/llamaindex-vs-langchain.md',
    '01.AI/02.AI 开发框架/01.LangChain/Recall Concept.md': 'ai/langchain/recall-concept.md',
    '01.AI/02.AI 开发框架/01.LangChain/Week1 Setup Core Concepts.md': 'ai/langchain/week1-setup-core-concepts.md',
    '01.AI/02.AI 开发框架/01.LangChain/Week16 Deployment Optimization.md': 'ai/langchain/week16-deployment-optimization.md',
    '01.AI/02.AI 开发框架/01.LangChain/Week2 Messages Prompts.md': 'ai/langchain/week2-messages-prompts.md',
    '01.AI/02.AI 开发框架/01.LangChain/Week3 Models Deep Dive.md': 'ai/langchain/week3-models-deep-dive.md',
    '01.AI/02.AI 开发框架/01.LangChain/Week4 Chains Basics.md': 'ai/langchain/week4-chains-basics.md',
    '01.AI/02.AI 开发框架/01.LangChain/Week5 Documents.md': 'ai/langchain/week5-documents.md',
    '01.AI/02.AI 开发框架/01.LangChain/Week6 Memory.md': 'ai/langchain/week6-memory.md',
    '01.AI/02.AI 开发框架/01.LangChain/Week7 Tools.md': 'ai/langchain/week7-tools.md',
    '01.AI/02.AI 开发框架/01.LangChain/Week8 Agents.md': 'ai/langchain/week8-agents.md',
    '01.AI/02.AI 开发框架/01.LangChain/index.md': 'ai/langchain/index.md',
    '01.AI/02.AI 开发框架/01.LangChain/第10周-RAG系统详解.md': 'ai/langchain/week10-rag-system.md',
    '01.AI/02.AI 开发框架/01.LangChain/第11周-Callbacks和监控系统.md': 'ai/langchain/week11-callbacks-monitoring.md',
    '01.AI/02.AI 开发框架/01.LangChain/第9周-Embeddings和Vector Stores详解.md': 'ai/langchain/week9-embeddings-vectorstores.md',
    // LangGraph
    '01.AI/02.AI 开发框架/02.LangGraph/Advanced Features.md': 'ai/langgraph/advanced-features.md',
    '01.AI/02.AI 开发框架/02.LangGraph/LangGraph API 参考手册.md': 'ai/langgraph/api-reference.md',
    '01.AI/02.AI 开发框架/02.LangGraph/LangGraph 与 LangChain 集成.md': 'ai/langgraph/langchain-integration.md',
    '01.AI/02.AI 开发框架/02.LangGraph/LangGraph 与其他框架对比分析.md': 'ai/langgraph/framework-comparison.md',
    '01.AI/02.AI 开发框架/02.LangGraph/LangGraph 企业级案例.md': 'ai/langgraph/enterprise-cases.md',
    '01.AI/02.AI 开发框架/02.LangGraph/LangGraph 入门介绍.md': 'ai/langgraph/introduction.md',
    '01.AI/02.AI 开发框架/02.LangGraph/LangGraph 基础案例实战.md': 'ai/langgraph/basic-examples.md',
    '01.AI/02.AI 开发框架/02.LangGraph/LangGraph 基础概念深度解析.md': 'ai/langgraph/basic-concepts.md',
    '01.AI/02.AI 开发框架/02.LangGraph/LangGraph 多智能体系统.md': 'ai/langgraph/multi-agent-system.md',
    '01.AI/02.AI 开发框架/02.LangGraph/LangGraph 学习指南 - 目录索引.md': 'ai/langgraph/index.md',
    '01.AI/02.AI 开发框架/02.LangGraph/LangGraph 学习路线图.md': 'ai/langgraph/learning-roadmap.md',
    '01.AI/02.AI 开发框架/02.LangGraph/LangGraph 实战案例详解.md': 'ai/langgraph/practical-examples.md',
    '01.AI/02.AI 开发框架/02.LangGraph/LangGraph 实际项目案例.md': 'ai/langgraph/real-projects.md',
    '01.AI/02.AI 开发框架/02.LangGraph/LangGraph 常见问题与解决方案.md': 'ai/langgraph/troubleshooting.md',
    '01.AI/02.AI 开发框架/02.LangGraph/LangGraph 快速入门指南.md': 'ai/langgraph/quickstart.md',
    '01.AI/02.AI 开发框架/02.LangGraph/LangGraph 性能优化指南.md': 'ai/langgraph/performance-optimization.md',
    '01.AI/02.AI 开发框架/02.LangGraph/LangGraph 性能测试与基准.md': 'ai/langgraph/performance-testing.md',
    '01.AI/02.AI 开发框架/02.LangGraph/LangGraph 最佳实践总结.md': 'ai/langgraph/best-practices.md',
    '01.AI/02.AI 开发框架/02.LangGraph/LangGraph 核心概念解析.md': 'ai/langgraph/core-concepts.md',
    '01.AI/02.AI 开发框架/02.LangGraph/LangGraph 核心概念详解与作用.md': 'ai/langgraph/core-concepts-detail.md',
    '01.AI/02.AI 开发框架/02.LangGraph/LangGraph 测试与调试.md': 'ai/langgraph/testing-debugging.md',
    '01.AI/02.AI 开发框架/02.LangGraph/LangGraph 测试策略与最佳实践.md': 'ai/langgraph/testing-strategy.md',
    '01.AI/02.AI 开发框架/02.LangGraph/LangGraph 状态管理详解.md': 'ai/langgraph/state-management.md',
    '01.AI/02.AI 开发框架/02.LangGraph/LangGraph 生产部署实践.md': 'ai/langgraph/production-deployment.md',
    '01.AI/02.AI 开发框架/02.LangGraph/LangGraph 节点开发指南.md': 'ai/langgraph/node-development.md',
    '01.AI/02.AI 开发框架/02.LangGraph/LangGraph 路由与控制流.md': 'ai/langgraph/routing-control.md',
    '01.AI/02.AI 开发框架/02.LangGraph/LangGraph 项目开发实战.md': 'ai/langgraph/project-development.md',
    '01.AI/02.AI 开发框架/02.LangGraph/LangGraph 项目模板与脚手架.md': 'ai/langgraph/project-template.md',
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
  },  },

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
        text: '后端技术',
        items: [
          { text: 'Java', link: '/docs/java/first' },
          { text: 'Spring', link: '/docs/spring/Bean/01.Spring Bean 的加载过程' },
          { text: 'SpringBoot', link: '/docs/spring/SpringBoot/01.SpringBoot日志配置' },
          { text: 'SpringCloud', link: '/docs/spring/SpringCloud/Gateway/01.Spring Cloud Gateway' },
          { text: 'MyBatis', link: '/docs/mybatis/first' },
          { text: 'Kafka', link: '/docs/kafka/00.prompts' },
        ]
      },
      {
        text: '数据库',
        items: [
          { text: 'MySQL', link: '/docs/database/mysql/prompts' },
          { text: 'Redis', link: '/docs/database/redis/data-structures/01.sds' },
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
          { text: 'Claude Code', link: '/docs/ai/Claude Code/01.模型切换' },
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
          { text: '场景题', link: '/system-design/short-url.html' },
          { text: '区块链', link: '/docs/blockchain/基础知识/00.比特币知识学习路线路' },
        ]
      },
      {
        text: '工具',
        items: [
          { text: 'Docker', link: '/docs/docker/01.CentOS安装Docker' },
          { text: 'Git', link: '/docs/工具/git/01.从项目中取消git版库跟踪' },
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
      '/docs/ai/': [
        {
          text: 'Claude Code',
          collapsed: false,
          items: [
            { text: '模型切换', link: '/docs/ai/Claude Code/01.模型切换' },
            { text: 'Claude-Code命令行操作手册', link: '/docs/ai/Claude Code/Claude-Code命令行操作手册' },
            { text: 'MCP高级配置和常见问题', link: '/docs/ai/Claude Code/MCP高级配置和常见问题' },
            {
              text: 'A2A',
              collapsed: false,
              items: [
                { text: 'A2A 进行详细的解读', link: '/docs/ai/Claude Code/A2A/A2A 对 A2A 进行详细的解读' }
              ]
            },
            {
              text: '架构详解',
              collapsed: true,
              items: [
                { text: '整体架构设计', link: '/docs/ai/Claude Code/架构详解/01-整体架构设计' },
                { text: '核心引擎实现', link: '/docs/ai/Claude Code/架构详解/02-核心引擎实现' },
                { text: '上下文管理系统', link: '/docs/ai/Claude Code/架构详解/03-上下文管理系统' },
                { text: '会话与状态管理', link: '/docs/ai/Claude Code/架构详解/04-会话与状态管理' },
                { text: '工具系统架构', link: '/docs/ai/Claude Code/架构详解/05-工具系统架构' },
                { text: '文件操作工具实现', link: '/docs/ai/Claude Code/架构详解/06-文件操作工具实现' },
                { text: '代码搜索与分析工具', link: '/docs/ai/Claude Code/架构详解/07-代码搜索与分析工具' },
                { text: '终端集成实现', link: '/docs/ai/Claude Code/架构详解/08-终端集成实现' },
                { text: 'MCP协议深入解析', link: '/docs/ai/Claude Code/架构详解/09-MCP协议深入解析' },
                { text: '浏览器自动化集成', link: '/docs/ai/Claude Code/架构详解/10-浏览器自动化集成' },
                { text: 'Agent工作流引擎', link: '/docs/ai/Claude Code/架构详解/11-Agent工作流引擎' },
                { text: '多模态交互实现', link: '/docs/ai/Claude Code/架构详解/12-多模态交互实现' },
                { text: '性能优化深度剖析', link: '/docs/ai/Claude Code/架构详解/13-性能优化深度剖析' },
                { text: '错误处理与容错设计', link: '/docs/ai/Claude Code/架构详解/14-错误处理与容错设计' },
                { text: '安全机制设计', link: '/docs/ai/Claude Code/架构详解/15-安全机制设计' },
                { text: '测试与质量保证', link: '/docs/ai/Claude Code/架构详解/16-测试与质量保证' },
                { text: '自定义工具开发实战', link: '/docs/ai/Claude Code/架构详解/17-自定义工具开发实战' },
                { text: 'MCP-Server开发实战', link: '/docs/ai/Claude Code/架构详解/18-MCP-Server开发实战' },
                { text: '集成第三方服务', link: '/docs/ai/Claude Code/架构详解/19-集成第三方服务' },
                { text: '构建自己的AI编程助手', link: '/docs/ai/Claude Code/架构详解/20-构建自己的AI编程助手' },
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
            { text: 'README', link: '/docs/ai/langchain/README' },
            { text: 'LlamaIndex与LangChain对比分析', link: '/docs/ai/langchain/LlamaIndex与LangChain对比分析' },
            { text: '召回(Recall)质量评测', link: '/docs/ai/langchain/召回(Recall)质量评测' },
            { text: '第1章-企业级智能的概念', link: '/docs/ai/langchain/第1章-企业级智能的概念' },
            { text: '第2章-Messages与Prompts', link: '/docs/ai/langchain/第2章-Messages与Prompts' },
            { text: '第3章-Models层', link: '/docs/ai/langchain/第3章-Models层' },
            { text: '第4章-Chains链式', link: '/docs/ai/langchain/第4章-Chains链式' },
            { text: '第5章-Documents文档处理', link: '/docs/ai/langchain/第5章-Documents文档处理' },
            { text: '第6章-Memory记忆系统', link: '/docs/ai/langchain/第6章-Memory记忆系统' },
            { text: '第7章-Tools工具集成', link: '/docs/ai/langchain/第7章-Tools工具集成' },
            { text: '第8章-Agents代理器', link: '/docs/ai/langchain/第8章-Agents代理器' },
            { text: '第9章-Embeddings与VectorStores', link: '/docs/ai/langchain/第9章-Embeddings与VectorStores' },
            { text: '第10章-RAG系统设计', link: '/docs/ai/langchain/第10章-RAG系统设计' },
            { text: '第11章-Callbacks与监控系统', link: '/docs/ai/langchain/第11章-Callbacks与监控系统' },
            { text: '第16章-企业级部署与优化', link: '/docs/ai/langchain/第16章-企业级部署与优化' },
            { text: '综合实战-企业级RAG系统', link: '/docs/ai/langchain/综合实战-企业级RAG系统' },
          ]
        },
        {
          text: 'LangGraph',
          collapsed: true,
          items: [
            { text: '目录概览', link: '/docs/ai/langgraph/00.目录概览' },
            { text: '快速开始指南', link: '/docs/ai/langgraph/01.快速开始指南' },
            { text: '核心概念讲解', link: '/docs/ai/langgraph/02.核心概念讲解' },
            { text: '图构建实践', link: '/docs/ai/langgraph/03.图构建实践' },
            { text: '状态管理详解', link: '/docs/ai/langgraph/04.状态管理详解' },
            { text: '路由与条件分支', link: '/docs/ai/langgraph/05.路由与条件分支' },
            { text: '节点开发指南', link: '/docs/ai/langgraph/06.节点开发指南' },
            { text: '项目实战应用', link: '/docs/ai/langgraph/07.项目实战应用' },
            { text: '错误处理机制', link: '/docs/ai/langgraph/08.错误处理机制' },
            { text: '性能优化指南', link: '/docs/ai/langgraph/09.性能优化指南' },
            { text: '高级特性探索', link: '/docs/ai/langgraph/10.高级特性探索' },
            { text: 'LangChain集成', link: '/docs/ai/langgraph/11.LangChain集成' },
            { text: '流式响应系统', link: '/docs/ai/langgraph/12.流式响应系统' },
            { text: '检查点与回滚', link: '/docs/ai/langgraph/13.检查点与回滚' },
            { text: '企业级应用', link: '/docs/ai/langgraph/14.企业级应用' },
            { text: 'API参考手册', link: '/docs/ai/langgraph/15.API参考手册' },
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

      '/docs/java/': [
        {
          text: 'Java 基础',
          collapsed: false,
          items: [
            { text: 'first', link: '/docs/java/first' },
            { text: 'arthas', link: '/docs/java/arthas' },
            { text: 'arthas-redefine', link: '/docs/java/arthas-redefine' },
            { text: 'concurrenct', link: '/docs/java/concurrenct' },
            { text: 'consumer', link: '/docs/java/consumer' },
            { text: 'functional', link: '/docs/java/functional' },
            { text: 'javathread', link: '/docs/java/javathread' },
            { text: 'jvmcommand', link: '/docs/java/jvmcommand' },
            { text: 'lock', link: '/docs/java/lock' },
          ]
        },
        {
          text: 'IO',
          collapsed: false,
          items: [
            { text: 'IO流程', link: '/docs/java/IO/01.IO流程' },
            { text: '五路复用', link: '/docs/java/IO/02.五路复用' },
          ]
        },
        {
          text: 'JUC',
          collapsed: false,
          items: [
            { text: 'AbstractQueuedSynchronizer', link: '/docs/java/JUC/02.AbstractQueuedSynchronizer' },
            {
              text: '原子类',
              items: [
                { text: '原子类', link: '/docs/java/JUC/原子类/01.原子类' },
              ]
            },
            {
              text: '锁',
              items: [
                { text: 'JUC中的锁', link: '/docs/java/JUC/锁/01.JUC中的锁' },
                { text: 'ReentrantLock源码（深度剖析）', link: '/docs/java/JUC/锁/02.ReentrantLockԴ源码（深度剖析）' },
                { text: '独占锁与共享锁', link: '/docs/java/JUC/锁/03.独占锁与共享锁' },
                { text: 'CountDownLatch 源码（深度剖析）', link: '/docs/java/JUC/锁/04.CountDownLatch 源码（深度剖析）' },
                { text: 'Semphore源码（深度剖析）', link: '/docs/java/JUC/锁/05.Semphore源码（深度剖析）' },
              ]
            },
            {
              text: '队列',
              items: [
                { text: 'JUC中的队列', link: '/docs/java/JUC/队列/01.JUC中的队列' },
              ]
            }
          ]
        },
        {
          text: 'JVM',
          collapsed: false,
          items: [
            { text: '线程FullGC问题排查', link: '/docs/java/JVM/01.线程FullGC问题排查' },
            { text: 'Java内存结构', link: '/docs/java/JVM/02.Java内存结构' },
            { text: '元空间 MetaSpace', link: '/docs/java/JVM/03.元空间 MetaSpace' },
          ]
        },
        {
          text: 'Maven',
          collapsed: false,
          items: [
            { text: '查看Maven依赖', link: '/docs/java/Maven/01查看Maven依赖' },
          ]
        },
        {
          text: 'Thread',
          collapsed: false,
          items: [
            {
              text: '线程',
              items: [
                { text: 'Thread', link: '/docs/java/Thread/线程/00.Thread' },
                { text: 'ThreadLocal', link: '/docs/java/Thread/线程/01.ThreadLocal' },
              ]
            },
            {
              text: '线程池',
              items: [
                { text: '自定义线程池工厂类', link: '/docs/java/Thread/线程池/01.自定义线程池工厂类' },
                { text: '线程池', link: '/docs/java/Thread/线程池/03.线程池' },
              ]
            }
          ]
        },
        {
          text: '集合',
          collapsed: false,
          items: [
            { text: 'HashMap 源码（深度剖析）', link: '/docs/java/集合/01.HashMap 源码（深度剖析）' },
            { text: 'ConcurrentHashMap 源码（深度剖析）', link: '/docs/java/集合/02.ConcurrentHashMap 源码（深度剖析）' },
          ]
        }
      ],

      '/docs/spring/': [
        {
          text: 'Spring',
          collapsed: false,
          items: [
            {
              text: 'Bean',
              items: [
                { text: 'Spring Bean 的加载过程', link: '/docs/spring/Bean/01.Spring Bean 的加载过程' },
              ]
            },
            {
              text: 'SpringBasic',
              items: [
                { text: 'Spring AOP实现机制', link: '/docs/spring/SpringBasic/01.Spring AOP实现机制' },
                { text: 'Spring扩展点', link: '/docs/spring/SpringBasic/02.Spring扩展点' },
              ]
            },
            {
              text: 'SpringBoot',
              items: [
                { text: 'SpringBoot日志配置', link: '/docs/spring/SpringBoot/01.SpringBoot日志配置' },
                { text: 'SpringBoot Starter', link: '/docs/spring/SpringBoot/02.SpringBoot Starter' },
                { text: 'Endpoint', link: '/docs/spring/SpringBoot/03.Endpoint' },
              ]
            },
            {
              text: 'SpringCloud',
              collapsed: true,
              items: [
                {
                  text: 'Gateway',
                  items: [
                    { text: 'Spring Cloud Gateway', link: '/docs/spring/SpringCloud/Gateway/01.Spring Cloud Gateway' },
                    { text: 'Spring Cloud Gateway的使用', link: '/docs/spring/SpringCloud/Gateway/02.Spring Cloud Gateway的使用' },
                    { text: '动态路由', link: '/docs/spring/SpringCloud/Gateway/03.动态路由' },
                  ]
                },
                {
                  text: 'OpenFeign',
                  items: [
                    { text: 'Spring Cloud OpenFeign', link: '/docs/spring/SpringCloud/OpenFeign/01.Spring Cloud OpenFeign' },
                  ]
                },
                {
                  text: '熔断器',
                  items: [
                    { text: 'Resilience4J', link: '/docs/spring/SpringCloud/熔断器/01. Resilience4J' },
                    { text: 'Resilience4J熔断器使用', link: '/docs/spring/SpringCloud/熔断器/02.Resilience4J熔断器使用' },
                    { text: 'Resilience4J限流器使用', link: '/docs/spring/SpringCloud/熔断器/03.Resilience4J限流器使用' },
                  ]
                },
                {
                  text: '注册中心',
                  items: [
                    { text: 'Eureka', link: '/docs/spring/SpringCloud/注册中心/01.Eureka' },
                    { text: 'Eureka核心机制', link: '/docs/spring/SpringCloud/注册中心/02.Eureka核心机制' },
                    { text: 'Consul', link: '/docs/spring/SpringCloud/注册中心/03.Consul' },
                    { text: 'Consul高级特性', link: '/docs/spring/SpringCloud/注册中心/04.Consul高级特性' },
                    { text: 'Raft算法', link: '/docs/spring/SpringCloud/注册中心/05.Raft算法' },
                  ]
                },
                {
                  text: '负载均衡',
                  items: [
                    { text: 'Ribbon', link: '/docs/spring/SpringCloud/负载均衡/01.Ribbon' },
                    { text: 'Spring Cloud LoadBalancer', link: '/docs/spring/SpringCloud/负载均衡/02.Spring Cloud LoadBalancer' },
                  ]
                }
              ]
            },
            {
              text: '事务管理',
              items: [
                { text: 'PlatformTransactionManager', link: '/docs/spring/事务管理/01.PlatformTransactionManager' },
              ]
            },
            {
              text: '缓存',
              items: [
                { text: '数据库连接池配置', link: '/docs/spring/缓存/01.数据库连接池配置' },
                { text: 'Redis缓存配置', link: '/docs/spring/缓存/02.Redis缓存配置' },
              ]
            }
          ]
        }
      ],

      '/docs/database/': [
        {
          text: 'MySQL',
          collapsed: false,
          items: [
            { text: '提示词', link: '/docs/database/mysql/prompts' },
            {
              text: '原理',
              items: [
                { text: '数据类型', link: '/docs/database/mysql/原理/01.数据类型' },
                { text: 'mvcc', link: '/docs/database/mysql/原理/02.mvcc' },
                { text: 'Group By 的执行原理', link: '/docs/database/mysql/原理/03.Group By 的执行原理' },
                { text: 'SQL 解析', link: '/docs/database/mysql/原理/04.SQL 解析' },
                { text: 'SQL 查询优化', link: '/docs/database/mysql/原理/05.SQL 查询优化' },
              ]
            },
            {
              text: '应用',
              items: [
                { text: '索引', link: '/docs/database/mysql/应用/01.索引' },
                { text: '创建用户及为用户授权', link: '/docs/database/mysql/应用/02.创建用户及为用户授权' },
              ]
            },
            {
              text: '性能优化类',
              items: [
                { text: 'Explain的使用', link: '/docs/database/mysql/性能优化类/01.Explain的使用' },
              ]
            },
            {
              text: '问题处理',
              items: [
                { text: '死锁', link: '/docs/database/mysql/问题处理/01.死锁' },
              ]
            }
          ]
        },
        {
          text: 'Redis',
          collapsed: false,
          items: [
            {
              text: 'C语言基础',
              items: [
                { text: '文件事件', link: '/docs/database/redis/C语言基础/01.文件事件' },
              ]
            },
            {
              text: '安装部署配置',
              items: [
                { text: 'MacOS安装 Redis', link: '/docs/database/redis/安装部署配置/01.MacOS安装 Redis' },
                { text: 'Redis Docker 部署', link: '/docs/database/redis/安装部署配置/02.Redis Docker 部署' },
              ]
            },
            {
              text: '数据结构',
              items: [
                { text: 'sds', link: '/docs/database/redis/data-structures/01.sds' },
                { text: 'Redis 持久化详解全', link: '/docs/database/redis/data-structures/02.Redis 持久化详解全' },
              ]
            },
            {
              text: '高可用',
              items: [
                { text: '哨兵模式', link: '/docs/database/redis/高可用/01.哨兵模式' },
              ]
            }
          ]
        }
      ],

      '/docs/kafka/': [
        {
          text: 'Kafka',
          collapsed: false,
          items: [
            { text: '提示词', link: '/docs/kafka/00.prompts' },
            { text: 'Kafka概述介绍', link: '/docs/kafka/01Kafka概述介绍' },
            { text: 'Kafka安装部署', link: '/docs/kafka/02Kafka安装部署' },
            { text: 'Kafka安装部署2', link: '/docs/kafka/02Kafka安装部署2' },
            {
              text: '源码解析',
              items: [
                { text: '前置准备', link: '/docs/kafka/源码解析/01前置准备' },
                {
                  text: 'Producer',
                  items: [
                    { text: '指标', link: '/docs/kafka/源码解析/Producer/01.指标' },
                    { text: '配置参数', link: '/docs/kafka/源码解析/Producer/02.配置参数' },
                    { text: '重要配置参数', link: '/docs/kafka/源码解析/Producer/03.重要配置参数' },
                    { text: '发送概览', link: '/docs/kafka/源码解析/Producer/04.发送概览' },
                    { text: '发送消息', link: '/docs/kafka/源码解析/Producer/05.发送消息' },
                    { text: '消息追加到消息累加器', link: '/docs/kafka/源码解析/Producer/06.消息追加到消息累加器' },
                    { text: '粘性分区', link: '/docs/kafka/源码解析/Producer/07.粘性分区' },
                  ]
                }
              ]
            },
            {
              text: '原理解析',
              items: [
                { text: 'Kafka为什么这么快', link: '/docs/kafka/原理解析/01.Kafka为什么这么快' },
                { text: '副本机制', link: '/docs/kafka/原理解析/02.副本机制' },
              ]
            }
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
          text: '系统设计',
          collapsed: false,
          items: [
            { text: '短URL系统设计', link: '/system-design/short-url' },
            { text: '注册中心的设计', link: '/system-design/service-registry' },
            { text: '负载均衡设计', link: '/system-design/load-balancing' },
            { text: '分布式熔断算法（滑动窗口）', link: '/system-design/circuit-breaker' },
            { text: 'OpenFeign详解', link: '/system-design/openfeign' },
            { text: '分布式限流算法（令牌桶算法）', link: '/system-design/rate-limiting' },
            { text: '深度解析短URL系统高级特性', link: '/system-design/short-url-detailed' },
            { text: '秒杀系统设计', link: '/system-design/flash-sale' },
            { text: '12306抢票系统设计', link: '/system-design/train-ticket-system' },
            { text: '详细分析数据库分库分表的方案', link: '/system-design/sharding' },
            { text: 'CDN实现原理详解', link: '/system-design/cdn' },
            { text: 'DNS实现原理详解', link: '/system-design/dns' },
            { text: 'ClickHouse和Doris技术选型对比', link: '/system-design/clickhouse-vs-doris' },
            { text: 'MySQL几种同步机制对比分析', link: '/system-design/mysql-replication' },
            { text: '技术选型标准与分析', link: '/system-design/tech-selection' },
            { text: '支付系统设计', link: '/system-design/payment-system' },
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

      '/docs/工具/': [
        {
          text: '工具',
          collapsed: false,
          items: [
            {
              text: 'git',
              items: [
                { text: '从项目中取消git版库跟踪', link: '/docs/工具/git/01.从项目中取消git版库跟踪' },
                { text: 'git分支相关操作', link: '/docs/工具/git/02.git分支相关操作' },
                { text: 'git撤销本地提交到远程仓库操作', link: '/docs/工具/git/03.git撤销本地提交到远程仓库操作' },
                { text: 'git重置本地的修改', link: '/docs/工具/git/04.git重置本地的修改' },
                { text: '代码提交规范', link: '/docs/工具/git/05.代码提交规范' },
              ]
            }
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
      { icon: 'github', link: 'https://github.com/vuejs/vitepress' }
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
