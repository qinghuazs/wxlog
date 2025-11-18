---
title: VSCode 插件开发 - Code Agent 前端实现
date: 2025-01-30
permalink: /ai/codeagent/vscode-extension.html
categories:
  - AI
  - CodeAgent
---

# VSCode 插件开发

## 一、插件初始化

### 1.1 创建项目

```bash
# 安装脚手架工具
npm install -g yo generator-code

# 创建插件项目
yo code

# 选择配置
? What type of extension do you want to create? New Extension (TypeScript)
? What's the name of your extension? code-agent
? What's the identifier of your extension? code-agent
? What's the description of your extension? AI-powered coding assistant
? Initialize a git repository? Yes
? Which package manager to use? npm

cd code-agent
npm install
```

### 1.2 项目结构

```
code-agent/
├── src/
│   ├── extension.ts          # 插件入口
│   ├── providers/            # 各种Provider实现
│   │   ├── CompletionProvider.ts
│   │   ├── ChatProvider.ts
│   │   ├── HoverProvider.ts
│   │   └── CodeActionProvider.ts
│   ├── services/             # 业务服务
│   │   ├── ApiClient.ts
│   │   ├── ContextBuilder.ts
│   │   └── CacheManager.ts
│   ├── views/                # Webview UI
│   │   ├── ChatPanel.tsx
│   │   └── DiffView.tsx
│   └── utils/                # 工具函数
│       ├── logger.ts
│       └── config.ts
├── media/                    # 静态资源
├── package.json              # 插件配置
└── tsconfig.json             # TypeScript配置
```

### 1.3 package.json 配置

```json
{
  "name": "code-agent",
  "displayName": "Code Agent",
  "description": "AI-powered coding assistant",
  "version": "0.0.1",
  "engines": {
    "vscode": "^1.80.0"
  },
  "categories": [
    "Programming Languages",
    "Machine Learning",
    "Snippets"
  ],
  "activationEvents": [
    "onStartupFinished"
  ],
  "main": "./out/extension.js",
  "contributes": {
    "commands": [
      {
        "command": "code-agent.chat",
        "title": "Code Agent: 打开聊天"
      },
      {
        "command": "code-agent.explain",
        "title": "Code Agent: 解释代码"
      },
      {
        "command": "code-agent.refactor",
        "title": "Code Agent: 重构代码"
      },
      {
        "command": "code-agent.generateTests",
        "title": "Code Agent: 生成测试"
      }
    ],
    "keybindings": [
      {
        "command": "code-agent.chat",
        "key": "ctrl+shift+a",
        "mac": "cmd+shift+a"
      }
    ],
    "menus": {
      "editor/context": [
        {
          "command": "code-agent.explain",
          "group": "code-agent@1",
          "when": "editorHasSelection"
        },
        {
          "command": "code-agent.refactor",
          "group": "code-agent@2",
          "when": "editorHasSelection"
        }
      ]
    },
    "configuration": {
      "title": "Code Agent",
      "properties": {
        "codeAgent.apiEndpoint": {
          "type": "string",
          "default": "http://localhost:8000",
          "description": "后端 API 地址"
        },
        "codeAgent.model": {
          "type": "string",
          "default": "gpt-4",
          "enum": ["gpt-4", "gpt-3.5-turbo", "claude-3", "ollama"],
          "description": "使用的 AI 模型"
        },
        "codeAgent.enableAutoComplete": {
          "type": "boolean",
          "default": true,
          "description": "启用自动补全"
        }
      }
    },
    "viewsContainers": {
      "activitybar": [
        {
          "id": "code-agent",
          "title": "Code Agent",
          "icon": "media/icon.svg"
        }
      ]
    },
    "views": {
      "code-agent": [
        {
          "id": "code-agent.chatView",
          "name": "Chat"
        }
      ]
    }
  },
  "scripts": {
    "vscode:prepublish": "npm run compile",
    "compile": "tsc -p ./",
    "watch": "tsc -watch -p ./",
    "lint": "eslint src --ext ts"
  },
  "devDependencies": {
    "@types/node": "^18.x",
    "@types/vscode": "^1.80.0",
    "@typescript-eslint/eslint-plugin": "^6.0.0",
    "@typescript-eslint/parser": "^6.0.0",
    "eslint": "^8.0.0",
    "typescript": "^5.0.0"
  },
  "dependencies": {
    "axios": "^1.6.0",
    "eventsource": "^2.0.2"
  }
}
```

## 二、核心功能实现

### 2.1 插件入口

```typescript
// src/extension.ts
import * as vscode from 'vscode';
import { CompletionProvider } from './providers/CompletionProvider';
import { ChatProvider } from './providers/ChatProvider';
import { HoverProvider } from './providers/HoverProvider';
import { CodeActionProvider } from './providers/CodeActionProvider';
import { ApiClient } from './services/ApiClient';
import { Logger } from './utils/logger';

let apiClient: ApiClient;
let logger: Logger;

export function activate(context: vscode.ExtensionContext) {
    logger = new Logger();
    logger.info('Code Agent 正在激活...');

    // 初始化 API 客户端
    const config = vscode.workspace.getConfiguration('codeAgent');
    apiClient = new ApiClient(config.get('apiEndpoint') || 'http://localhost:8000');

    // 注册代码补全提供者
    if (config.get('enableAutoComplete')) {
        const completionProvider = new CompletionProvider(apiClient, logger);
        context.subscriptions.push(
            vscode.languages.registerInlineCompletionItemProvider(
                { pattern: '**' },
                completionProvider
            )
        );
        logger.info('代码补全已启用');
    }

    // 注册 Hover 提供者
    const hoverProvider = new HoverProvider(apiClient);
    context.subscriptions.push(
        vscode.languages.registerHoverProvider(
            { scheme: 'file' },
            hoverProvider
        )
    );

    // 注册 Code Action 提供者
    const codeActionProvider = new CodeActionProvider(apiClient);
    context.subscriptions.push(
        vscode.languages.registerCodeActionsProvider(
            { scheme: 'file' },
            codeActionProvider
        )
    );

    // 注册聊天视图
    const chatProvider = new ChatProvider(context, apiClient);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(
            'code-agent.chatView',
            chatProvider
        )
    );

    // 注册命令
    registerCommands(context);

    logger.info('Code Agent 激活完成');
}

function registerCommands(context: vscode.ExtensionContext) {
    // 打开聊天
    context.subscriptions.push(
        vscode.commands.registerCommand('code-agent.chat', () => {
            vscode.commands.executeCommand('workbench.view.extension.code-agent');
        })
    );

    // 解释代码
    context.subscriptions.push(
        vscode.commands.registerCommand('code-agent.explain', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) return;

            const selection = editor.selection;
            const text = editor.document.getText(selection);

            if (!text) {
                vscode.window.showWarningMessage('请先选择要解释的代码');
                return;
            }

            const panel = vscode.window.createWebviewPanel(
                'codeExplanation',
                '代码解释',
                vscode.ViewColumn.Beside,
                { enableScripts: true }
            );

            panel.webview.html = getLoadingHtml();

            try {
                const explanation = await apiClient.explainCode(text, editor.document.languageId);
                panel.webview.html = getExplanationHtml(explanation);
            } catch (error) {
                panel.webview.html = getErrorHtml(String(error));
            }
        })
    );

    // 重构代码
    context.subscriptions.push(
        vscode.commands.registerCommand('code-agent.refactor', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) return;

            const selection = editor.selection;
            const text = editor.document.getText(selection);

            if (!text) {
                vscode.window.showWarningMessage('请先选择要重构的代码');
                return;
            }

            const instruction = await vscode.window.showInputBox({
                prompt: '请输入重构指令',
                placeHolder: '例如：提取函数、重命名变量、优化性能...'
            });

            if (!instruction) return;

            try {
                vscode.window.withProgress({
                    location: vscode.ProgressLocation.Notification,
                    title: '正在重构代码...',
                    cancellable: false
                }, async () => {
                    const refactored = await apiClient.refactorCode(
                        text,
                        instruction,
                        editor.document.languageId
                    );

                    // 显示差异
                    const diffEditor = await vscode.commands.executeCommand(
                        'vscode.diff',
                        vscode.Uri.parse('code-agent:original'),
                        vscode.Uri.parse('code-agent:refactored'),
                        '重构预览'
                    );

                    // 提供应用选项
                    const apply = await vscode.window.showInformationMessage(
                        '是否应用重构结果?',
                        '应用',
                        '取消'
                    );

                    if (apply === '应用') {
                        editor.edit(editBuilder => {
                            editBuilder.replace(selection, refactored);
                        });
                    }
                });
            } catch (error) {
                vscode.window.showErrorMessage(`重构失败: ${error}`);
            }
        })
    );

    // 生成测试
    context.subscriptions.push(
        vscode.commands.registerCommand('code-agent.generateTests', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) return;

            const selection = editor.selection;
            const text = editor.document.getText(selection);

            if (!text) {
                vscode.window.showWarningMessage('请先选择要生成测试的代码');
                return;
            }

            try {
                vscode.window.withProgress({
                    location: vscode.ProgressLocation.Notification,
                    title: '正在生成测试...',
                    cancellable: false
                }, async () => {
                    const tests = await apiClient.generateTests(
                        text,
                        editor.document.languageId
                    );

                    // 创建新文件显示测试
                    const doc = await vscode.workspace.openTextDocument({
                        content: tests,
                        language: editor.document.languageId
                    });

                    await vscode.window.showTextDocument(doc, vscode.ViewColumn.Beside);
                });
            } catch (error) {
                vscode.window.showErrorMessage(`生成测试失败: ${error}`);
            }
        })
    );
}

export function deactivate() {
    logger?.info('Code Agent 已停用');
}

// HTML 模板函数
function getLoadingHtml(): string {
    return `
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body { padding: 20px; font-family: var(--vscode-font-family); }
                .loading { text-align: center; padding: 50px; }
            </style>
        </head>
        <body>
            <div class="loading">
                <p>正在分析代码...</p>
            </div>
        </body>
        </html>
    `;
}

function getExplanationHtml(explanation: string): string {
    return `
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {
                    padding: 20px;
                    font-family: var(--vscode-font-family);
                    color: var(--vscode-foreground);
                }
                pre {
                    background: var(--vscode-textCodeBlock-background);
                    padding: 10px;
                    border-radius: 5px;
                    overflow-x: auto;
                }
                code {
                    font-family: var(--vscode-editor-font-family);
                }
            </style>
        </head>
        <body>
            <h2>代码解释</h2>
            <div>${explanation}</div>
        </body>
        </html>
    `;
}

function getErrorHtml(error: string): string {
    return `
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body { padding: 20px; font-family: var(--vscode-font-family); }
                .error { color: var(--vscode-errorForeground); }
            </style>
        </head>
        <body>
            <div class="error">
                <h3>错误</h3>
                <p>${error}</p>
            </div>
        </body>
        </html>
    `;
}
```

### 2.2 代码补全Provider

```typescript
// src/providers/CompletionProvider.ts
import * as vscode from 'vscode';
import { ApiClient } from '../services/ApiClient';
import { Logger } from '../utils/logger';
import { ContextBuilder } from '../services/ContextBuilder';

export class CompletionProvider implements vscode.InlineCompletionItemProvider {
    private cache = new Map<string, string>();
    private pendingRequest: AbortController | null = null;

    constructor(
        private apiClient: ApiClient,
        private logger: Logger
    ) {}

    async provideInlineCompletionItems(
        document: vscode.TextDocument,
        position: vscode.Position,
        context: vscode.InlineCompletionContext,
        token: vscode.CancellationToken
    ): Promise<vscode.InlineCompletionItem[] | vscode.InlineCompletionList | null> {
        // 取消之前的请求
        if (this.pendingRequest) {
            this.pendingRequest.abort();
        }

        // 获取当前行
        const line = document.lineAt(position.line);
        const prefix = line.text.substring(0, position.character);

        // 如果是空行或只有空格，不补全
        if (!prefix.trim()) {
            return null;
        }

        // 检查缓存
        const cacheKey = this.getCacheKey(document, position);
        const cached = this.cache.get(cacheKey);
        if (cached) {
            this.logger.debug('使用缓存补全');
            return [new vscode.InlineCompletionItem(cached)];
        }

        // 构建上下文
        const contextBuilder = new ContextBuilder();
        const promptContext = await contextBuilder.buildContext({
            document,
            position,
            type: 'completion'
        });

        // 创建新的请求控制器
        this.pendingRequest = new AbortController();

        try {
            // 调用 API 获取补全
            const completion = await this.apiClient.getCompletion(
                promptContext,
                {
                    signal: this.pendingRequest.signal,
                    onProgress: (chunk) => {
                        // 实时更新补全（可选）
                        this.logger.debug(`收到补全片段: ${chunk}`);
                    }
                }
            );

            // 缓存结果
            this.cache.set(cacheKey, completion);

            // 限制缓存大小
            if (this.cache.size > 100) {
                const firstKey = this.cache.keys().next().value;
                this.cache.delete(firstKey!);
            }

            return [new vscode.InlineCompletionItem(completion)];

        } catch (error: any) {
            if (error.name === 'AbortError') {
                this.logger.debug('补全请求被取消');
                return null;
            }

            this.logger.error('补全失败:', error);
            return null;
        } finally {
            this.pendingRequest = null;
        }
    }

    private getCacheKey(document: vscode.TextDocument, position: vscode.Position): string {
        const line = document.lineAt(position.line);
        const prefix = line.text.substring(0, position.character);
        return `${document.uri.toString()}:${position.line}:${prefix}`;
    }
}
```

### 2.3 聊天视图Provider

```typescript
// src/providers/ChatProvider.ts
import * as vscode from 'vscode';
import { ApiClient } from '../services/ApiClient';

interface Message {
    role: 'user' | 'assistant';
    content: string;
}

export class ChatProvider implements vscode.WebviewViewProvider {
    private view?: vscode.WebviewView;
    private messages: Message[] = [];

    constructor(
        private context: vscode.ExtensionContext,
        private apiClient: ApiClient
    ) {}

    resolveWebviewView(
        webviewView: vscode.WebviewView,
        context: vscode.WebviewViewResolveContext,
        token: vscode.CancellationToken
    ): void {
        this.view = webviewView;

        webviewView.webview.options = {
            enableScripts: true
        };

        webviewView.webview.html = this.getHtml(webviewView.webview);

        // 处理来自 webview 的消息
        webviewView.webview.onDidReceiveMessage(async message => {
            switch (message.type) {
                case 'sendMessage':
                    await this.handleUserMessage(message.text);
                    break;
                case 'clearHistory':
                    this.messages = [];
                    this.updateChat();
                    break;
            }
        });
    }

    private async handleUserMessage(text: string) {
        // 添加用户消息
        this.messages.push({ role: 'user', content: text });
        this.updateChat();

        try {
            // 调用 API
            let assistantMessage = '';

            await this.apiClient.chat(
                this.messages,
                {
                    onChunk: (chunk) => {
                        assistantMessage += chunk;
                        // 实时更新
                        this.view?.webview.postMessage({
                            type: 'streamChunk',
                            content: chunk
                        });
                    }
                }
            );

            // 保存助手消息
            this.messages.push({
                role: 'assistant',
                content: assistantMessage
            });

            this.updateChat();

        } catch (error) {
            vscode.window.showErrorMessage(`聊天失败: ${error}`);
        }
    }

    private updateChat() {
        this.view?.webview.postMessage({
            type: 'updateMessages',
            messages: this.messages
        });
    }

    private getHtml(webview: vscode.Webview): string {
        const scriptUri = webview.asWebviewUri(
            vscode.Uri.joinPath(this.context.extensionUri, 'media', 'chat.js')
        );
        const styleUri = webview.asWebviewUri(
            vscode.Uri.joinPath(this.context.extensionUri, 'media', 'chat.css')
        );

        return `
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="${styleUri}" rel="stylesheet">
</head>
<body>
    <div id="chat-container">
        <div id="messages"></div>
        <div id="input-container">
            <textarea id="user-input" placeholder="输入消息..."></textarea>
            <button id="send-btn">发送</button>
            <button id="clear-btn">清空</button>
        </div>
    </div>
    <script src="${scriptUri}"></script>
</body>
</html>
        `;
    }
}
```

### 2.4 API 客户端

```typescript
// src/services/ApiClient.ts
import axios, { AxiosInstance } from 'axios';
import EventSource from 'eventsource';

export class ApiClient {
    private client: AxiosInstance;

    constructor(private baseUrl: string) {
        this.client = axios.create({
            baseURL: baseUrl,
            timeout: 30000,
            headers: {
                'Content-Type': 'application/json'
            }
        });
    }

    async getCompletion(
        context: any,
        options?: {
            signal?: AbortSignal;
            onProgress?: (chunk: string) => void;
        }
    ): Promise<string> {
        if (options?.onProgress) {
            // 流式请求
            return this.streamRequest('/completion/stream', context, options.onProgress, options.signal);
        }

        // 普通请求
        const response = await this.client.post('/completion', context, {
            signal: options?.signal
        });

        return response.data.completion;
    }

    async chat(
        messages: any[],
        options?: {
            onChunk?: (chunk: string) => void;
        }
    ): Promise<string> {
        if (options?.onChunk) {
            return this.streamRequest('/chat/stream', { messages }, options.onChunk);
        }

        const response = await this.client.post('/chat', { messages });
        return response.data.message;
    }

    async explainCode(code: string, language: string): Promise<string> {
        const response = await this.client.post('/explain', {
            code,
            language
        });
        return response.data.explanation;
    }

    async refactorCode(code: string, instruction: string, language: string): Promise<string> {
        const response = await this.client.post('/refactor', {
            code,
            instruction,
            language
        });
        return response.data.refactored;
    }

    async generateTests(code: string, language: string): Promise<string> {
        const response = await this.client.post('/generate-tests', {
            code,
            language
        });
        return response.data.tests;
    }

    private streamRequest(
        endpoint: string,
        data: any,
        onChunk: (chunk: string) => void,
        signal?: AbortSignal
    ): Promise<string> {
        return new Promise((resolve, reject) => {
            const url = `${this.baseUrl}${endpoint}`;
            const eventSource = new EventSource(url, {
                headers: {
                    'Content-Type': 'application/json'
                },
                method: 'POST',
                body: JSON.stringify(data)
            } as any);

            let fullResponse = '';

            eventSource.onmessage = (event) => {
                const chunk = event.data;
                fullResponse += chunk;
                onChunk(chunk);
            };

            eventSource.onerror = (error) => {
                eventSource.close();
                reject(error);
            };

            eventSource.addEventListener('done', () => {
                eventSource.close();
                resolve(fullResponse);
            });

            // 处理取消
            signal?.addEventListener('abort', () => {
                eventSource.close();
                reject(new DOMException('Aborted', 'AbortError'));
            });
        });
    }
}
```

## 三、调试与测试

### 3.1 调试配置

```json
// .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "运行扩展",
            "type": "extensionHost",
            "request": "launch",
            "args": [
                "--extensionDevelopmentPath=${workspaceFolder}"
            ],
            "outFiles": [
                "${workspaceFolder}/out/**/*.js"
            ],
            "preLaunchTask": "${defaultBuildTask}"
        },
        {
            "name": "扩展测试",
            "type": "extensionHost",
            "request": "launch",
            "args": [
                "--extensionDevelopmentPath=${workspaceFolder}",
                "--extensionTestsPath=${workspaceFolder}/out/test/suite/index"
            ],
            "outFiles": [
                "${workspaceFolder}/out/test/**/*.js"
            ],
            "preLaunchTask": "${defaultBuildTask}"
        }
    ]
}
```

### 3.2 编写测试

```typescript
// src/test/suite/extension.test.ts
import * as assert from 'assert';
import * as vscode from 'vscode';

suite('Extension Test Suite', () => {
    vscode.window.showInformationMessage('开始所有测试。');

    test('Extension should be present', () => {
        assert.ok(vscode.extensions.getExtension('yourpublisher.code-agent'));
    });

    test('Should activate', async () => {
        const ext = vscode.extensions.getExtension('yourpublisher.code-agent');
        await ext?.activate();
        assert.ok(ext?.isActive);
    });

    test('Should register commands', async () => {
        const commands = await vscode.commands.getCommands();
        assert.ok(commands.includes('code-agent.chat'));
        assert.ok(commands.includes('code-agent.explain'));
    });
});
```

## 四、打包与发布

### 4.1 打包插件

```bash
# 安装 vsce
npm install -g @vscode/vsce

# 打包
vsce package

# 生成 code-agent-0.0.1.vsix
```

### 4.2 发布到市场

```bash
# 创建发布者账号（在 https://marketplace.visualstudio.com）

# 创建个人访问令牌

# 登录
vsce login yourpublisher

# 发布
vsce publish
```

---

**下一步**: 查看 [03.代码补全实现](./03.代码补全实现.md) 深入了解补全系统！
