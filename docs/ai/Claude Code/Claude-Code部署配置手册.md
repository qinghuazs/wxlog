---
title: Claude-Code部署配置手册
date: 2025-11-17
categories:
  - AI
  - Claude Code
---

# 更新 Homebrew
```bash
brew update
```

# 安装 Node.js
```bash
brew install node
```

# 检查安装是否成功
```bash
node --version
npm --version
```

# 全局安装 Claude Code
```bash
npm install -g @anthropic-ai/claude-code
```

安装完成后，输入以下命令检查是否安装成功：
```bash
claude --version
```

# 环境变量配置

```
export ANTHROPIC_BASE_URL="http://39.106.25.20:3600/api
export ANTHROPIC_AUTH_TOKEN="你的API密钥"
```





