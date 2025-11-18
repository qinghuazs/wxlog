# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 文件信息

完成文档编写后，需要补充文档的具体信息，类似：
```
---
title: 比特币系统用到的数据结构
date: 2025-09-29
permalink: /blog/bitcoin-data-structures.html
categories:
  - Blockchain
  - Bitcoin
---
```
permalink都以 `.html` 结尾

## 完成文档编程后，需要在 .vitepress/config.mts 中添加导航连接，并使用rewrites将中文文件名导航到英文 URL，英文 URL 和源文件中的 permalink 保持一直。


## Project Overview

This is a VitePress-based technical blog called "冬眠日记" (Winter Sleep Diary). It contains comprehensive documentation covering backend development (Java, Spring, Kafka, Redis, MySQL), frontend technologies (React, TailwindCSS), AI/LLM topics (LangChain, LangGraph, Claude Code, MCP), blockchain, system design, and algorithm solutions.

## Development Commands

```bash
# Start development server (default port 5173)
npm run docs:dev

# Build for production
npm run docs:build

# Preview production build
npm run docs:preview
```

## Architecture

### VitePress Configuration

- **Config**: `.vitepress/config.mts` - Main configuration with navigation, sidebar, and theme settings
- **Mermaid Support**: Uses `vitepress-plugin-mermaid` for diagram rendering
- **Markdown Theme**: Configured with `github-light` and `github-dark` themes for syntax highlighting

### Content Organization

The codebase has **dual directory structures** for content:

1. **Root-level directories** (legacy): `ai/`, `blockchain/`, `java/`, `kafka/`, `mysql/`, `redis/`, `spring/`, `springboot/`, `springcloud/`, `leetcode/`, etc.
2. **Docs directory**: `docs/` with organized subdirectories mirroring the root structure

**Important**: When editing content, check if files exist in BOTH locations. The `.vitepress/config.mts` sidebar configuration references paths under `/docs/`, so navigation is primarily driven by the `docs/` structure.

### Navigation Structure

The site has a complex multi-level navigation defined in `.vitepress/config.mts`:

- **Top Nav**: Organized by technology category (后端技术, 数据库, 前端技术, AI, 算法, 系统设计, 工具, 其他)
- **Sidebar**: Extensive sidebar configs for each section with collapsible groups
- Files are referenced with paths like `/docs/ai/langchain/README` or `/system-design/short-url`

### Special Features

1. **Mermaid Diagrams**: Full support via `vitepress-plugin-mermaid`
2. **Search**: Local search enabled via `search.provider: 'local'`
3. **Syntax Highlighting**: Shiki 2.5.0 with 300+ language support including `java`, `sql`, `bash`, `dotenv`, `yaml`, etc.

## Content Guidelines

### Adding New Articles

1. Determine the category (AI, Java, System Design, etc.)
2. Create the markdown file in the appropriate `docs/` subdirectory
3. Update `.vitepress/config.mts`:
   - Add to `nav` if creating a new top-level category
   - Add to `sidebar` configuration for the relevant path

### Syntax Highlighting

Use correct language identifiers for code blocks:
- Java: `java`
- SQL/MySQL: `sql` (no specific `mysql` identifier)
- Environment files: `dotenv` (NOT `.env`)
- Shell scripts: `bash`, `sh`, `shell`
- Spring configs: `properties` or `yaml`
- XML (Maven, MyBatis): `xml`

Avoid using unsupported identifiers (e.g., `.env`) as they will trigger fallback warnings.

### File Naming Convention

Most files use numbered prefixes for ordering:
- `00.提示词.md` or `00.目录概览.md` - Index/overview pages
- `01.xxx.md`, `02.xxx.md` - Sequential content
- English files: `week1-setup-core-concepts.md` or descriptive names

## Migration Context

The repository underwent a permalink migration (see `migrate_permalinks.py` and `migration_log.txt`). These scripts are historical and should be ignored for regular development.

## Key Configuration Details

### Sidebar Architecture

The sidebar configuration is path-based. For example:
- `/docs/ai/` path activates the AI sidebar with sections for Claude Code, CodeAgent, LangChain, LangGraph, MCP, Dify, Ollama, and 提示词
- Each section can have nested `items` arrays for multi-level navigation
- Use `collapsed: true/false` to control default expansion state

### Theme Configuration

```javascript
themeConfig: {
  nav: [...],
  sidebar: {...},
  socialLinks: [...],
  search: { provider: 'local' },
  footer: { message: '...', copyright: '...' },
  lastUpdated: { text: '最后更新于', formatOptions: {...} }
}
```

## Common Patterns

### Bilingual Content

Some content exists in both Chinese and English versions (e.g., `第1周-环境搭建与核心概念.md` and `week1-setup-core-concepts.md`). When updating content, check for parallel files.

### Collapsed Sections

Long sidebar sections (like SpringCloud, LangGraph) use `collapsed: true` to improve initial page load UX. Keep this pattern when adding similar lengthy sections.

### Index Pages

Many directories have index/overview pages:
- `00.目录概览.md`
- `README.md`
- `index.md`

These serve as landing pages for their sections.
