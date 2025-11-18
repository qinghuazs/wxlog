---
title: 12 Multimodal Interaction
date: 2025-11-18
permalink: /ai/claude-code/architecture-12-multimodal-interaction.html
categories:
  - AI
---

# 第12篇:多模态交互实现

## 引言

在现代AI应用中,仅仅处理文本已经远远不够。开发者经常需要:
- 分析UI截图并生成对应的代码
- 解析PDF文档提取关键信息
- 理解代码高亮后的显示效果
- 预览Markdown文档的渲染结果

Claude Code通过强大的多模态能力,支持图像、PDF、代码渲染等多种媒体类型的处理,让AI能够"看到"和"理解"更丰富的信息。

### 本文目标

学完本文,你将掌握:
- Claude Vision API的使用方法
- 截图分析和OCR实现
- PDF文档的智能解析
- 代码高亮和Markdown渲染
- 多模态上下文管理策略


## 二、图像处理能力(Claude Vision API)

### 2.1 Vision API基础

Claude 3.5 Sonnet支持原生的图像理解能力,可以:
- 识别图像内容
- 提取图像中的文本(OCR)
- 理解UI布局
- 分析图表和数据可视化

**API调用格式**:

```typescript
import Anthropic from '@anthropic-ai/sdk';
import fs from 'fs';

// 图像内容类型定义
interface ImageContent {
  type: 'image';
  source: {
    type: 'base64';
    media_type: 'image/png' | 'image/jpeg' | 'image/gif' | 'image/webp';
    data: string; // Base64编码的图像数据
  };
}

// Vision API调用
class VisionAPI {
  private client: Anthropic;

  constructor(apiKey: string) {
    this.client = new Anthropic({ apiKey });
  }

  /**
   * 分析图像
   * @param imagePath 图像文件路径
   * @param prompt 分析提示词
   */
  async analyzeImage(imagePath: string, prompt: string): Promise<string> {
    // 读取图像文件
    const imageBuffer = fs.readFileSync(imagePath);
    const base64Image = imageBuffer.toString('base64');

    // 检测图像类型
    const mediaType = this.detectMediaType(imagePath);

    // 构造消息
    const response = await this.client.messages.create({
      model: 'claude-3-5-sonnet-20250929',
      max_tokens: 4096,
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'image',
              source: {
                type: 'base64',
                media_type: mediaType,
                data: base64Image
              }
            },
            {
              type: 'text',
              text: prompt
            }
          ]
        }
      ]
    });

    return response.content[0].type === 'text'
      ? response.content[0].text
      : '';
  }

  /**
   * 检测图像MIME类型
   */
  private detectMediaType(filePath: string):
    'image/png' | 'image/jpeg' | 'image/gif' | 'image/webp' {
    const ext = filePath.toLowerCase().split('.').pop();

    switch (ext) {
      case 'png': return 'image/png';
      case 'jpg':
      case 'jpeg': return 'image/jpeg';
      case 'gif': return 'image/gif';
      case 'webp': return 'image/webp';
      default: return 'image/png';
    }
  }

  /**
   * 批量分析多张图像
   */
  async analyzeMultipleImages(
    images: Array<{ path: string; description?: string }>,
    prompt: string
  ): Promise<string> {
    const imageContents: Array<ImageContent | { type: 'text'; text: string }> = [];

    // 添加所有图像
    for (const img of images) {
      const imageBuffer = fs.readFileSync(img.path);
      const base64Image = imageBuffer.toString('base64');
      const mediaType = this.detectMediaType(img.path);

      imageContents.push({
        type: 'image',
        source: {
          type: 'base64',
          media_type: mediaType,
          data: base64Image
        }
      });

      // 如果有描述,添加文本说明
      if (img.description) {
        imageContents.push({
          type: 'text',
          text: `[图像说明: ${img.description}]`
        });
      }
    }

    // 添加分析提示
    imageContents.push({
      type: 'text',
      text: prompt
    });

    const response = await this.client.messages.create({
      model: 'claude-3-5-sonnet-20250929',
      max_tokens: 8000,
      messages: [
        {
          role: 'user',
          content: imageContents
        }
      ]
    });

    return response.content[0].type === 'text'
      ? response.content[0].text
      : '';
  }
}
```

### 2.2 图像预处理

在某些情况下,需要对图像进行预处理以提升识别效果:

```typescript
import sharp from 'sharp';

/**
 * 图像预处理工具
 */
class ImagePreprocessor {
  /**
   * 调整图像大小(保持宽高比)
   */
  async resize(
    inputPath: string,
    outputPath: string,
    maxWidth: number = 1568,
    maxHeight: number = 1568
  ): Promise<void> {
    await sharp(inputPath)
      .resize(maxWidth, maxHeight, {
        fit: 'inside',
        withoutEnlargement: true
      })
      .toFile(outputPath);
  }

  /**
   * 转换图像格式
   */
  async convert(
    inputPath: string,
    outputPath: string,
    format: 'png' | 'jpeg' | 'webp'
  ): Promise<void> {
    const image = sharp(inputPath);

    switch (format) {
      case 'png':
        await image.png().toFile(outputPath);
        break;
      case 'jpeg':
        await image.jpeg({ quality: 90 }).toFile(outputPath);
        break;
      case 'webp':
        await image.webp({ quality: 90 }).toFile(outputPath);
        break;
    }
  }

  /**
   * 增强对比度(提升OCR效果)
   */
  async enhanceContrast(
    inputPath: string,
    outputPath: string
  ): Promise<void> {
    await sharp(inputPath)
      .normalize() // 归一化
      .sharpen()   // 锐化
      .toFile(outputPath);
  }

  /**
   * 转换为灰度图(减小文件大小)
   */
  async toGrayscale(
    inputPath: string,
    outputPath: string
  ): Promise<void> {
    await sharp(inputPath)
      .grayscale()
      .toFile(outputPath);
  }

  /**
   * 获取图像元数据
   */
  async getMetadata(imagePath: string): Promise<sharp.Metadata> {
    return await sharp(imagePath).metadata();
  }
}
```


## 四、PDF文档解析

### 4.1 PDF处理基础

Claude可以直接处理PDF文档,提取文本和理解布局:

```typescript
import pdf from 'pdf-parse';
import { PDFDocument } from 'pdf-lib';

/**
 * PDF文档处理器
 */
class PDFProcessor {
  private vision: VisionAPI;

  constructor(apiKey: string) {
    this.vision = new VisionAPI(apiKey);
  }

  /**
   * 提取PDF文本
   */
  async extractText(pdfPath: string): Promise<{
    text: string;
    pages: number;
    metadata: any;
  }> {
    const dataBuffer = fs.readFileSync(pdfPath);
    const data = await pdf(dataBuffer);

    return {
      text: data.text,
      pages: data.numpages,
      metadata: data.metadata
    };
  }

  /**
   * 将PDF转换为图像进行分析
   */
  async analyzePDFPage(
    pdfPath: string,
    pageNumber: number,
    prompt: string
  ): Promise<string> {
    // 1. 提取PDF页面为图像
    const imagePath = await this.pdfPageToImage(pdfPath, pageNumber);

    // 2. 使用Vision API分析
    const result = await this.vision.analyzeImage(imagePath, prompt);

    // 3. 清理临时文件
    fs.unlinkSync(imagePath);

    return result;
  }

  /**
   * 将PDF页面转换为图像
   */
  private async pdfPageToImage(
    pdfPath: string,
    pageNumber: number
  ): Promise<string> {
    // 使用pdf-poppler或ghostscript转换
    // 这里简化处理,实际需要调用外部工具

    const outputPath = `/tmp/pdf-page-${Date.now()}.png`;

    // 调用pdftoppm或gs命令
    const { execSync } = require('child_process');
    execSync(
      `pdftoppm -f ${pageNumber} -l ${pageNumber} -png "${pdfPath}" "${outputPath.replace('.png', '')}"`
    );

    return outputPath;
  }

  /**
   * 智能提取PDF中的表格
   */
  async extractTables(pdfPath: string): Promise<Array<{
    page: number;
    headers: string[];
    rows: string[][];
  }>> {
    const { numpages } = await this.extractText(pdfPath);
    const tables: Array<{
      page: number;
      headers: string[];
      rows: string[][];
    }> = [];

    // 遍历每一页
    for (let page = 1; page <= numpages; page++) {
      const prompt = `
请识别这一页中的所有表格。

以JSON数组格式输出:
\`\`\`json
[
  {
    "headers": ["列1", "列2"],
    "rows": [["数据1", "数据2"]]
  }
]
\`\`\`
`;

      const result = await this.analyzePDFPage(pdfPath, page, prompt);

      try {
        const jsonMatch = result.match(/```json\n([\s\S]*?)```/);
        if (jsonMatch) {
          const pageTables = JSON.parse(jsonMatch[1]);
          pageTables.forEach((table: any) => {
            tables.push({ page, ...table });
          });
        }
      } catch (error) {
        console.error(`Failed to parse tables on page ${page}:`, error);
      }
    }

    return tables;
  }

  /**
   * 提取PDF文档结构
   */
  async extractStructure(pdfPath: string): Promise<{
    title: string;
    sections: Array<{
      title: string;
      page: number;
      content: string;
    }>;
    references: string[];
  }> {
    // 提取完整文本
    const { text, pages } = await this.extractText(pdfPath);

    // 使用Claude分析文档结构
    const client = new Anthropic({
      apiKey: process.env.ANTHROPIC_API_KEY!
    });

    const response = await client.messages.create({
      model: 'claude-3-5-sonnet-20250929',
      max_tokens: 8000,
      messages: [
        {
          role: 'user',
          content: `
请分析以下PDF文档的文本内容,提取:
1. 文档标题
2. 章节结构(标题和页码)
3. 参考文献

文档内容:
${text.slice(0, 50000)} // 限制长度

以JSON格式输出:
\`\`\`json
{
  "title": "文档标题",
  "sections": [
    {
      "title": "章节标题",
      "page": 1,
      "content": "章节摘要"
    }
  ],
  "references": ["参考文献1"]
}
\`\`\`
`
        }
      ]
    });

    const resultText = response.content[0].type === 'text'
      ? response.content[0].text
      : '';

    const jsonMatch = resultText.match(/```json\n([\s\S]*?)```/);
    if (!jsonMatch) {
      throw new Error('Failed to extract document structure');
    }

    return JSON.parse(jsonMatch[1]);
  }
}
```

### 4.2 Read工具的PDF支持

Claude Code的Read工具原生支持PDF:

```typescript
/**
 * 增强的Read工具,支持PDF
 */
class EnhancedReadTool {
  private pdfProcessor: PDFProcessor;

  constructor(apiKey: string) {
    this.pdfProcessor = new PDFProcessor(apiKey);
  }

  /**
   * 读取文件(支持PDF)
   */
  async read(filePath: string, options?: {
    offset?: number;
    limit?: number;
  }): Promise<string> {
    const ext = filePath.toLowerCase().split('.').pop();

    if (ext === 'pdf') {
      return this.readPDF(filePath, options);
    } else {
      return this.readTextFile(filePath, options);
    }
  }

  /**
   * 读取PDF文件
   */
  private async readPDF(
    filePath: string,
    options?: { offset?: number; limit?: number }
  ): Promise<string> {
    const { text, pages, metadata } = await this.pdfProcessor.extractText(filePath);

    // 按页分割
    const pageTexts = this.splitByPages(text, pages);

    // 应用偏移和限制
    const offset = options?.offset || 0;
    const limit = options?.limit || pageTexts.length;
    const selectedPages = pageTexts.slice(offset, offset + limit);

    // 格式化输出
    let result = `PDF文档: ${filePath}\n`;
    result += `总页数: ${pages}\n`;
    result += `作者: ${metadata?.Author || 'Unknown'}\n`;
    result += `创建时间: ${metadata?.CreationDate || 'Unknown'}\n\n`;
    result += `显示页面: ${offset + 1} - ${offset + selectedPages.length}\n`;
    result += '='.repeat(50) + '\n\n';

    selectedPages.forEach((pageText, idx) => {
      result += `[第 ${offset + idx + 1} 页]\n`;
      result += pageText;
      result += '\n\n' + '-'.repeat(50) + '\n\n';
    });

    return result;
  }

  /**
   * 按页分割PDF文本
   */
  private splitByPages(text: string, pages: number): string[] {
    // 简单的页面分割策略
    // 实际PDF库可能提供更精确的分页信息
    const lines = text.split('\n');
    const linesPerPage = Math.ceil(lines.length / pages);

    const pageTexts: string[] = [];
    for (let i = 0; i < pages; i++) {
      const start = i * linesPerPage;
      const end = Math.min((i + 1) * linesPerPage, lines.length);
      pageTexts.push(lines.slice(start, end).join('\n'));
    }

    return pageTexts;
  }

  /**
   * 读取普通文本文件
   */
  private async readTextFile(
    filePath: string,
    options?: { offset?: number; limit?: number }
  ): Promise<string> {
    const content = fs.readFileSync(filePath, 'utf-8');
    const lines = content.split('\n');

    const offset = options?.offset || 0;
    const limit = options?.limit || lines.length;
    const selectedLines = lines.slice(offset, offset + limit);

    return selectedLines
      .map((line, idx) => `${offset + idx + 1}\t${line}`)
      .join('\n');
  }
}
```


## 六、Markdown实时预览

### 6.1 Markdown渲染

```typescript
import MarkdownIt from 'markdown-it';
import hljs from 'highlight.js';

/**
 * Markdown渲染器
 */
class MarkdownRenderer {
  private md: MarkdownIt;

  constructor() {
    this.md = new MarkdownIt({
      html: true,
      linkify: true,
      typographer: true,
      highlight: (str, lang) => {
        if (lang && hljs.getLanguage(lang)) {
          try {
            return hljs.highlight(str, { language: lang }).value;
          } catch (error) {
            console.error('Highlight error:', error);
          }
        }
        return '';
      }
    });

    // 添加插件
    this.md.use(require('markdown-it-emoji'));
    this.md.use(require('markdown-it-footnote'));
    this.md.use(require('markdown-it-task-lists'));
    this.md.use(require('markdown-it-anchor'));
    this.md.use(require('markdown-it-table-of-contents'));
  }

  /**
   * 渲染Markdown为HTML
   */
  render(markdown: string): string {
    return this.md.render(markdown);
  }

  /**
   * 渲染Markdown为纯文本
   */
  renderToText(markdown: string): string {
    // 移除Markdown语法,只保留纯文本
    let text = markdown;

    // 移除代码块
    text = text.replace(/```[\s\S]*?```/g, '[代码块]');

    // 移除行内代码
    text = text.replace(/`[^`]+`/g, match => match.slice(1, -1));

    // 移除链接
    text = text.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1');

    // 移除图片
    text = text.replace(/!\[([^\]]*)\]\([^)]+\)/g, '[图片: $1]');

    // 移除标题标记
    text = text.replace(/^#{1,6}\s+/gm, '');

    // 移除加粗和斜体
    text = text.replace(/\*\*([^*]+)\*\*/g, '$1');
    text = text.replace(/\*([^*]+)\*/g, '$1');
    text = text.replace(/__([^_]+)__/g, '$1');
    text = text.replace(/_([^_]+)_/g, '$1');

    return text;
  }

  /**
   * 提取Markdown目录
   */
  extractTOC(markdown: string): Array<{
    level: number;
    text: string;
    anchor: string;
  }> {
    const headings: Array<{ level: number; text: string; anchor: string }> = [];
    const lines = markdown.split('\n');

    lines.forEach(line => {
      const match = line.match(/^(#{1,6})\s+(.+)$/);
      if (match) {
        const level = match[1].length;
        const text = match[2];
        const anchor = text
          .toLowerCase()
          .replace(/[^\w\s-]/g, '')
          .replace(/\s+/g, '-');

        headings.push({ level, text, anchor });
      }
    });

    return headings;
  }

  /**
   * 生成完整的HTML页面
   */
  renderToHTMLPage(markdown: string, options?: {
    title?: string;
    css?: string;
    theme?: 'light' | 'dark';
  }): string {
    const {
      title = 'Markdown Preview',
      css = '',
      theme = 'light'
    } = options || {};

    const content = this.render(markdown);
    const toc = this.extractTOC(markdown);

    return `
<!DOCTYPE html>
<html lang="zh-CN" data-theme="${theme}">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${title}</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.2.0/github-markdown.min.css">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/styles/github${theme === 'dark' ? '-dark' : ''}.min.css">
  <style>
    body {
      margin: 0;
      padding: 20px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background-color: ${theme === 'dark' ? '#0d1117' : '#ffffff'};
    }
    .container {
      max-width: 980px;
      margin: 0 auto;
    }
    .markdown-body {
      padding: 45px;
      background-color: ${theme === 'dark' ? '#0d1117' : '#ffffff'};
    }
    .toc {
      position: fixed;
      right: 20px;
      top: 20px;
      width: 200px;
      max-height: 80vh;
      overflow-y: auto;
      background: ${theme === 'dark' ? '#161b22' : '#f6f8fa'};
      border-radius: 6px;
      padding: 16px;
    }
    .toc a {
      text-decoration: none;
      color: ${theme === 'dark' ? '#58a6ff' : '#0969da'};
      display: block;
      padding: 4px 0;
    }
    ${css}
  </style>
</head>
<body>
  <div class="toc">
    <h3>目录</h3>
    ${toc.map(item => `
      <a href="#${item.anchor}" style="padding-left: ${(item.level - 1) * 12}px">
        ${item.text}
      </a>
    `).join('')}
  </div>
  <div class="container">
    <div class="markdown-body">
      ${content}
    </div>
  </div>
</body>
</html>
`;
  }
}
```

### 6.2 Markdown文件监听和实时预览

```typescript
import chokidar from 'chokidar';
import express from 'express';
import { Server } from 'socket.io';
import http from 'http';

/**
 * Markdown实时预览服务器
 */
class MarkdownPreviewServer {
  private renderer: MarkdownRenderer;
  private app: express.Application;
  private server: http.Server;
  private io: Server;
  private watcher: chokidar.FSWatcher | null = null;

  constructor() {
    this.renderer = new MarkdownRenderer();
    this.app = express();
    this.server = http.createServer(this.app);
    this.io = new Server(this.server);

    this.setupRoutes();
    this.setupWebSocket();
  }

  /**
   * 设置HTTP路由
   */
  private setupRoutes(): void {
    this.app.get('/', (req, res) => {
      res.send(`
<!DOCTYPE html>
<html>
<head>
  <title>Markdown Preview</title>
  <script src="/socket.io/socket.io.js"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.2.0/github-markdown.min.css">
  <style>
    body { margin: 0; padding: 20px; }
    .markdown-body { max-width: 980px; margin: 0 auto; padding: 45px; }
  </style>
</head>
<body>
  <div class="markdown-body" id="content">
    Loading...
  </div>
  <script>
    const socket = io();
    socket.on('update', (html) => {
      document.getElementById('content').innerHTML = html;
    });
  </script>
</body>
</html>
      `);
    });
  }

  /**
   * 设置WebSocket连接
   */
  private setupWebSocket(): void {
    this.io.on('connection', (socket) => {
      console.log('Client connected');

      socket.on('disconnect', () => {
        console.log('Client disconnected');
      });
    });
  }

  /**
   * 启动预览服务器
   */
  start(filePath: string, port: number = 3000): void {
    // 读取并渲染初始内容
    this.updatePreview(filePath);

    // 监听文件变化
    this.watcher = chokidar.watch(filePath, {
      persistent: true,
      ignoreInitial: true
    });

    this.watcher.on('change', () => {
      console.log('File changed, updating preview...');
      this.updatePreview(filePath);
    });

    // 启动服务器
    this.server.listen(port, () => {
      console.log(`Preview server running at http://localhost:${port}`);
      console.log(`Watching: ${filePath}`);
    });
  }

  /**
   * 更新预览内容
   */
  private updatePreview(filePath: string): void {
    try {
      const markdown = fs.readFileSync(filePath, 'utf-8');
      const html = this.renderer.render(markdown);

      // 通过WebSocket推送更新
      this.io.emit('update', html);
    } catch (error) {
      console.error('Failed to update preview:', error);
    }
  }

  /**
   * 停止服务器
   */
  stop(): void {
    if (this.watcher) {
      this.watcher.close();
    }
    this.server.close();
  }
}
```


## 八、音频处理(未来支持)

虽然目前Claude Code主要专注于文本、图像和文档,但音频处理也是多模态的重要方向:

```typescript
/**
 * 音频处理工具(概念实现)
 */
class AudioProcessor {
  /**
   * 语音转文字(Speech-to-Text)
   */
  async transcribe(audioPath: string): Promise<{
    text: string;
    confidence: number;
    language: string;
  }> {
    // 使用Whisper API或其他STT服务
    // 这里是概念实现
    return {
      text: '转录的文字内容',
      confidence: 0.95,
      language: 'zh-CN'
    };
  }

  /**
   * 文字转语音(Text-to-Speech)
   */
  async synthesize(text: string, options?: {
    voice?: string;
    language?: string;
    speed?: number;
  }): Promise<Buffer> {
    // 使用TTS服务生成音频
    return Buffer.from([]);
  }

  /**
   * 音频分析
   */
  async analyze(audioPath: string): Promise<{
    duration: number;
    sampleRate: number;
    channels: number;
    format: string;
  }> {
    // 分析音频文件
    return {
      duration: 0,
      sampleRate: 44100,
      channels: 2,
      format: 'mp3'
    };
  }
}
```


## 十、最佳实践

### 10.1 图像处理最佳实践

**1. 图像大小优化**

```typescript
// ✅ 好的做法:调整图像大小以节省Token
const preprocessor = new ImagePreprocessor();
await preprocessor.resize(
  originalPath,
  processedPath,
  1568, // Claude Vision最佳宽度
  1568
);

// ❌ 不好的做法:直接使用超大图像
// 会浪费Token,影响性能
```

**2. 批量处理策略**

```typescript
// ✅ 好的做法:批量处理相关图像
await vision.analyzeMultipleImages([
  { path: 'screen1.png', description: '首页' },
  { path: 'screen2.png', description: '详情页' }
], '请对比这两个页面的布局差异');

// ❌ 不好的做法:多次单独调用
// 会丢失上下文,增加成本
```

**3. 提示词优化**

```typescript
// ✅ 好的做法:具体、明确的提示词
const prompt = `
请分析这个登录页面的UI设计,重点关注:
1. 表单布局是否合理
2. 按钮位置和样式
3. 是否符合无障碍标准
4. 配色方案

请以JSON格式输出分析结果。
`;

// ❌ 不好的做法:模糊的提示词
const prompt = '分析这个页面'; // 太笼统,结果不可控
```

### 10.2 PDF处理最佳实践

**1. 大型PDF分页处理**

```typescript
// ✅ 好的做法:分页处理大型PDF
async function processBigPDF(pdfPath: string) {
  const { pages } = await pdfProcessor.extractText(pdfPath);

  for (let i = 1; i <= pages; i += 10) {
    // 每次处理10页
    const result = await processBatch(pdfPath, i, Math.min(i + 9, pages));
    await saveBatch(result);
  }
}

// ❌ 不好的做法:一次性处理所有页面
// 可能超出Token限制
```

**2. 缓存提取结果**

```typescript
// ✅ 好的做法:缓存PDF文本提取结果
class CachedPDFProcessor {
  private cache: Map<string, { text: string; mtime: number }> = new Map();

  async extractText(pdfPath: string) {
    const stat = fs.statSync(pdfPath);
    const cached = this.cache.get(pdfPath);

    if (cached && cached.mtime === stat.mtimeMs) {
      return cached; // 返回缓存
    }

    const result = await this.actualExtract(pdfPath);
    this.cache.set(pdfPath, { ...result, mtime: stat.mtimeMs });
    return result;
  }
}
```

### 10.3 代码渲染最佳实践

**1. 语言自动检测**

```typescript
// ✅ 好的做法:自动检测语言
const language = highlighter.detectLanguage(code, fileName);
const highlighted = highlighter.highlight(code, language);

// ❌ 不好的做法:硬编码语言
const highlighted = highlighter.highlight(code, 'javascript');
// 如果实际是Python代码,高亮会错误
```

**2. 增量渲染**

```typescript
// ✅ 好的做法:只渲染可见部分
class LazyCodeRenderer {
  renderVisible(code: string, startLine: number, endLine: number) {
    const lines = code.split('\n');
    const visibleLines = lines.slice(startLine, endLine);
    return this.highlight(visibleLines.join('\n'));
  }
}

// ❌ 不好的做法:渲染整个大文件
// 对于数万行代码,会很慢
```

### 10.4 多模态上下文管理

**1. Token预算分配**

```typescript
// ✅ 好的做法:为不同模态分配Token预算
class MultimodalContextManager {
  private readonly TOKEN_BUDGET = 150000;
  private readonly BUDGET_ALLOCATION = {
    text: 0.5,    // 50%给文本
    image: 0.3,   // 30%给图像
    pdf: 0.2      // 20%给PDF
  };

  allocate(content: MultimodalContent) {
    const textBudget = this.TOKEN_BUDGET * this.BUDGET_ALLOCATION.text;
    const imageBudget = this.TOKEN_BUDGET * this.BUDGET_ALLOCATION.image;
    // ...
  }
}
```

**2. 优先级排序**

```typescript
// ✅ 好的做法:根据相关性排序
interface MultimodalItem {
  type: 'text' | 'image' | 'pdf';
  content: any;
  relevance: number; // 0-1
  tokens: number;
}

function prioritize(items: MultimodalItem[]): MultimodalItem[] {
  return items.sort((a, b) => {
    // 优先级 = 相关性 / Token数(性价比)
    const aPriority = a.relevance / a.tokens;
    const bPriority = b.relevance / b.tokens;
    return bPriority - aPriority;
  });
}
```


## 十二、总结

多模态交互是Claude Code的核心竞争力之一,通过强大的Vision API和文档处理能力,实现了:

✅ **图像理解**: 截图分析、OCR、UI转代码
✅ **文档解析**: PDF提取、结构分析、表格识别
✅ **代码渲染**: 语法高亮、行号、主题切换
✅ **实时预览**: Markdown、Jupyter Notebook
✅ **上下文融合**: 多模态信息综合理解

**核心要点回顾**:

1. **Vision API使用**: Base64编码图像 + 文本提示词
2. **PDF处理**: 文本提取 + 页面图像分析
3. **代码高亮**: Prism.js等库实现语法高亮
4. **Markdown渲染**: markdown-it + 文件监听
5. **最佳实践**: 图像预处理、Token预算管理、缓存优化

**下一步建议**:

1. 实践图像转代码功能
2. 尝试解析技术PDF文档
3. 开发Markdown实时预览工具
4. 探索更多多模态应用场景


## 下一篇预告

在下一篇文章中,我们将深入探讨**[性能优化深度剖析](./13-性能优化深度剖析.md)**,包括:
- 请求批处理和合并策略
- 智能缓存系统设计
- Token使用优化技巧
- 内存和网络优化

敬请期待!

---

**如果觉得这篇文章对你有帮助,欢迎分享给更多的朋友!**
