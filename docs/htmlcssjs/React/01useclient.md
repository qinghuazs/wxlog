---
title: useClient
date: 2025-04-03
permalink: /react/useclient/
categories:
  - React
  - Frontend
---

在 React 中，'use client' 是一个特殊的指令，用于指示某个文件或组件应该在客户端运行，而不是在服务器端运行。它主要用于 Next.js 等框架中，帮助开发者更好地控制代码的执行环境，特别是在 Server Components（服务器组件）和 Client Components（客户端组件）的场景下。

## Server Components 和 Client Components

在 Next.js 13 及更高版本中，引入了 Server Components 的概念。Server Components 是一种可以在服务器端渲染的组件，而 Client Components 是在客户端运行的组件。默认情况下，Next.js 会尝试将组件渲染为 Server Components，以提高性能和优化服务器端渲染。

Server Components：在服务器端渲染，不包含交互逻辑，主要用于展示数据。

Client Components：在客户端运行，可以包含交互逻辑（如事件处理、状态管理等）。

### 性能对比

Server Components 和 Client Components 在性能方面有显著差异：

1. **初始加载性能**
   - Server Components：更快的页面加载，更小的 JavaScript 包体积
   - Client Components：需要等待 JavaScript 下载和执行

2. **数据获取**
   - Server Components：直接在服务器端获取数据，减少客户端请求
   - Client Components：需要在客户端发起请求，可能增加延迟

3. **SEO 友好度**
   - Server Components：更好的 SEO 支持，搜索引擎可直接爬取内容
   - Client Components：需要额外配置才能支持 SEO

### 最佳实践

1. **Server Components 适用场景**：
   - 静态内容展示
   - 数据库查询展示
   - SEO 关键页面
   ```jsx
   // pages/blog/[id].tsx
   async function BlogPost({ id }) {
     const post = await db.post.findUnique({ where: { id } });
     return <article>{post.content}</article>;
   }
   ```

2. **Client Components 适用场景**：
   - 用户交互界面
   - 状态管理
   - 浏览器 API 调用

## 使用 'use client'

当你希望某个组件在客户端运行时，可以在文件的顶部添加 'use client' 指令。这样，Next.js 会将该组件及其依赖的代码打包为客户端代码，可使用 useState 来管理状态。

### 实际示例

1. **交互式导航栏**
```jsx
'use client';

import { useState } from 'react';

export default function NavBar() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <nav>
      <button onClick={() => setIsOpen(!isOpen)}>
        {isOpen ? '关闭' : '打开'} 菜单
      </button>
      {isOpen && (
        <ul>
          <li>首页</li>
          <li>关于</li>
        </ul>
      )}
    </nav>
  );
}
```

2. **表单处理**
```jsx
'use client';

import { useState } from 'react';

export default function ContactForm() {
  const [formData, setFormData] = useState({ name: '', email: '' });

  const handleSubmit = async (e) => {
    e.preventDefault();
    await fetch('/api/contact', {
      method: 'POST',
      body: JSON.stringify(formData)
    });
  };

  return (
    <form onSubmit={handleSubmit}>
      <input
        type="text"
        value={formData.name}
        onChange={(e) => setFormData({ ...formData, name: e.target.value })}
      />
      <button type="submit">提交</button>
    </form>
  );
}
```




## 注意事项

使用 'use client' 指令时，需要注意以下几点：

1. 'use client' 指令只能在文件的顶部使用，且是文件的第一行代码，不能在函数内部使用。
2. 如果组件包含交互逻辑（如事件处理、状态管理等），应该使用 'use client'。
3. 如果组件仅用于展示数据，且不需要交互，可以不使用 'use client'，让其作为 Server Component。
4. 'use client' 指令只能用于 Next.js 13 及更高版本。
5. 'use client' 指令不会影响服务器端代码的执行，只会影响客户端代码的打包和执行。
6. 'use client' 指令不会影响组件的渲染方式，它只是告诉 Next.js 该组件应该在客户端运行。

## 常见问题和故障排除

1. **混合使用时的问题**
   - 问题：Server Component 中导入 Client Component 报错
   - 解决：确保 Client Component 被正确标记且只在客户端组件中使用客户端特有的 API

2. **状态管理问题**
   - 问题：Server Component 无法使用 useState
   - 解决：将需要状态管理的部分提取为单独的 Client Component

3. **数据获取问题**
   - 问题：Client Component 中的数据获取导致瀑布流
   - 解决：考虑使用 React Suspense 或将数据获取移至 Server Component

## 与其他框架的对比

1. **Remix**
   - 同样支持服务器端渲染
   - 使用不同的数据加载策略
   - 不需要显式声明客户端组件

2. **Gatsby**
   - 静态站点生成（SSG）为主
   - 构建时生成页面
   - 不区分服务器和客户端组件
