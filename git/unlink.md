---
title: 本地项目取消git仓库关联
date: 2025-04-03
categories:
  - Git
  - Tools
---

## 本地项目取消git版本控制

切换到项目根目录下，删除 .git 文件夹即可

或者在项目根目录打开终端或者命令行工具

```bash
ls -a 

rm -rf .git 
```

## 本地项目取消与git仓库的关联

在项目根目录打开终端或者命令行工具

```bash
git remote remove origin
```
检查是否取消成功

```bash
git remote -v
```
如果输出为空，则表示成功
