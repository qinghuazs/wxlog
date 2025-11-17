---
title: Git 推送本地提交到远程仓库
date: 2025/03/28
categories:
  - Git
  - Tools
---

### 查看当前状态
```bash
git status
```
### 添加更改到暂存区
```bash
git add .  # 添加所有更改

git add 文件名  # 添加特定文件
```

### 提交更改
```bash
git commit -m "提交说明"
```

### 查看远程仓库信息
```bash
git remote -v
```

### 推送到远程仓库
```bash
git push origin main  # main 是分支名，根据实际分支名替换
```

如果是首次推送到远程仓库，可能需要：

添加远程仓库
```bash
git remote add origin 仓库地址
```

设置上游分支并推送
```bash
git push -u origin main
```

如果遇到冲突：

先拉取远程更新
```bash
git pull origin main
```
解决冲突后再次提交和推送
```bash
git add .
git commit -m "解决冲突"
git push origin main
```