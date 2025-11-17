---
title: JVM 问题排查常用命令
date: 2024/04/17
categories:
  - Java
  - JVM
---

### 查看进程pid

```shell
jps
```

### 打印线程堆栈信息

```java
jstack 82 > /home/dump82
```

### 统计各个状态的线程个数

```java
grep java.lang.Thread.State dump82 | awk '{print $2$3$4$5}' | sort | uniq -c
```

