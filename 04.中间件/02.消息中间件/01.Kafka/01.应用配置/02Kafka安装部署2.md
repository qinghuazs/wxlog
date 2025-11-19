---
title: Kafka安装部署
date: 2025-04-07
permalink: /kafka/Kafka安装部署2/
tags:
  - Kafka
categories:
  - Kafka
---

## 环境准备

### 前置条件
- JDK 1.8+
- ZooKeeper 3.4.6+
- 操作系统：Linux（推荐CentOS 7+）
- 内存：最少4GB RAM
- 磁盘：根据数据量规划，建议50GB以上

### 下载安装包
1. 访问[Kafka官网](https://kafka.apache.org/downloads)下载最新稳定版本
2. 选择二进制版本，例如：kafka_2.13-3.9.0.tgz

解压

```bash
tar -xzf kafka_2.13-3.9.0.tgz
```
重命名
```bash
mv kafka_2.13-3.9.0 kafka390
```
进入目录
```bash
cd kafka390
```

## 单机版启动方式

### KRaft模式



### Zookeeper模式

1. 启动ZooKeeper服务：
```bash
bin/zookeeper-server-start.sh config/zookeeper.properties
```

2. 启动Kafka服务：
```bash
bin/kafka-server-start.sh config/server.properties
```







